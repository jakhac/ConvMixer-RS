import yaml
import argparse
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from torch.nn import DataParallel as DP

from ben_dataset import *
from conv_mixer import *
from training_utils import *
from benedict import benedict

def _parse_args():
    """Parse arguments and return ArgumentParser obj

    Returns:
        ArgumentParser: ArgumentParser
    """

    parser = argparse.ArgumentParser(description="ConvMixer Parameters")

    # Training parameters
    parser.add_argument('--new_epochs', type=int, default=None,
                        help='number additional epochs to train')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='path to model directory')
    parser.add_argument('--ckpt_filename', type=str, default=None,
                        help='name of model (usually <epoch>.ckpt)')
    parser.add_argument('--dry_run', default=False,
                        help='limit ds size and epochs for testing purpose')

    return parser.parse_args()


def main():


    args = _parse_args()
    assert args.new_epochs, "Need to specify number of additional epochs to train."
    assert args.model_dir, "Need to specify model_dir."
    assert args.ckpt_filename, "Need to specify ckpt_filename."

    with open(args.model_dir + "/args.yaml", "r") as file:
        args_dict = yaml.load(file, Loader=yaml.FullLoader)

    print("Dict", args_dict)

    args.batch_size = args_dict['batch_size']
    args.ds_size = args_dict['ds_size']
    args.augmentation = args_dict['augmentation']
    args.h = args_dict['h']
    args.depth = args_dict['depth']
    args.k_size = args_dict['k_size']
    args.p_size = args_dict['p_size']
    args.k_dilation = args_dict['k_dilation']
    args.activation = args_dict['activation']
    args.n_params = args_dict['n_params']
    args.n_params_trainable = args_dict['n_params_trainable']
    args.lr = args_dict['lr']
    args.momentum = args_dict['momentum']
    args.optimizer = args_dict['optimizer']
    args.decay = None
    args.epochs = int(args_dict['epochs']) + int(args.new_epochs)
    args.save_training = args_dict['save_training']
    args.run_tests_n = args_dict['run_tests_n']
    args.run_tests = args_dict['run_tests']

    #### Path and further hparams settings ###
    # setup_paths_and_hparams(args)
    load_dotenv('../.env')
    args.BEN_LMDB_PATH = os.environ.get("BEN_LMDB_PATH")
    args.TRAIN_CSV_FILE = os.environ.get("TRAIN_CSV")
    args.VALID_CSV_FILE = os.environ.get("VAL_CSV")
    args.TEST_CSV_FILE = os.environ.get("TEST_CSV")
    args.PATH_TO_RUNS = os.environ.get("PATH_TO_RUNS")
    args.SPLIT = os.environ.get("SPLIT")

    if args.dry_run:
        args.ds_size = 40
        args.batch_size = 20
        args.lr = 0.1
        args.run_tests = True
        # args.run_tests_n = 2

    args.model_ckpt_dir = args.model_dir + '/ckpt'
    assert os.path.isdir(args.model_ckpt_dir)

    args.path_to_ckpt_file = args.model_ckpt_dir + '/' + args.ckpt_filename
    assert os.path.isfile(args.path_to_ckpt_file)


    writer = SummaryWriter(log_dir=args.model_dir)
    resume_training(args, writer)

    if args.run_tests: run_tests(args, writer)
    writer.close()


def resume_training(args, writer,):

    print("Start training ...")
    torch.manual_seed(42)

    train_loader = get_dataloader(args, args.TRAIN_CSV_FILE, True, shuffle=True)
    valid_loader = get_dataloader(args, args.VALID_CSV_FILE, True, shuffle=True)

    # Create model and move to GPU(s)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvMixer(
        10,
        args.h,
        args.depth,
        kernel_size=args.k_size,
        patch_size=args.p_size,
        n_classes=19,
        activation=args.activation,
        dilation=args.k_dilation
    )

    torch.cuda.empty_cache()
    model.to(device)
    model = DP(model)

    optimizer = get_optimizer(model, args, len(train_loader))
    criterion = nn.BCEWithLogitsLoss()

    state_dict = torch.load(args.path_to_ckpt_file)
    model.load_state_dict(state_dict['model_state'])
    optimizer.load_state_dict(state_dict['optimizer'])
    args.start_epoch = state_dict['epoch']

    # Main training loop
    val_loss_min = np.inf
    print(f'Resume main training loop from epochs {args.start_epoch} to {args.epochs}.')
    for e in range(args.start_epoch, args.epochs):

        print(f'\n[{e+1:3d}/{args.epochs:3d}]')

        # Training
        model.train()
        loss, train_yyhat = train_batches(train_loader, model, optimizer, criterion, device)
        write_metrics(writer, 'train', train_yyhat, loss, e)

        # Validation
        model.eval()
        loss, valid_yyhat = valid_batches(valid_loader, model, criterion, device)
        write_metrics(writer, 'valid', valid_yyhat, loss, e)

        # Checkpoints
        if args.save_training and loss < val_loss_min:
            print(f'\tval_loss decreased ({val_loss_min:.6f} --> {loss:.6f}). Saving this model ...')
            save_checkpoint(args, model, optimizer, e+1)
            val_loss_min = loss

    print('Finished training.\n')
    
    if args.save_training:
        print('Saving final model ...')
        save_checkpoint(args, model, optimizer, args.epochs, exist_ok=True)


def run_tests(args, writer):
    """Run the {run_test_n} best models based on validation loss
    on testdata.

    Args:
        args (ArgumentParser): args
        writer (SummaryWriter): SummaryWriter TB
    """

    print('\nStart testing.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvMixer(
        10,
        args.h,
        args.depth,
        kernel_size=args.k_size,
        patch_size=args.p_size,
        n_classes=19,
        activation=args.activation,
        dilation=args.k_dilation
    )
    criterion = nn.BCEWithLogitsLoss()

    torch.cuda.empty_cache()
    model.to(device)

    test_loader = get_dataloader(args, args.TEST_CSV_FILE, False, shuffle=False)
    ckpt_names = get_sorted_ckpt_filenames(args.model_ckpt_dir)
    model_names = list(reversed(ckpt_names))[:args.run_tests_n]
    print(f'Following epochs are tested: {model_names}')

    for model_name in model_names:

        if int(model_name) <= args.start_epoch:
            print("Skip test model. Already tested model-epoch", model_name)
            continue

        p = f'{args.model_ckpt_dir}/{model_name}.ckpt'
        load_checkpoint(model, p)

        loss, yyhat = valid_batches(test_loader, model, criterion, device)
        print(f'{model_name}.pt scores:')

        e = int(model_name)
        write_metrics(writer, 'test', yyhat, loss, e)


    print('Finished testing.')



if __name__ == '__main__':
    main()