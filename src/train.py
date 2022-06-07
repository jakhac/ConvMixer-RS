from functools import total_ordering
import yaml
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.distributed as dist
from torch.nn import DataParallel as DP

from ben_dataset import *
from conv_mixer import *
from training_utils import *


def _parse_args():
    """Parse arguments and return ArgumentParser obj

    Returns:
        ArgumentParser: ArgumentParser
    """

    parser = argparse.ArgumentParser(description="ConvMixer Parameters")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (for train, valid, test)')
    parser.add_argument('--ds_size', type=int, default=None,
                        help='limit number of overall samples (i.e. for dry runs)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of optimizer')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='one of \'SGD\', \'Adam\', \'AdamW\', \'Ranger21\', \'LAMB\'')
    parser.add_argument('--activation', type=str, default='GELU',
                        help='GELU or ReLU')

    # Config parameters
    parser.add_argument('--dry_run', default=False,
                        help='limit ds size and epochs for testing purpose')
    parser.add_argument('--exp_name', type=str, default='test',
                        help='save several runs of an experiment in one dir')
    parser.add_argument('--save_training', default=True,
                        help='save checkpoints when valid_loss decreases')
    parser.add_argument('--run_tests', type=bool, default=True,
                        help='run best models on test data')
    parser.add_argument('--run_tests_n', type=int, default=5,
                        help='test the n best models on test data')


    # Model parameters
    parser.add_argument('--h', type=int, default=128,
                        help='hidden dimension')
    parser.add_argument('--depth', type=int, default=8,
                        help='number of ConvMixer layers')
    parser.add_argument('--k_size', type=int, default=9,
                        help='kernel size in ConvMixer layers')
    parser.add_argument('--p_size', type=int, default=7,
                        help='patch size')
    parser.add_argument('--k_dilation', type=int, default=1,
                        help='dilation of convolutions in ConvMixer layers')

    return parser.parse_args()


def main():

    args = _parse_args()


    #### Path and further hparams settings ###
    setup_paths_and_hparams(args)

    writer = SummaryWriter(log_dir=args.model_dir)
    run_training(args, writer)

    if args.run_tests: run_tests(args, writer)
    writer.close()


def run_training(args, writer):

    print("Start training ...")
    torch.manual_seed(42)

    train_loader = get_dataloader(args, args.TRAIN_CSV_FILE, shuffle=True)
    valid_loader = get_dataloader(args, args.VALID_CSV_FILE, shuffle=True)

    # Create model and move to GPU(s)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvMixer(
        10,
        args.h,
        args.depth,
        kernel_size=args.k_size,
        patch_size=args.p_size,
        n_classes=19,
        activation=args.activation
    )
    optimizer = get_optimizer(model, args, len(train_loader))
    criterion = nn.BCEWithLogitsLoss()

    args.n_params = sum(p.numel() for p in model.parameters())
    args.n_params_trainable = sum(p.numel() for p in model.parameters())

    # Dump arguments into yaml file
    with open(f'{args.model_dir}/args.yaml', 'w') as outfile:
        yaml.dump(args.__dict__, outfile, default_flow_style=False)


    torch.cuda.empty_cache()
    model.to(device)
    model = DP(model)


    # Main training loop
    val_loss_min = np.inf
    print('Start main training loop.')
    for e in range(args.epochs):

        print(f'\n[{e+1:3d}/{args.epochs:3d}]')

        # Training
        model.train()
        loss, train_yyhat = train_batches(train_loader, model, optimizer, criterion)
        write_metrics(writer, 'train', train_yyhat, loss, e)

        # Validation
        model.eval()
        loss, valid_yyhat = valid_batches(valid_loader, model, criterion)
        write_metrics(writer, 'valid', valid_yyhat, loss, e)


        # Checkpoints
        if args.save_training and loss < val_loss_min:
            print(f'\tval_loss decreased ({val_loss_min:.6f} --> {loss:.6f}). Saving this model ...')
            save_checkpoint(args, model, optimizer, e+1)
            val_loss_min = loss

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, e)
            writer.add_histogram('{}.grad'.format(name), param.grad, e)


    print('Finished training.\n')
    
    if args.save_training:
        print('Saving final model ...')
        save_checkpoint(args, model, optimizer, args.epochs)




def run_tests(args, writer):
    """Run the {run_test_n} best models based on validation loss
    on testdata.

    Args:
        args (ArgumentParser): args
        writer (SummaryWriter): SummaryWriter TB
    """

    print('\nStart testing.')

    model = ConvMixer(
        10,
        args.h,
        args.depth,
        kernel_size=args.k_size,
        patch_size=args.p_size,
        n_classes=19,
        activation=args.activation
    ).cuda()
    criterion = nn.BCEWithLogitsLoss()

    test_loader = get_dataloader(args, args.TEST_CSV_FILE, False)
    ckpt_names = get_sorted_ckpt_filenames(args.model_ckpt_dir)
    model_names = list(reversed(ckpt_names))[:args.run_tests_n]
    print(f'Following epochs are tested: {model_names}')

    for model_name in model_names:
        p = f'{args.model_ckpt_dir}/{model_name}.ckpt'
        load_checkpoint(model, p)

        loss, yyhat = valid_batches(test_loader, model, criterion)
        print(f'{model_name}.pt scores:')

        e = int(model_name)
        write_metrics(writer, 'test', yyhat, loss, e)


    print('Finished testing.')



if __name__ == '__main__':
    main()