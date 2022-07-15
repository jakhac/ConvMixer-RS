import yaml
import argparse
import numpy as np

import pytorch_warmup as warmup
import torch
from torch.utils.tensorboard import SummaryWriter

from torch.nn import DataParallel as DP

from ben_dataset import *
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
    parser.add_argument('--decay', type=float, default=None,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_schedule', type=int, default=1,
                        help='Apply LR scheduling')
    parser.add_argument('--lr_warmup', type=int, default=2000,
                        help='number of warump steps')
    parser.add_argument('--lr_warmup_fn', type=str, default='linear',
                        help='warmup function, e.g. \'linear\', \'exp\'')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='one of \'SGD\', \'Adam\', \'AdamW\', \'Ranger21\', \'LAMB\'')
    parser.add_argument('--activation', type=str, default='GELU',
                        help='GELU or ReLU')
    parser.add_argument('--augmentation', type=int, default=0,
                        help='version of augmentation in [0, 4]')


    # Config parameters
    parser.add_argument('--dry_run', default=False,
                        help='limit ds size and epochs for testing purpose')
    parser.add_argument('--exp_name', type=str, default='test',
                        help='save several runs of an experiment in one dir')
    parser.add_argument('--save_training', type=int, default=1,
                        help='save checkpoints when valid_loss decreases')
    parser.add_argument('--run_tests', type=int, default=1,
                        help='run best models on test data')
    parser.add_argument('--run_tests_n', type=int, default=5,
                        help='test the n best models on test data')
    parser.add_argument('--img_size', type=int, default=120,
                        help='image resolution that is processed in models')
    parser.add_argument('--arch', type=str, default='CvMx',
                        help='specify the model, \'CvMx\', \'CvChMx\', \'ChMx\', \'ResNet[18, 50]\', \'ViT\', \'Swin\'')


    # (CvMx-) Model parameters
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
    parser.add_argument('--residual', type=int, default=1,
                        help='set 0-th bit for depthwise (=1) / set 1-th bit (=2) for pointwise / both for both (=3)')
    parser.add_argument('--drop', type=float, default=0.,
                        help='set dropout probability in mixer layers, default is 0.')


    # ViT Model parameters
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='patch embedding dimension')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='number of self-attention heads')
    parser.add_argument('--attn_drop', type=float, default=0.,
                        help='attention dropout rate')
    parser.add_argument('--path_drop', type=float, default=0.,
                        help='attention dropout rate')

    parser.add_argument('--window', type=int, default=8,
                        help='window size for SwinTransformer')


    return parser.parse_args()


def main():

    args = _parse_args()

    # Path and further hparams settings
    setup_paths_and_hparams(args)

    writer = SummaryWriter(log_dir=args.model_dir)
    run_training(args, writer)

    if args.run_tests: run_tests(args, writer)
    writer.close()


def run_training(args, writer):

    print("Start training ...")
    torch.manual_seed(42)

    train_loader = get_dataloader(args, args.TRAIN_CSV_FILE, True, shuffle=True)
    valid_loader = get_dataloader(args, args.VALID_CSV_FILE, True, shuffle=True)

    # Create model and move to GPU(s)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args)
    torch.cuda.empty_cache()
    model.to(device)
    model = DP(model)

    optimizer = get_optimizer(model, args, len(train_loader))
    num_steps = len(train_loader) * args.epochs
    lr_scheduler, warmup_scheduler = get_scheduler(args, optimizer, num_steps)
    criterion = nn.BCEWithLogitsLoss()

    args.n_params = sum(p.numel() for p in model.parameters())
    args.n_params_trainable = sum(p.numel() for p in model.parameters())

    # Dump arguments into yaml file
    with open(f'{args.model_dir}/args.yaml', 'w') as outfile:
        yaml.dump(args.__dict__, outfile, default_flow_style=False)

    # Main training loop
    val_loss_min = np.inf
    print('Start main training loop.')
    for e in range(args.epochs):

        print(f'\n[{e+1:3d}/{args.epochs:3d}]')

        # Training
        model.train()
        loss, train_yyhat = train_batches(train_loader, model, optimizer, criterion, device,
            scheduler=lr_scheduler, warmup_scheduler=warmup_scheduler)
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
        save_checkpoint(args, model, optimizer, args.epochs)


def run_tests(args, writer):
    """Run {args.run_test_n} best models based on validation loss
    on test data.

    Args:
        args (ArgumentParser): args
        writer (SummaryWriter): SummaryWriter TB
    """

    print('\nStart testing.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(args)
    criterion = nn.BCEWithLogitsLoss()

    torch.cuda.empty_cache()
    model.to(device)

    test_loader = get_dataloader(args, args.TEST_CSV_FILE, False, shuffle=False)
    ckpt_names = get_sorted_ckpt_filenames(args.model_ckpt_dir)
    model_names = list(reversed(ckpt_names))[:args.run_tests_n]
    print(f'Following epochs are tested: {model_names}')

    for model_name in model_names:
        p = f'{args.model_ckpt_dir}/{model_name}.ckpt'
        load_checkpoint(model, p)

        loss, yyhat = valid_batches(test_loader, model, criterion, device)
        print(f'{model_name}.pt scores:')

        e = int(model_name)
        write_metrics(writer, 'test', yyhat, loss, e)


    print('Finished testing.')



if __name__ == '__main__':
    main()