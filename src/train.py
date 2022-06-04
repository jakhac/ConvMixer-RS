import os
from re import M
import yaml
import argparse
import socket
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ben_dataset import *
from conv_mixer import *
from training_utils import *


def _parse_args():
    """Parse arguments and return ArgumentParser obj

    Returns:
        ArgumentParser: ArgumentParser
    """

    parser = argparse.ArgumentParser(description="ConvMixer Parameters")

    # DDP config
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--num_nodes', default='1', type=int, 
                        help='number of available nodes')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--ds_size', type=int, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='SGD or Adam')
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
    parser.add_argument('--h', type=int, default=128)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--k_size', type=int, default=9)
    parser.add_argument('--p_size', type=int, default=7)
    parser.add_argument('--k_dilation', type=int, default=1)

    return parser.parse_args()


def main():

    args = _parse_args()

    ### DDP settings ###
    args.world_size = args.num_nodes * 2 # every node TUB slurm cluster as 2 GPU devices
    args.distributed = args.world_size > 1

    if args.distributed:
        assert 'SLURM_PROCID' in os.environ

        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.gpu = 0
        args.rank = 0


    args.host = socket.gethostname()
    args.is_master = int(args.rank) == 0
    args.id_string = f'Rank {args.rank} on {args.host}@cuda:{args.gpu}:'

    print(f"Register: {args.id_string}")
    if args.is_master:
        print(f"Configured DDP settings for {args.num_nodes} nodes with 2 GPUs each ...")


    #### Path and further hparams settings ###
    setup_paths_and_hparams(args)

    # Dump arguments into yaml file
    if args.is_master:
        with open(f'{args.model_dir}/args.yaml', 'w') as outfile:
            yaml.dump(args.__dict__, outfile, default_flow_style=False)


    writer = SummaryWriter(log_dir=args.model_dir)
    run_training(args, writer)
    dist.barrier()

    if args.run_tests: run_tests(args, writer)
    writer.close()

    if args.distributed:
        dist.destroy_process_group()


def run_training(args, writer):

    print(f"{args.id_string} Start training ...")
    torch.manual_seed(42)

    ### Create model and move to GPU(s) ###
    model = ConvMixer(
        10,
        args.h,
        args.depth,
        kernel_size=args.k_size, 
        patch_size=args.p_size,
        n_classes=19, 
        activation=args.activation
    )
    optimizer = get_optimizer(model, args)
    criterion = nn.BCEWithLogitsLoss()

    args.n_params = sum(p.numel() for p in model.parameters())
    args.n_params_trainable = sum(p.numel() for p in model.parameters())


    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.distributed:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()
        model.cuda(args.gpu)
        # TODO use torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) ?
        model = DDP(model, device_ids=[args.gpu], gradient_as_bucket_view=True)
    else:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    ### Configure dataloaders ###
    train_loader = get_dataloader(args, args.TRAIN_CSV_FILE, shuffle=True)
    valid_loader = get_dataloader(args, args.VALID_CSV_FILE, shuffle=True)


    #### MAIN TRAINING LOOP ####
    val_loss_min = np.inf
    print(f'{args.id_string} Start main training loop.')
    for e in range(args.epochs):

        print(f'\n{args.id_string} [{e+1:3d}/{args.epochs:3d}]', end=" ")

        if args.distributed:
            assert train_loader.sampler is not None, "Error: no sampler set."
            train_loader.sampler.set_epoch(e)


        ### TRAINING ###
        model.train()
        train_loss, train_acc = train_batches(train_loader, model, optimizer, criterion, args.gpu)

        # Log all metrics per process
        print(f'{args.id_string} train_loss={train_loss:.4f} train_acc={train_acc:.4f}')
        writer.add_scalar(f"Per-GPU-Loss/train_gpu{args.rank}", train_loss, e)
        writer.add_scalar(f"Per-GPU-Acc/train_gpu{args.rank}", train_acc, e)

        # Gather global metrics to log average later
        global_train_loss = global_metric_avg(args, train_loss)
        global_train_acc = global_metric_avg(args, train_acc)


        ### VALIDATION ###
        model.eval()
        valid_loss, valid_acc = valid_batches(valid_loader, model, criterion, args.gpu)

        print(f'{args.id_string} valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}')
        writer.add_scalar(f"Per-GPU-Loss/valid_gpu{args.rank}", valid_loss, e)
        writer.add_scalar(f"Per-GPU-Acc/valid_gpu{args.rank}", valid_acc, e)

        global_valid_loss = global_metric_avg(args, valid_loss)
        global_valid_acc = global_metric_avg(args, valid_acc)


        ### Logging ###
        # After each epoch, the master handles logging and checkpoint-saving
        if args.is_master:
            print(f'Epoch {e} train_loss={global_train_loss:.4f} train_acc={global_train_acc:.4f}')
            print(f'Epoch {e} val_loss={global_valid_loss:.4f} val_acc={global_valid_acc:.4f}')

            writer.add_scalar("Loss/train", global_train_loss, e)
            writer.add_scalar("Loss/valid", global_valid_loss, e)
            writer.add_scalar("Acc/train", global_train_acc, e)
            writer.add_scalar("Acc/valid", valid_acc, e)

            # Save checkpoint model if validation loss improves
            if args.save_training and global_valid_loss.item() < val_loss_min:
                print(f'\tval_loss decreased ({val_loss_min:.6f} --> {global_valid_loss:.6f}). Saving this model ...')
                save_checkpoint(args, model, optimizer, e+1)
                val_loss_min = global_valid_loss


    if args.is_master:

        # TODO Add losses, accs, etc to dict
        args.val_loss_min = val_loss_min.item()
        writer.add_hparams(args.__dict__, {'Zero':0.0})

        if args.save_training:
            print('Saving final model ...')
            save_checkpoint(args, model, optimizer, args.epochs)


    print(f'{args.id_string} Finished training.')


def run_tests(args, writer):
    """Run the {run_test_n} best models based on validation loss
    on testdata.

    Args:
        args (ArgumentParser): args
        writer (SummaryWriter): SummaryWriter TB
    """

    print(f'{args.id_string} Start testing.')

    model = ConvMixer(
        10,
        args.h,
        args.depth,
        kernel_size=args.k_size, 
        patch_size=args.p_size,
        n_classes=19,
        activation=args.activation
    ).cuda(args.gpu)
    criterion = nn.BCEWithLogitsLoss()

    test_loader = get_dataloader(args, args.TEST_CSV_FILE, False)
    ckpt_names = get_sorted_ckpt_filenames(args.model_ckpt_dir)
    model_names = list(reversed(ckpt_names))[:args.run_tests_n]

    print(f'Following epochs are tested: {model_names}')

    # Forward testdata and compute metrics 
    # for top-{args.run_tests_n} models
    for model_name in model_names:
        p = f'{args.model_ckpt_dir}/{model_name}.ckpt'
        load_checkpoint(model, p, args.gpu, args.distributed)

        test_loss, test_acc = valid_batches(test_loader, model, criterion, args.gpu)
        print(f'{model_name}.pt scores test_loss={test_loss:.4f} test_acc={test_acc:.4f}')

        if args.is_master:
            writer.add_scalar("Acc/test", test_acc, int(model_name))
            writer.add_scalar("Loss/test", test_loss, int(model_name))


    print(f'{args.id_string} Finished testing.')




if __name__ == '__main__':
    main()