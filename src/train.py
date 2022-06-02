import os
from re import M
import yaml
import argparse
import socket

from pathlib import Path
from dotenv import load_dotenv
from natsort import natsorted

import numpy as np


import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
    parser.add_argument('--timestamp', type=str, default="",
                        help='unix timestamp to create unique folder')
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
    assert "WORLD_SIZE" in os.environ

    args.world_size = int(os.environ["WORLD_SIZE"])
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

    if args.is_master:
        print("Configured DDP settings ...")

    print(f"Registered {args.id_string}.")


    #### Path settings ###
    load_dotenv('../.env')
    args.BEN_LMDB_PATH = os.environ.get("BEN_LMDB_PATH")
    args.TRAIN_CSV_FILE = os.environ.get("TRAIN_CSV")
    args.VALID_CSV_FILE = os.environ.get("VAL_CSV")
    args.TEST_CSV_FILE = os.environ.get("TEST_CSV")
    args.PATH_TO_RUNS = os.environ.get("PATH_TO_RUNS")
    args.SPLIT = os.environ.get("SPLIT")

    assert os.path.isdir(args.BEN_LMDB_PATH)
    assert os.path.isfile(args.TRAIN_CSV_FILE)
    assert os.path.isfile(args.VALID_CSV_FILE)
    assert os.path.isdir(args.PATH_TO_RUNS)

    args.model_name = get_model_name(args)
    args.model_dir = f'{args.PATH_TO_RUNS}/{args.SPLIT}/{args.exp_name}/{args.model_name}'
    args.model_ckpt_dir = args.model_dir + '/ckpt'

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.model_ckpt_dir, exist_ok=True)

    # Allow dry runs for quick testing purpose
    if args.dry_run:
        args.epochs = 5
        args.ds_size = 100
        args.batch_size = 1
        args.lr = 0.1
        args.run_tests_n = 2


    # Dump arguments into yaml file
    if args.is_master:
        with open(f'{args.model_dir}/args.yaml', 'w') as outfile:
            yaml.dump(args.__dict__, outfile, default_flow_style=False)


    writer = SummaryWriter(log_dir=args.model_dir)
    model = run_training(args, writer)

    # dist.barrier()
    if args.run_tests:
        run_tests(args, writer, model)

    writer.close()
    dist.destroy_process_group()


def run_training(args, writer=None):

    print(f"{args.id_string} Start training ...")
    torch.manual_seed(42)

    ### model ###
    model = ConvMixer(
        10, args.h, args.depth, kernel_size=args.k_size, 
        patch_size=args.p_size, n_classes=19, 
        activation=args.activation
    )
    optimizer = get_optimizer(model, args)

    args.n_params = sum(p.numel() for p in model.parameters())
    args.n_params_trainable = sum(p.numel() for p in model.parameters())


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = DDP(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = DDP(model)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")


    ### Configure dataloaders for distributed training ###
    train_ds = BenDataset(args.TRAIN_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              sampler=train_sampler, drop_last=True)

    valid_ds = BenDataset(args.VALID_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)
    valid_sampler = DistributedSampler(valid_ds, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              sampler=valid_sampler, drop_last=True)


    #### MAIN TRAINING LOOP ####

    val_loss_min = np.inf
    print(f'{args.id_string} Start main training loop.')
    for e in range(args.epochs):

        print(f'\n{args.id_string} [{e+1:3d}/{args.epochs:3d}]', end=" ")

        if args.distributed:
            train_loader.sampler.set_epoch(e)


        ### TRAINING ###
        model.train()
        train_loss, train_acc = train_batches(train_loader, model, optimizer, args.gpu)

        # Log local metrics
        print(f'{args.id_string} train_loss={train_loss:.4f} train_acc={train_acc:.4f}')
        writer.add_scalar(f"Per-GPU-Loss/train_gpu{args.rank}", train_loss, e)
        writer.add_scalar(f"Per-GPU-Acc/train_gpu{args.rank}", train_acc, e)

        # Gather global metrics to log average later
        global_train_losses = torch.ones(args.world_size, dtype=torch.float32).cuda(args.gpu)
        global_train_accs = torch.ones(args.world_size, dtype=torch.float32).cuda(args.gpu)
        if args.is_master:
            dist.gather(
                torch.tensor(train_loss).cuda(args.gpu),
                gather_list=[a.to(torch.float) for a in global_train_losses]
            )

            dist.gather(
                torch.tensor(train_acc).cuda(args.gpu),
                gather_list=[a.to(torch.float) for a in global_train_accs]
            )
        else:
            dist.gather(torch.tensor(train_loss).cuda(args.gpu))
            dist.gather(torch.tensor(train_acc).cuda(args.gpu))
        
            
        ### VALIDATION ###
        model.eval()
        valid_loss, valid_acc = valid_batches(valid_loader, model, args.gpu)

        # Log local metrics
        print(f'{args.id_string} train_loss={valid_loss:.4f} train_acc={valid_acc:.4f}')
        writer.add_scalar(f"Per-GPU-Loss/valid_gpu{args.rank}", valid_loss, e)
        writer.add_scalar(f"Per-GPU-Acc/valid_gpu{args.rank}", valid_acc, e)

        # Gather global metrics to log average later
        global_valid_losses = torch.ones(args.world_size, dtype=torch.float32).cuda(args.gpu)
        global_valid_accs = torch.ones(args.world_size, dtype=torch.float32).cuda(args.gpu)
        if args.is_master:
            dist.gather(torch.tensor(valid_loss).cuda(args.gpu),
                        gather_list=[a.to(torch.float) for a in global_valid_losses])

            dist.gather(torch.tensor(valid_acc).cuda(args.gpu),
                        gather_list=[a.to(torch.float) for a in global_valid_accs])
        else:
            dist.gather(torch.tensor(valid_loss).cuda(args.gpu))
            dist.gather(torch.tensor(valid_acc).cuda(args.gpu))


        # After each epoch, the master handles logging and checkpoint saving
        if args.is_master:

            global_train_loss = torch.mean(global_train_losses)
            global_valid_loss = torch.mean(global_valid_losses)
            global_train_acc = torch.mean(global_train_accs)
            global_valid_acc = torch.mean(global_valid_accs)

            print(f'Epoch {e} train_loss={global_train_loss:.4f} train_acc={global_train_acc:.4f}')
            print(f'Epoch {e} val_loss={global_valid_loss:.4f} val_acc={global_valid_acc:.4f}')

            writer.add_scalar("Loss/train", global_train_loss, e)
            writer.add_scalar("Loss/valid", global_valid_loss, e)
            writer.add_scalar("Acc/train", global_train_acc, e)
            writer.add_scalar("Acc/valid", valid_acc, e)
        
            # Save checkpoint model if validation loss improves
            if args.save_training and global_valid_loss < val_loss_min:
                print(f'\tval_loss decreased ({val_loss_min:.6f} --> {global_valid_loss:.6f}). Saving this model ...')
                p = f'{args.model_ckpt_dir}/{e+1}.ckpt'
                torch.save(model.state_dict(), p)
                
                val_loss_min = global_valid_loss


    if args.is_master and writer is not None:
        writer.add_hparams(args.__dict__, {'0':0.0})

        if args.save_training:
            print('Saving final model ...')
            p = f'{args.model_ckpt_dir}/{args.epochs}.ckpt'
            torch.save(model.state_dict(), p)



    print(f'{args.id_string} Finished training.')
    dist.barrier()

    return model


def run_tests(args, writer, model):

    print(f'{args.id_string} Start testing.')

    ckpt_names = natsorted([Path(f).stem for f in os.listdir(args.model_ckpt_dir)])
    print("Found following (sorted!) model checkpoints:", ckpt_names)

    test_ds = BenDataset(args.TEST_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             sampler=None)

    for model_name in list(reversed(ckpt_names))[:args.run_tests_n]:
        print(f'\nLoad {args.model_ckpt_dir}/{model_name}.ckpt for testing.')
        p = f'{args.model_ckpt_dir}/{model_name}.ckpt'

        # state_dict = torch.load(p, map_location='cuda:' + str(args.gpu))
        # print("SATE999", state_dict['model_state_dict'].keys())
        # model.load_state_dict(state_dict['model_state_dict'])
        model.load_state_dict(
            torch.load(p, map_location='cuda:' + str(args.gpu)))

        test_loss, test_acc = valid_batches(test_loader, model, args.gpu)
        print(f'{model_name}.pt scores test_loss={test_loss:.4f} test_acc={test_acc:.4f}')

        if args.is_master:
            writer.add_scalar("Acc/test", test_acc, int(model_name))
            writer.add_scalar("Loss/test", test_loss, int(model_name))


    print(f'{args.id_string} Finished testing.')


if __name__ == '__main__':
    main()