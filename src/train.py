import os
import argparse
import yaml
import pprint

import lmdb
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch
from pathlib import Path
import importlib
from dotenv import load_dotenv
from datetime import datetime
from natsort import natsorted

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from torchmetrics import Accuracy

import torch
from torch import Tensor, cat, stack
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset_class import *
from conv_mixer import *
from training_utils import get_logging_dirs, get_model_name, get_optimizer, load_complete_model, train_batches, valid_batches, save_complete_model, get_history_plots

import time
import socket



def main():

    args = _parse_args()


    ### DDP settings ###
    assert "WORLD_SIZE" in os.environ

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        assert 'SLURM_PROCID' in os.environ

        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    args.host = socket.gethostname()
    print("RANKRTYPE", type(args.rank))
    args.is_master = int(args.rank) == 0
    args.id_string = f'Rank {args.rank} on {args.host}@cuda:{args.gpu}:'

    if args.is_master:
        print("Configure DDP settings ...")

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

    """
    runs/
        default/
        serbia_summer/
            exp1
                /full_model_hps1/
                    args.yml
                    ckpts/
                /full_model_hps2
                    args.yml
                    ckpts/
            exp2
                /full_model_hps1/
                    args.yml
                    ckpts/
                /full_model_hps2
                    args.yml
                    ckpts/
    """
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


    if args.is_master:
        # Dump arguments into yaml file
        # pprint.pprint(args.__dict__)
        with open(f'{args.model_dir}/args.yaml', 'w') as outfile:
            yaml.dump(args.__dict__, outfile, default_flow_style=False)


    writer = SummaryWriter(log_dir=args.model_dir) if args.is_master else None
    run_training(args, writer)
    # run_tests(args, writer)
    
    if args.is_master:
        writer.close()


def run_training(args, writer=None):

    print(f"{args.id_string} Start training ...")
    torch.manual_seed(42) 

    ### model ###
    model = ConvMixer(
        10, args.h, args.depth, kernel_size=args.k_size, 
        patch_size=args.p_size, n_classes=19, 
        activation=args.activation
    )
    loss_fn = nn.BCEWithLogitsLoss()
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
            # model_without_ddp = model.module
        else:
            model.cuda()
            model = DDP(model)
            # model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")


    ### Configure dataloaders for distributed training ###
    train_ds = BenDataset(args.TRAIN_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              sampler=train_sampler, drop_last=True)

    # validation happens non distributed
    valid_ds = BenDataset(args.VALID_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)
    valid_sampler = None
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)


    #### Training ####
    val_loss_min = np.inf
    train_acc_hist = []
    valid_acc_hist = []
    train_loss_hist = []
    valid_loss_hist = []

    # Main training loop
    print(f'{args.id_string} Start main training loop.')
    for e in range(args.epochs):

        print(f'\n{args.id_string} [{e+1:3d}/{args.epochs:3d}]', end=" ")

        if args.distributed:
            train_loader.sampler.set_epoch(e)

        # Train one set of batches
        model.train()
        train_loss, train_acc = train_batches(train_loader, model, optimizer, loss_fn, args.gpu)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
            
        # Validate one set of batches
        model.eval()
        valid_loss, valid_acc = valid_batches(valid_loader, model, loss_fn, args.gpu)
        valid_loss_hist.append(valid_loss)
        valid_acc_hist.append(valid_acc)

        print(f'{args.id_string} train_loss={train_loss:.4f} train_acc={train_acc:.4f}', end=" ")
        print(f'val_loss={valid_loss:.4f} val_acc={valid_acc:.4f}')


        # After each epoch, the master handles logging and checkpoint saving
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, e)
            writer.add_scalar("Loss/valid", valid_loss, e)
            writer.add_scalar("Acc/train", train_acc, e)
            writer.add_scalar("Acc/valid", valid_acc, e)
        
            # Save checkpoint model if validation loss improves
            if args.save_training and valid_loss < val_loss_min:
                print(f'\tval_loss decreased ({val_loss_min:.6f} --> {valid_loss:.6f}). Saving this model ...')

                p = f'{args.model_ckpt_dir}/{e+1}.pt'
                save_complete_model(p, model)
                
                val_loss_min = valid_loss


    print(f'{args.id_string} Finished Training')
    if args.is_master and writer is not None:
        writer.add_hparams(args.__dict__, {'0':0.0})

        if args.save_training:
            print('Saving final model ...')
            p = f'{args.model_ckpt_dir}/{args.epochs}.pt'
            save_complete_model(p, model)


# def run_tests(args, writer=None):

#     print('\n\nStart testing phase on master node.')

#     ckpt_names = natsorted(
#         [Path(f).stem for f in os.listdir(args.model_ckpt_dir)]
#     )
#     print("Found following (sorted!) model checkpoints:", ckpt_names)

#     loss_fn = nn.BCEWithLogitsLoss()
#     test_ds = BenDataset(args.TEST_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)
#     test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

#     for model_name in list(reversed(ckpt_names))[:args.run_tests_n]:
#         print(f'\nLoad {args.model_ckpt_dir}/{model_name}.pt for testing.')

#         model = load_complete_model(f'{args.model_ckpt_dir}/{model_name}.pt')
#         model = model.cuda(args.gpu)
#         test_loss, test_acc = valid_batches(test_loader, model, loss_fn, args.gpu)
        
#         print(f'{model_name}.pt scores test_loss={test_loss:.4f} test_acc={test_acc:.4f}')

#         if writer is not None:
#             writer.add_scalar("Acc/test", test_acc, int(model_name))
#             writer.add_scalar("Loss/test", test_loss, int(model_name))


#     print('\nFinished testing phase.')



def _parse_args():
    """Parse arguments and return ArgumentParser obj

    Returns:
        ArgumentParser: obj
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
                        help='SGD or Adam') #TODO add LAMB
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


if __name__ == '__main__':
    main()