import os
import argparse
import yaml

import lmdb
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch
from pathlib import Path
import importlib
from dotenv import load_dotenv
from datetime import datetime

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

from dataset_class import *
from conv_mixer import *
from training_utils import train_batch, validate_batch, save_complete_model



def main():
    
    ### CONFIG ###
    parser = argparse.ArgumentParser(description="ConvMixer Parameters")

    # HPC Parametes
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--ngpus_per_node', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    

    # Training parameters
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--ds_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--RLROP', type=bool, default=False)
    parser.add_argument('--save_training', type=bool, default=True,
                        help='save plots and checkpoints to model directory')

    # Model Parameters
    parser.add_argument('--h', type=int, default=128)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--k_size', type=int, default=9)
    parser.add_argument('--p_size', type=int, default=7)

    args = parser.parse_args()

    # World size = total #gpus
    args.world_size = args.ngpus_per_node * args.nodes
    print("World size", args.world_size)
    
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    os.environ['MASTER_PORT'] = '8080'
    print('Masteraddr', os.environ['MASTER_ADDR'])

    # Get and assert paths
    load_dotenv('../.env')
    BEN_LMDB_PATH = os.environ.get("BEN_LMDB_PATH")
    TRAIN_CSV_FILE = os.environ.get("TRAIN_CSV")
    VAL_CSV_FILE = os.environ.get("VAL_CSV")
    PATH_TO_MODELS = os.environ.get("PATH_TO_MODELS")
    # TEST_CSV_FILE = os.environ.get("TEST_CSV")

    assert os.path.isdir(BEN_LMDB_PATH)
    assert os.path.isfile(TRAIN_CSV_FILE)
    assert os.path.isdir(PATH_TO_MODELS)
    
    args.train_csv_file = TRAIN_CSV_FILE
    args.val_csv_file = VAL_CSV_FILE
    args.ben_lmdb_path = BEN_LMDB_PATH

    # Create folder and model name for this training run
    timestamp = datetime.now().strftime('%m-%d_%H%M')
    model_name = f'ConvMx-{args.h}-{args.depth}'
    model_dir = PATH_TO_MODELS + '/' + timestamp + '-' + model_name

    assert not os.path.isdir(model_dir)
    os.mkdir(model_dir) # Store all model specific files here
    
    args.model_name = model_name
    args.model_dir = model_dir


    # Decide which gpu and if distributed is feasible
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Dump arguments into yaml file
    with open(f'{args.model_dir}/config.yaml', 'w') as outfile:
        yaml.dump(args.__dict__, outfile, default_flow_style=False)
    
    
    print(args.__dict__)
    print("Spawn")
    mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))



def train(gpu, args):
    
    print("train on", gpu)
    
    rank = args.nr * args.ngpus_per_node + gpu
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=args.world_size, rank=rank)


    if gpu == 0:
        writer = SummaryWriter()

    # Model prep    
    model = ConvMixer(10, args.h, args.depth, kernel_size=args.k_size,
                      patch_size=args.p_size, n_classes=19)
    loss_fn = nn.BCEWithLogitsLoss().cuda(gpu)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Dataloader prep
    # TODO: Add WeightedSampler to reduce overfit?
    train_ds = BenDataset(args.train_csv_file, args.ben_lmdb_path, args.ds_size)
    val_ds = BenDataset(args.val_csv_file, args.ben_lmdb_path, args.ds_size)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, sampler=train_sampler)
    
    val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=rank)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, sampler=val_sampler)


    val_loss_min = np.inf

    train_loss_hist = []
    train_acc_hist = []
    val_loss_hist = []
    val_acc_hist = []

    # Main training loop
    print('Start main training loop.')
    for e in range(args.epochs):

        print(f'\nEpoch{e+1:3d}/{args.epochs:3d}')

        # Inference, backpropagation, weight adjustments
        model.train()
        train_loss, train_acc = train_batch(val_loader, model, optimizer, 
                                            loss_fn, gpu)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
            
        
        # Evaluate on validation data
        model.eval()
        val_loss, val_acc = validate_batch(train_loader, model,
                                        loss_fn, gpu)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        if gpu == 0:
            print(f'train_loss={train_loss:.4f} train_acc={train_acc:.4f}', end=" ")
            print(f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
        
            writer.add_scalar("train_loss", train_loss, e)
            writer.add_scalar("val_loss", val_loss, e)
            writer.add_scalar("train_acc", train_acc, e)
            writer.add_scalar("val_acc", val_acc, e)
        
        scheduler.step(val_loss)
        
        # Save checkpoint model if validation loss improves
        if args.save_training and val_loss < val_loss_min:
            print(f'val_loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}). Saving model weights ...')

            p = f'{args.model_dir}/e{e+1}_{args.model_name}.pt'
            save_complete_model(p, model)
            
            val_loss_min = val_loss

        

    if gpu == 0:
        print('Finished Training')
        if args.save_training:
            print('Saving final model ...')
            p = f'{args.model_dir}/{args.model_name}.pt'
            save_complete_model(p, model)
            
        writer.close()


if __name__ == '__main__':
    main()