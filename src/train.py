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
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--local_ranks', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    # parser.add_argument('--ip_adress', type=str, required=True,
    #                     help='ip address of the host node')
    parser.add_argument('--ngpus', default=1, type=int,
                        help='number of gpus per node')
    

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
    args.world_size = args.ngpus * args.nodes
    print("World size", args.world_size)
    
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    os.environ['MASTER_PORT'] = '8080'
    os.environ['WORLD_SIZE'] = str(args.world_size)
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

    # assert not os.path.isdir(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    args.model_name = model_name
    args.model_dir = model_dir


    # Decide which gpu and if distributed is feasible
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Dump arguments into yaml file
    with open(f'{args.model_dir}/config.yaml', 'w') as outfile:
        yaml.dump(args.__dict__, outfile, default_flow_style=False)
    
    
    print(args.__dict__)
    print("Spawn")
    mp.spawn(train, nprocs=args.ngpus, args=(args,))



def train(gpu, args):
    
    args.gpu = gpu
    print('args.gpu:', args.gpu)
    print('train on gpu:', gpu)
    
    # rank = args.local_ranks * args.ngpus + gpu
    rank = int(os.environ.get("SLURM_NODEID")) * args.ngpus + gpu
    print("Rank", rank)
    dist.init_process_group(backend='nccl', init_method='env://', 
                            world_size=args.world_size, rank=rank)

    torch.manual_seed(0)
    # if gpu == 0:
    #     writer = SummaryWriter()


    # if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)


    # Dataloader prep
    train_ds = BenDataset(args.train_csv_file, args.ben_lmdb_path, args.ds_size)
    train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False, sampler=train_sampler)
    
    # val_ds = BenDataset(args.val_csv_file, args.ben_lmdb_path, args.ds_size)
    # val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=rank)
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size,
    #                         shuffle=False, sampler=val_sampler)


    # Model prep    
    model = ConvMixer(10, args.h, args.depth, kernel_size=args.k_size,
                      patch_size=args.p_size, n_classes=19).cuda(args.gpu)
    loss_fn = nn.BCEWithLogitsLoss().cuda(args.gpu)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])


    # val_loss_min = np.inf

    train_loss_hist = []
    train_acc_hist = []
    # val_loss_hist = []
    # val_acc_hist = []

    # Main training loop
    print('Start main training loop.')
    for e in range(args.epochs):

        print(f'\nEpoch{e+1:3d}/{args.epochs:3d}')

        # Inference, backpropagation, weight adjustments
        model.train()
        train_loss, train_acc = train_batch(train_loader, model, optimizer, 
                                            loss_fn, args.gpu)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
            
        
        # # Evaluate on validation data
        # model.eval()
        # val_loss, val_acc = validate_batch(val_loader, model,
        #                                 loss_fn, gpu)
        # val_loss_hist.append(val_loss)
        # val_acc_hist.append(val_acc)

        if args.gpu == 0:
            print(f'train_loss={train_loss:.4f} train_acc={train_acc:.4f}')
            # print(f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
        
        #     writer.add_scalar("train_loss", train_loss, e)
        #     writer.add_scalar("val_loss", val_loss, e)
        #     writer.add_scalar("train_acc", train_acc, e)
        #     writer.add_scalar("val_acc", val_acc, e)
        
        # scheduler.step(val_loss)
        
        # # Save checkpoint model if validation loss improves
        # if args.save_training and val_loss < val_loss_min:
        #     print(f'val_loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}). Saving model weights ...')

        #     p = f'{args.model_dir}/e{e+1}_{args.model_name}.pt'
        #     save_complete_model(p, model)
            
        #     val_loss_min = val_loss

        

    if gpu == 0:
        print('Finished Training')
        # if args.save_training:
        #     print('Saving final model ...')
        #     p = f'{args.model_dir}/{args.model_name}.pt'
        #     save_complete_model(p, model)
            
        # writer.close()


if __name__ == '__main__':
    main()