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
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel as DP

from dataset_class import *
from conv_mixer import *
from training_utils import train_batch, validate_batch, save_complete_model, get_history_plots



### CONFIG ###

parser = argparse.ArgumentParser(description="ConvMixer Parameters")

# Training parameters
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--ds_size', type=int, default=None)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--activation', type=str, default='GELU')


parser.add_argument('--save_training', default=True,
                    help='save plots and checkpoints to model directory')

# Model Parameters
parser.add_argument('--h', type=int, default=512)
parser.add_argument('--depth', type=int, default=8)
parser.add_argument('--k_size', type=int, default=9)
parser.add_argument('--p_size', type=int, default=7)
parser.add_argument('--k_dilation', type=int, default=1)

args = parser.parse_args()


#### Data Loading ####
load_dotenv('../.env')

BEN_LMDB_PATH = os.environ.get("BEN_LMDB_PATH")

TRAIN_CSV_FILE = os.environ.get("TRAIN_CSV")
TEST_CSV_FILE = os.environ.get("TEST_CSV")
VAL_CSV_FILE = os.environ.get("VAL_CSV")
PATH_TO_MODELS = os.environ.get("PATH_TO_MODELS")

assert os.path.isdir(BEN_LMDB_PATH)
assert os.path.isfile(TRAIN_CSV_FILE)
assert os.path.isdir(PATH_TO_MODELS)

timestamp = datetime.now().strftime('%m-%d_%H%M_%S')
model_name = f'ConvMx-{args.h}-{args.depth}-{args.k_size}-{args.p_size}'
model_type_dir = PATH_TO_MODELS + '/' + model_name
model_dir = model_type_dir + '/' + timestamp

# Store all model specific files in {model_dir}
os.makedirs(model_dir, exist_ok=True)

env = lmdb.open(BEN_LMDB_PATH, readonly=True, readahead=False, lock=False)
txn = env.begin()
cur = txn.cursor()

# Dump arguments into yaml file
with open(f'{model_dir}/config.yaml', 'w') as outfile:
    yaml.dump(args.__dict__, outfile, default_flow_style=False)

writer = SummaryWriter(log_dir=model_dir)

val_ds = BenDataset(VAL_CSV_FILE, BEN_LMDB_PATH, args.ds_size)
train_ds = BenDataset(TRAIN_CSV_FILE, BEN_LMDB_PATH, args.ds_size)

val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)


#### Training Preparation ####
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvMixer(10, args.h, args.depth, kernel_size=args.k_size, 
                  patch_size=args.p_size, n_classes=19)
loss_fn = nn.BCEWithLogitsLoss().to(dev)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

args.n_params = sum(p.numel() for p in model.parameters())
args.n_params_trainable = sum(p.numel() for p in model.parameters())

#### Training ####
val_loss_min = np.inf

train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

if torch.cuda.is_available():
    model = model.to(dev)
    model = DP(model)

# Main training loop
print('Start main training loop.')
for e in range(args.epochs):

    # Inference, backpropagation, weight adjustments
    model.train()
    train_loss, train_acc = train_batch(val_loader, model, optimizer, loss_fn, dev)
    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
        
    # Evaluate on validation data
    model.eval()
    val_loss, val_acc = validate_batch(train_loader, model, loss_fn, dev)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    print(f'train_loss={train_loss:.4f} train_acc={train_acc:.4f}', end=" ")
    print(f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')
    
    writer.add_scalar("train_loss", train_loss, e)
    writer.add_scalar("val_loss", val_loss, e)
    writer.add_scalar("train_acc", train_acc, e)
    writer.add_scalar("val_acc", val_acc, e)
    
    # Save checkpoint model if validation loss improves
    if args.save_training and val_loss < val_loss_min:
        print(f'val_loss decreased ({val_loss_min:.6f} --> {val_loss:.6f}). Saving model weights ...')

        p = f'{model_dir}/e{e+1}_{model_name}.pt'
        save_complete_model(p, model)
        
        val_loss_min = val_loss
        args.val_loss_min = val_loss_min



print('Finished Training')
if args.save_training:
    print('Saving final model ...')
    p = f'{model_dir}/{model_name}.pt'
    save_complete_model(p, model)


writer.add_figure("Loss / Acc plots", get_history_plots(val_loss_hist, train_loss_hist,
                                                        val_acc_hist, train_acc_hist))



writer.add_hparams(args.__dict__, {'metric':0.0})
writer.close()

