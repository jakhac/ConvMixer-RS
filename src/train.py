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

from dataset_class import *
from conv_mixer import *
from training_utils import train_batch, validate_batch, save_complete_model




### CONFIG ###

parser = argparse.ArgumentParser(description="ConvMixer Parameters")

# Training parameters
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--ds_size', type=int, default=None)


parser.add_argument('--save_training', default=True,
                    help='save plots and checkpoints to model directory')

# Model Parameters
parser.add_argument('--h', type=int, default=512)
parser.add_argument('--depth', type=int, default=8)
parser.add_argument('--k_size', type=int, default=9)
parser.add_argument('--p_size', type=int, default=7)

args = parser.parse_args()



#### Data Loading ####

writer = SummaryWriter()

# Get paths
load_dotenv('../.env')

BEN_LMDB_PATH = os.environ.get("BEN_LMDB_PATH")

TRAIN_CSV_FILE = os.environ.get("TRAIN_CSV")
TEST_CSV_FILE = os.environ.get("TEST_CSV")
VAL_CSV_FILE = os.environ.get("VAL_CSV")
PATH_TO_MODELS = os.environ.get("PATH_TO_MODELS")

assert os.path.isdir(BEN_LMDB_PATH)
assert os.path.isfile(TRAIN_CSV_FILE)
assert os.path.isdir(PATH_TO_MODELS)

timestamp = datetime.now().strftime('%m-%d_%H%M')
model_name = f'ConvMx-{args.h}-{args.depth}'
model_dir = PATH_TO_MODELS + '/' + timestamp + '-' + model_name

# Store all model specific files in {model_dir}
assert not os.path.isdir(model_dir)
os.mkdir(model_dir)

env = lmdb.open(BEN_LMDB_PATH, readonly=True, readahead=False, lock=False)
txn = env.begin()
cur = txn.cursor()

# Dump arguments into yaml file
with open(f'{model_dir}/config.yaml', 'w') as outfile:
    yaml.dump(args.__dict__, outfile, default_flow_style=False)

val_ds = BenDataset(VAL_CSV_FILE, BEN_LMDB_PATH, args.ds_size)
train_ds = BenDataset(TRAIN_CSV_FILE, BEN_LMDB_PATH, args.ds_size)

val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)


#### Training Preparation ####
device = torch.device('cuda:0')

model = ConvMixer(10, args.h, args.depth, kernel_size=args.k_size, 
                  patch_size=args.p_size, n_classes=19)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


#### Training ####
val_loss_min = np.inf

train_loss_hist = []
train_acc_hist = []
val_loss_hist = []
val_acc_hist = []

if torch.cuda.is_available():
    model = model.to(device)

# Main training loop
print('Start main training loop.')
for e in range(args.epochs):

    try:
        print(f'\nEpoch{e+1:3d}/{args.epochs:3d}')

        # Inference, backpropagation, weight adjustments
        model.train()
        train_loss, train_acc = train_batch(val_loader, model, optimizer, loss_fn)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
            
        
        # Evaluate on validation data
        model.eval()
        val_loss, val_acc = validate_batch(train_loader, model, loss_fn)
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

    except KeyboardInterrupt:
        print('\n\nAbort training.')
        break
    

print('Finished Training')
if args.save_training:
    print('Saving final model ...')
    p = f'{model_dir}/{model_name}.pt'
    save_complete_model(p, model)


writer.close()

#### Save Training History ####
if args.save_training:
    fig = plt.figure(figsize=(16,4))
    ax = fig.add_subplot(121)
    ax.plot(val_loss_hist, label='val')
    ax.plot(train_loss_hist, label='train')
    ax.legend(loc="upper right")
    # ax.set_ylim([0, 1])
    ax.set_title("loss")
    ax.set_xlabel("epochs")

    ax = fig.add_subplot(122)
    ax.plot([v.cpu().detach().numpy() for v in val_acc_hist], label='val')
    ax.plot([v.cpu().detach().numpy() for v in train_acc_hist], label='train')
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1])
    ax.set_title("accuracy")
    ax.set_xlabel("epochs")

    plt.savefig(f'{model_dir}/{model_name}.pdf')


