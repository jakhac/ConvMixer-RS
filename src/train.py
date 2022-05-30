import os
import argparse
import yaml

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
from torch.nn import DataParallel as DP

from dataset_class import *
from conv_mixer import *
from training_utils import get_activation, get_logging_dirs, get_optimizer, load_complete_model, train_batches, valid_batches, save_complete_model, get_history_plots


def main():

    args = _parse_args()

    #### Data Loading ####
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

    args.SPLIT = 'test' if args.test_run else args.SPLIT
    model_arch, model_name = get_logging_dirs(args)
    args.model_arch_dir = args.PATH_TO_RUNS + '/' + args.SPLIT + '/' + model_arch
    args.model_name_dir = args.model_arch_dir + '/' + model_name
    args.model_ckpt_dir = args.model_name_dir + '/ckpt'
    print('model_arch_dir', args.model_arch_dir)
    print('model_name_dir', args.model_name_dir)

    os.makedirs(args.model_name_dir, exist_ok=False)
    os.makedirs(args.model_ckpt_dir, exist_ok=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dump arguments into yaml file
    with open(f'{args.model_name_dir}/args.yaml', 'w') as outfile:
        yaml.dump(args.__dict__, outfile, default_flow_style=False)


    writer = SummaryWriter(log_dir=args.model_name_dir)
    run_training(args, writer, dev)

    if args.run_tests and args.run_tests_n > 0:
        run_tests(args, writer, dev)


    writer.close()


def run_training(args, writer, dev):

    model = ConvMixer(10, args.h, args.depth, kernel_size=args.k_size, 
                      patch_size=args.p_size, n_classes=19, 
                      activation=args.activation)

    if torch.cuda.is_available():
        model = model.to(dev)
        model = DP(model)

    loss_fn = nn.BCEWithLogitsLoss().to(dev)
    optimizer = get_optimizer(model, args)

    args.n_params = sum(p.numel() for p in model.parameters())
    args.n_params_trainable = sum(p.numel() for p in model.parameters())

    valid_ds = BenDataset(args.VALID_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)
    train_ds = BenDataset(args.TRAIN_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)

    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    #### Training ####
    val_loss_min = np.inf
    train_loss_hist = []
    valid_loss_hist = []
    valid_acc_hist = []
    train_acc_hist = []

    # Main training loop
    print('Start main training loop.')
    for e in range(args.epochs):

        print(f'\n[{e+1:3d}/{args.epochs:3d}]', end=" ")

        # Inference and 
        model.train()
        train_loss, train_acc = train_batches(train_loader, model, optimizer, loss_fn, dev)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
            
        # Evaluate on validation data
        model.eval()
        valid_loss, valid_acc = valid_batches(valid_loader, model, loss_fn, dev)
        valid_loss_hist.append(valid_loss)
        valid_acc_hist.append(valid_acc)

        print(f'train_loss={train_loss:.4f} train_acc={train_acc:.4f}', end=" ")
        print(f'val_loss={valid_loss:.4f} val_acc={valid_acc:.4f}')
        
        writer.add_scalar("Loss/train", train_loss, e)
        writer.add_scalar("Loss/valid", valid_loss, e)
        writer.add_scalar("Acc/train", train_acc, e)
        writer.add_scalar("Acc/valid", valid_acc, e)
        
        # Save checkpoint model if validation loss improves
        if args.save_training and valid_loss < val_loss_min:
            print(f'\tval_loss decreased ({val_loss_min:.6f} --> {valid_loss:.6f}). Saving this model ...')

            p = f'{args.model_name_dir}/ckpt/{e+1}.pt'
            save_complete_model(p, model)
            
            val_loss_min = valid_loss


    print('Finished Training')
    if args.save_training:
        print('Saving final model ...')
        p = f'{args.model_name_dir}/{args.epochs}.pt'
        save_complete_model(p, model)


    writer.add_figure("matplotlib", get_history_plots(
        valid_loss_hist, train_loss_hist,
        valid_acc_hist, train_acc_hist)
    )

    writer.add_hparams(args.__dict__, {'0':0.0})


def run_tests(args, writer, dev):

    print('\n\nStart testing phase.')

    ckpt_names = natsorted(
        [Path(f).stem for f in os.listdir(args.model_ckpt_dir)]
    )
    print("Found following (sorted!) model checkpoints:", ckpt_names)

    loss_fn = nn.BCEWithLogitsLoss().to(dev)
    test_ds = BenDataset(args.TEST_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    for model_name in list(reversed(ckpt_names))[:args.run_tests_n]:
        print(f'\nLoad {args.model_ckpt_dir}/{model_name}.pt for testing.')

        model = load_complete_model(f'{args.model_ckpt_dir}/{model_name}.pt')
        test_loss, test_acc = valid_batches(test_loader, model, loss_fn, dev)
        
        print(f'{model_name}.pt scores test_loss={test_loss:.4f} test_acc={test_acc:.4f}')

        writer.add_scalar("Acc/test", test_acc, int(model_name))
        writer.add_scalar("Loss/test", test_loss, int(model_name))


    print('\nFinished testing phase.')


def _parse_args():
    """Parse arguments and return ArgumentParser obj

    Returns:
        ArgumentParser: obj
    """

    parser = argparse.ArgumentParser(description="ConvMixer Parameters")

    # Training parameters
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--ds_size', type=int, default=None)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--activation', type=str, default='GELU')

    # Config Parameters
    parser.add_argument('--test_run', default=False,
                        help='log into run/test directory')
    parser.add_argument('--save_training', default=True,
                        help='save checkpoints when valid_loss decreases')
    parser.add_argument('--run_tests', type=bool, default=True,
                        help='run best models on test data')
    parser.add_argument('--run_tests_n', type=int, default=5,
                        help='test the n best models on test data')

    # Model Parameters
    parser.add_argument('--h', type=int, default=512)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--k_size', type=int, default=9)
    parser.add_argument('--p_size', type=int, default=7)
    parser.add_argument('--k_dilation', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    main()