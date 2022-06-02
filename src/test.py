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

from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel as DP
import torch.distributed as dist


from dataset_class import *
from conv_mixer import *
from training_utils import _parse_args, get_model_name, load_complete_model, valid_batches



def main():

    args = _parse_args()

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

    assert os.path.isdir(args.model_ckpt_dir)
    print("Model ckt dir", args.model_ckpt_dir)
    print('\nStart testing phase.')

    writer = SummaryWriter()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_names = natsorted(
        [Path(f).stem for f in os.listdir(args.model_ckpt_dir)]
    )
    print("Found following (sorted!) model checkpoints:", ckpt_names)

    loss_fn = nn.BCEWithLogitsLoss()
    test_ds = BenDataset(args.TEST_CSV_FILE, args.BEN_LMDB_PATH, args.ds_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    for model_name in list(reversed(ckpt_names))[:args.run_tests_n]:
        print(f'\nLoad {args.model_ckpt_dir}/{model_name}.pt for testing.')

        model = load_complete_model(f'{args.model_ckpt_dir}/{model_name}.pt')
        model = model.to(dev)
        test_loss, test_acc = valid_batches(test_loader, model, loss_fn, dev)
        
        print(f'{model_name}.pt scores test_loss={test_loss:.4f} test_acc={test_acc:.4f}')

        writer.add_scalar("Acc/test", test_acc, int(model_name))
        writer.add_scalar("Loss/test", test_loss, int(model_name))


    print('\nFinished testing phase.')


if __name__ == '__main__':
    main()