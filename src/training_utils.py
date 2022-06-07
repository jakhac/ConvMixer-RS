import os
from pydoc import apropos
from dotenv import load_dotenv
from natsort import natsorted
from pathlib import Path
import numpy as np

import torch
from datetime import datetime

from sklearn.metrics import average_precision_score, f1_score, accuracy_score

from ranger21 import Ranger21
import torch.optim as optim
import torch_optimizer as torch_optim
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import torch.distributed as dist

from ben_dataset import BenDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def train_batches(train_loader, model, optimizer, criterion):
    """Perform a training step.
    Args:
        train_loader (DataLoader): data loader with training images
        model (Model): model
        optimizer (optimizer): optimizer
        criterion (criterion): loss functin
    Returns:
        float, List[(y, y_hat)]: loss and label-output tuples
    """

    # total_acc = 0.0
    total_loss = 0.0
    n_batches = len(train_loader)
    yyhat_tuples = []

    model.train()
    for X, y in train_loader:
        X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()

        y_hat = model(X)
        y_hat_sigmoid = torch.sigmoid(y_hat)
        yyhat_tuples += zip(y.cpu().detach().numpy(),
                            y_hat_sigmoid.cpu().detach().numpy())

        # Loss
        loss = criterion(y_hat, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return (total_loss / n_batches), np.asarray(yyhat_tuples)


def valid_batches(val_loader, model, criterion):
    """Perform a validation step.
    Args:
        train_loader (DataLoader): data loader with training images
        model (Model): model
        optimizer (optimizer): optimizer
        criterion (criterion): loss functin
    Returns:
        float, List[(y, y_hat)]: loss and label-output tuples
    """

    n_batches = len(val_loader)
    total_loss = 0.0
    yyhat_tuples = []

    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.cuda(), y.cuda()

            y_hat = model(X)
            y_hat_sigmoid = torch.sigmoid(y_hat)
            yyhat_tuples += zip(y.cpu().detach().numpy(),
                                y_hat_sigmoid.cpu().detach().numpy())

            # Add loss to accumulator
            loss = criterion(y_hat, y)
            total_loss += loss.item()
            

    return (total_loss / n_batches), np.asarray(yyhat_tuples)


def get_predictions_for_batch(outputs):
    """Apply sigmoid layer to scale into [0, 1], then round
    such that all entries are in { 0, 1 }.
    
    Args:
        outputs (Tensor): batch of outputs by model
    Returns:
        (Tensor): transformed tensor with one-hot predictions
    """
    outputs_sig = torch.sigmoid(outputs)
    predictions = torch.round(outputs_sig)
    
    return predictions


def get_model_name(args):
    """Create a unique model consisting of timestamp + model_architecture + model_config

    Args:
        args (ArgumentParser): Argumentparser

    Returns:
        str: unique model name
    """

    timestamp = datetime.now().strftime('%m-%d_%H%M_%S')

    model_arch = f'CvMx-h={args.h}-d={args.depth}-k={args.k_size}-p={args.p_size}'
    model_config = f'batch={args.batch_size}_lr={args.lr}_mom={args.momentum}_{args.activation}_{args.optimizer}'

    return f'{timestamp}_{model_arch}_{model_config}'


def get_optimizer(model, args, n_batches_per_e=None):
    """Rreturns an optimizer as defined in args.optimizer.

    Args:
        model (Model): pytorch model
        args (ArgumentParser): ArgumentParser
        n_batches_per_e (int, optional): Number of batches per epoch, used for Ranger21. Defaults to None.

    Returns:
        torch.optimizer: optimizer
    """

    if args.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Lamb':
        return torch_optim.Lamb(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Ranger21':
        assert n_batches_per_e
        return Ranger21(model.parameters(), lr=args.lr, 
                        num_epochs=args.epochs, num_batches_per_epoch=n_batches_per_e)
    else:
        assert False, "Error: get_optimizer() did not find a matching optimizer."


def get_activation(activation):
    """Return an activation function. Either 'GELU' or 'ReLU'.

    Args:
        activation (str): One of 'GELU' or 'ReLU'

    Returns:
        nn.Optimizer: optimzer function
    """

    if activation == 'GELU':
        return nn.GELU()
    elif activation == 'ReLU':
        return nn.ReLU()
    else:
        print("Error: get_activation() did not find a matching activation fn.")
        assert False


def get_dataloader(args, csv_file, shuffle):
    """Return a dataloader. If distributed, the dataloader uses a
    DistributedSampler.

    Args:
        args (ArgumentParser): ArgumentParser
        csv_file (str): Absolute path to csv file
        shuffle (bool): True if data needs to be shuffled
        shuffle (bool): True if data is distributed across multiple devices (false for testing!)

    Returns:
        DataLoader: dataloader
    """

    ds = BenDataset(csv_file, args.BEN_LMDB_PATH, args.ds_size)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, 
                      drop_last=True, pin_memory=True)


def setup_paths_and_hparams(args):
    """Add csv and data paths to given ArgumentParser

    Args:
        args (ArgumentParser): ArgumentParser
    """

    load_dotenv('../.env')
    args.BEN_LMDB_PATH = os.environ.get("BEN_LMDB_PATH")
    args.TRAIN_CSV_FILE = os.environ.get("TRAIN_CSV")
    args.VALID_CSV_FILE = os.environ.get("VAL_CSV")
    args.TEST_CSV_FILE = os.environ.get("TEST_CSV")
    args.PATH_TO_RUNS = os.environ.get("PATH_TO_RUNS")
    args.SPLIT = os.environ.get("SPLIT")

    assert os.path.isdir(args.BEN_LMDB_PATH)
    assert os.path.isdir(args.PATH_TO_RUNS)
    assert os.path.isfile(args.TRAIN_CSV_FILE)
    assert os.path.isfile(args.VALID_CSV_FILE)

    args.model_name = get_model_name(args)
    args.model_dir = f'{args.PATH_TO_RUNS}/{args.SPLIT}/{args.exp_name}/{args.model_name}'
    args.model_ckpt_dir = args.model_dir + '/ckpt'

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.model_ckpt_dir, exist_ok=True)

    # Allow dry runs for quick testing purpose
    if args.dry_run:
        args.epochs = 1
        args.ds_size = 100
        args.batch_size = 10
        args.lr = 0.1
        args.run_tests_n = 2


def save_checkpoint(args, model, optimizer, epoch):
    """Save a checkpoint containing model weights, optimizer state and 
    last finished epoch.

    Args:
        args (ArgumentParser): ArgumentParser
        model (Model): Model
        optimizer (Optimizer): Optimizer
        epoch (int): Last finished epoch
    """

    p = f'{args.model_ckpt_dir}/{epoch}.ckpt'
    torch.save({
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch},
        p)


def load_checkpoint(model, path):
    """Load weights from a checkpoint into model on the specified GPU.
    Distributed flag is required to consume prefixes from DDP state dicts.

    Args:
        model (Model): Model
        path (str): path/to/file.ckpt
        distributed (bool): True if model that is loading weights is DDP, else false
    """

    state_dict = torch.load(path)
    consume_prefix_in_state_dict_if_present(state_dict['model_state'], "module.")
    model.load_state_dict(state_dict['model_state'])


def get_sorted_ckpt_filenames(path):
    """For a given path, returns a sorted list of filenames.

    Args:
        path (str): path/to/dir

    Returns:
        List[str]: List of sorted filenames
    """

    assert os.path.isdir(path)

    ckpt_names = natsorted([Path(f).stem for f in os.listdir(path)])
    print("Found following (sorted!) model checkpoints:", ckpt_names)
    return ckpt_names


def write_metrics(writer, tag, yy_hat, loss, e):
    """Log metrics to stdout and to Tensorboard writer.

    Args:
        writer (SummaryWriter): TB SummaryWriter
        tag (str): phase, either train, valid or test
        yy_hat ([(y, y_hat)]): List of (y, y_hat) tuples
        loss (float): loss value
        e (int): current epoch
    """

    print(f"{tag} metrics:")

    y = yy_hat[:, 0, :]
    y_hat_sigmoid = yy_hat[:, 1, :]
    y_hat_predict = np.round(y_hat_sigmoid)

    assert not np.isnan(np.sum(y))
    assert not np.isnan(np.sum(y_hat_sigmoid))
    assert not np.isinf(np.sum(y))
    assert not np.isinf(np.sum(y_hat_sigmoid))
    assert np.any(y)
    assert np.any(y_hat_sigmoid)

    # mAP
    writer.add_scalar(f"mAP-micro/{tag}", average_precision_score(y, y_hat_sigmoid, average='micro'), e)
    writer.add_scalar(f"mAP-macro/{tag}", average_precision_score(y, y_hat_sigmoid, average='macro'), e)
    print(f"mAP-micro/{tag} {average_precision_score(y, y_hat_sigmoid, average='micro'):.4f}")
    print(f"mAP-macro/{tag} {average_precision_score(y, y_hat_sigmoid, average='macro'):.4f}")
    print(f"AP_class/{tag} {np.round(average_precision_score(y, y_hat_sigmoid, average=None), 4)}")

    # F1
    writer.add_scalar(f"F1-micro/{tag}", f1_score(y, y_hat_predict, average='micro'), e)
    writer.add_scalar(f"F1-macro/{tag}", f1_score(y, y_hat_predict, average='macro'), e)
    print(f"F1-micro/{tag} {f1_score(y, y_hat_predict, average='micro'):.4f}")
    print(f"F1-macro/{tag} {f1_score(y, y_hat_predict, average='macro'):.4f}")
    print(f"F1-class/{tag} {np.round(f1_score(y, y_hat_predict, average=None), 4)}")

    # Loss
    writer.add_scalar(f"Loss/{tag}", loss, e)
    print(f"Loss/{tag} {loss:.4f}")
    
    # Accuracy
    acc = accuracy_score(y_hat_predict, y)
    writer.add_scalar(f"Acc/{tag}", acc, e)
    print(f"Acc/{tag} {acc:.4f}")
    print()
