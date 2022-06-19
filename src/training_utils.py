import os
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
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

from ben_dataset import BenDataset, get_transformation_chain
from torch.utils.data import DataLoader


def train_batches(train_loader, model, optimizer, criterion, dev, scheduler=None, warmup_scheduler=None):
    """Perform a training step.
    Args:
        train_loader (DataLoader): data loader with training images
        model (Model): model
        optimizer (optimizer): optimizer
        criterion (criterion): loss function
        dev (str): device, either cuda or cpu
    Returns:
        float, List[(y, y_hat)]: loss and label-output tuples
    """

    total_loss = 0.0
    n_batches = len(train_loader)
    yyhat_tuples = []

    model.train()
    for X, y in train_loader:
        X, y = X.to(dev), y.to(dev)
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

        with warmup_scheduler.dampening():
            scheduler.step()

    return (total_loss / n_batches), np.asarray(yyhat_tuples)


def valid_batches(val_loader, model, criterion, dev):
    """Perform a validation step.
    Args:
        train_loader (DataLoader): data loader with training images
        model (Model): model
        optimizer (optimizer): optimizer
        criterion (criterion): loss function
        dev (str): device, either 'cuda' or 'cpu'
    Returns:
        float, List[(y, y_hat)]: loss and label-output tuples
    """

    n_batches = len(val_loader)
    total_loss = 0.0
    yyhat_tuples = []

    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(dev), y.to(dev)

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

    weight_decay = "" if args.decay is None else f"_dec={args.decay}"
    lr_policy = "" if args.lr_policy is None else f"_lr-pol={args.lr_policy}"
    res = "" if args.residual == 1 else f"_res={args.residual}"
    drop = "" if args.drop == 0.0 else f"_drop={args.drop}"
    warmup_fn = f"_wfn={args.lr_warmup_fn}-{args.lr_warmup}"

    model_arch = f'CvMx-h={args.h}-d={args.depth}-k={args.k_size}-p={args.p_size}'
    model_config = f'batch={args.batch_size}_lr={args.lr}_mom={args.momentum}_{args.activation}_{args.optimizer}_aug={args.augmentation}{weight_decay}{lr_policy}{res}{drop}{warmup_fn}'

    return f'{timestamp}_{model_arch}_{model_config}'


def get_optimizer(model, args, iterations=None):
    """Rreturns an optimizer as defined in args.optimizer.

    Args:
        model (Model): pytorch model
        args (ArgumentParser): ArgumentParser
        n_batches_per_e (int, optional): Number of batches per epoch, used for Ranger21. Defaults to None.

    Returns:
        torch.optimizer: optimizer
    """

    adamW_weight_decay = args.decay if args.decay is not None else 1e-2

    if args.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=adamW_weight_decay)
    elif args.optimizer == 'Lamb':
        return torch_optim.Lamb(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Ranger21':
        assert iterations
        return Ranger21(model.parameters(), lr=args.lr, momentum=args.momentum,
                        num_epochs=args.epochs, num_batches_per_epoch=iterations)
    else:
        assert False, "Error: get_optimizer() did not find a matching optimizer."


def get_scheduler(optimizer, args, iterations):

    if args.lr_policy == "1CycleLR":
        return OneCycleLR(optimizer, epochs=args.epochs,
            max_lr=1e-3, steps_per_epoch=iterations, verbose=True)
    elif args.lr_policy == "RLROP":
        return ReduceLROnPlateau(optimizer, patience=3, threshold=5e-4,
            verbose=True, factor=0.25)


    return None



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


def get_dataloader(args, csv_file, apply_transforms, shuffle):
    """Return a dataloader.

    Args:
        args (ArgumentParser): ArgumentParser
        csv_file (str): Absolute path to csv file
        shuffle (bool): True if data needs to be shuffled

    Returns:
        DataLoader: dataloader
    """

    transforms = None
    if apply_transforms:
        transforms = get_transformation_chain(args.augmentation)


    ds = BenDataset(csv_file, args.BEN_LMDB_PATH, transforms=transforms, size=args.ds_size)
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
        args.ds_size = 40
        args.batch_size = 20
        args.lr = 0.1
        args.run_tests = True
        # args.run_tests_n = 2


def save_checkpoint(args, model, optimizer, epoch, exist_ok=False):
    """Save a checkpoint containing model weights, optimizer state and 
    last finished epoch.

    Args:
        args (ArgumentParser): ArgumentParser
        model (Model): Model
        optimizer (Optimizer): Optimizer
        epoch (int): Last finished epoch
    """

    p = f'{args.model_ckpt_dir}/{epoch}.ckpt'
    # assert exist_ok or not os.path.isfile(p)

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
        yy_hat ([(y, y_hat_sigmoid)]): List of (y, y_hat_sigmoid) tuples
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
    acc = accuracy_score(y, y_hat_predict)
    writer.add_scalar(f"Acc/{tag}", acc, e)
    print(f"Acc/{tag} {acc:.4f}")
    print()
