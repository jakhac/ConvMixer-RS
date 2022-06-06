import os
from pydoc import apropos
from dotenv import load_dotenv
from natsort import natsorted
from pathlib import Path
import numpy as np

import torch
from torchmetrics.functional import accuracy
from datetime import datetime

from sklearn.metrics import average_precision_score, f1_score, precision_score

from ranger21 import Ranger21
import torch.optim as optim
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import torch.distributed as dist

from ben_dataset import BenDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def train_batches(train_loader, model, optimizer, criterion, gpu):
    """Perform a full training step for given batches in train_loader.
    Args:
        train_loader (DataLoader): data loader with training images
        model (Model): model
        optimizer (optimizer): optimizer
        loss_fn (criterion): loss function
        accuracy (accuracy): accuracy object
    Returns:
        (float, float): (loss, accuracy) for this data_loader
    """

    total_acc = 0.0
    total_loss = 0.0
    n_batches = len(train_loader)
    yyhat_tuples = []

    model.train()
    for X, y in train_loader:
        X, y = X.cuda(gpu), y.cuda(gpu)
        optimizer.zero_grad()

        y_hat = model(X)

        ### METRICS ###
        y_hat_sigmoid = torch.sigmoid(y_hat)
        yyhat_tuples += zip(y.cpu().detach().numpy(), y_hat_sigmoid.cpu().detach().numpy())

        # Accuracy
        y_predictions = torch.round(y_hat_sigmoid)
        total_acc += accuracy(y_predictions.to(torch.int), y.to(torch.int), subset_accuracy=True)

        # Loss
        loss = criterion(y_hat, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return (total_loss / n_batches), (total_acc / n_batches), np.asarray(yyhat_tuples)


def valid_batches(val_loader, model, criterion, gpu):
    """Perform validation on val_loader images.
    Args:
        val_loader (DataLoader): data loader with validation data
        model (Model): model
        loss_fn (criterion): loss function
        accuracy (accuracy): accuracy object
        device (string): cuda or cpu
    Returns:
        (float, float): (loss, accuracy) for data loader
    """

    n_batches = len(val_loader)
    total_loss = 0.0
    total_acc = 0.0
    yyhat_tuples = []

    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.cuda(gpu), y.cuda(gpu)
            y_hat = model(X)

            y_hat_sigmoid = torch.sigmoid(y_hat)
            yyhat_tuples += zip(y.cpu().detach().numpy(), y_hat_sigmoid.cpu().detach().numpy())

            # Accuracy
            y_predictions = torch.round(y_hat_sigmoid)
            total_acc += accuracy(y_predictions.to(torch.int), y.to(torch.int),
                                  subset_accuracy=True)   
            
            # Add loss to accumulator
            loss = criterion(y_hat, y)
            total_loss += loss.item()
            

    return (total_loss / n_batches), (total_acc / n_batches), yyhat_tuples


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
    timestamp = datetime.now().strftime('%m-%d_%H%M_%S')

    model_arch = f'CvMx-h={args.h}-d={args.depth}-k={args.k_size}-p={args.p_size}'
    model_config = f'batch={args.batch_size}_lr={args.lr}_mom={args.momentum}_{args.activation}_{args.optimizer}'

    return f'{timestamp}_{model_arch}_{model_config}'


def get_optimizer(model, args, n_batches_per_e=None):
    if args.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Ranger21':
        assert n_batches_per_e
        return Ranger21(model.parameters(), lr=args.lr, 
                        num_epochs=args.epochs, num_batches_per_epoch=n_batches_per_e)
    else:
        print("Error: get_optimizer() did not find a matching optimizer.")
        assert False


def get_activation(activation):
    if activation == 'GELU':
        return nn.GELU()
    elif activation == 'ReLU':
        return nn.ReLU()
    else:
        print("Error: get_activation() did not find a matching activation fn.")
        assert False


def get_global_yyhat(args, samples_per_gpu, yyhat):


    global_yyhat = torch.zeros((args.world_size, samples_per_gpu, 2, 19), dtype=torch.float32).cuda(args.gpu)
    yyhat = np.array(yyhat) # Faster, according to pytorch user warning

    if args.is_master:
        dist.gather(torch.tensor(yyhat).cuda(args.gpu), [g.to(torch.float32) for g in global_yyhat], dst=0)
    else:
        dist.gather(torch.tensor(yyhat).cuda(args.gpu), dst=0)

    return global_yyhat


def global_metric_avg(args, metric):
    """Gather all metric tensors across processes and return mean.

    Args:
        args (ArgumentParser): ArgumentParser
        metric (Tensor): Tensor to be synchronized

    Returns:
        Tensor: Mean of all tensors
    """

    global_metric = torch.ones(args.world_size, dtype=torch.float32).cuda(args.gpu)

    # Process 0 stores all results in a list, remaining
    # processes only call gather().
    if args.is_master:
        dist.gather(torch.tensor(metric).cuda(args.gpu),
                    gather_list=[g.to(torch.float) for g in global_metric]) # ??
    else:
        dist.gather(torch.tensor(metric).cuda(args.gpu))

    return torch.mean(global_metric)


def get_dataloader(args, csv_file, shuffle, distribute=True):
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

    sampler = None
    if distribute:
        sampler = DistributedSampler(ds, shuffle=shuffle, drop_last=True)
    

    shuffle_in_dl = not distribute and shuffle
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle_in_dl,
                              sampler=sampler, drop_last=True, pin_memory=True)


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
        args.epochs = 2
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


def load_checkpoint(model, path, gpu, distributed):
    """Load weights from a checkpoint into model on the specified GPU.
    Distirbuted flag is required to consume prefixes from DDP state dicts.

    Args:
        model (Model): Model
        path (str): path/to/file.ckpt
        gpu (int): CUDA GPU device
        distributed (bool): True if model that is loading weights is DDP, else false
    """

    state_dict = torch.load(path, map_location='cuda:' + str(gpu))

    if distributed:
        consume_prefix_in_state_dict_if_present(state_dict['model_state'], 'module.')
    
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


def write_metrics(writer, tag, yy_hat, loss, acc, e, from_gpu=True):

    if from_gpu:
        yy_hat = yy_hat.cpu().detach().numpy()
        yy_hat = np.concatenate((yy_hat), axis=0)
    else:
        yy_hat = np.asarray(yy_hat)

    y = yy_hat[:, 0, :]
    y_hat_sig = yy_hat[:, 1, :]

    # mAP
    writer.add_scalar(f"mAP-micro/{tag}", average_precision_score(y, y_hat_sig, average='micro'))
    writer.add_scalar(f"mAP-macro/{tag}", average_precision_score(y, y_hat_sig, average='macro'))
    print(f"mAP-micro/{tag} {average_precision_score(y, y_hat_sig, average='micro'):.4f}")
    print(f"mAP-macro/{tag} {average_precision_score(y, y_hat_sig, average='macro'):.4f}")
    print(f"AP_class/{tag} {average_precision_score(y, y_hat_sig, average=None)}")

    # F1
    writer.add_scalar(f"F1-micro/{tag}", f1_score(y, np.round(y_hat_sig), average='micro'))
    writer.add_scalar(f"F1-macro/{tag}", f1_score(y, np.round(y_hat_sig), average='macro'))
    print(f"F1-micro/{tag} {f1_score(y, np.round(y_hat_sig), average='micro'):.4f}")
    print(f"F1-macro/{tag} {f1_score(y, np.round(y_hat_sig), average='macro'):.4f}")
    print(f"F1-class/{tag} {f1_score(y, np.round(y_hat_sig), average=None)}")

    # Loss
    writer.add_scalar(f"Loss/{tag}", loss, e)
    print(f"Loss/{tag} {loss:.4f}")
    
    # Accuracy
    writer.add_scalar(f"Acc/{tag}", acc, e)
    print(f"Acc/{tag} {acc:.4f}")
