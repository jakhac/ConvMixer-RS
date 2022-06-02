import argparse

import torch
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
import matplotlib.pyplot as plt
from datetime import datetime

import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist


def train_batches(train_loader, model, optimizer, gpu):
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
    print('train n_batches', n_batches)

    model.train()
    for X, y in train_loader:
        X, y = X.cuda(gpu), y.cuda(gpu)
        optimizer.zero_grad()
        outputs = model(X)
        
        # Keep track of accuracy in this epoch
        y_pred = get_predictions_for_batch(outputs)
        total_acc += accuracy(y_pred.to(torch.int), y.to(torch.int),
                              subset_accuracy=True)
        
        # Add loss to accumulator
        loss = model.module.loss(outputs, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    return (total_loss / n_batches), (total_acc / n_batches)


def valid_batches(val_loader, model, gpu):
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
    
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.cuda(gpu), y.cuda(gpu)
            outputs = model(X)
            
            # Keep track of accuracy in this epoch
            y_pred = get_predictions_for_batch(outputs)
            total_acc += accuracy(y_pred.to(torch.int), y.to(torch.int),
                                  subset_accuracy=True)   
            
            # Add loss to accumulator
            loss = model.module.loss(outputs, y)
            total_loss += loss.item()
            

    return (total_loss / n_batches), (total_acc / n_batches)


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


def get_history_plots(val_loss_hist, train_loss_hist, val_acc_hist, train_acc_hist):
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
    
    return fig


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
    parser.add_argument('--timestamp', type=str, default="",
                        help='unix timestamp to create unique folder')
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


def get_model_name(args):
    assert args.timestamp != ""

    timestamp = datetime.utcfromtimestamp(float(args.timestamp)).strftime('%m-%d_%H%M_%S')
    model_arch = f'CvMx-h={args.h}-d={args.depth}-k={args.k_size}-p={args.p_size}'
    model_config = f'batch={args.batch_size}_lr={args.lr}_mom={args.momentum}_{args.activation}_{args.optimizer}'

    return f'{timestamp}_{model_arch}_{model_config}'


def get_optimizer(model, args):
    if args.optimizer == 'SGD':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr)
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