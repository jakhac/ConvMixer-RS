import torch
from torchmetrics.functional import accuracy
from datetime import datetime

import torch.optim as optim
import torch.nn as nn


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


def get_model_name(args):
    timestamp = datetime.now().strftime('%m-%d_%H%M_%S')

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
    
