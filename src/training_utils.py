import torch
from sklearn.metrics import accuracy_score
from torchmetrics.functional import accuracy
import matplotlib.pyplot as plt
from datetime import datetime

import torch.optim as optim
import torch.nn as nn


def train_batches(train_loader, model, optimizer, loss_fn, dev):
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
    
    n_batches = len(train_loader)
    
    train_loss_accu = 0.0
    train_acc_accu = 0.0
    model.train()
    for batch_idx, (X, y) in enumerate(train_loader):

        # Transfer to GPU if available
        if torch.cuda.is_available():
            X, y = X.to(dev), y.to(dev)
        
        # Clear gradients and pass data through network
        optimizer.zero_grad()
        outputs = model(X)
        
        # Keep track of accuracy in this epoch
        y_pred = get_predictions_for_batch(outputs)
        train_acc_accu += get_accuracy_for_batch(y_pred.to(torch.int), y.to(torch.int))
        
        # Add loss to accumulator
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        
        # Keep track of loss
        train_loss_accu += loss.item()
        
        
    train_loss = train_loss_accu / n_batches
    train_acc = train_acc_accu / n_batches
    
    return train_loss, train_acc


def valid_batches(val_loader, model, loss_fn, dev):
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
    
    val_loss_accu = 0.0
    val_acc_accu = 0.0
    
    model.eval()
    with torch.no_grad():
        
        for X, y in val_loader:
            
            # Transfer to GPU if available
            if torch.cuda.is_available():
                X, y = X.to(dev), y.to(dev)
        
            outputs = model(X)
            
            # Keep track of accuracy in this epoch
            y_pred = get_predictions_for_batch(outputs)
            val_acc_accu += get_accuracy_for_batch(y_pred.to(torch.int), y.to(torch.int))
                
            # Add loss to accumulator
            loss = loss_fn(outputs, y)
            val_loss_accu += loss.item()
            
        
    val_loss = val_loss_accu / n_batches
    val_acc = val_acc_accu / n_batches
    
    return val_loss, val_acc


def save_complete_model(path, model):
    """Save a complete model. Loading does not require architecture beforehand.add()
    Args:
        path (string): path/to/model-file.pt
        model (Model): model to save
    """
    
    torch.save(model, path)
    

def load_complete_model(path):
    """Load complete model from path.add()
    Args:
        path (string): path/to/model-file.pt
    Returns:
        Model: complete model
    """
    
    return torch.load(path)


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


def get_accuracy_for_batch(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    
    return accuracy(y_pred, y_true, subset_accuracy=True)


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


def get_logging_dirs(args):
    """Determine model-architecture directory (4 hyperparameters) and
    the model name based on training (..) parameters

    Args:
        args (args): args

    Returns:
        (string, string): model_arch, model_name
    """

    timestamp = datetime.now().strftime('%m-%d_%H%M_%S')
    model_arch = f'CvMx-{args.h}-{args.depth}-{args.k_size}-{args.p_size}'
    model_name = timestamp + f'-batch={args.batch_size}_lr={args.lr}_mom={args.momentum}_{args.activation}_{args.optimizer}'

    return model_arch, model_name

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