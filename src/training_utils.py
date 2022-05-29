import torch
from sklearn.metrics import accuracy_score
from torchmetrics.functional import accuracy


def train_batch(train_loader, model, optimizer, loss_fn, gpu):
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
            X, y = X.cuda(gpu), y.cuda(gpu)
        
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


def validate_batch(val_loader, model, loss_fn, gpu):
    """Perform validation on val_loader images.

    Args:
        val_loader (DataLoader): data loader with validation data
        model (Model): model
        loss_fn (criterion): loss function
        accuracy (accuracy): accuracy object

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
                X, y = X.cuda(gpu), y.cuda(gpu)
        
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


def save_general_checkpoint(path, epoch, model, optimizer, val_loss):
    """Save a general checkpoint for either inference/further training.
    Loading this checkpoint requires the model's architecture beforehand.

    Args:
        path (string): path/to/checkpoint-file.ckpt
        epoch (int): epoch
        model (Model): model
        optimizer (optimizer): optimizer
        val_loss (float): validation loss
    """
    
    dict_state = {
        'epoch': epoch,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(dict_state, path)
    
    
def load_general_checkpoint(path):
    """Load a general checkpoint and return its dictionary.
    For usage see https://pytorch.org/tutorials/beginner/saving_loading_models.html#save

    Args:
        path (string): path/to/checkpoint-file.ckpt

    Returns:
        dict: state dictionary
    """
    
    return torch.load(path)


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
