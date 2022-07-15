import torch
import torch.nn as nn
import training_utils
from torchmetrics.functional import accuracy



class MaxPoolEmbedding(nn.Module):
    """Generate input embedding by 1x1-convs to increase number of channels and then applies
    patch_size x patch_size pooling to generate patches.

    Args:
        nn (nn.Module): nn.Modue
    """

    def __init__(self, input_dim, h, patch_size, activation):
        super().__init__()

        # Patch embeddings:
        # - Pointwise convolutions to increase number of channels
        # - 5x5-Pooling to generate patches and also include some kind of spatial-operation
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(input_dim, h, kernel_size=1),
            nn.MaxPool2d((patch_size, patch_size)),
            training_utils.get_activation(activation),
            nn.BatchNorm2d(h)
        )
        
        
    def forward(self, x):
        return self.patch_embedding(x)


class ConvEmbedding(nn.Module):
    """Generate input embedding by convolutions with adjusted stride.

    Args:
        nn (nn.Module): nn.Modue
    """

    def __init__(self, input_dim, h, patch_size, activation):
        super().__init__()

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(input_dim, h, kernel_size=patch_size, stride=patch_size),
            training_utils.get_activation(activation),
            nn.BatchNorm2d(h)
        )
        
    def forward(self, x):
        return self.patch_embedding(x)



class ConvChannelMixerLayer(nn.Module):
    """Layer that only applies a pointwise convolution (only channel-mixing).

    Args:
        nn (nn.Module): nn.Module
    """

    def __init__(self, h, activation='GELU'):
        super().__init__()

        # Pointwise convolution layer
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=1),
            training_utils.get_activation(activation),
            nn.BatchNorm2d(h),
        )
        
    def forward(self, x):
        return self.pointwise_conv(x)


class ConvMixerLayer(nn.Module):
    """Layer that applies depthwise- and pointwise-convolutions. Based on original laer
    from "Patches Is All You Need?" paper.

    Args:
        nn (nn.Module): nn.Module
    """
    
    def __init__(self, h, kernel_size=9, dilation=1, activation='GELU'):
        super().__init__()

        # Depthwise convolution layer
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=kernel_size, 
                groups=h, padding='same', dilation=dilation),
            training_utils.get_activation(activation),
            nn.BatchNorm2d(h),
        )
        
        # Pointwise convolution layer
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=1),
            training_utils.get_activation(activation),
            nn.BatchNorm2d(h),
        )
        
        
    def forward(self, x):
        # Combine both layers and (default) add a residual connection after depthwise convolution
        x = x + self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class ClassificationHead(nn.Module):
    """Classification head to produce logits.

    Args:
        nn (nn.Module): nn.Module
    """

    def __init__(self, h, n_classes):
        super().__init__()

        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(h, n_classes)
        )
        
    def forward(self, x):
        return self.classification(x)