import torch
import torch.nn as nn
from training_utils import get_activation
from torchmetrics.functional import accuracy


class ConvMixerLayer(nn.Module):
    
    def __init__(self, h, kernel_size=9, dilation=1, activation='GELU', residual=1, drop=0.0):
        super().__init__()

        self.residual = residual
        
        # Depthwise convolution layer
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=kernel_size, 
                groups=h, padding='same', dilation=dilation),
            get_activation(activation),
            nn.BatchNorm2d(h),
            nn.Dropout(p=drop, inplace=True)
        )
        
        # Pointwise convolution layer
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=1),
            get_activation(activation),
            nn.BatchNorm2d(h),
            nn.Dropout(p=drop, inplace=True)
        )
        
        
    def forward(self, x):
        # Combine both layers and (default) add a residual connection after depthwise convolution
        if self.residual & 1: x = x + self.depthwise_conv(x)
        else: x = self.depthwise_conv(x)

        if self.residual & 2: x = x + self.pointwise_conv(x)
        else: x = self.pointwise_conv(x)

        return x   


class ConvMixer(nn.Module):

    def __init__(self, input_dim, h, depth, kernel_size=9, patch_size=7,
                 n_classes=19, activation='GELU', dilation=1, residual=1, drop=0.0):
        super().__init__()

        # Patch embeddings as convolutions
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(input_dim, h, kernel_size=patch_size, stride=patch_size),
            get_activation(activation),
            nn.BatchNorm2d(h)
        )
        
        # Add depth-many ConvMixerLayer blocks
        self.ConvMixerLayers = nn.ModuleList([])
        for _ in range(depth):
            self.ConvMixerLayers.append(ConvMixerLayer(h=h, kernel_size=kernel_size, 
                activation=activation, dilation=dilation, residual=residual, drop=drop)
            )

        # Unroll patches and add classification layer
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(h, n_classes)
        )
        
        
    def forward(self,x):
        x = self.patch_embedding(x)
        for cml in self.ConvMixerLayers: x = cml(x)
        x = self.head(x)
        
        return x