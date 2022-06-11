import torch
import torch.nn as nn
from training_utils import get_activation
# from torchmetrics.functional import accuracy
from torchmetrics import Accuracy, F1Score, AveragePrecision

class ConvMixerLayer(nn.Module):
    
    def __init__(self, h, kernel_size=9, activation='GELU'):
        super().__init__()
        
        # Depthwise convolution layer
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=kernel_size, groups=h, padding='same'),
            get_activation(activation),
            nn.BatchNorm2d(h)
        )
        
        # Pointwise convolution layer
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=1),
            get_activation(activation),
            nn.BatchNorm2d(h)
        )
        
        
    def forward(self,x):
        # Combine both layers and add a residual connection after depthwise convolution
        x = x + self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x   


class ConvMixer(nn.Module):

    def __init__(self, input_dim, h, depth, kernel_size=9, patch_size=7,
                 n_classes=19, activation='GELU'):
        super().__init__()


        self.accuracy = Accuracy(subset_accuracy=True)

        self.mAP_micro = AveragePrecision(num_classes=19, average='micro', multiclass=True)
        self.mAP_macro = AveragePrecision(num_classes=19, average='macro', multiclass=True)
        self.mAP_class = AveragePrecision(num_classes=19, average=None, multiclass=True)

        self.f1_micro = F1Score(num_classes=19, mdmc_average='samplewise')
        self.f1_macro = F1Score(num_classes=19, average='macro', mdmc_average='global', multiclass=True)
        self.f1_class = F1Score(num_classes=19, average=None, mdmc_average='global', multiclass=True)

        
        # Patch embeddings as convolutions
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(input_dim, h, kernel_size=patch_size, stride=patch_size),
            get_activation(activation),
            nn.BatchNorm2d(h)
        )
        
        # Add depth-many ConvMixerLayer blocks
        self.ConvMixerLayers = nn.ModuleList([])
        for _ in range(depth):
            self.ConvMixerLayers.append(
                ConvMixerLayer(h=h, kernel_size=kernel_size, activation=activation)
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