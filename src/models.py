import torch
import torch.nn as nn
import layers


class ConvMixer(nn.Module):
    """Original ConvMixer as presented in "Patches Are All You Need?"

    - ConvEmbedding
    - Depthwise-Convolutions + Pointwise-Convolutions
    - Classification Head

    Args:
        nn (nn.Module): nn.Module
    """

    def __init__(self, input_dim, h, depth, kernel_size=9, patch_size=7,
                 n_classes=19, activation='GELU', dilation=1, residual=1, drop=0.0):
        super().__init__()

        # Patch embeddings as convolutions
        self.patch_embedding = layers.ConvEmbedding(input_dim, h, patch_size, activation)
        
        # Add depth-many ConvMixerLayer blocks
        self.ConvMixerLayers = nn.ModuleList([])
        for _ in range(depth):
            self.ConvMixerLayers.append(layers.ConvMixerLayer(
                h=h,
                kernel_size=kernel_size,
                activation=activation,
                dilation=dilation,
                residual=residual,
                drop=drop)
            )

        # Unroll patches and add classification layer
        self.head = layers.ClassificationHead(h, n_classes)
        
        
    def forward(self,x):
        x = self.patch_embedding(x)
        for cml in self.ConvMixerLayers: x = cml(x)
        return self.head(x)


class ConvChannelMixer(nn.Module):
    """Modified ConvMixer model that only applies pointwise-convolutions.

    - ConvEmbedding
    - Pointwise-Convolutions
    - Classification Head

    Args:
        nn (nn.Module): nn.Module
    """

    def __init__(self, input_dim, h, depth, patch_size=7, n_classes=19, activation='GELU'):
        super().__init__()

        # Patch embeddings
        self.patch_embedding = layers.ConvEmbedding(input_dim, h, patch_size, activation)
        
        # Add depth-many mixing blocks
        self.MixingLayers = nn.ModuleList([])
        for _ in range(depth):
            self.MixingLayers.append(layers.ConvChannelMixerLayer(h, activation))

        # Unroll patches and add classification layer
        self.head = layers.ClassificationHead(h, n_classes)
        
        
    def forward(self,x):
        x = self.patch_embedding(x)
        for mix in self.MixingLayers: x = mix(x)
        return self.head(x)


class ChannelMixer(nn.Module):
    """Modified ConvMixer where input embeddings are produced by pooling and 1x1 convolutions. Only channel mixing is applied.

    - MaxPoolEmbedding
    - Pointwise-Convolutions
    - Classification Head

    Args:
        nn (nn.Module): nn.Module
    """

    def __init__(self, input_dim, h, depth, patch_size=7, n_classes=19, activation='GELU'):
        super().__init__()

        # Patch embeddings
        self.patch_embedding = layers.MaxPoolEmbedding(input_dim, h, patch_size, activation)
        
        # Add depth-many mixing blocks
        self.MixingLayers = nn.ModuleList([])
        for _ in range(depth):
            self.MixingLayers.append(layers.ConvChannelMixerLayer(h, activation))

        # Unroll patches and add classification layer
        self.head = layers.ClassificationHead(h, n_classes)
        
        
    def forward(self,x):
        x = self.patch_embedding(x)
        for mix in self.MixingLayers: x = mix(x)
        return self.head(x)
