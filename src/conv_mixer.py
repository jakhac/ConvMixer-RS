import torch.nn as nn

# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x):
#         return self.fn(x) + x


# def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
#     return nn.Sequential(
#         nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
#                     nn.GELU(),
#                     nn.BatchNorm2d(dim)
#                 )),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)],
#         nn.AdaptiveAvgPool2d((1,1)),
#         nn.Flatten(),
#         nn.Linear(dim, n_classes)
#     )
    
    
class ConvMixerLayer(nn.Module):
    
    def __init__(self, h, kernel_size=9):
        super().__init__()
        
        # Depthwise convolution layer
        self.depthwise_conv =  nn.Sequential(
            nn.Conv2d(h, h, kernel_size=kernel_size, groups=h, padding='same'),
            nn.GELU(),
            nn.BatchNorm2d(h)
        )
        
        # Pointwise convolution layer
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(h)
        )
        
        
    def forward(self,x):
        # Combine both layers and add a residual connection after depthwise convolutions
        x = x + self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x   


class ConvMixer(nn.Module):
    
    def __init__(self, input_dim, h, depth, kernel_size=9, patch_size=7, n_classes=19):
        super().__init__()
        
        # Patch embeddings as convolutions
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(input_dim, h, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(h)
        )
        
        # Add depth-many ConvMixerLayer blocks
        self.ConvMixerLayers = nn.ModuleList([])
        for _ in range(depth):
            self.ConvMixerLayers.append(ConvMixerLayer(h=h, kernel_size=kernel_size))

        # Unroll patches and add classification layer
        self.head =  nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(h, n_classes)
        )
        
        
    def forward(self,x):

        x = self.patch_embedding(x)
        for cml in self.ConvMixerLayers: x = cml(x)
        x = self.head(x)
        
        return x