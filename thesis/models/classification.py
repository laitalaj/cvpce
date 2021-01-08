import torch
from torch import nn
from torchvision import models as tmodels
from torchvision.models import VGG

class MACVGG(VGG): # TODO: Grad off from unnecessary layers
    def __init__(self, features, convs_per_block = [2, 2, 3, 3, 3], batch_norm = True):
        super().__init__(features, init_weights=False)
        layers_per_conv = 3 if batch_norm else 2 # conv, batch norm (conditionally), relu
        layers_per_block = [convs * layers_per_conv + 1 for convs in convs_per_block] # +1 for max pool
        self.cutoff_1 = sum(layers_per_block[:-1]) - 1 # last relu of second-to-last block
        self.cutoff_2 = sum(layers_per_block) - 1 # last relu of last block
    def forward(self, x):
        x = self.features[:self.cutoff_1](x)
        desc_1 = x.amax(dim=(-2, -1))
        x = self.features[self.cutoff_1 : self.cutoff_2](x)
        desc_2 = x.amax(dim=(-2, -1))
        return torch.cat(desc_1, desc_2, dim=1)
