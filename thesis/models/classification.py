import torch
from torch import nn
from torchvision.models import utils as tvutils
from torchvision.models import vgg

from .pix2pix.models import networks as p2pn

class MACVGG(vgg.VGG): # TODO: Grad off from unnecessary layers
    def __init__(self, model = 'vgg16_bn', convs_per_block = [2, 2, 3, 3, 3], batch_norm = True):
        super().__init__(vgg.make_layers(vgg.cfgs[model]), init_weights=False)

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

class DIHE(nn.Module):
    def __init__(self, embedder):
        super().__init__()
        self.generator = p2pn.define_G(3, 3, 64, 'unet_256')
        self.discriminator = p2pn.define_D(3, 64, 'basic')
        self.embedder = embedder
    def forward(self, positive, negative, reference): # TODO: Hierarchy stuff
        # positive, negative: in vitro samples for generator and embedder
        # reference: in situ sample for discriminator
        anchor = self.generator(positive)
        anchor_score = self.discriminator(anchor)
        reference_score = self.discriminator(reference) #TODO: Diskriminaattorii halutaan ehkä päivittää kahessa stepissä? Mahdollisesti osana tätä forwardii?

        anchor = self.embedder(anchor)
        positive = self.embedder(positive)
        negative = self.embedder(negative)

        return anchor, positive, negative


def macvgg_embedder(model = 'vgg16_bn', pretrained = True, progress = True):
    if model != 'vgg16_bn':
        raise NotImplementedError(f'MACVGG convs_per_block not implemented for {model}')

    model = MACVGG(model)
    if pretrained:
        state_dict = tvutils.load_state_dict_from_url(vgg.model_urls[model], progress=progress)
        model.load_state_dict(state_dict)
    
    return model

def standard_dihe():
    return DIHE(macvgg_embedder())
