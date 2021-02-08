import torch
from torch import nn, linalg as tla
from torch.nn import functional as nnf
from torchvision.models import utils as tvutils
from torchvision.models import vgg

from .pix2pix.models import networks as p2pn # TODO: this doesn't work, maybe fork it and fix the models init?

class AveragingPatchGAN(nn.Module):
    def __init__(self, channels = 3, initial_filters = 64, typ = 'basic'):
        super().__init__()
        self.module = p2pn.define_D(channels, initial_filters, typ)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.module(x)
        x = self.activation(x)
        return x.mean((1, 2))

class MACVGG(vgg.VGG): # TODO: Grad off from unnecessary layers
    def __init__(self, config = 'D', convs_per_block = [2, 2, 3, 3, 3], batch_norm = True):
        super().__init__(vgg.make_layers(vgg.cfgs[config], batch_norm=batch_norm), init_weights=False)

        layers_per_conv = 3 if batch_norm else 2 # conv, batch norm (conditionally), relu
        layers_per_block = [convs * layers_per_conv + 1 for convs in convs_per_block] # +1 for max pool
        self.cutoff_1 = sum(layers_per_block[:-1]) - 1 # last relu of second-to-last block
        self.cutoff_2 = sum(layers_per_block) - 1 # last relu of last block
    def forward(self, x, eps = 1e-8):
        x = self.features[:self.cutoff_1](x)
        desc_1 = x.amax(dim=(-2, -1))
        x = self.features[self.cutoff_1 : self.cutoff_2](x)
        desc_2 = x.amax(dim=(-2, -1))
        desc = torch.cat((desc_1, desc_2), dim=1)
        return desc / max(tla.norm(desc), eps)

def distance(emb1, emb2, dim = 1):
    return 1 - nnf.cosine_similarity(emb1, emb2, dim=dim)

def nearest_neighbors(anchors, queries, k=1):
    anchor_indices = torch.arange(len(anchors), device=anchors.device)
    query_indices = torch.arange(len(queries), device=queries.device)
    q_mesh, a_mesh = torch.meshgrid(query_indices, anchor_indices)
    distances = distance(anchors[a_mesh], queries[q_mesh], dim=-1)
    if k == 1:
        return distances.argmin(dim=-1)
    else:
        return distances.argsort(dim=-1)[:, :k]

def macvgg_embedder(model = 'vgg16_bn', pretrained = True, progress = True):
    model_to_config = {
        'vgg16_bn': 'D'
    }
    if model not in model_to_config:
        raise NotImplementedError(f'MACVGG not implemented for {model}')

    embedder = MACVGG(model_to_config[model])
    if pretrained:
        state_dict = tvutils.load_state_dict_from_url(vgg.model_urls[model], progress=progress)
        embedder.load_state_dict(state_dict)
    
    return embedder

def unet_generator():
    return p2pn.define_G(3, 3, 64, 'unet_256')

def patchgan_discriminator():
    return AveragingPatchGAN()
