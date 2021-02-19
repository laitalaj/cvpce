import torch
from torch import nn, linalg as tla
from torch.nn import functional as nnf
from torchvision.models import utils as tvutils
from torchvision.models import vgg
from torchvision.transforms import functional as ttf

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

class MACVGG(nn.Module): # TODO: calc convs_per_block based on the actual config
    def __init__(self, config = 'D', convs_per_block = [2, 2, 3, 3, 3], batch_norm = True, vgg_state_dict = None):
        super().__init__()

        source_vgg = vgg.VGG(vgg.make_layers(vgg.cfgs[config], batch_norm=batch_norm), init_weights=vgg_state_dict is None)
        if vgg_state_dict is not None:
            source_vgg.load_state_dict(vgg_state_dict)

        layers_per_conv = 3 if batch_norm else 2 # conv, batch norm (conditionally), relu
        layers_per_block = [convs * layers_per_conv + 1 for convs in convs_per_block] # +1 for max pool
        cutoff_1 = sum(layers_per_block[:-1]) - 1 # last relu of second-to-last block
        cutoff_2 = sum(layers_per_block) - 1 # last relu of last block

        self.block1 = source_vgg.features[:cutoff_1]
        self.block2 = source_vgg.features[cutoff_1 : cutoff_2]
    def forward(self, x, eps = 1e-8):
        x = self.block1(x)
        desc_1 = x.amax(dim=(-2, -1))
        x = self.block2(x)
        desc_2 = x.amax(dim=(-2, -1))
        desc = torch.cat((desc_1, desc_2), dim=1)
        return desc / tla.norm(desc, dim=1, keepdim=True).clamp(min=eps)

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
        'vgg16': ('D', False),
        'vgg16_bn': ('D', True),
    }
    if model not in model_to_config:
        raise NotImplementedError(f'MACVGG not implemented for {model}')

    state_dict = tvutils.load_state_dict_from_url(vgg.model_urls[model], progress=progress) if pretrained else None
    config, batchnorm = model_to_config[model]
    embedder = MACVGG(config, vgg_state_dict=state_dict, batch_norm=batchnorm)

    return embedder

def unet_generator():
    return p2pn.define_G(3, 3, 64, 'unet_256')

def patchgan_discriminator():
    return AveragingPatchGAN()
