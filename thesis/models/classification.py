import torch
from torch import nn, linalg as tla
from torch.nn import functional as nnf
from torchvision.models import utils as tvutils
from torchvision.models import vgg, resnet
from torchvision.transforms import functional as ttf

from .pix2pix.models import networks as p2pn

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

    embedding_size = 512 * 2

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
        # Mean and STD as expected by pretrained Torch models (from https://pytorch.org/docs/stable/torchvision/models.html )
        # but scaled to match the -1 to 1 scale
        x = ttf.normalize(x,
            [0.485 * 2 - 1, 0.456 * 2 - 1, 0.406 * 2 - 1],
            [0.229 * 2, 0.224 * 2, 0.225 * 2]
        )

        x = self.block1(x)
        desc_1 = x.amax(dim=(-2, -1))
        x = self.block2(x)
        desc_2 = x.amax(dim=(-2, -1))
        desc = torch.cat((desc_1, desc_2), dim=1)
        return desc / tla.norm(desc, dim=1, keepdim=True).clamp(min=eps)

class MACResNet(nn.Module):
    layer_output_sizes = [64, 256, 512, 1024, 2048]
    def __init__(self, source_resnet, descriptor_layers=[2, 3]):
        super().__init__()

        layers = [
            nn.Sequential(
                source_resnet.conv1,
                source_resnet.bn1,
                source_resnet.relu,
                source_resnet.maxpool,
            ),
            source_resnet.layer1,
            source_resnet.layer2,
            source_resnet.layer3,
            source_resnet.layer4,
        ]

        prev = 0
        self.blocks = nn.ModuleList()
        for l in descriptor_layers:
            self.blocks.append(nn.Sequential(*layers[prev:l + 1]))
            prev = l + 1

        self.embedding_size = sum([self.layer_output_sizes[l] for l in descriptor_layers])
    def forward(self, x, eps = 1e-8):
        desc = []
        for b in self.blocks:
            x = b(x)
            b_desc = x.amax(dim=(-2, -1))
            desc.append(b_desc)
        desc = torch.cat(desc, dim=1)
        return desc / tla.norm(desc, dim=1, keepdim=True).clamp(min=eps)

def distance(emb1, emb2, dim = 1):
    return 1 - nnf.cosine_similarity(emb1, emb2, dim=dim)

def nearest_neighbors(anchors, queries, k=1):
    anchor_indices = torch.arange(len(anchors), device=anchors.device)
    query_indices = torch.arange(len(queries), device=queries.device)
    q_mesh, a_mesh = torch.meshgrid(query_indices, anchor_indices)
    distances = distance(anchors[a_mesh], queries[q_mesh], dim=-1)
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

def macresnet_encoder(model = 'resnet50', pretrained = True, progress = True, batch_norm = True, desc_layers = [2, 3]):
    model_to_config = {
        'resnet50': (resnet.Bottleneck, [3, 4, 6, 3]),
    }
    if model not in model_to_config:
        raise NotImplementedError(f'MACResNet not implemented for {model}')

    block, layers = model_to_config[model]
    norm_layer = nn.BatchNorm2d if batch_norm else nn.Identity
    source_resnet = resnet._resnet(model, block, layers, pretrained, progress, norm_layer = norm_layer)
    return MACResNet(source_resnet, desc_layers)

def unet_generator(masked = False):
    return p2pn.define_G(4 if masked else 3, 3, 64, 'unet_256')

def patchgan_discriminator():
    return AveragingPatchGAN()
