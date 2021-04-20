import torch
from torch import nn
from torch.nn import functional as nnf
from torchvision import models as tmodels
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

from .. import utils

class StateLoggingLayer(ExtraFPNBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.p6p7 = LastLevelP6P7(in_channels, out_channels)
    def forward(self, p, c, names):
        p, names = self.p6p7(p, c, names)
        print(f"names: {names}, c: {len(c)}, p: {len(p)}")
        print(', '.join(f'C{i}: {c[i].shape}' for i in range(len(c))))
        print(', '.join(f'P{i}: {p[i].shape}' for i in range(len(p))))
        return p, names

class LoggingTransform(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform
    def postprocess(self, result, image_shapes, original_image_sizes):
        return self.base_transform.postprocess(result, image_shapes, original_image_sizes)
    def forward(self, images, targets):
        def pretty_targets(t):
            return {k: v.shape if torch.is_tensor(v) else v for k, v in t.items()}
        print(f'Before: {[i.shape for i in images]}, {[pretty_targets(t) for t in targets]}')
        images, targets = self.transform(images, targets)
        for t in images.tensors:
            utils.show(t)
        print(f'After: {images.tensors.shape} w/ {images.image_sizes}, {[pretty_targets(t) for t in targets]}')
        return images, targets

class SizeCapturingTransform(nn.Module):
    def __init__(self, base_transform):
        super().__init__()
        self.base_transform = base_transform
        self.image_sizes = None
    def postprocess(self, result, image_shapes, original_image_sizes):
        return self.base_transform.postprocess(result, image_shapes, original_image_sizes)
    def forward(self, images, targets):
        image_list, targets = self.base_transform(images, targets)
        self.image_sizes = image_list.image_sizes
        return image_list, targets

class GaussianLayerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)

class GaussianLayer(nn.Module):
    def __init__(self, c_channels, p_channels):
        super().__init__()
        self.lateral = nn.Conv2d(c_channels, p_channels, 1)
        self.block1 = GaussianLayerBlock(p_channels, p_channels//2)
        self.block2 = GaussianLayerBlock(p_channels//2, p_channels//4)
        self.up = nn.Upsample(scale_factor=2)

        nn.init.xavier_normal_(self.lateral.weight)
        nn.init.constant_(self.lateral.bias, 0)
    def forward(self, x, p):
        x = self.lateral(x) + self.up(p)
        x = self.block1(x)
        x = self.block2(x)
        return self.up(x)

class GaussianSubnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, tanh=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=1 if kernel > 1 else 0)
        self.activation = nn.Tanh() if tanh else nn.ReLU()

        if tanh:
            nn.init.xavier_normal_(self.conv.weight, gain=nn.init.calculate_gain('tanh'))
        else:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)
    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)

class GaussianSubnet(nn.Module):
    def __init__(self, in_channels, tanh=False):
        super().__init__()
        self.blocks = nn.Sequential(
            GaussianSubnetBlock(in_channels, in_channels//2, 3),
            GaussianSubnetBlock(in_channels//2, in_channels//2, 3),
            GaussianSubnetBlock(in_channels//2, in_channels//4, 3),
            GaussianSubnetBlock(in_channels//4, in_channels//4, 1),
            GaussianSubnetBlock(in_channels//4, 1, 1, tanh),
        )
    def forward(self, x):
        return self.blocks.forward(x)

class BackboneWithFPNAndGaussians(BackboneWithFPN): # todo: gaussian layer on-off switch
    # see https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py#L11
    def __init__(self, backbone, extra_fpn_block=LastLevelP6P7, tanh=False):
        return_layers = {f'layer{i + 1}': str(i) for i in range(4)}

        fpn_layers = [2, 3, 4]
        in_channels_list = [(backbone.inplanes // 8) * 2 ** (i - 1) for i in fpn_layers]

        out_channels = 256
        extra_blocks = extra_fpn_block(out_channels, out_channels)

        super().__init__(backbone, return_layers, in_channels_list, out_channels, extra_blocks)

        c2_channels = 256
        self.gaussian_layer = GaussianLayer(c2_channels, out_channels)
        self.gaussian_subnet = GaussianSubnet(out_channels//4, tanh)
        self.gaussians = None
    def get_gaussians(self):
        g = self.gaussians
        self.gaussians = None #making sure these are used only once and the memory is cleared
        return g
    def forward(self, x):
        x = self.body(x)

        _, first_features = x.popitem(last = False)
        p = self.fpn(x) # missing the c2 = first feature maps at this point, as expected by RetinaNet

        gl_features = self.gaussian_layer(first_features, next(iter(p.values())))
        self.gaussians = self.gaussian_subnet(gl_features)

        return p

def gaussian_loss(predictions, targets, sizes, tanh=False, negative_threshold=0.0, positive_threshold=0.1, min_negatives=1000, negatives_per_positive=3):
    batch_targets = torch.full_like(predictions, -1) if tanh else torch.zeros_like(predictions) # can't be done on dataset level due to "unpredictability" -> probably could tune RetinaNet a bit
    for i, target_and_size in enumerate(zip(targets, sizes)):
        target, size = target_and_size
        target = target[None, None]
        size = tuple(s // 2 for s in size)
        target = nnf.interpolate(target, size=size, mode='bilinear', align_corners=False)
        batch_targets[i, 0, :size[0], :size[1]] = target

    negative_mask = batch_targets <= negative_threshold
    positive_mask = batch_targets >= positive_threshold

    se = nnf.mse_loss(predictions, batch_targets, reduction='none')
    positive_se = se[positive_mask]
    negative_se = se[negative_mask]

    top_negatives = max(min_negatives, negatives_per_positive * len(positive_se))
    top_indices = negative_se.argsort(descending = True)[:top_negatives]

    return (positive_se.sum() + negative_se[top_indices].sum()) / (len(positive_se) + len(top_indices))

class GaussianLayerNetwork(RetinaNet):
    def __init__(self, resnet, num_classes, extra_fpn_block=LastLevelP6P7, transform_wrapper=SizeCapturingTransform, gaussian_loss_params={}, tanh=False, **kwargs):
        # detections_per_img: 1000 > 576 in SKU110K train, 718 in val, 533 in test > 300 (default)
        super().__init__(BackboneWithFPNAndGaussians(resnet, extra_fpn_block, tanh=tanh), num_classes, detections_per_img=1000, **kwargs)
        self.transform = transform_wrapper(self.transform)
        self.gaussian_loss_params = gaussian_loss_params
    def compute_loss(self, targets, head_outputs, anchors):
        loss = super().compute_loss(targets, head_outputs, anchors)

        predicted_gaussians = self.backbone.get_gaussians()
        loss['gaussian'] = gaussian_loss(predicted_gaussians, [t['gaussians'] for t in targets], self.transform.image_sizes, **self.gaussian_loss_params)

        return loss
    def forward(self, images, targets = None):
        res = super().forward(images, targets)
        if not self.training:
            for r, g in zip(res, self.backbone.get_gaussians()):
                r['gaussians'] = g
        return res

def gln_backbone(trainable_layers=5, pretrained=True):
    backbone = tmodels.resnet50(pretrained=pretrained, norm_layer=FrozenBatchNorm2d)

    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return backbone

def state_logging_gln(num_classes = 1, trainable_layers=5):
    model = GaussianLayerNetwork(
        gln_backbone(trainable_layers),
        num_classes,
        extra_fpn_block=StateLoggingLayer,
        transform_wrapper=lambda t: SizeCapturingTransform(LoggingTransform(t))
    )
    return model

def gln(num_classes = 1, trainable_layers=4, pretrained_backbone=True, tanh=False, gaussian_loss_params = {}):
    return GaussianLayerNetwork(gln_backbone(trainable_layers, pretrained_backbone), num_classes, tanh=tanh, gaussian_loss_params=gaussian_loss_params)
