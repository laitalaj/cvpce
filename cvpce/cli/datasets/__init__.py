import click

from .gp import gp
from .grozi import grozi
from .internal import internal
from .sku110k import sku110k
from .misc import visualize_coco, visualize_coco_gaussians

@click.group()
def datasets():
    '''
    Dataset visualization and interaction.

    This command group contains further groups and commands
    for visualizing and fixing or otherwise interacting with the various datasets.

    The actual commands under this don't contain help texts,
    sorry about that!
    I'll try to have time to add those in the future.
    '''
    pass

groups = (gp, grozi, internal, sku110k)
commands = (visualize_coco, visualize_coco_gaussians)
for c in groups + commands:
    datasets.add_command(c)
