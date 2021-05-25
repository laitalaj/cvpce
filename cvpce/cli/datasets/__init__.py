import click

from .gp import gp
from .grozi import grozi
from .internal import internal
from .sku110k import sku110k
from .misc import visualize_coco, visualize_coco_gaussians

@click.group()
def datasets():
    pass

groups = (gp, grozi, internal, sku110k)
commands = (visualize_coco, visualize_coco_gaussians)
for c in groups + commands:
    datasets.add_command(c)
