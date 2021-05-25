import warnings
# Ray likes to tell you that you don't have everything offered by installed when you just want to use Tune
warnings.filterwarnings('ignore', message='Not all Ray CLI dependencies were found')

import click

from .datasets import datasets
from .dihe import dihe
from .gln import gln
from .misc import misc
from .eval import eval_product_detection, rebuild_scene, eval_planograms, plot_planogram_eval

# TODO: A bunch of options could be moved up from commands to groups, eliminating a lot of duplicate code.

@click.group()
def cli():
    '''
    Computer vision based planogram compliance evaluation.
    Code for the master's thesis of Julius Laitala, University of Helsinki, 2021.

    This CLI tool is structured as a hierarchy;
    most of the commands contain further subcommands.
    Try

    \b
    cvpce <command> --help

    for more info on each command!
    '''
    pass

groups = (datasets, dihe, gln, misc)
commands = (eval_product_detection, rebuild_scene, eval_planograms, plot_planogram_eval)
for c in groups + commands:
    cli.add_command(c)

if __name__ == '__main__':
    cli()
