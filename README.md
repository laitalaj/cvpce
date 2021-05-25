# cvpce

Computer vision based planogram compliance evaluation.
Code for the master's thesis of Julius Laitala,
University of Helsinki, 2021

## Installation

Currently, the functions in cvpce are set up to run only on CUDA.
Therefore,
you'll need a NVidia card to run cvpce.
We suggest using [Conda](https://docs.conda.io/en/latest/)
to avoid CUDA installation pains.
To create a Conda environment for cvpce,
simply utilize the provided [`environment.yml`](./environment.yml):
```sh
conda env create -f environment.yml
conda activate cvpce
```

With the Conda environment set up and activated,
cvpce can be installed with setuptools:
```sh
pip install .
```
If you wish to tweak cvpce a bit,
the `-e` flag is your friend!

## Usage

cvpce is a command line tool,
and a bunch of usage instructions can be accessed with the `--help` option.
Go ahead and
```sh
cvpce --help
```
after installing to explore the available commands!

## Pre-trained weights

Pre-trained model weights are available in
[the releases.](https://github.com/laitalaj/cvpce/releases/)
