# cvpce

Computer vision based planogram compliance evaluation.
Code for the master's thesis of Julius Laitala,
University of Helsinki, 2021.
The thesis is available at http://urn.fi/URN:NBN:fi:hulib-202106092585 .

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

## Datasets

The following public datasets were used for training and testing cvpce:
* GLN training and product proposal generation testing: [SKU-110K of Goldman et al. (2019)](https://github.com/eg4000/SKU110K_CVPR19)
* DIHE training: [the 2019 version of GP-180](https://alessiotonioni.github.io/publication/DIHE)
(Tonioni et al. 2017)
* Classification and product detection testing: [the Grocery Products dataset](https://github.com/tobiagru/ObjectDetectionGroceryProducts)
of George et al. (2014) with annotations from [GP-180 (the 2017 version)](https://alessiotonioni.github.io/publication/planogram)
* Planogram compliance testing: [planograms from GP-180 (2017 version)](https://alessiotonioni.github.io/publication/planogram)
with [fixes from this gist](https://gist.github.com/laitalaj/09778eab24c0d16b8447d6ca3360c7b2)
