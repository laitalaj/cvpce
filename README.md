# cvpce

Computer vision based planogram compliance evaluation.
Code for the master's thesis of Julius Laitala,
University of Helsinki, 2021

This README's under construction,
and the repo needs a bit of refactoring.
Come back later!

## Installation

Currently, the functions in cvpce are set up to run only on CUDA.
Therefore,
you'll need a NVidia card to run cvpce.
We suggest using [Conda](https://docs.conda.io/en/latest/)
to avoid CUDA installation pains.
To create a Conda environment for cvpce,
simply utilize the provided `environment.yml`:
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
