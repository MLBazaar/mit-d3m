<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“mit-d3m” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>


[![Travis](https://travis-ci.org/HDI-Project/mit-d3m.svg?branch=master)](https://travis-ci.org/HDI-Project/mit-d3m)
[![PyPi Shield](https://img.shields.io/pypi/v/mit-d3m.svg)](https://pypi.python.org/pypi/mit-d3m)


# mit-d3m

- License: MIT
- Documentation: https://HDI-Project.github.io/mit-d3m/
- Homepage: https://github.com/HDI-Project/mit-d3m

# Overview

MIT tools to work with D3M datasets.

# Install

## Requirements

**mit-d3m** has been developed and tested on [Python 3.5, 3.6 and 3.7](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **mit-d3m** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **mit-d3m**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) mit-d3m-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source mit-d3m-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **mit-d3m**!

## Install with pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **mit-d3m**:

```bash
pip install mit-d3m
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from source

Alternatively, with your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:HDI-Project/mit-d3m.git
cd mit-d3m
git checkout stable
make install
```

For development, you can use `make install-develop` instead in order to install all
the required dependencies for testing and code linting.
