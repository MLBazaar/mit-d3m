# -*- coding: utf-8 -*-

"""Top-level package for mit-d3m."""

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.2.1-dev0'

import os
import shutil
import tarfile

import boto3
import botocore
import botocore.config
from funcy import memoize

from mit_d3m.dataset import D3MDS
from mit_d3m.loaders import get_loader
from mit_d3m.metrics import METRICS_DICT
from mit_d3m.utils import contains_files

__all__ = (
    'DATA_PATH',
    'BUCKET',
    'load_d3mds',
    'load_dataset',
)


DATA_PATH = 'data'
BUCKET = 'd3m-data-dai'


@memoize
def get_client():
    config = botocore.config.Config(signature_version=botocore.UNSIGNED)
    return boto3.client('s3', config=config)


def get_dataset_tarfile_path(datapath, dataset):
    return os.path.join(datapath, '{dataset}.tar.gz'.format(dataset=dataset))


def get_dataset_dir(datapath, dataset):
    return os.path.join(datapath, dataset)


def get_dataset_s3_key(dataset):
    return 'datasets/{dataset}.tar.gz'.format(dataset=dataset)


def download_dataset(bucket, key, filename):
    print("Downloading dataset from s3://{bucket}".format(bucket=bucket))
    client = get_client()
    client.download_file(Bucket=bucket, Key=key, Filename=filename)


def extract_dataset(src, dst):
    print("Extracting {}".format(src))
    shutil.rmtree(dst, ignore_errors=True)
    with open(src, 'r') as f:
        tf = tarfile.TarFile(mode='r:gz', fileobj=f)
        tf.extractall(dst)


def load_d3mds(dataset, root=DATA_PATH, force_download=False):
    if not os.path.exists(root):
        os.makedirs(root)

    if dataset.endswith('_dataset_TRAIN'):
        dataset = dataset[:len('_dataset_TRAIN')]

    dataset_dir = get_dataset_dir(root, dataset)
    dataset_tarfile = get_dataset_tarfile_path(dataset_dir, dataset)
    dataset_key = get_dataset_s3_key(dataset)

    if force_download or not os.path.exists(dataset_tarfile):
        download_dataset(BUCKET, dataset_key, dataset_tarfile)

    if force_download or not os.path.exists(dataset_dir) or not contains_files(dataset_dir):
        extract_dataset(dataset_tarfile, dataset_dir)

    phase_root = os.path.join(dataset_dir, 'TRAIN')
    dataset_path = os.path.join(phase_root, 'dataset_TRAIN')
    problem_path = os.path.join(phase_root, 'problem_TRAIN')

    return D3MDS(dataset=dataset_path, problem=problem_path)


def load_dataset(dataset, root=DATA_PATH, force_download=False, ignore_memory_errors=False):

    d3mds = load_d3mds(dataset, root, force_download=force_download)

    loader = get_loader(
        d3mds.get_data_modality(),
        d3mds.get_task_type()
    )

    dataset = None
    try:
        dataset = loader.load(d3mds)
        dataset.scorer = METRICS_DICT[d3mds.get_metric()]
    except MemoryError:
        if not ignore_memory_errors:
            raise

    return dataset
