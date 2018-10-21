# -*- coding: utf-8 -*-

"""Top-level package for mit-d3m."""

__author__ = """MIT Data To AI Lab"""
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.0'

import os
import shutil
import tarfile

import boto3

from mit_d3m.dataset import D3MDS
from mit_d3m.loaders import get_loader
from mit_d3m.metrics import METRICS_DICT

DATA_PATH = 'data'
BUCKET = 'd3m-data-dai'


def download_dataset(bucket, dataset, root_dir):
    client = boto3.client('s3')

    print("Downloading dataset {}".format(dataset))

    key = 'datasets/' + dataset + '.tar.gz'
    filename = root_dir + '.tar.gz'

    print("Getting file {} from S3 bucket {}".format(key, bucket))
    client.download_file(Bucket=bucket, Key=key, Filename=filename)

    shutil.rmtree(root_dir, ignore_errors=True)

    print("Extracting {}".format(filename))
    with tarfile.open(filename, 'r:gz') as tf:
        tf.extractall(os.path.dirname(root_dir))


def load_d3mds(dataset, force_download=False):
    if dataset.endswith('_dataset_TRAIN'):
        dataset = dataset[:-14]

    root_dir = os.path.join(DATA_PATH, dataset)

    if force_download or not os.path.exists(root_dir):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        download_dataset(BUCKET, dataset, root_dir)

    phase_root = os.path.join(root_dir, 'TRAIN')
    dataset_path = os.path.join(phase_root, 'dataset_TRAIN')
    problem_path = os.path.join(phase_root, 'problem_TRAIN')

    return D3MDS(dataset=dataset_path, problem=problem_path)


def load_dataset(dataset, force_download=False):

    d3mds = load_d3mds(dataset, force_download)

    loader = get_loader(
        d3mds.get_data_modality(),
        d3mds.get_task_type()
    )

    dataset = loader.load(d3mds)

    dataset.scorer = METRICS_DICT[d3mds.get_metric()]

    return dataset
