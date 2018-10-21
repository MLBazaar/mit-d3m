# -*- coding: utf-8 -*-

import argparse
import logging
import os

import pandas as pd

from mit_d3m.config import build_config
from mit_d3m.dataset import D3MDS
from mit_d3m.utils import disk_usage, logging_setup, make_abs

LOGGER = logging.getLogger(__name__)


def get_d3mds(dataset, path, phase, problem):
    config = build_config(dataset, path, phase, problem)
    dataset_key = 'training' if phase == 'TRAIN' else 'test'
    d3mds = D3MDS(
        dataset=config[dataset_key + '_data_root'],
        problem=config['problem_root']
    )
    return d3mds


def get_dataset_stats(dataset, path, problem):
    train_d3mds = get_d3mds(dataset, path, 'TRAIN', problem)
    test_d3mds = get_d3mds(dataset, path, 'TEST', problem)

    train_shape = train_d3mds.get_data()[0].shape
    test_shape = test_d3mds.get_data()[0].shape

    size = disk_usage(os.path.join(path, dataset, dataset + '_dataset'))
    size_human = disk_usage(os.path.join(path, dataset, dataset + '_dataset'), True)

    if problem:
        dataset = dataset + '_' + problem

    return {
        'dataset': dataset,
        'dataset_id': train_d3mds.dataset_id,
        'problem_id': train_d3mds.problem_id,
        'data_modality': train_d3mds.get_data_modality(),
        'task_type': train_d3mds.get_task_type(),
        'task_subtype': train_d3mds.get_task_subtype(),
        'metric': train_d3mds.get_metric(),
        'target': train_d3mds.target_column,
        'train_samples': train_shape[0],
        'train_features': train_shape[1],
        'test_samples': test_shape[0],
        'test_features': test_shape[1],
        'size': size,
        'size_human': size_human
    }


def get_problems(dataset, path):
    dataset_path = os.path.join(path, dataset)
    folders = os.listdir(dataset_path)
    problems = []
    for folder in folders:
        if folder == 'TRAIN':
            problems.append(None)

        if folder.startswith('TRAIN_'):
            problems.append(folder.replace('TRAIN_', ''))

    return problems


def get_stats(datasets, path):
    data = []
    for dataset in datasets:
        for problem in get_problems(dataset, path):
            try:
                stats = get_dataset_stats(dataset, path, problem)
            except Exception as e:
                LOGGER.exception("Exception in dataset %s", dataset)
                stats = {
                    'dataset': dataset,
                    'error': str(e),
                }

            data.append(stats)

    return pd.DataFrame(data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get Dataset summary')
    parser.add_argument('-i', '--input', default='data/datasets', nargs='?')
    parser.add_argument('-o', '--output', nargs='?')
    parser.add_argument('datasets', nargs='*')

    args = parser.parse_args()

    logging_setup()

    args.input = make_abs(args.input, os.getcwd())

    if not args.datasets:
        args.datasets = os.listdir(args.input)

    print("Processing Datasets: {}".format(args.datasets))

    output = get_stats(args.datasets, args.input)

    if args.output:
        print("Storing report as {}".format(args.output))
        output.to_csv(args.output, index=False)

    else:
        print(output)
