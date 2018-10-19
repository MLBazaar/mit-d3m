# -*- coding: utf-8 -*-

import argparse
import json
import os

from mit_d3m.utils import make_abs


def build_config(dataset, datasets_dir, phase, problem=None, output_dir='data/output'):
    """
    root@d3m-example-pod:/# cat /input/185_baseball/test_config.json
    {
      "problem_schema": "/input/TEST/problem_TEST/problemDoc.json",
      "problem_root": "/input/TEST/problem_TEST",
      "dataset_schema": "/input/TEST/dataset_TEST/datasetDoc.json",
      "test_data_root": "/input/TEST/dataset_TEST",
      "results_root": "/output/predictions",
      "executables_root": "/output/executables",
      "temp_storage_root": "/output/supporting_files"
    }
    root@d3m-example-pod:/# cat /input/185_baseball/search_config.json
    {
      "problem_schema": "/input/TRAIN/problem_TRAIN/problemDoc.json",
      "problem_root": "/input/TRAIN/problem_TRAIN",
      "dataset_schema": "/input/TRAIN/dataset_TRAIN/datasetDoc.json",
      "training_data_root": "/input/TRAIN/dataset_TRAIN",
      "pipeline_logs_root": "/output/pipelines",
      "executables_root": "/output/executables",
      "user_problems_root": "/output/user_problems",
      "temp_storage_root": "/output/supporting_files"
    }

    """

    if problem:
        full_phase = phase + '_' + problem
    else:
        full_phase = phase

    root_dir = os.path.join(datasets_dir, dataset, full_phase)
    problem_root = os.path.join(root_dir, 'problem_' + phase)
    data_root = os.path.join(root_dir, 'dataset_' + phase)

    config = {
        'problem_root': problem_root,
        'problem_schema': os.path.join(problem_root, 'problemDoc.json'),
        'dataset_schema': os.path.join(data_root, 'datasetDoc.json'),
        'executables_root': os.path.join(output_dir, 'executables'),
        'temp_storage_root': os.path.join(output_dir, 'supporting_files'),
    }

    if phase == 'TRAIN':
        config['training_data_root'] = data_root
        config['pipeline_logs_root'] = os.path.join(output_dir, 'pipelines')
    else:
        config['test_data_root'] = data_root
        config['results_root'] = os.path.join(output_dir, 'predictions')

    return config


PHASES = {
    'TRAIN': 'search',
    'TEST': 'test'
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate D3M dataset config files')
    parser.add_argument('-a', '--absolute', action='store_true')
    parser.add_argument('-b', '--base-dir', default='data', nargs='?')
    parser.add_argument('-d', '--datasets', default='datasets', nargs='?')
    parser.add_argument('-o', '--output', default='output', nargs='?')
    parser.add_argument('-c', '--config-dir', required=True)
    parser.add_argument('-p', '--problem', default='', nargs='?')
    parser.add_argument('dataset', nargs='+')

    args = parser.parse_args()

    if args.absolute:
        base_dir = make_abs(args.base_dir, os.getcwd())
        datasets = make_abs(args.datasets, base_dir)
        output = make_abs(args.output, base_dir)
    else:
        base_dir = args.base_dir
        datasets = os.path.join(base_dir, args.datasets)
        output = os.path.join(base_dir, args.output)

    for dataset in args.dataset:
        for phase, phase_filename in PHASES.items():
            config = build_config(dataset, datasets, phase, args.problem, output)

            filename = '{}_{}.json'.format(dataset, phase_filename)
            with open(os.path.join(args.config_dir, filename), 'w') as f:
                json.dump(config, f, indent=4)
