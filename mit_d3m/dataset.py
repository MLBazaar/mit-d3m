# -*- coding: utf-8 -*-

import json
import logging
import os
import re
import warnings
from urllib.parse import urlparse

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

RE_PYTHONIZE = re.compile(r'[A-Z]')


def pythonize(name):
    pythonized = re.sub('[A-Z]', '_\g<0>', name).lower()
    if pythonized.startswith('_'):
        pythonized = pythonized[1:]

    return pythonized


DATASET_SCHEMA_VERSION = '3.0'
PROBLEM_SCHEMA_VERSION = '3.0'


class D3MDataset:
    dsHome = None
    dsDoc = None
    learningDataFile = None

    def _get_learning_data_path(self):
        """
        Returns the path of learningData.csv in a dataset
        """
        for res in self.dsDoc['dataResources']:
            resPath = res['resPath']
            resType = res['resType']

            dirname = os.path.basename(os.path.normpath(os.path.dirname(resPath)))

            if resType == 'table' and dirname == 'tables':
                if 'learningData.csv' in res['resPath']:
                    return os.path.join(self.dsHome, resPath)

        # if the for loop is over and learningDoc is not found, then return None
        raise RuntimeError('could not find learningData file the dataset')

    def __init__(self, dataset):
        # handle uris
        logger.info("Loading dataset: %s", dataset)
        dataset = urlparse(dataset).path
        self.dsHome = dataset

        # read the schema in dsHome
        if os.path.isdir(dataset):
            self.dsHome = dataset
            _dsDoc = os.path.join(self.dsHome, 'datasetDoc.json')
        else:
            self.dsHome = os.path.dirname(dataset)
            _dsDoc = dataset

        assert os.path.exists(_dsDoc), _dsDoc
        with open(_dsDoc, 'r') as f:
            self.dsDoc = json.load(f)

        # make sure the versions line up
        if self.get_dataset_schema_version() != DATASET_SCHEMA_VERSION:
            warnings.warn("the datasetSchemaVersions in the API and datasetDoc do not match!")

        # locate the special learningData file
        self.learningDataFile = self._get_learning_data_path()

    def get_datasetID(self):
        """Get the datasetID from datasetDoc."""
        return self.dsDoc['about']['datasetID']

    def get_dataset_schema_version(self):
        """Get the dataset schema version that was used to create this dataset."""
        return self.dsDoc['about']['datasetSchemaVersion']

    def get_learning_data(self):
        """Get the contents of learningData.doc as a DataFrame."""
        return pd.read_csv(self.learningDataFile, index_col='d3mIndex')

    def _get_learning_data_resource(self):
        """
        Returns the path of learningData.csv in a dataset
        """
        for res in self.dsDoc['dataResources']:
            resPath = res['resPath']
            resType = res['resType']
            if resType == 'table':
                if 'learningData.csv' in resPath:
                    return res
                else:
                    raise RuntimeError('could not find learningData.csv')

        # if the for loop is over and learningDoc is not found, then return None
        raise RuntimeError('could not find learningData resource')

    def get_learning_data_columns(self):
        res = self._get_learning_data_resource()
        return res['columns']

    def get_resource_types(self):
        return [dr["resType"] for dr in self.dsDoc['dataResources']]

    def get_data_modality(self):
        """Detect the data modality based on the resource_types.

        resource_types == ['table'] => 'single_table'
        resource_types == ['something_else'...] => 'something_else'   # this is not likely
        resource_types == ['table', 'table'...] => 'multi_table'
        resource_types == ['table', 'something_else'...] => 'something_else'
        """
        resource_types = self.get_resource_types()
        first_type = resource_types[0]
        if first_type != 'table':
            return first_type

        elif len(resource_types) == 1:
            return 'tabular'

        else:
            second_type = resource_types[1]
            if second_type == 'table':
                return 'tabular'

            return second_type

    def get_image_path(self):
        """
        Returns the path of the directory containing images if they exist in this dataset.
        """
        for res in self.dsDoc['dataResources']:
            resPath = res['resPath']
            resType = res['resType']
            isCollection = res['isCollection']

            if resType == 'image' and isCollection:
                return os.path.join(self.dsHome, resPath)

        # if the for loop is over and no image directory is found, then return None
        raise RuntimeError('could not find learningData file the dataset')

    def get_graph_resources(self):
        return [r for r in self.dsDoc['dataResources'] if r["resType"] == "graph"]

    def get_graphs_as_nx(self):
        graph_res = self.get_graph_resources()

        graphs = {}
        # todo allow more than one graph resource
        for g in graph_res:
            graph_path = os.path.join(self.dsHome, g["resPath"])
            try:
                graphs[g['resID']] = nx.read_gml(graph_path)
            except nx.exception.NetworkXError:
                graphs[g['resID']] = nx.read_gml(graph_path, label='id')

        return graphs

    def _get_resources_by_type(self, resource_type):
        """
        Returns the list of resources that are of the indicated type
        """
        resources = []
        for res in self.dsDoc['dataResources']:
            if res['resType'] == resource_type:
                resources.append(res)

        return resources

    def get_related_resource_names(self, resource_type):
        related_names = dict()
        related_resources = self._get_resources_by_type(resource_type)
        related_resources = {r['resID'] for r in related_resources}

        for column in self.get_learning_data_columns():
            refers_to = column.get('refersTo')
            if refers_to:
                res_id = refers_to['resID']
                if res_id in related_resources:
                    related_names[column['colName']] = res_id

        return related_names

    def get_text_path(self):
        """
        Returns the path of the directory containing text if they exist in this dataset.
        """
        for res in self.dsDoc['dataResources']:
            resPath = res['resPath']
            resType = res['resType']
            isCollection = res['isCollection']
            if resType == 'text' and isCollection:
                return os.path.join(self.dsHome, resPath)

        # if the for loop is over and no image directory is found, then return None
        raise RuntimeError('could not find learningData file the dataset')


class D3MProblem:
    prHome = None
    prDoc = None
    splitsFile = None

    def __init__(self, problem):
        if isinstance(problem, dict):
            self.prDoc = problem
        else:
            self.prHome = problem

            # read the schema in prHome
            _prDoc = os.path.join(self.prHome, 'problemDoc.json')
            assert os.path.exists(_prDoc), _prDoc
            with open(_prDoc, 'r') as f:
                self.prDoc = json.load(f)

        # make sure the versions line up
        if self.get_problem_schema_version() != PROBLEM_SCHEMA_VERSION:
            warnings.warn("the problemSchemaVersions in the API and datasetDoc do not match!")

    def get_task_type(self):
        return self.prDoc["about"].get("taskType", "")

    def get_task_subtype(self):
        return self.prDoc["about"].get("taskSubType", "")

    def get_problem_id(self):
        """Get the problemID from problemDoc."""
        return self.prDoc['about']['problemID']

    def get_problem_schema_version(self):
        """Get the problem schema version that was used to create this dataset."""
        return self.prDoc['about']['problemSchemaVersion']

    def get_performance_metrics(self):
        return self.prDoc['inputs']['performanceMetrics']

    def get_target_column_names(self):
        targets = self.prDoc['inputs']['data'][0]['targets']
        target_columns = []
        for target in targets:
            target_columns.append(target['colName'])

        return target_columns


class D3MDS:
    dataset = None
    problem = None

    def __init__(self, dataset, problem):
        if isinstance(dataset, D3MDataset):
            self.dataset = dataset
        else:
            self.dataset = D3MDataset(dataset)

        if isinstance(problem, D3MProblem):
            self.problem = problem
        else:
            self.problem = D3MProblem(problem)

        self.dataset_doc = self.dataset.dsDoc
        self.problem_doc = self.problem.prDoc
        self.dataset_root = self.dataset.dsHome
        self.dataset_id = self.dataset.get_datasetID()
        self.problem_id = self.problem.get_problem_id()
        self.target_column = self.problem.get_target_column_names()[0]
        self.targets = self.problem.get_target_column_names()

    def get_data(self):
        X = self.dataset.get_learning_data()

        try:
            if len(self.targets) == 1:
                y = X[self.targets[0]]
            else:
                y = X[self.targets]

            X = X.drop(self.targets, axis=1, errors='ignore')
        except KeyError:
            y = pd.DataFrame(index=X.index)

        return X, y

    def get_columns(self):
        return self.dataset.get_learning_data_columns()

    def get_resources_dir(self, data_modality):
        if data_modality == 'image':
            return self.dataset.get_image_path()
        if data_modality == 'text':
            return self.dataset.get_text_path()

    def get_related_resources(self, data_modality):
        return self.dataset.get_related_resource_names(data_modality)

    def load_graphs(self):
        return self.dataset.get_graphs_as_nx()

    def get_data_modality(self):
        return self.dataset.get_data_modality()

    def get_problem_id(self):
        return self.problem.get_problem_id()

    def get_task_type(self):
        return pythonize(self.problem.get_task_type())

    def get_task_subtype(self):
        return pythonize(self.problem.get_task_subtype())

    def get_metric(self):
        return self.problem.get_performance_metrics()[0]['metric']
