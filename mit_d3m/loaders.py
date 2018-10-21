# -*- coding: utf-8 -*-

import logging
import os
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img

from mit_d3m.utils import available_memory, used_memory

LOGGER = logging.getLogger(__name__)


class Dataset:

    def __init__(self, name, X=None, y=None, context=None):
        self.name = name
        self.X = X
        self.y = y
        self.context = context or dict()

    def __repr__(self):
        attributes = ["'{}'".format(self.name)]
        for attribute in ['X', 'y', 'context']:
            if getattr(self, attribute) is not None:
                attributes.append(attribute)

        return "Dataset({})".format(', '.join(attributes))

    def get_split(self, indexes):
        X = self.X
        if hasattr(X, 'iloc'):
            X = X.iloc[indexes]
        else:
            X = X[indexes]

        y = self.y
        if y is not None:
            if hasattr(y, 'iloc'):
                y = y.iloc[indexes]
            else:
                y = y[indexes]

        return X, y


class Loader(object):

    def __init__(self, data_modality, task_type):
        self.data_modality = data_modality
        self.task_type = task_type

    def load(self, d3mds):
        """Load X, y and context from D3MDS."""
        X, y = d3mds.get_data()

        return Dataset(d3mds.dataset_id, X, y)

    def to_dict(self):
        return {
            'data_modality': self.data_modality,
            'task_type': self.task_type,
        }

def features_by_type(column_types, columns):
    if not isinstance(column_types, list):
        column_types = [column_types]

    features = []
    for column in columns:
        is_of_type = column['colType'] in column_types
        target = column['role'] == ['suggestedTarget']
        if is_of_type and not target:
            features.append(column['colName'])

    return features


class TabularLoader(Loader):

    @staticmethod
    def find_privileged_features(dataset_doc, tables):
        privileged_features = dict()
        for quality in dataset_doc.get('qualities', []):
            privileged_quality = quality['qualName'] == 'privilegedFeature'
            privileged_true = quality['qualValue'] == 'True'
            restricted_to = quality.get('restrictedTo')

            if privileged_quality and privileged_true and restricted_to:

                res_id = restricted_to['resID']
                privileged_feature = privileged_features.setdefault(res_id, list())

                res_component = restricted_to.get('resComponent')
                if res_component is not None:
                    column_name = res_component.get('columnName')
                    if column_name is None:
                        column_index = res_component.get('columnIndex')
                        if column_index is not None:
                            column_name = tables[res_id]['columns'][column_index]['columnName']

                    if column_name:
                        privileged_feature.append(column_name)

        return privileged_features

    @classmethod
    def remove_privileged_features(cls, dataset_doc, tables):
        privileged_features = cls.find_privileged_features(dataset_doc, tables)
        for res_id, columns in privileged_features.items():
            if columns and res_id in tables:
                tables[res_id]['data'].drop(columns, axis=1, inplace=True)

    @staticmethod
    def map_dtype_to_d3m_type(dtype):
        if 'int' in str(dtype):
            return 'integer'
        elif 'float' in str(dtype):
            return 'real'
        elif 'str' in str(dtype):
            return 'string'
        elif 'object' in str(dtype):
            return 'categorical'
        elif 'date' in str(dtype):
            return 'dateTime'
        elif 'bool' in str(dtype):
            return 'boolean'
        else:
            return 'categorical'

    @classmethod
    def analyze_columns(cls, columns_list, data):
        columns = dict()
        index = None
        time_index = None
        targets = []

        for column in columns_list:
            column_name = column['colName']

            if 'suggestedTarget' in column['role']:
                targets.append(column_name)

            else:
                columns[column_name] = column

                if 'index' in column['role']:
                    if index:
                        raise ValueError("Multiple indexes found")

                    index = column_name

                if 'timeIndicator' in column['role']:
                    if time_index:
                        raise ValueError("Multiple indexes found")

                    time_index = column_name

        if index:
            data.set_index(index, drop=False, inplace=True)

        if targets:
            data.drop(targets, axis=1, errors='ignore', inplace=True)

        return columns, index, time_index

    @classmethod
    def build_columns(cls, data, name):
        index = cls.make_index(data, name)

        columns = {
            column_name: {
                'colIndex': column_index,
                'colName': column_name,
                'colType': cls.map_dtype_to_d3m_type(data[column_name].dtype)
            }
            for column_index, column_name in enumerate(data)
        }

        time_index = None
        if 'time' in data.columns:
            time_index = 'time'

        return columns, index, time_index

    @classmethod
    def get_columns(cls, resource, data, name):
        columns = resource.get('columns')
        if columns:
            columns, index, time_index = cls.analyze_columns(columns, data)
            if not index:
                index = cls.make_index(data, name)

            return columns, index, time_index

        else:
            return cls.build_columns(data, name)

    @classmethod
    def load_table(cls, dataset_root, resource):
        table_path = os.path.join(dataset_root, resource['resPath'])
        table_name = os.path.basename(table_path).split('.')[0]

        dirname = os.path.basename(os.path.normpath(os.path.dirname(table_path)))
        if dirname != 'tables':
            raise ValueError("Found a table out of the tables folder!")

        data = pd.read_csv(table_path)

        columns, index, time_index = cls.get_columns(resource, data, table_name)

        return {
            'resource_id': resource['resID'],
            'table_name': table_name,
            'columns': columns,
            'data': data,
            'index': index,
            'time_index': time_index
        }

    @staticmethod
    def get_parent(resource_id, tables):
        for table in tables.values():
            for column, details in table['columns'].items():
                refers_to = details.get('refersTo', dict()).get('resID')
                if refers_to == resource_id:
                    return table, column

    @staticmethod
    def get_collection_details(dataset_root, resource):
        collection_path = os.path.join(dataset_root, resource['resPath'])
        if collection_path.endswith('/'):
            collection_path = collection_path[:-1]

        collection_name = os.path.basename(collection_path).split('.')[0]

        return collection_name, collection_path

    @classmethod
    def load_collection_data(cls, path, parent_table, parent_column):

        parent_data = parent_table['data']
        parent_index_name = parent_table['index']

        dataframes = []
        for parent_index, row in parent_data.iterrows():
            filename = row[parent_column]
            df = pd.read_csv(os.path.join(path, filename))
            df[parent_index_name] = parent_index
            dataframes.append(df)

        del parent_data[parent_column]

        return pd.concat(dataframes, ignore_index=True)

    @staticmethod
    def make_index(data, name):
        index_name = name + '_id'
        while index_name in data.columns:
            index_name += '_id'

        data.index.name = index_name

        data.reset_index(inplace=True, drop=False)

        return index_name

    @classmethod
    def load_collection(cls, tables, dataset_root, resource):
        parent_table, parent_column = cls.get_parent(
            resource['resID'],
            tables
        )

        table_name, path = cls.get_collection_details(dataset_root, resource)

        data = cls.load_collection_data(path, parent_table, parent_column)

        columns, index, time_index = cls.get_columns(resource, data, table_name)

        return {
            'resource_id': resource['resID'],
            'table_name': table_name,
            'columns': columns,
            'data': data,
            'index': index,
            'time_index': time_index
        }

        return table

    @staticmethod
    def get_resources(d3mds):
        main_table = None
        resources = list()

        for resource in d3mds.dataset_doc['dataResources']:
            if 'learningData.csv' in resource['resPath']:
                main_table = resource
            else:
                resources.append(resource)

        if main_table is None:
            raise RuntimeError('Main table not found')

        return main_table, resources

    @classmethod
    def load_tables(cls, d3mds):
        main_table, resources = cls.get_resources(d3mds)

        dataset_root = d3mds.dataset_root
        main_table = cls.load_table(dataset_root, main_table)

        tables = {
            main_table['resource_id']: main_table
        }

        for resource in resources:
            resource_type = resource['resType']
            is_collection = resource['isCollection']

            if resource_type == 'table' and not is_collection:
                table = cls.load_table(dataset_root, resource)
            elif resource_type == 'timeseries' or is_collection:
                table = cls.load_collection(tables, dataset_root, resource)
            else:
                raise ValueError("I don't know what to do with this")

            tables[table['resource_id']] = table

        cls.remove_privileged_features(d3mds.dataset_doc, tables)

        return tables

    @staticmethod
    def get_relationships(tables):
        relationships = []
        table_names = {
            table['resource_id']: table['table_name']
            for table in tables.values()
        }

        for table in tables.values():
            columns = table['columns']
            df = table['data']
            table_name = table['table_name']

            for column_name, column in columns.items():
                refers_to = column.get('refersTo')

                if refers_to:
                    res_id = refers_to['resID']
                    res_obj = refers_to['resObject']

                    foreign_table_name = table_names[res_id]

                    if column_name in df.columns and isinstance(res_obj, dict):

                        foreign_table_name = table_names[res_id]

                        if 'columnIndex' in res_obj:
                            column_index = res_obj['columnIndex']
                            foreign_column_name = table['columns'][column_index]['colName']

                        else:
                            foreign_column_name = res_obj['columnName']

                        relationships.append((
                            foreign_table_name,
                            foreign_column_name,
                            table_name,
                            column_name,
                        ))

                    # elif table['table_name'] == 'learningData' and res_obj == 'item':
                    elif res_obj == 'item':
                        foreign_column_name = 'd3mIndex'
                        column_name = 'd3mIndex'

                        relationships.append((
                            table_name,
                            column_name,
                            foreign_table_name,
                            foreign_column_name,
                        ))

        return relationships

    def load(self, d3mds):
        X, y = d3mds.get_data()

        tables = self.load_tables(d3mds)
        relationships = self.get_relationships(tables)

        entities = dict()
        for table in tables.values():
            entities[table['table_name']] = (
                table['data'],
                table['index'],
                table['time_index']
            )

        context = {
            'target_entity': 'learningData',
            'entities': entities,
            'relationships': relationships
        }

        return Dataset(d3mds.dataset_id, X, y, context)


class ResourceLoader(Loader):

    def load_resources(self, resources_names, d3mds):
        raise NotImplementedError

    def get_context(self, X, y):
        return None

    def load(self, d3mds):
        """Load X, y and context from D3MDS."""
        X, y = d3mds.get_data()

        resource_columns = d3mds.get_related_resources(self.data_modality)
        for resource_column in resource_columns:
            X = self.load_resources(X, resource_column, d3mds)

        context = self.get_context(X, y)

        return Dataset(d3mds.dataset_id, X, y, context=context)


class ImageLoader(ResourceLoader):

    INPUT_SHAPE = [224, 224, 3]
    EPOCHS = 1

    def load_resources(self, X, resource_column, d3mds):
        LOGGER.info("Loading %s images", len(X))

        image_dir = d3mds.get_resources_dir('image')
        images = []

        for filename in X[resource_column]:
            if used_memory() > available_memory():
                raise MemoryError()

            filename = os.path.join(image_dir, filename)
            image = load_img(filename)
            image = image.resize(tuple(self.INPUT_SHAPE[0:2]))
            image = img_to_array(image)
            image = image / 255.0  # Quantize images.
            images.append(image)

        return np.array(images)


class TextLoader(ResourceLoader):

    def load_resources(self, X, resource_column, d3mds):
        texts_dir = d3mds.get_resources_dir('text')
        texts = []
        for filename in X.pop(resource_column):
            with open(os.path.join(texts_dir, filename), 'r') as text_file:
                texts.append(text_file.read())

        X['texts'] = texts

        return X


class GraphLoader(Loader):

    def load_graphs(self, d3mds, max_graphs=2):
        graphs = d3mds.load_graphs()
        node_columns = d3mds.get_related_resources(self.data_modality)

        graph_names = OrderedDict()
        for _, (column, graph_id) in zip(range(max_graphs), node_columns.items()):
            graph_names[column] = nx.Graph(graphs[graph_id])

        return graph_names

    def get_context(self, X, d3mds):
        if self.task_type == 'community_detection':
            graphs = self.load_graphs(d3mds, 1)
            column, graph = list(graphs.items())[0]
            context = {
                'graph': graph,
            }

        elif self.task_type == 'link_prediction':
            graphs = self.load_graphs(d3mds, 2)
            columns = list(graphs.keys())
            context = {
                'node_columns': columns,
                'graph': graphs[columns[-1]]
            }

        elif self.task_type == 'vertex_nomination':
            graphs = self.load_graphs(d3mds, 1)
            context = {
                'graphs': graphs
            }

        elif self.task_type == 'graph_matching':
            graphs = self.load_graphs(d3mds, 2)
            columns = list(graphs.keys())
            graph_0, graph_1 = tuple(graphs.values())

            pairs = X[columns].values
            graph = graph_0.copy()
            graph.add_nodes_from(graph_1.nodes(data=True))
            graph.add_edges_from(graph_1.edges)
            graph.add_edges_from(pairs)

            context = {
                'node_columns': columns,
                'graph': graph,
                'graphs': graphs
            }

        return context

    def load(self, d3mds):
        X, y = d3mds.get_data()

        context = self.get_context(X, d3mds)

        return Dataset(d3mds.dataset_id, X, y, context=context)


_LOADERS = {
    'tabular': TabularLoader,
    'timeseries': TabularLoader,
    'image': ImageLoader,
    'text': TextLoader,
    'graph': GraphLoader,
}


def get_loader(data_modality, task_type):
    loader_class = _LOADERS.get(data_modality, Loader)
    return loader_class(data_modality, task_type)
