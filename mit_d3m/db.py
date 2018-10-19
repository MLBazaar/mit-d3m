# -*- coding: utf-8 -*-

import getpass
import json
import logging

from pymongo import MongoClient

LOGGER = logging.getLogger(__name__)


def get_db(database=None, config=None, **kwargs):
    if config:
        with open(config, 'r') as f:
            config = json.load(f)
    else:
        config = kwargs

    host = config.get('host', 'localhost')
    port = config.get('port', 27017)
    user = config.get('user')
    password = config.get('password')
    database = database or config.get('database', 'test')
    auth_database = config.get('auth_database', 'admin')

    if user and not password:
        password = getpass.getpass(prompt='Please insert database password: ')

    client = MongoClient(
        host=host,
        port=port,
        username=user,
        password=password,
        authSource=auth_database
    )

    LOGGER.info("Setting up a MongoClient %s", client)

    return client[database]
