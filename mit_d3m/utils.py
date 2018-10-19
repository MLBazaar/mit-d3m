# -*- coding: utf-8 -*-

import logging
import os
import subprocess

import psutil


def make_abs(path, base_dir=None):
    base_dir = base_dir or os.getcwd()
    if not os.path.isabs(path):
        return os.path.join(base_dir, path)

    return path


def used_memory():
    return psutil.Process(os.getpid()).memory_info().rss


def available_memory():
    return psutil.virtual_memory().available


def disk_usage(path, human=False):
    """disk usage in bytes or human readable format (e.g. '2,1GB')"""
    command = ['du', '-s', path]
    if human:
        command.append('-h')

    return subprocess.check_output(command).split()[0].decode('utf-8')


def walk(document, transform):
    if not isinstance(document, dict):
        return document

    new_doc = dict()
    for key, value in document.items():
        if isinstance(value, dict):
            value = walk(value, transform)
        elif isinstance(value, list):
            value = [walk(v, transform) for v in value]

        new_key, new_value = transform(key, value)
        new_doc[new_key] = new_value

    return new_doc


def remove_dots(document):
    return walk(document, lambda key, value: (key.replace('.', '-'), value))


def restore_dots(document):
    return walk(document, lambda key, value: (key.replace('-', '.'), value))


def logging_setup(verbosity=1, logfile=None, logger_name=None):
    logger = logging.getLogger(logger_name)
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)
    logger.setLevel(log_level)
    logger.propagate = False

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
