#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

install_requires = [
    'baytune==0.2.1',
    'boto==2.48.0',
    'featuretools==0.2.0',
    'mlblocks==0.2.3',
    'mlprimitives==0.1.2',
    'networkx==2.1',
    'numpy==1.15.2',
    'pandas==0.23.4',
    'pymongo==3.7.2',
    'requests==2.18.4',
    'scikit-learn==0.20.0',
    'scipy==1.1.0',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

tests_require = [
    'coverage>=4.5.1',
    'pytest>=3.4.2',
    'tox>=2.9.1',
]

development_requires = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8>=1.3.5',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',
]


setup(
    author="MIT Data To AI Lab",
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="MIT tools to work with D3M datasets.",
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='mit-d3m',
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='mit-d3m',
    packages=find_packages(include=['mit_d3m', 'mit_d3m.*']),
    python_requires='>=3.4',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-Project/mit-d3m',
    version='0.1.0',
    zip_safe=False,
)
