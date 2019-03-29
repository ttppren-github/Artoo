#!/usr/bin/env python
import os
from os.path import dirname, join

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='artoo',
    version='0.0.1',
    description='A wraper of GYM',
    author='LFY',
    author_email='ttppren@163.com',
    license='Apache License v2',
    url='',
    install_requires=read("requirements.txt").strip(),
    packages=find_packages(),
    package_data={
        '': ['*.*'],
        'artoo': ['config/episode.yaml'],
    },
)
