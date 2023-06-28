# -*- coding: utf-8 -*-
from io import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='PhotonCount',
    version='1.0.0',
    description='Photon Counting',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kjl0025/PhotonCount',
    author='Bijan Nemati, Sam Miller, Kevin Ludwick',
    author_email='bijan.nemati@uah.edu, sam.miller@uah.edu, kjl0025@uah.edu',
    classifiers=[
        #'Development Status :: 4 - Beta',
        #'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages(),
    #package_data={'': ['metadata.yaml']},
    #include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'mpmath'
    ]
)