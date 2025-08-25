#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'src', 'mineralML', '_version.py'), encoding='utf-8') as f:
    exec(f.read())

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mineralML",
    version=__version__,
    author="Sarah C. Shi",
    author_email="sarah.c.shi@gmail.com",
    description="mineralML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarahshi/mineralML",
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required

    package_data={
        # Include all pickle files
        "": ["*.npz", "*.pt"],
    },
    install_requires=[
            'pandas',
            'numpy<2',
            'scipy',
            'seaborn', 
            'matplotlib',
            'scikit-learn',
            'torch',
            'hdbscan', 
            'python-ternary',
            ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
