# mineralML
[![PyPI](https://badgen.net/pypi/v/mineralML)](https://pypi.org/project/mineralML/)
[![Build Status](https://github.com/SarahShi/mineralML/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/SarahShi/mineralML/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/mineralml/badge/?version=latest)](https://mineralml.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/SarahShi/mineralML/branch/main/graph/badge.svg)](https://codecov.io/gh/SarahShi/mineralML/branch/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SarahShi/mineralML/blob/main/mineralML_colab.ipynb)
[![Python 3.7](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

We present mineralML (mineral classification using Machine Learning) for classifying common igneous minerals based on oxide data collected by EPMA, with functions for calculating stoichiometries and crystallographic sites based on this classification. Utilizing this package allows for the identification of misclassified mineral phases and poor-quality data. We streamline data processing and cleaning to allow for the rapid transition to usable data, improving the utility of data curated in these databases and furthering computing and modeling capabilities. 

## Documentation
Read the [documentation](https:/mineralML.readthedocs.io/en/latest/) for a run-through of the mineralML code. 

## Run on the Cloud 
If you do not have Python installed locally, run mineralML on [Google Colab](https://colab.research.google.com/github/SarahShi/mineralML/blob/main/mineralML_colab.ipynb). The Cloud-based version runs rapidly, with test cases of >10,000 microanalyses classified within 4 seconds. 

## Run and Install Locally
Obtain a version of Python between 3.7 and 3.11 if you do not already have it installed. mineralML can be installed with one line. Open terminal and type the following:

```
pip install mineralML
```

Make sure that you keep up with the latest version of mineralML. To upgrade to the latest version of mineralML, open terminal and type the following: 

```
pip install mineralML --upgrade
```

Mac/Linux installation will be straightforward. Windows installations will require the additional setup of WSL.