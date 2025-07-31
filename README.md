# mineralML
[![PyPI](https://badgen.net/pypi/v/mineralML)](https://pypi.org/project/mineralML/)
[![Build Status](https://github.com/SarahShi/mineralML/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/SarahShi/mineralML/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/mineralml/badge/?version=latest)](https://mineralml.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/SarahShi/mineralML/branch/main/graph/badge.svg)](https://codecov.io/gh/SarahShi/mineralML/branch/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SarahShi/mineralML/blob/main/mineralML_colab.ipynb)
[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

We present mineralML (mineral classification using Machine Learning) for classifying common igneous minerals based on oxide data collected by EPMA, with functions for calculating stoichiometries and crystallographic sites based on this classification. Utilizing this package allows for the identification of misclassified mineral phases and poor-quality data. We streamline data processing and cleaning to allow for the rapid transition to usable data, improving the utility of data curated in these databases and furthering computing and modeling capabilities. 

## Documentation
Read the [documentation](https://mineralml.readthedocs.io/en/latest/?badge=latest) for a run-through of the mineralML code. 

## Citation
If you use mineralML in your work, please cite this abstract. This package represents a significant time investment. Proper citation helps support continued development and academic recognition.

```console
Shi, S., Wieser, P., Toth, N., Antoshechkina, P.M., Lehnert, K., (2023) “MIN-ML: Leveraging Machine Learning for Probabilistic Mineral Classification in Geochemical Databases”. In AGU Fall Meeting Abstracts (Vol. 2023, pp. V54A-07).
```

```
@inproceedings{Shietal2023,
  title     = {MIN-ML: Leveraging Machine Learning for Probabilistic Mineral Classification in Geochemical Databases},
  author    = {Shi, Sarah C and Wieser, Penny E and Toth, Norbert and Antoshechkina, Paula M and Lehnert, Kerstin},
  booktitle = {AGU Fall Meeting Abstracts},
  volume.   = {2023},
  pages.    = {V54A--07},
  year.     = {2023}
}
```

## Run on the Cloud 
If you do not have Python installed locally, run mineralML on [Google Colab](https://colab.research.google.com/github/SarahShi/mineralML/blob/main/mineralML_colab.ipynb). The Cloud-based version runs rapidly, with test cases of >10,000 microanalyses classified within 4 seconds. 

## Run and Install Locally
Obtain a version of Python between 3.8 and 3.12 if you do not already have it installed. mineralML can be installed with one line. Open terminal and type the following:

```
pip install mineralML
```

Make sure that you keep up with the latest version of mineralML. To upgrade to the latest version of mineralML, open terminal and type the following: 

```
pip install mineralML --upgrade
```

Mac/Linux installation will be straightforward. Windows installations will require the additional setup of WSL.