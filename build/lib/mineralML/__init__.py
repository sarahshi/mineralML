# %% 
__author__ = 'Sarah Shi'

import os
import copy
import numpy as np
import pandas as pd
import warnings

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
from matplotlib import pyplot as plt
mpl.use('pgf')

from mineralML.core import *
from mineralML.stoichiometry import *
from mineralML.unsupervised import *
from mineralML.supervised import *
from mineralML.confusion_matrix import *


from ._version import __version__
