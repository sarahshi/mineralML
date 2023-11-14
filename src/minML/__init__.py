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

from minML.core import *
from minML.stoichiometry import *
from minML.unsupervised import *
from minML.supervised import *
from minML.confusion_matrix import *


from ._version import __version__
