# %% 

""" Created on February 16, 2023 // @author: Sarah Shi """

import numpy as np
import pandas as pd
import scipy

import os
import time
import json
import random
import pickle
import warnings

from scipy.sparse import (SparseEfficiencyWarning)
warnings.simplefilter('ignore', category=(FutureWarning,SparseEfficiencyWarning))

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import scale, normalize, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support
# from imblearn.over_sampling import RandomOverSampler

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.express as px

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 14})
plt.rcParams['pdf.fonttype'] = 42

# %% 

df = pd.read_csv('lepr_allphases.csv')


unique = np.unique(df['Mineral'])
print(unique)

lim_phases = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Cristobalite', 'Garnet', 'Muscovite', 'Olivine', 'Orthopyroxene', 'Oxide', 'Plagioclase', 'KFeldspar', 'Quartz', 'Rutile', 'Spinel', 'Tridymite', 'Zircon'])

df_lim = df.loc[df.Mineral.isin(lim_phases), :]
unique_new = np.unique(df_lim['Mineral'])
print(unique_new)

df_lim['Mineral'] = df['Mineral'].replace('Cristobalite', 'Quartz')
df_lim['Mineral'] = df['Mineral'].replace('Tridymite', 'Quartz')
df_lim['Mineral'] = df['Mineral'].replace('Oxide', 'FeTiOxide')

df_lim['FeOT'] = np.nan
df_lim['Fe2O3T'] = np.nan

# %%

def Fe_Conversion(df):

    """
    Handle inconsistent Fe speciation in PetDB datasets by converting all to FeOT. 

    Parameters
    --------------
    df:class:`pandas.DataFrame`
        Array of oxide compositions.

    Returns
    --------
    df:class:`pandas.DataFrame`
        Array of oxide compositions with corrected Fe.
    """

    fe_conv = 1.1113
    conditions = [~np.isnan(df['FeO']) & np.isnan(df['FeOT']) & np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3T']]),
    ~np.isnan(df['FeOT']) & np.isnan(df['FeO']) & np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3T']]), 
    ~np.isnan(df['Fe2O3']) & np.isnan(df['Fe2O3T']) & np.isnan(df['FeO']) & np.isnan([df['FeOT']]), # 2
    ~np.isnan(df['Fe2O3T']) & np.isnan(df['Fe2O3']) & np.isnan(df['FeO']) & np.isnan([df['FeOT']]), # 2
    ~np.isnan(df['FeO']) & ~np.isnan(df['Fe2O3']) & np.isnan(df['FeOT']) & np.isnan([df['Fe2O3T']]), # 3
    ~np.isnan(df['FeO']) & ~np.isnan(df['FeOT']) & ~np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3T']]), # 4
    ~np.isnan(df['FeO']) & ~np.isnan(df['Fe2O3']) & ~np.isnan(df['Fe2O3T']) & np.isnan([df['FeOT']]), # 5
    ~np.isnan(df['FeOT']) & ~np.isnan(df['Fe2O3']) & np.isnan(df['Fe2O3T']) & np.isnan([df['FeO']]), # 6
    ~np.isnan(df['Fe2O3']) & ~np.isnan(df['Fe2O3T']) & np.isnan(df['FeO']) & np.isnan([df['FeOT']]) ] # 7

    choices = [ (df['FeO']), (df['FeOT']),
    (df['Fe2O3']),(df['Fe2O3T']),
    (df['FeO'] + (df['Fe2O3'] / fe_conv)), # 3
    (df['FeOT']), # 4 of interest
    (df['Fe2O3T'] / fe_conv), # 5
    (df['FeOT']), # 6
    (df['Fe2O3T'] / fe_conv) ] # 7

    df.insert(9, 'FeOT_F', np.select(conditions, choices))

    return df 

# %%

df_new = Fe_Conversion(df_lim)


# %% 


df_new_lim = df_new[['entity_id', 'Mineral', 'method', 'SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOT_F', 'MnO', 'MgO', 'NiO', 'CoO', 'CaO', 'Na2O', 'K2O', 'P2O5']]
df_new_lim.rename(columns={'FeOT_F': 'FeOt'}, inplace = True)

df_new_lim.to_csv('LEPR_allphases_lim.csv')


# %%

df_new_lim



# %%
