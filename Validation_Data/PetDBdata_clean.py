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

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 14})
plt.rcParams['pdf.fonttype'] = 42

# %% 

amp = pd.read_csv('PetDB_allminerals/mineral_amphibole.csv', index_col=0)
cpx = pd.read_csv('PetDB_allminerals/mineral_clinopyroxene.csv', index_col=0)
spar = pd.read_csv('PetDB_allminerals/mineral_feldspars.csv', index_col=0)
ol = pd.read_csv('PetDB_allminerals/mineral_olivine.csv', index_col=0)
opx = pd.read_csv('PetDB_allminerals/mineral_orthopyroxene.csv', index_col=0)
sp = pd.read_csv('PetDB_allminerals/mineral_spinels.csv', index_col=0)
rest = pd.read_csv('PetDB_allminerals/mineral_therest.csv', index_col=0)

amphibole = amp[amp.Mineral=='AMPHIBOLE']
apatite = rest[rest.Mineral=='APATITE']
biotite = rest[rest.Mineral=='BIOTITE']
clinopyroxenes = cpx[cpx.Mineral=='CLINOPYROXENE']
garnet = rest[rest.Mineral=='GARNET']
kspar = spar[spar.Mineral=='K-FELDSPAR']
muscovite = rest[rest.Mineral=='MUSCOVITE']
olivine = ol[ol.Mineral=='OLIVINE']
orthopyroxene = opx[opx.Mineral=='ORTHOPYROXENE']
ilmenite = rest[rest.Mineral=='ILMENITE']
magnetite = sp[sp.Mineral=='MAGNETITE']
plagioclase = spar[spar.Mineral=='PLAGIOCLASE']
quartz = rest[rest.Mineral=='QUARTZ']
rutile = rest[rest.Mineral=='RUTILE']
spinel = sp[sp.Mineral=='SPINEL']
zircon = rest[rest.Mineral=='ZIRCON']

oxideslab = ['Code', 'SiO2 (wt%)', 'TiO2 (wt%)', 'Al2O3 (wt%)', 'FeO (wt%)', 'FeOT (wt%)', 'Fe2O3 (wt%)', 'MnO (wt%)', 'MgO (wt%)', 
             'CaO (wt%)', 'Na2O (wt%)', 'K2O (wt%)', 'P2O5 (wt%)', 'Cr2O3 (wt%)', 'NiO (wt%)', 'Mineral']
new_column_names = {'Code': 'Sample Name', 'SiO2 (wt%)': 'SiO2', 'TiO2 (wt%)': 'TiO2', 'Al2O3 (wt%)': 'Al2O3', 'FeO (wt%)': 'FeO', 'FeOT (wt%)': 'FeOT', 
                    'Fe2O3 (wt%)': 'Fe2O3', 'MnO (wt%)': 'MnO', 'MgO (wt%)': 'MgO', 'CaO (wt%)': 'CaO', 'Na2O (wt%)': 'Na2O', 'K2O (wt%)': 'K2O', 
                    'P2O5 (wt%)': 'P2O5', 'Cr2O3 (wt%)': 'Cr2O3', 'NiO (wt%)': 'NiO'}
oxideslab1 = ['Sample Name', 'SiO2', 'TiO2', 'Al2O3', 'FeO', 'FeOT', 'Fe2O3', 'MnO', 'MgO', 
             'CaO', 'Na2O', 'K2O', 'P2O5', 'Cr2O3', 'NiO', 'Mineral']


amp_comp_filt = amphibole.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
ap_comp_filt = apatite.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
bt_comp_filt = biotite.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
cpx_comp_filt = clinopyroxenes.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
gt_comp_filt = garnet.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
ksp_comp_filt = kspar.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
ms_comp_filt = muscovite.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
ol_comp_filt = olivine.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
opx_comp_filt = orthopyroxene.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
il_comp_filt = ilmenite.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
mt_comp_filt = magnetite.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
plag_comp_filt = plagioclase.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
qz_comp_filt = quartz.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
rt_comp_filt = rutile.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
sp_comp_filt = spinel.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))
zr_comp_filt = zircon.rename(columns=lambda x: next((new_column_names[k] for k in new_column_names if k in x), x))


petdb_comp = pd.concat([amp_comp_filt[oxideslab1], ap_comp_filt[oxideslab1], bt_comp_filt[oxideslab1], cpx_comp_filt[oxideslab1], gt_comp_filt[oxideslab1], 
                        ksp_comp_filt[oxideslab1], ms_comp_filt[oxideslab1], ol_comp_filt[oxideslab1], opx_comp_filt[oxideslab1], il_comp_filt[oxideslab1], 
                        mt_comp_filt[oxideslab1], plag_comp_filt[oxideslab1], qz_comp_filt[oxideslab1], rt_comp_filt[oxideslab1], sp_comp_filt[oxideslab1], 
                        zr_comp_filt[oxideslab1]], axis = 0)

def to_title_case(s):
    return s.title()

petdb_comp['Mineral'] = petdb_comp['Mineral'].apply(to_title_case)
petdb_comp['Fe2O3T'] = np.nan

petdb_comp.to_csv('petdb_allphases.csv', index = False)

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

petdb_comp_conv = Fe_Conversion(petdb_comp)

petdb_comp_fe = petdb_comp_conv[['Sample Name', 'Mineral', 'SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOT_F', 'MnO', 'MgO', 'NiO', 'CaO', 'Na2O', 'K2O', 'P2O5']]
petdb_comp_fe.rename(columns={'FeOT_F': 'FeOt'}, inplace = True)

petdb_comp_fe.to_csv('PetDB_allphases_fe.csv')

# %%
