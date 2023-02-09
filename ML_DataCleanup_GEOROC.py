# %% 

""" Created on November 9, 2022 // @author: Sarah Shi and Penny Wieser """

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import numpy as np

import os 
import json 
import pickle
import pygmt
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

from sklearn.preprocessing import scale, normalize

import TAS_Functions as tas
import Thermobar as pt

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.path as mpath
import matplotlib.colors as mcolors

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 14})
plt.rcParams['pdf.fonttype'] = 42

pt.__version__

# %% 

path_parent = os.getcwd()

output_dir = ["/GEOROC_minerals_filt", "/GEOROC_minerals_orig", "/GEOROC_minerals_lim"] 
for ii in range(len(output_dir)):
    if not os.path.exists(path_parent + output_dir[ii]):
        os.makedirs(path_parent + output_dir[ii], exist_ok=True)

# %%

amphiboles = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_AMPHIBOLES.csv', encoding='latin-1')
apatites = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_APATITES.csv', encoding='latin-1')
carbonates = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_CARBONATES.csv', encoding='latin-1')
# chalcogenides = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_CHALCOGENIDES.csv', encoding='latin-1')
# clays = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_CLAY_MINERALS.csv', encoding='latin-1')
clinopyroxenes = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_CLINOPYROXENES.csv', encoding='latin-1')
feldspars = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_FELDSPARS.csv', encoding='latin-1')
# feldspathoids = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_FELDSPATHOIDES.csv', encoding='latin-1')
garnets = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_GARNETS.csv', encoding='latin-1')
ilmenites = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_ILMENITES.csv', encoding='latin-1')
micas = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_MICA.csv', encoding='latin-1')
olivines = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_OLIVINES.csv', encoding='latin-1')
orthopyroxenes = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_ORTHOPYROXENES.csv', encoding='latin-1')
perovskites = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_PEROVSKITES.csv', encoding='latin-1')
pyroxenes = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_PYROXENES.csv', encoding='latin-1')
quartz = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_QUARTZ.csv', encoding='latin-1')
spinels = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_SPINELS.csv', encoding='latin-1')
titanites = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_TITANITES.csv', encoding='latin-1')
zircons = pd.read_csv('./GEOROC_minerals_orig/2022-09-SGFTFN_ZIRCONS.csv', encoding='latin-1')

# %%

amphiboles = amphiboles.dropna(subset=['SIO2(WT%)'], how='all')
apatites = apatites.dropna(subset=['SIO2(WT%)'], how='all')
carbonates = carbonates.dropna(subset=['SIO2(WT%)'], how='all')
# chalcogenides = chalcogenides.dropna(subset=['SIO2(WT%)'], how='all')
# clays = clays.dropna(subset=['SIO2(WT%)'], how='all')
clinopyroxenes = clinopyroxenes.dropna(subset=['SIO2(WT%)'], how='all')
feldspars = feldspars.dropna(subset=['SIO2(WT%)'], how='all')
# feldspathoids = feldspathoides.dropna(subset=['SIO2(WT%)'], how='all')
garnets = garnets.dropna(subset=['SIO2(WT%)'], how='all')
ilmenites = ilmenites.dropna(subset=['SIO2(WT%)'], how='all')
micas = micas.dropna(subset=['SIO2(WT%)'], how='all')
olivines = olivines.dropna(subset=['SIO2(WT%)'], how='all')
orthopyroxenes = orthopyroxenes.dropna(subset=['SIO2(WT%)'], how='all')
perovskites = perovskites.dropna(subset=['SIO2(WT%)'], how='all')
pyroxenes = pyroxenes.dropna(subset=['SIO2(WT%)'], how='all')
quartz = quartz.dropna(subset=['SIO2(WT%)'], how='all')
spinels = spinels.dropna(subset=['SIO2(WT%)'], how='all')
titanites = titanites.dropna(subset=['SIO2(WT%)'], how='all')
zircons = zircons.dropna(subset=['SIO2(WT%)'], how='all')

# %%

amphiboles.to_csv('./GEOROC_minerals_filt/GEOROC_amphibole.csv')
apatites.to_csv('./GEOROC_minerals_filt/GEOROC_apatite.csv')
carbonates.to_csv('./GEOROC_minerals_filt/GEOROC_carbonate.csv')
# chalcogenides.to_csv('./GEOROC_minerals_filt/GEOROC_chalcogenide.csv')
# clays.to_csv('./GEOROC_minerals_filt/GEOROC_clay.csv')
clinopyroxenes.to_csv('./GEOROC_minerals_filt/GEOROC_clinopyroxene.csv')
feldspars.to_csv('./GEOROC_minerals_filt/GEOROC_feldspar.csv')
# feldspathoides.to_csv('./GEOROC_minerals_filt/GEOROC_feldspathoid.csv')
garnets.to_csv('./GEOROC_minerals_filt/GEOROC_garnet.csv')
ilmenites.to_csv('./GEOROC_minerals_filt/GEOROC_ilmenite.csv')
micas.to_csv('./GEOROC_minerals_filt/GEOROC_mica.csv')
olivines.to_csv('./GEOROC_minerals_filt/GEOROC_olivine.csv')
orthopyroxenes.to_csv('./GEOROC_minerals_filt/GEOROC_orthopyroxene.csv')
perovskites.to_csv('./GEOROC_minerals_filt/GEOROC_perovskite.csv')
pyroxenes.to_csv('./GEOROC_minerals_filt/GEOROC_pyroxene.csv')
quartz.to_csv('./GEOROC_minerals_filt/GEOROC_quartz.csv')
spinels.to_csv('./GEOROC_minerals_filt/GEOROC_spinel.csv')
titanites.to_csv('./GEOROC_minerals_filt/GEOROC_titanite.csv')
zircons.to_csv('./GEOROC_minerals_filt/GEOROC_zircon.csv')


# %%

lim_oxides = ['CITATION', 'SAMPLE NAME', 'TECTONIC SETTING', 'LATITUDE (MIN.)', 'LATITUDE (MAX.)',
    'LONGITUDE (MIN.)', 'LONGITUDE (MAX.)', 'ROCK NAME', 'ROCK TEXTURE', 'ALTERATION', 'MINERAL',
    'SIO2(WT%)', 'TIO2(WT%)', 'AL2O3(WT%)', 'FE2O3T(WT%)', 'FE2O3(WT%)', 'FEOT(WT%)', 'FEO(WT%)', 'MNO(WT%)', 
    'MGO(WT%)', 'CAO(WT%)', 'NA2O(WT%)', 'K2O(WT%)', 'P2O5(WT%)', 'CR2O3(WT%)', 'NIO(WT%)']

lim_oxides_nofe3 = ['CITATION', 'SAMPLE NAME', 'TECTONIC SETTING', 'LATITUDE (MIN.)', 'LATITUDE (MAX.)',
    'LONGITUDE (MIN.)', 'LONGITUDE (MAX.)', 'ROCK NAME', 'ROCK TEXTURE', 'ALTERATION', 'MINERAL',
    'SIO2(WT%)', 'TIO2(WT%)', 'AL2O3(WT%)', 'FEOT(WT%)', 'FEO(WT%)', 'MNO(WT%)', 
    'MGO(WT%)', 'CAO(WT%)', 'NA2O(WT%)', 'K2O(WT%)', 'P2O5(WT%)', 'CR2O3(WT%)', 'NIO(WT%)']

amphiboles_lim = amphiboles.loc[:, lim_oxides]
apatites_lim = apatites.loc[:, lim_oxides]
carbonates_lim = carbonates.loc[:, lim_oxides]
# chalcogenides_lim = chalcogenides.loc[:, lim_oxides_nofe3]
# clays_lim = clays.loc[:, lim_oxides]
clinopyroxenes_lim = clinopyroxenes.loc[:, lim_oxides]
feldspars_lim = feldspars.loc[:, lim_oxides]
# feldspathoides_lim = feldspathoides.loc[:, lim_oxides]
garnets_lim = garnets.loc[:, lim_oxides]
ilmenites_lim = ilmenites.loc[:, lim_oxides]
micas_lim = micas.loc[:, lim_oxides]
olivines_lim = olivines.loc[:, lim_oxides]
orthopyroxenes_lim = orthopyroxenes.loc[:, lim_oxides]
perovskites_lim = perovskites.loc[:, lim_oxides]
pyroxenes_lim = pyroxenes.loc[:, lim_oxides]
quartz_lim = quartz.loc[:, lim_oxides]
spinels_lim = spinels.loc[:, lim_oxides]
titanites_lim = titanites.loc[:, lim_oxides]
zircons_lim = zircons.loc[:, lim_oxides_nofe3]

# %% 

def remove_alt(df):
    df_lim = df[ (df['ALTERATION']!='almost totally altered') & (df['ALTERATION']!='extensively altered') & (df['ALTERATION']!='moderately altered')]
    return df_lim

amphiboles_limalt = remove_alt(amphiboles_lim)
apatites_limalt = remove_alt(apatites_lim)
carbonates_limalt = remove_alt(carbonates_lim)
# chalcogenides_limalt = remove_alt(chalcogenides_lim)
# clays_limalt = remove_alt(clays_lim)
clinopyroxenes_limalt = remove_alt(clinopyroxenes_lim)
feldspars_limalt = remove_alt(feldspars_lim)
# feldspathoides_limalt = remove_alt(feldspathoides_lim)
garnets_limalt = remove_alt(garnets_lim)
ilmenites_limalt = remove_alt(ilmenites_lim)
micas_limalt = remove_alt(micas_lim)
olivines_limalt = remove_alt(olivines_lim)
orthopyroxenes_limalt = remove_alt(orthopyroxenes_lim)
perovskites_limalt = remove_alt(perovskites_lim)
pyroxenes_limalt = remove_alt(pyroxenes_lim)
quartz_limalt = remove_alt(quartz_lim)
spinels_limalt = remove_alt(spinels_lim)
titanites_limalt = remove_alt(titanites_lim)
zircons_limalt = remove_alt(zircons_lim)

# %% 
