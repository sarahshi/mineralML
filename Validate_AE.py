# %% 

""" Created on March 22, 2023 // @author: Sarah Shi """

import numpy as np
import pandas as pd
import scipy

import os
import time
import random
import pickle
import warnings
warnings.simplefilter('ignore', category=(FutureWarning,UserWarning))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import hdbscan
from hdbscan.flat import (HDBSCAN_flat, approximate_predict_flat)

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import scale, normalize, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.cm as mcm
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns

import Thermobar as pt

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

# %% 

min_df = pd.read_csv('Training_Data/mindf_filt.csv')

oxideslab = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3', 'Mineral']
oxides = oxideslab[:-1]

df = min_df[oxideslab].copy()
df = df.fillna(0)
df_ox = df[oxides]
mins = np.unique(df['Mineral'])

phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'FeTiOxide',
        'Garnet', 'KFeldspar', 'Muscovite', 'Olivine', 'Orthopyroxene',
        'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline',
        'Zircon'])

# fig, ax = plt.subplots(4, 4, figsize=(18, 18))
# ax = ax.flatten()
# for i in range(len(phase)): 
#     if (df['Mineral'] == phase[i]).sum() > 0:

#         ax[i].violinplot(df[df['Mineral']==phase[i]][oxides], positions = np.linspace(0, 10, 11), showmeans = True, showextrema= False)
#         ax[i].set_title(phase[i])
#         ax[i].set_xticks(np.linspace(0, 10, 11))
#         ax[i].set_xticklabels(oxides, rotation = 45, fontsize = 15)
#         ax[i].set_ylim([-5, 105])
#         ax[i].set_yticklabels(ax[i].get_yticks(), fontsize = 15)
# fig.suptitle('Training')
# plt.tight_layout()
# # plt.savefig('Training_Min.pdf')

lepr_df = pd.read_csv('Validation_Data/lepr_allphases_lim.csv')
lepr_df = lepr_df.dropna(subset=['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3'], thresh = 6)

lepr_df = lepr_df[oxideslab].copy()
lepr_df = lepr_df.fillna(0)
lepr_df_ox = lepr_df[oxides]
mins = np.unique(lepr_df['Mineral'])

# fig, ax = plt.subplots(4, 4, figsize=(18, 18))
# ax = ax.flatten()

# for i in range(len(phase)):
#     if phase[i] in mins:
#         ax[i].violinplot(lepr_df[lepr_df['Mineral']==phase[i]][oxides], positions = np.linspace(0, 10, 11), showmeans = True, showextrema= False)
#         ax[i].set_title(phase[i])
#         ax[i].set_xticks(np.linspace(0, 10, 11))
#         ax[i].set_xticklabels(oxides, rotation = 45, fontsize = 15)
#         ax[i].set_ylim([-5, 105])
#         ax[i].set_yticklabels(ax[i].get_yticks(), fontsize = 15)
#     else:
#         ax[i].axis('off')  # create empty subplots for minerals that don't exist
# fig.suptitle('LEPR')
# plt.tight_layout()
# # plt.savefig('LEPR_Min.pdf')


georoc_df = pd.read_csv('Validation_Data/GEOROC_validationdata_Fe.csv')
georoc_df = georoc_df[georoc_df.Mineral.isin(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide', 'Ilmenite', '(Al)Kalifeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene','Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])]
georoc_df['Mineral'] = georoc_df['Mineral'].replace('(Al)Kalifeldspar', 'KFeldspar')


georoc_df = georoc_df[oxideslab].copy()
georoc_df = georoc_df.fillna(0)
georoc_df_ox = georoc_df[oxides]
mins = np.unique(georoc_df['Mineral'])


# %% 



# cpx_tern = pt.tern_points_px(px_comps=cpx_df.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_tern, edgecolor="k", marker="^", facecolor="tab:blue", label='GEOROC Correct Cpx', s=75, alpha = 0.25, rasterized=True)


# %% 

# Now calculate the amphibole components
# cat_23ox = pt.calculate_Leake_Diagram_Class(amp_comps=amp_df.add_suffix('_Amp'))

# fig, (ax1) = plt.subplots(1, figsize=(10, 8), sharey=True)
# pt.add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=12, color=[0.3, 0.3, 0.3], linewidth=0.5, lower_text=0.3, upper_text=0.8, text_labels=True)
# ax1.scatter(cat_23ox['Si_Amp_cat_23ox'], cat_23ox['Mgno_Amp'], c='tab:blue', edgecolor="k", alpha=0.25, label ='GEOROC Correct Amphibole')
# ax1.set_ylabel('Mg# Amphibole')
# ax1.set_xlabel('Si (apfu)')
# ax1.set_xlim([5, 9])
# ax1.invert_xaxis()

# ax1.legend(prop={'size': 10}, loc=(1.0, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# oxide_sum = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3t', 'Fe2O3', 'FeOt', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Cr2O3', 'NiO']

# amp_corr['Total'] = amp_corr[oxide_sum].sum(axis=1)
# amp_incorr['Total'] = amp_incorr[oxide_sum].sum(axis=1)

# print('Correct Amp: ' + str(round(amp_corr['Total'].mean(), 2)) + ', ' + str(round(amp_corr['Total'].std(),2)))

# print('Incorrect Amp: ' + str(round(amp_incorr['Total'].mean(),2)) + ', ' + str(round(amp_incorr['Total'].std(),2)))


# %% 



# fig, ax = plt.subplots(4, 4, figsize=(18, 18))
# ax = ax.flatten()

# for i in range(len(phase)):
#     if phase[i] in mins:
#         ax[i].violinplot(georoc_df[georoc_df['Mineral']==phase[i]][oxides], positions = np.linspace(0, 10, 11), showmeans = True, showextrema= False)
#         ax[i].set_title(phase[i])
#         ax[i].set_xticks(np.linspace(0, 10, 11))
#         ax[i].set_xticklabels(oxides, rotation = 45, fontsize = 15)
#         ax[i].set_ylim([-5, 105])
#         ax[i].set_yticklabels(ax[i].get_yticks(), fontsize = 15)
#     else:
#         ax[i].axis('off')  # create empty subplots for minerals that don't exist
# fig.suptitle('GEOROC')
# plt.tight_layout()
# # plt.savefig('GEOROC_Min.pdf')

# %% 


min_df = pd.read_csv('Training_Data/mindf_filt.csv')
oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
label = ['Mineral']

min = min_df[label]
wt = min_df[oxides].fillna(0).to_numpy()

ss = StandardScaler()
array_norm = ss.fit_transform(wt)

# %% 

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
georoc = pd.read_csv('Validation_Data/GEOROC_validationdata_Fe.csv', index_col=0)
georoc_df = georoc.dropna(subset=oxides, thresh = 6)

georoc_df = georoc_df[georoc_df.Mineral.isin(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide', 'Ilmenite', '(Al)Kalifeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene','Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])]
georoc_df['Mineral'] = georoc_df['Mineral'].replace('(Al)Kalifeldspar', 'KFeldspar')

lepr = pd.read_csv('Validation_Data/lepr_allphases_lim.csv', index_col=0)
lepr_df = lepr.dropna(subset=oxides, thresh = 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = 'mindf_256_64_16_noP'
path = 'autoencoder_parametermatrix/' + name + '_tanh_params.pt'
model = Tanh_Autoencoder(input_dim=10, hidden_layer_sizes=(256, 64, 16)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
load_model(model, optimizer, path)

georoc_wt = georoc_df[oxides].fillna(0)
georoc_wt = georoc_wt.to_numpy()
georoc_norm_wt = ss.transform(georoc_wt)

# ss1 = StandardScaler()
# georoc_norm_wt = ss1.fit_transform(georoc_wt)
z_georoc = getLatent(model, georoc_norm_wt)

lepr_wt = lepr_df[oxides].fillna(0)
lepr_wt = lepr_wt.to_numpy()
# ss2 = StandardScaler()
# lepr_norm_wt = ss2.fit_transform(lepr_wt)
lepr_norm_wt = ss.transform(lepr_wt)
z_lepr = getLatent(model, lepr_norm_wt)


cpx_df = pd.read_csv('Cpx_compilation_April23.csv')
amp_df = pd.read_csv('Amp_compilation_April23.csv')

cpx_df = cpx_df[oxideslab].copy()
cpx_df = cpx_df.fillna(0)

amp_df = amp_df[oxideslab].copy()
amp_df = amp_df.fillna(0)

cpx_wt = cpx_df[oxides].fillna(0)
amp_wt = amp_df[oxides].fillna(0)
cpx_amp_df_concat = pd.concat([cpx_df, amp_df])

cpx_wt['Cr2O3'] = pd.to_numeric(cpx_wt['Cr2O3'], errors='coerce')
amp_wt['Cr2O3'] = pd.to_numeric(amp_wt['Cr2O3'], errors='coerce')

cpx_amp_concat = pd.concat([cpx_wt, amp_wt])


# ss3 = StandardScaler()
# cpx_amp_norm_wt = ss3.fit_transform(cpx_amp_concat)
cpx_amp_norm_wt = ss.transform(cpx_amp_concat)
z_cpxamp = getLatent(model, cpx_amp_norm_wt)


# %% 

name = 'mindf_256_64_16'
min_df = pd.read_csv('Training_Data/mindf_filt.csv')
z = np.load('autoencoder_parametermatrix/' + name + '_tanh.npz')['z']

array, params = feature_normalisation(z, return_params = True)

clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels = clusterer.labels_
probs = clusterer.probabilities_

np.savez('autoencoder_parametermatrix/' + name + '_hdbscan_flat.npz', labels = labels, probs = probs)

z_scores_df = pd.DataFrame(columns = ['LV1', 'LV2']) 
z_scores_df['LV1'] = z[:,0]
z_scores_df['LV2'] = z[:,1]

fig = plt.figure(figsize = (14, 14))
gs = GridSpec(4, 4)
ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_x = fig.add_subplot(gs[0,0:3])
ax_hist_y = fig.add_subplot(gs[1:4, 3])

phase = list(set(min_df['Mineral']))
tab = plt.get_cmap('tab20')
label_plot = list(set(labels))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
labelscalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

for i in range(len(label_plot)):
    indx = labels == i
    ax_scatter.scatter(z[indx, 0], z[indx, 1], s=15, color=labelscalarMap.to_rgba(i), lw=1, label=label_plot[i], rasterized = True)
ax_scatter.set_xlabel("Latent Variable 1")
ax_scatter.set_ylabel("Latent Variable 2")
ax_scatter.set_xlim([-1.5, 2.0])
ax_scatter.set_ylim([-2.5, 2.5])
ax_scatter.legend(prop={'size': 8})

pc1_sns = sns.kdeplot(data = z_scores_df, x = 'LV1', color = 'k', ax = ax_hist_x)
pc1_sns.set_xlim([-1.5, 2.0])
pc1_sns.set(xlabel = None)

pc2_sns = sns.kdeplot(data = z_scores_df, y = 'LV2', color = 'k')
pc2_sns.set_ylim([-2.5, 2.5])
pc2_sns.set(ylabel = None)
plt.tight_layout()

# %% HDBSCAN_flat implementation for LEPR 

name = 'mindf_256_64_16'
min_df = pd.read_csv('Training_Data/mindf_filt.csv')
z = np.load('autoencoder_parametermatrix/' + name + '_tanh.npz')['z']
z_lepr = getLatent(model, lepr_norm_wt)

array, params = feature_normalisation(z, return_params = True)
clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels, probs = clusterer.labels_, clusterer.probabilities_
array_filt = array[labels!=-1]
labels_filt = labels[labels!=-1]
probs_filt = probs[labels!=-1]
labels[labels==7] = 5 # combine opx
labels[labels==4] = 3 # combine olivine

array_lepr, params_lepr = feature_normalisation(z_lepr, return_params = True)
labels_lepr, probs_lepr = approximate_predict_flat(clusterer, array_lepr, cluster_selection_epsilon=0.025)
array_lepr_filt = array_lepr[labels_lepr!=-1]
labels_lepr_filt = labels_lepr[labels_lepr!=-1]
z_lepr_filt = z_lepr[labels_lepr!=-1]
df_lepr_err = lepr_df[labels_lepr==-1]

labels_lepr_filt[labels_lepr_filt==7] = 5
labels_lepr_filt[labels_lepr_filt==4] = 3

tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_lepr_plot = list(set(labels_lepr_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}


fig, ax = plt.subplots(1, 3, figsize = (30, 10))
ax = ax.flatten()

for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas = probs[indx]
        ax[0].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, rasterized=True)

for i in range(len(phase)):
    indx = lepr_df['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[1].scatter(z_lepr[indx, 0], z_lepr[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized=True)

for label in label_lepr_plot:
    indx = labels_lepr_filt == label
    if np.any(indx):  # Add this condition
        alphas=probs_lepr
        ax[2].scatter(z_lepr_filt[indx, 0], z_lepr_filt[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, rasterized=True)

ax[0].set_title("Train/Validate - HDBSCAN_flat")
ax[0].set_xlabel("Latent Variable 1")
ax[0].set_ylabel("Latent Variable 2")
ax[0].set_xlim([-1.5, 2.0])
ax[0].set_ylim([-2.5, 2.5])
ax[0].legend(prop={'size': 8})

ax[1].set_title("Test Dataset - LEPR Labels")
ax[1].set_xlabel("Latent Variable 1")
ax[1].set_ylabel("Latent Variable 2")
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].legend(prop={'size': 8})

ax[2].set_title("Test Dataset - approximate_predict_flat prob")
ax[2].set_xlabel("Latent Variable 1")
ax[2].set_ylabel("Latent Variable 2")
ax[2].set_xlim([-1.5, 2.0])
ax[2].set_ylim([-2.5, 2.5])
ax[2].legend(prop={'size': 8})
plt.tight_layout()
# plt.savefig('LEPR_HDBSCAN_flat_probs.png', bbox_inches='tight', pad_inches = 0.025, dpi=300)

# %% 
# %% HDBSCAN_flat implementation for GEOROC 

name = 'mindf_256_64_16'
min_df = pd.read_csv('Training_Data/mindf_filt.csv')
z = np.load('autoencoder_parametermatrix/' + name + '_tanh.npz')['z']
z_georoc = getLatent(model, georoc_norm_wt)

train_idx, test_idx = train_test_split(np.arange(len(georoc_df)), test_size=0.2, stratify = georoc_df['Mineral'], random_state=42)
georoc_df_lim = georoc_df.iloc[test_idx]
z_georoc_lim = z_georoc[test_idx]

array, params = feature_normalisation(z, return_params = True)
clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels, probs = clusterer.labels_, clusterer.probabilities_
array_filt = array[labels!=-1]
labels_filt = labels[labels!=-1]
probs_filt = probs[labels!=-1]
labels[labels==7] = 5 # combine opx
labels[labels==4] = 3 # combine olivine

# array_georoc, params_georoc = feature_normalisation(z_georoc, return_params = True)
array_georoc, params_georoc = feature_normalisation(z_georoc_lim, return_params = True)
labels_georoc, probs_georoc = approximate_predict_flat(clusterer, array_georoc, cluster_selection_epsilon=0.025)
array_georoc_filt = array_georoc[labels_georoc!=-1]
labels_georoc_filt = labels_georoc[labels_georoc!=-1]
# z_georoc_filt = z_georoc[labels_georoc!=-1]
z_georoc_filt = z_georoc_lim[labels_georoc!=-1]
labels_georoc_filt[labels_georoc_filt==7] = 5
labels_georoc_filt[labels_georoc_filt==4] = 3

tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_georoc_plot = list(set(labels_georoc_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}

fig, ax = plt.subplots(1, 3, figsize = (30, 10))
ax = ax.flatten()

for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas=probs
        ax[0].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, rasterized=True)

for i in range(len(phase)):
    # indx = georoc_df['Mineral'] == phase[i]
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        # ax[1].scatter(z_georoc[indx, 0], z_georoc[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized=True)
        ax[1].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized=True)

for label in label_georoc_plot:
    indx = labels_georoc_filt == label
    if np.any(indx):  # Add this condition
        alphas=probs_georoc
        ax[2].scatter(z_georoc_filt[indx, 0], z_georoc_filt[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, rasterized=True)

ax[0].set_title("Train/Validate - HDBSCAN_flat")
ax[0].set_xlabel("Latent Variable 1")
ax[0].set_ylabel("Latent Variable 2")
ax[0].set_xlim([-1.5, 2.0])
ax[0].set_ylim([-2.5, 2.5])
ax[0].legend(prop={'size': 8})

ax[1].set_title("Test Dataset - GEOROC Labels")
ax[1].set_xlabel("Latent Variable 1")
ax[1].set_ylabel("Latent Variable 2")
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].legend(prop={'size': 8})

ax[2].set_title("Test Dataset - approximate_predict_flat")
ax[2].set_xlabel("Latent Variable 1")
ax[2].set_ylabel("Latent Variable 2")
ax[2].set_xlim([-1.5, 2.0])
ax[2].set_ylim([-2.5, 2.5])
ax[2].legend(prop={'size': 8})
plt.tight_layout()
# plt.savefig('GEOROC_HDBSCAN_flat_probs.png', bbox_inches='tight', pad_inches = 0.025, dpi=300)


# %% 

# %% EGU PLOTTING

phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])

min_df = pd.read_csv('Training_Data/mindf_filt.csv')

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
label = ['Mineral']

min = min_df[label]
wt = min_df[oxides].fillna(0).to_numpy()
wt_scale = StandardScaler().fit_transform(wt)

start = time.time()
pca_for_wt = PCA(n_components = 3)
pca_for_z = PCA(n_components = 3)
wt_pca = pca_for_wt.fit_transform(wt)
wt_z_pca = pca_for_z.fit_transform(wt_scale)
end = time.time()
print(str(round(end-start, 2)) + ' seconds elapsed')

cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

fig, ax = plt.subplots(1, 2, figsize = (16, 8))
ax = ax.flatten()
for i in range(len(phase)):
    indx = min['Mineral'] == phase[i]
    ax[0].scatter(wt_z_pca[indx][:, 0], wt_z_pca[indx][:, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], edgecolor='k', linewidth=0.1, rasterized = True)

handles, labels = ax[0].get_legend_handles_labels()
order = [11, 18, 16, 14, 17, 8, 12, 9, 15, 3, 5, 2, 6, 0, 10, 1, 13] 
ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 10}, loc = 'lower right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
ax[0].annotate("PCA: Train/Validate Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].set_xlim([-4, 10])
ax[0].set_ylim([-3, 5])

array, params = feature_normalisation(z, return_params = True)
clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels, probs = clusterer.labels_, clusterer.probabilities_
array_filt = array[labels!=-1]
labels_filt = labels[labels!=-1]
probs_filt = probs[labels!=-1]
labels[labels==7] = 5 
labels[labels==4] = 3 

array_lepr, params_lepr = feature_normalisation(z_lepr, return_params = True)
labels_lepr, probs_lepr = approximate_predict_flat(clusterer, array_lepr, cluster_selection_epsilon=0.025)
array_lepr_filt = array_lepr[labels_lepr!=-1]
labels_lepr_filt = labels_lepr[labels_lepr!=-1]
z_lepr_filt = z_lepr[labels_lepr!=-1]
df_lepr_err = lepr_df[labels_lepr==-1]
labels_lepr_filt[labels_lepr_filt==7] = 5
labels_lepr_filt[labels_lepr_filt==4] = 3

# oxides_sum = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3', 'NiO', 'P2O5']
# df_lepr_err['Total'] = df_lepr_err[oxides_sum].sum(axis=1)
# df_lepr_err.to_csv('lepr_err.csv')

tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_lepr_plot = list(set(labels_lepr_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}

for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas = probs[indx]
        ax[1].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, linewidth=0.1,edgecolor='k',rasterized = True)
ax[1].annotate("Autoencoder: Train/Validate Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].set_xlabel("Latent Variable 1")
ax[1].set_ylabel("Latent Variable 2")
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])

error = [0.0,0.5,1.0]
h = [plt.scatter([],[],s=55, c=(0,0,0,i), edgecolors='k') for i in error]
lg = ax[1].legend(h, error, prop={'size': 10}, loc = 'lower right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
lg.set_title('Cluster\nConfidence',prop={'size':10})
plt.tight_layout()
# plt.savefig('pcaae.pdf', dpi = 300, transparent = True, bbox_inches='tight', pad_inches = 0.025)


# %%
# %% 

fig, ax = plt.subplots(1, 2, figsize = (16, 8))
ax = ax.flatten()

array, params = feature_normalisation(z, return_params = True)
clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels, probs = clusterer.labels_, clusterer.probabilities_
array_filt = array[labels!=-1]
labels_filt = labels[labels!=-1]
probs_filt = probs[labels!=-1]
labels[labels==7] = 5 
labels[labels==4] = 3 

array_lepr, params_lepr = feature_normalisation(z_lepr, return_params = True)
labels_lepr, probs_lepr = approximate_predict_flat(clusterer, array_lepr, cluster_selection_epsilon=0.025)
array_lepr_filt = array_lepr[labels_lepr!=-1]
labels_lepr_filt = labels_lepr[labels_lepr!=-1]
z_lepr_filt = z_lepr[labels_lepr!=-1]
df_lepr_err = lepr_df[labels_lepr==-1]
labels_lepr_filt[labels_lepr_filt==7] = 5
labels_lepr_filt[labels_lepr_filt==4] = 3

tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_lepr_plot = list(set(labels_lepr_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}

for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas = probs[indx]
        ax[0].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, linewidth=0.1,edgecolor='k',rasterized = True)
handles, labels = ax[0].get_legend_handles_labels()

order = [9, 16, 14, 12, 15, 6, 10, 7, 13, 3, 4, 2, 5, 0, 8, 1, 11] # [11, 18, 16, 14, 17, 8, 12, 9, 15, 3, 5, 2, 6, 0, 10, 1, 13] 
leg1 = ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 10}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
for lh in leg1.legendHandles: 
    lh.set_alpha(1)

ax[0].annotate("Autoencoder: Train/Validate Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].set_xlabel("Latent Variable 1")
ax[0].set_ylabel("Latent Variable 2")
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].set_xlim([-1.5, 2.0])
ax[0].set_ylim([-2.5, 2.5])

error = [0.0,0.5,1.0]
h = [plt.scatter([],[],s=55, c=(0,0,0,i), edgecolors='k') for i in error]
# lg1 = ax[0].legend(h, error, prop={'size': 10}, loc = 'lower left', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
# lg.set_title('Cluster\nConfidence',prop={'size':10})


for i in range(len(phase)):
    indx = lepr_df['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[1].scatter(z_lepr[indx, 0], z_lepr[indx, 1], s=15, color=scalarMap.to_rgba(i), linewidth=0.1, edgecolor='k', label=phase[i], rasterized=True)
ax[1].annotate("Autoencoder: Test LEPR Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].set_xlabel("Latent Variable 1")
# ax[1].set_ylabel("Latent Variable 2")
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)

# ax[1].legend(prop={'size': 8})

plt.tight_layout()
# plt.savefig('ae_train_lepr.pdf', dpi = 300, transparent = True, bbox_inches='tight', pad_inches = 0.025)

# %% 



fig, ax = plt.subplots(1, 2, figsize = (16, 8))
ax = ax.flatten()

array, params = feature_normalisation(z, return_params = True)
clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels, probs = clusterer.labels_, clusterer.probabilities_
array_filt = array[labels!=-1]
labels_filt = labels[labels!=-1]
probs_filt = probs[labels!=-1]
labels[labels==7] = 5 
labels[labels==4] = 3 

train_idx, test_idx = train_test_split(np.arange(len(georoc_df)), test_size=0.1, stratify = georoc_df['Mineral'], random_state=42)
georoc_df_lim = georoc_df.iloc[test_idx]
z_georoc_lim = z_georoc[test_idx]

array_georoc, params_georoc = feature_normalisation(z_georoc_lim, return_params = True)
labels_georoc, probs_georoc = approximate_predict_flat(clusterer, array_georoc, cluster_selection_epsilon=0.025)
array_georoc_filt = array_georoc[labels_georoc!=-1]
labels_georoc_filt = labels_georoc[labels_georoc!=-1]
z_georoc_lim_filt = z_georoc_lim[labels_georoc!=-1]
df_georoc_err = georoc_df_lim[labels_georoc==-1]
labels_georoc_filt[labels_georoc_filt==7] = 5
labels_georoc_filt[labels_georoc_filt==4] = 3

tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_georoc_plot = list(set(labels_georoc_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}

for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas = probs[indx]
        ax[0].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, linewidth=0.1,edgecolor='k',rasterized = True)
handles, labels = ax[0].get_legend_handles_labels()

order = [9, 16, 14, 12, 15, 6, 10, 7, 13, 3, 4, 2, 5, 0, 8, 1, 11] # [11, 18, 16, 14, 17, 8, 12, 9, 15, 3, 5, 2, 6, 0, 10, 1, 13] 
leg1 = ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 10}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
for lh in leg1.legendHandles: 
    lh.set_alpha(1)

ax[0].annotate("Autoencoder: Train/Validate Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].set_xlabel("Latent Variable 1")
ax[0].set_ylabel("Latent Variable 2")
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].set_xlim([-1.5, 2.0])
ax[0].set_ylim([-2.5, 2.5])

error = [0.0,0.5,1.0]
h = [plt.scatter([],[],s=55, c=(0,0,0,i), edgecolors='k') for i in error]
# lg1 = ax[0].legend(h, error, prop={'size': 10}, loc = 'lower left', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
# lg.set_title('Cluster\nConfidence',prop={'size':10})

for i in range(len(phase)):
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[1].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=15, color=scalarMap.to_rgba(i), linewidth=0.1, edgecolor='k', label=phase[i],rasterized = True)
ax[1].annotate("Autoencoder: Test GEOROC Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].set_xlabel("Latent Variable 1")
# ax[1].set_ylabel("Latent Variable 2")
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
# ax[1].legend(prop={'size': 8})

plt.tight_layout()
# plt.savefig('ae_train_georoc.pdf', dpi = 300, transparent = True, bbox_inches='tight', pad_inches = 0.025)



# %%


df=pd.read_csv('mineralML_Cluster/Cascades_CpxAmp_NN.csv')
df_Cpx=df.loc[df['Mineral']=='Clinopyroxene']
S_Cpx=df_Cpx['NN_Labels']=='Clinopyroxene'
S_Amp=df_Cpx['NN_Labels']=='Amphibole'
S_Opx=df_Cpx['NN_Labels']=='Orthopyroxene'
S_Gt=df_Cpx['NN_Labels']=='Garnet'
S_Zr=df_Cpx['NN_Labels']=='Zircon'
S_Plg=df_Cpx['NN_Labels']=='Plagioclase'
df_Amp=df.loc[df['Mineral']=='Amphibole']
SA_Cpx=df_Amp['NN_Labels']=='Clinopyroxene'
SA_Amp=df_Amp['NN_Labels']=='Amphibole'
SA_Opx=df_Amp['NN_Labels']=='Orthopyroxene'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
x='SiO2'
y='Na2O'
ax1.set_title('Published Classifications')
ax1.plot(df_Cpx[x], df_Cpx[y], 'ok', mfc='tab:red', label='Published Cpx', rasterized=True)
ax1.plot(df_Amp[x], df_Amp[y], 'ok', mfc='tab:blue', label='Published Amp', rasterized=True)
ax1.legend(prop={'size': 12}, labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0)
ax1.tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax1.tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax2.set_title('Neural Network Classifications')
ax2.plot(df_Cpx[x].loc[S_Cpx], df_Cpx[y].loc[S_Cpx], 'ok', mfc='tab:red', label='Pub Cpx=NN Cpx', alpha=0.3, rasterized=True)
ax2.plot(df_Amp[x].loc[SA_Amp], df_Amp[y].loc[SA_Amp], 'ok', mfc='tab:blue', label='Pub Amp=NN Amp', alpha=0.3, rasterized=True)

ax2.plot(df_Amp[x].loc[SA_Cpx], df_Amp[y].loc[SA_Cpx], 'dk', mfc='cyan', lw=1, mec='black', label='Pub Amp; NN Cpx', ms=8, rasterized=True)
ax2.plot(df_Cpx[x].loc[S_Amp], df_Cpx[y].loc[S_Amp], 'dk',  mfc='yellow', lw=1, mec='black', label='Pub Cpx; NN Amp', ms=8, rasterized=True)

ax1.set_xlim([35, 60])
ax1.set_ylim([-0.1, 4.1])
ax2.legend(prop={'size': 12}, labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0)
ax2.yaxis.set_tick_params(which='both', labelbottom=True)
ax1.set_xlabel('SiO$_\mathrm{2}$ (wt.%)')
ax1.set_ylabel('Na$_\mathrm{2}$O (wt.%)')
ax2.set_xlabel('SiO$_\mathrm{2}$ (wt.%)')
ax2.set_ylabel('Na$_\mathrm{2}$O (wt.%)')
ax2.tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax2.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('nnvspub.pdf', bbox_inches='tight', pad_inches = 0.025)


# %% 

fig, ax = plt.subplots(1, 2, figsize = (16, 8))

for i in range(len(phase)):
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[0].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=15, color=scalarMap.to_rgba(i), linewidth=0.1, edgecolor='k', label=phase[i],rasterized = True)
ax[0].annotate("Autoencoder: Test GEOROC Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].set_xlabel("Latent Variable 1")
ax[0].set_ylabel("Latent Variable 2")
ax[0].set_xlim([-1.5, 2.0])
ax[0].set_ylim([-2.5, 2.5])
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
# ax[1].legend(prop={'size': 8})

z_cpx=z_cpxamp[df['Mineral']=='Clinopyroxene']
z_amp=z_cpxamp[df['Mineral']=='Amphibole']


tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_lepr_plot = list(set(labels_lepr_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}

for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas = probs[indx]
        ax[1].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, alpha=alphas, linewidth=0.1,edgecolor='k',rasterized = True)
ax[1].annotate("Autoencoder: Train/Validate Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].set_xlabel("Latent Variable 1")


ax[1].scatter(z_cpx[:, 0][S_Cpx], z_cpx[:, 1][S_Cpx], color='tab:red', edgecolor ='k', linewidth=0.2, label='Pub Cpx=NN Cpx')
ax[1].scatter(z_amp[:, 0][SA_Amp], z_amp[:,1][SA_Amp], color='tab:blue', edgecolor ='k', linewidth=0.2, label='Pub Amp=NN Amp')

ax[1].scatter(z_amp[:,0][SA_Cpx], z_amp[:,1][SA_Cpx], color='cyan', marker = 'd', lw=1, edgecolor='black', label='Pub Amp; NN Cpx', s=80, rasterized=True)
ax[1].scatter(z_cpx[:,0][S_Amp], z_cpx[:,1][S_Amp],  color='yellow', marker = 'd', lw=1, edgecolor='black', label='Pub Cpx; NN Amp', s=80, rasterized=True)

ax[1].set_xlabel("Latent Variable 1")
ax[1].set_ylabel("Latent Variable 2")
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
# ax[1].legend(prop={'size': 8})
plt.tight_layout()


# %%# %% 


fig, ax = plt.subplots(1, 2, figsize = (16, 8))

for i in range(len(phase)):
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[0].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=15, color=scalarMap.to_rgba(i), linewidth=0.1, edgecolor='k', label=phase[i],rasterized = True)
ax[0].annotate("Autoencoder: Test GEOROC Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].set_xlabel("Latent Variable 1")
ax[0].set_ylabel("Latent Variable 2")
ax[0].set_xlim([-1.5, 2.0])
ax[0].set_ylim([-2.5, 2.5])
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
# ax[1].legend(prop={'size': 8})

z_cpx=z_cpxamp[df['Mineral']=='Clinopyroxene']
z_amp=z_cpxamp[df['Mineral']=='Amphibole']

for i in range(len(phase)):
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[1].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=15, color=scalarMap.to_rgba(i), linewidth=0.1, edgecolor='k', alpha=0.2, rasterized = True)
ax[1].scatter(z_cpx[:, 0][S_Cpx], z_cpx[:, 1][S_Cpx], color='tab:red', edgecolor ='k', linewidth=0.2, label='Pub Cpx=NN Cpx')
ax[1].scatter(z_amp[:, 0][SA_Amp], z_amp[:,1][SA_Amp], color='tab:blue', edgecolor ='k', linewidth=0.2, label='Pub Amp=NN Amp')

ax[1].scatter(z_amp[:,0][SA_Cpx], z_amp[:,1][SA_Cpx], color='cyan', marker = 'd', lw=1, edgecolor='black', label='Pub Amp; NN Cpx', s=80, rasterized=True)
ax[1].scatter(z_cpx[:,0][S_Amp], z_cpx[:,1][S_Amp],  color='yellow', marker = 'd', lw=1, edgecolor='black', label='Pub Cpx; NN Amp', s=80, rasterized=True)

ax[1].set_xlabel("Latent Variable 1")
ax[1].set_ylabel("Latent Variable 2")
ax[1].set_xlim([0.4, 1.2])
ax[1].set_ylim([-0.5, 0.75])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(loc = 'lower right', prop={'size': 12})
plt.tight_layout()



# %% TRY ON OTHER GEOROC: 


name = 'mindf_256_64_16'
min_df = pd.read_csv('Training_Data/mindf_filt.csv')
z = np.load('autoencoder_parametermatrix/' + name + '_tanh.npz')['z']
z_georoc = getLatent(model, georoc_norm_wt)

train_idx, test_idx = train_test_split(np.arange(len(georoc_df)), test_size=0.2, stratify = georoc_df['Mineral'], random_state=42)
georoc_df_lim = georoc_df.iloc[test_idx]
z_georoc_lim = z_georoc[test_idx]

array, params = feature_normalisation(z, return_params = True)
clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels, probs = clusterer.labels_, clusterer.probabilities_
array_filt = array[labels!=-1]
labels_filt = labels[labels!=-1]
probs_filt = probs[labels!=-1]
labels[labels==7] = 5 # combine opx
labels[labels==4] = 3 # combine olivine

# array_georoc, params_georoc = feature_normalisation(z_georoc, return_params = True)
array_georoc, params_georoc = feature_normalisation(z_georoc_lim, return_params = True)
labels_georoc, probs_georoc = approximate_predict_flat(clusterer, array_georoc, cluster_selection_epsilon=0.025)
array_georoc_filt = array_georoc[labels_georoc!=-1]
labels_georoc_filt = labels_georoc[labels_georoc!=-1]
# z_georoc_filt = z_georoc[labels_georoc!=-1]
z_georoc_filt = z_georoc_lim[labels_georoc!=-1]
labels_georoc_filt[labels_georoc_filt==7] = 5
labels_georoc_filt[labels_georoc_filt==4] = 3

tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_georoc_plot = list(set(labels_georoc_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}

fig, ax = plt.subplots(1, 3, figsize = (30, 10))
ax = ax.flatten()

for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas=probs
        ax[0].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, rasterized=True)

for i in range(len(phase)):
    # indx = georoc_df['Mineral'] == phase[i]
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        # ax[1].scatter(z_georoc[indx, 0], z_georoc[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized=True)
        ax[1].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized=True)

for label in label_georoc_plot:
    indx = labels_georoc_filt == label
    if np.any(indx):  # Add this condition
        alphas=probs_georoc
        ax[2].scatter(z_georoc_filt[indx, 0], z_georoc_filt[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, rasterized=True)

ax[0].set_title("Train/Validate - HDBSCAN_flat")
ax[0].set_xlabel("Latent Variable 1")
ax[0].set_ylabel("Latent Variable 2")
ax[0].set_xlim([-1.5, 2.0])
ax[0].set_ylim([-2.5, 2.5])
ax[0].legend(prop={'size': 8})

ax[1].set_title("Test Dataset - GEOROC Labels")
ax[1].set_xlabel("Latent Variable 1")
ax[1].set_ylabel("Latent Variable 2")
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].legend(prop={'size': 8})

ax[2].set_title("Test Dataset - approximate_predict_flat")
ax[2].set_xlabel("Latent Variable 1")
ax[2].set_ylabel("Latent Variable 2")
ax[2].set_xlim([-1.5, 2.0])
ax[2].set_ylim([-2.5, 2.5])
ax[2].legend(prop={'size': 8})
plt.tight_layout()
# plt.savefig('GEOROC_HDBSCAN_flat_probs.png', bbox_inches='tight', pad_inches = 0.025, dpi=300)


# %% 

amp_mask = (labels_georoc==18) & (georoc_df_lim.Mineral=='Amphibole') 
cpx_mask = (labels_georoc==14) & (georoc_df_lim.Mineral=='Clinopyroxene') 
amp_good = georoc_df_lim[amp_mask]
cpx_good = georoc_df_lim[cpx_mask]

amp_mask_bad = (labels_georoc==14) & (georoc_df_lim.Mineral=='Amphibole') 
cpx_mask_bad = (labels_georoc==18) & (georoc_df_lim.Mineral=='Clinopyroxene') 
amp_bad = georoc_df_lim[amp_mask_bad]
cpx_bad = georoc_df_lim[cpx_mask_bad]

amp_mask_ae = (amp_good['SiO2'] >= 35) &  (amp_good['SiO2'] <= 50) & (amp_good['Na2O'] >= 0.5) &  (amp_good['Na2O'] <= 4) 
amp_good_filt = amp_good[amp_mask_ae]

cpx_mask_ae = (cpx_good['SiO2'] >= 45) &  (cpx_good['SiO2'] <= 57.5) & (cpx_good['Na2O'] > 0.1) &  (cpx_good['Na2O'] <= 2.5) 
cpx_good_filt = cpx_good[cpx_mask_ae]

georoc_cpx = georoc_df_lim[georoc_df_lim.Mineral=='Clinopyroxene']
georoc_amp = georoc_df_lim[georoc_df_lim.Mineral=='Amphibole']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
x='SiO2'
y='Na2O'
ax1.set_title('Published Classifications')
ax1.plot(georoc_cpx[x], georoc_cpx[y], 'ok', mfc='tab:red', label='Published Cpx', alpha=0.2)
ax1.plot(georoc_amp[x], georoc_amp[y], 'ok', mfc='tab:blue', label='Published Amp', alpha=0.2)
ax1.legend(prop={'size': 12}, labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0)
ax1.tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax1.tick_params(axis="y", direction='in', length=5, pad = 6.5)


ax2.set_title('Autoencoder Classifications')
ax2.plot(cpx_good[x], cpx_good[y], 'ok', mfc='tab:red', label='Pub Cpx; AE Cpx', alpha=0.2, rasterized=True)
ax2.plot(amp_good[x], amp_good[y], 'ok', mfc='tab:blue', label='Pub Amp; AE Amp', alpha=0.2, rasterized=True)

ax2.plot(cpx_bad[x], cpx_bad[y], 'dk', mfc='cyan', label='Pub Amp; AE Cpx')
ax2.plot(amp_bad[x], amp_bad[y], 'dk', mfc='yellow', label='Pub Cpx; AE Amp')


ax1.set_xlim([35, 60])
ax1.set_ylim([-0.1, 4.1])
ax2.legend(prop={'size': 12}, labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0)
ax2.yaxis.set_tick_params(which='both', labelbottom=True)
ax1.set_xlabel('SiO$_\mathrm{2}$ (wt.%)')
ax1.set_ylabel('Na$_\mathrm{2}$O (wt.%)')
ax2.set_xlabel('SiO$_\mathrm{2}$ (wt.%)')
ax2.set_ylabel('Na$_\mathrm{2}$O (wt.%)')
ax2.tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax2.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('nnvspub.pdf', bbox_inches='tight', pad_inches = 0.025)

# %% 



amp_mask_good = (labels_georoc==18) & (georoc_df_lim.Mineral=='Amphibole') 
cpx_mask_good = (labels_georoc==14) & (georoc_df_lim.Mineral=='Clinopyroxene') 
# amp_good = georoc_df_lim[amp_mask]
# cpx_good = georoc_df_lim[cpx_mask]
amp_good = z_georoc_lim[amp_mask_good]
cpx_good = z_georoc_lim[cpx_mask_good]
cpx_good_lim = cpx_good[cpx_good[:,0]>0.87]


amp_mask_bad = (labels_georoc==14) & (georoc_df_lim.Mineral=='Amphibole') 
cpx_mask_bad = (labels_georoc==18) & (georoc_df_lim.Mineral=='Clinopyroxene') 
amp_bad = z_georoc_lim[amp_mask_bad]
cpx_bad = z_georoc_lim[cpx_mask_bad]


# amp_mask = (z_georoc_err[:,0] > 0.8) & (z_georoc_err[:,0] < 1.2) & (z_georoc_err[:,1] > 0.25) & (z_georoc_err[:,1] < 0.75)
# cpx_mask = (z_georoc_err[:,0] > 0.5) & (z_georoc_err[:,0] < 0.8) & (z_georoc_err[:,1] > -0.5) & (z_georoc_err[:,1] < -0.1)

fig, ax = plt.subplots(1, 2, figsize = (16, 8))

for i in range(len(phase)):
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[0].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=15, color=scalarMap.to_rgba(i), linewidth=0.1, edgecolor='k', label=phase[i],rasterized = True)
ax[0].annotate("Autoencoder: Test GEOROC Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].set_xlabel("Latent Variable 1")
ax[0].set_ylabel("Latent Variable 2")
ax[0].set_xlim([-1.5, 2.0])
ax[0].set_ylim([-2.5, 2.5])
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
# ax[1].legend(prop={'size': 8})

z_cpx=z_cpxamp[df['Mineral']=='Clinopyroxene']
z_amp=z_cpxamp[df['Mineral']=='Amphibole']

# for i in range(len(phase)):
#     indx = georoc_df_lim['Mineral'] == phase[i]
#     if np.any(indx):  # Add this condition
for i in range(len(phase)):
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[1].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=5, color=scalarMap.to_rgba(i), linewidth=0.1, edgecolor='k', rasterized = True, alpha=0.1)

ax[1].scatter(amp_good[:, 0], amp_good[:, 1], s=20, color='tab:blue', label='Pub Amp; AE Amp', linewidth=0.1, edgecolor='k', rasterized=True)
ax[1].scatter(cpx_good_lim[:, 0], cpx_good_lim[:, 1], s=20, color='tab:red', label='Pub Cpx; AE Cpx',linewidth=0.1, edgecolor='k', rasterized=True)

ax[1].scatter(amp_bad[:, 0], amp_bad[:, 1], s=50, marker = 'd', color='cyan', label='Pub Amp; AE Cpx', linewidth=1, edgecolor='k', rasterized=True)
ax[1].scatter(cpx_bad[:, 0], cpx_bad[:, 1], s=50, marker = 'd', color='yellow', label='Pub Amp; AE Cpx', linewidth=1, edgecolor='k', rasterized=True)

ax[1].set_xlabel("Latent Variable 1")
ax[1].set_ylabel("Latent Variable 2")
ax[1].set_xlim([0.5, 1.3])
ax[1].set_ylim([-0.5, 0.75])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(prop={'size': 12})
plt.tight_layout()



# %% 


# %% 





# %%





# %%

# %%

# # %% visualize LEPR and GEOROC cpx

# cpx_tern_mine = pt.tern_points_px(px_comps=min_df[min_df.Mineral=='Clinopyroxene'].rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))

# cpx_tern_alllepr = pt.tern_points_px(px_comps=lepr_df[lepr_df.Mineral=='Clinopyroxene'].rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))

# cpx_tern_allgeoroc = pt.tern_points_px(px_comps=georoc_df[georoc_df.Mineral=='Clinopyroxene'].rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_tern_mine, edgecolor="k", marker="^", facecolor="tab:green", label='Train/Validate All Cpx', s=75, alpha = 0.25)
# plt.legend(prop={'size': 10}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_tern_alllepr, edgecolor="k", marker="^", facecolor="tab:blue", label='LEPR All Cpx', s=75, alpha = 0.25)
# plt.legend(prop={'size': 10}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_tern_allgeoroc, edgecolor="k", marker="^", facecolor="tab:orange", label='GEOROC All Cpx', s=75, alpha = 0.25)
# plt.legend(prop={'size': 10}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# # %% visualize LEPR and GEOROC amph

# # Now calculate the amphibole components
# cat_23ox_mine = pt.calculate_Leake_Diagram_Class(amp_comps=min_df[min_df.Mineral=='Amphibole'].add_suffix('_Amp'))
# cat_23ox_alllepr = pt.calculate_Leake_Diagram_Class(amp_comps=lepr_df[lepr_df.Mineral=='Amphibole'].add_suffix('_Amp'))
# cat_23ox_allgeoroc = pt.calculate_Leake_Diagram_Class(amp_comps=georoc_df[georoc_df.Mineral=='Amphibole'].add_suffix('_Amp'))

# fig, (ax1) = plt.subplots(1, figsize=(10, 8), sharey=True)
# pt.add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=12, color=[0.3, 0.3, 0.3], linewidth=0.5, lower_text=0.3, upper_text=0.8, text_labels=True)
# ax1.scatter(cat_23ox_mine['Si_Amp_cat_23ox'], cat_23ox_mine['Mgno_Amp'], c='tab:green', edgecolor="k", alpha=0.25, label ='Train/Validate All Amphibole')
# ax1.set_ylabel('Mg# Amphibole')
# ax1.set_xlabel('Si (apfu)')
# ax1.invert_xaxis()
# ax1.legend(prop={'size': 10}, loc=(1.0, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# fig, (ax1) = plt.subplots(1, figsize=(10, 8), sharey=True)
# pt.add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=12, color=[0.3, 0.3, 0.3], linewidth=0.5, lower_text=0.3, upper_text=0.8, text_labels=True)
# ax1.scatter(cat_23ox_alllepr['Si_Amp_cat_23ox'], cat_23ox_alllepr['Mgno_Amp'], c='tab:blue', edgecolor="k", alpha=0.25, label ='LEPR All Amphibole')
# ax1.set_ylabel('Mg# Amphibole')
# ax1.set_xlabel('Si (apfu)')
# ax1.invert_xaxis()
# ax1.legend(prop={'size': 10}, loc=(1.0, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# fig, (ax1) = plt.subplots(1, figsize=(10, 8), sharey=True)
# pt.add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=12, color=[0.3, 0.3, 0.3], linewidth=0.5, lower_text=0.3, upper_text=0.8, text_labels=True)
# ax1.scatter(cat_23ox_allgeoroc['Si_Amp_cat_23ox'], cat_23ox_allgeoroc['Mgno_Amp'], c='tab:orange', edgecolor="k", alpha=0.25, label ='GEOROC All Amphibole')
# ax1.set_ylabel('Mg# Amphibole')
# ax1.set_xlabel('Si (apfu)')
# ax1.invert_xaxis()
# ax1.legend(prop={'size': 10}, loc=(1.0, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# fig, (ax1) = plt.subplots(1, figsize=(10, 8), sharey=True)
# pt.add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=12, color=[0.3, 0.3, 0.3], linewidth=0.5, lower_text=0.3, upper_text=0.8, text_labels=True)

# # Now add these components to the axis, you can change symbol size, plot multiple amphioble populations in different colors.
# ax1.scatter(cat_23ox_allgeoroc['Si_Amp_cat_23ox'], cat_23ox_allgeoroc['Mgno_Amp'], c='tab:orange', edgecolor="k", alpha=0.25, label ='GEOROC All Amphibole')
# ax1.set_ylabel('Mg# Amphibole')
# ax1.set_xlabel('Si (apfu)')
# ax1.set_xlim([4.5, 8.5])
# ax1.invert_xaxis()
# ax1.legend(prop={'size': 10}, loc=(1.0, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# # %% 


# # %% error analysis 

# amp_min_df = min_df[min_df.Mineral=='Amphibole']
# cpx_min_df = min_df[min_df.Mineral=='Clinopyroxene']
# opx_min_df = min_df[min_df.Mineral=='Orthopyroxene']

# oxides_sum = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3', 'P2O5']
# df_lepr_err['Total'] = df_lepr_err[oxides_sum].sum(axis=1)
# amp_lepr_err = df_lepr_err[df_lepr_err.Mineral=='Amphibole']
# cpx_lepr_err = df_lepr_err[df_lepr_err.Mineral=='Clinopyroxene']
# opx_lepr_err = df_lepr_err[df_lepr_err.Mineral=='Orthopyroxene']

# df_georoc_err['Total'] = df_georoc_err[oxides_sum].sum(axis=1)
# amp_georoc_err = df_georoc_err[df_georoc_err.Mineral=='Amphibole']
# cpx_georoc_err = df_georoc_err[df_georoc_err.Mineral=='Clinopyroxene']
# opx_georoc_err = df_georoc_err[df_georoc_err.Mineral=='Orthopyroxene']


# cpx_min_tern=pt.tern_points_px(px_comps=cpx_min_df.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
# cpx_lepr_err_tern=pt.tern_points_px(px_comps=cpx_lepr_err.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
# cpx_georoc_err_tern=pt.tern_points_px(px_comps=cpx_georoc_err.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))


# opx_min_tern=pt.tern_points_px(px_comps=opx_min_df.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
# opx_lepr_err_tern=pt.tern_points_px(px_comps=opx_lepr_err.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
# opx_georoc_err_tern=pt.tern_points_px(px_comps=opx_georoc_err.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))


# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", facecolor="green", label='Train/Validate Cpx', s=75)
# tax.scatter(cpx_lepr_err_tern, edgecolor="k", marker="^", facecolor="red", label='LEPR Cpx Errors', s=75, alpha=0.5)
# plt.legend(prop={'size': 10}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(opx_min_tern, edgecolor="k", marker="^", facecolor="green", label='Train/Validate Opx', s=75)
# tax.scatter(opx_lepr_err_tern, edgecolor="k", marker="^", facecolor="red", label='LEPR Opx Errors', s=75, alpha=0.5)
# plt.legend(prop={'size': 10}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", facecolor="green", label='Train/Validate Cpx', s=75)
# tax.scatter(cpx_georoc_err_tern, edgecolor="k", marker="^", facecolor="red", label='GEOROC Cpx Errors', s=75, alpha=0.5)
# plt.legend(prop={'size': 10}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(opx_min_tern, edgecolor="k", marker="^", facecolor="green", label='Train/Validate Opx', s=75)
# tax.scatter(opx_georoc_err_tern, edgecolor="k", marker="^", facecolor="red", label='GEOROC Opx Errors', s=75, alpha=0.5)
# plt.legend(prop={'size': 10}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# # %% 

# sorttotals = cpx_lepr_err['Total'].argsort()
# cpx_lepr_err_sort = cpx_lepr_err.sort_values(by=['Total'], ascending=False)
# cpx_lepr_tern_sort = cpx_lepr_err_tern[sorttotals[::-1]]

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", facecolor="#A9A9A9", label='Train/Validate Cpx', s=75, alpha=0.7)
# tax.scatter(cpx_lepr_tern_sort, edgecolor="k", marker="^", c=cpx_lepr_err_sort['Total'],vmin=np.min(cpx_lepr_err_sort['Total']), vmax=np.max(cpx_lepr_err_sort['Total']), label='LEPR Cpx Errors', s=75, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Total"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", c=cpx_min_df['Cr2O3'],vmin=np.min(cpx_min_df['Cr2O3']), vmax=np.max(cpx_min_df['Cr2O3']), facecolor="#A9A9A9", label='Train/Validate Cpx', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Cr2O3"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_lepr_tern_sort, edgecolor="k", marker="^", c=cpx_lepr_err_sort['Cr2O3'],vmin=np.min(cpx_lepr_err_sort['Cr2O3']), vmax=np.max(cpx_lepr_err_sort['Cr2O3']), label='LEPR Cpx Errors', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Cr2O3"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", c=cpx_min_df['Na2O'],vmin=np.min(cpx_min_df['Na2O']), vmax=np.max(cpx_min_df['Na2O']), facecolor="#A9A9A9", label='Train/Validate Cpx', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Na2O"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_lepr_tern_sort, edgecolor="k", marker="^", c=cpx_lepr_err_sort['Na2O'],vmin=np.min(cpx_lepr_err_sort['Na2O']), vmax=np.max(cpx_lepr_err_sort['Na2O']), label='LEPR Cpx Errors', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Na2O"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", c=cpx_min_df['K2O'],vmin=np.min(cpx_min_df['K2O']), vmax=np.max(cpx_min_df['K2O']), facecolor="#A9A9A9", label='Train/Validate Cpx', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "K2O"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_lepr_tern_sort, edgecolor="k", marker="^", c=cpx_lepr_err_sort['K2O'],vmin=np.min(cpx_lepr_err_sort['K2O']), vmax=np.max(cpx_lepr_err_sort['K2O']), label='LEPR Cpx Errors', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "K2O"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", c=cpx_min_df['Na2O'],vmin=np.min(cpx_min_df['Na2O']), vmax=np.max(cpx_min_df['Na2O']), facecolor="#A9A9A9", label='Train/Validate Cpx', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Na2O"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_lepr_tern_sort, edgecolor="k", marker="^", c=cpx_lepr_err_sort['Na2O'],vmin=np.min(cpx_lepr_err_sort['Na2O']), vmax=np.max(cpx_lepr_err_sort['Na2O']), label='LEPR Cpx Errors', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Na2O"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# # %%

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(opx_min_tern, edgecolor="k", marker="^", facecolor="#A9A9A9", label='Train/Validate Opx', s=75)
# tax.scatter(opx_lepr_err_tern, edgecolor="k", marker="^", facecolor="red", label='LEPR Opx Errors', s=75)
# plt.legend(prop={'size': 10}, loc = (0.90, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# # %% 

# sorttotals = cpx_georoc_err['Total'].argsort()
# cpx_georoc_err_sort = cpx_georoc_err.sort_values(by=['Total'], ascending=False)
# cpx_georoc_tern_sort = cpx_georoc_err_tern[sorttotals[::-1]]

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", facecolor="#A9A9A9", label='Train/Validate Cpx', s=75, alpha=0.7)
# tax.scatter(cpx_georoc_err_tern, edgecolor="k", marker="^", c=cpx_georoc_err_sort['Total'],vmin=np.min(cpx_georoc_err_sort['Total']), vmax=np.max(cpx_georoc_err_sort['Total']), label='GEOROC Cpx Errors', s=75, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Total"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", c=cpx_min_df['Cr2O3'],vmin=np.min(cpx_min_df['Cr2O3']), vmax=np.max(cpx_min_df['Cr2O3']), facecolor="#A9A9A9", label='Train/Validate Cpx', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Cr2O3"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_georoc_err_tern, edgecolor="k", marker="^", c=cpx_georoc_err_sort['Cr2O3'],vmin=np.min(cpx_georoc_err_sort['Cr2O3']), vmax=np.max(cpx_georoc_err_sort['Cr2O3']), label='GEOROC Cpx Errors', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Cr2O3"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_min_tern, edgecolor="k", marker="^", c=cpx_min_df['Na2O'],vmin=np.min(cpx_min_df['Na2O']), vmax=np.max(cpx_min_df['Na2O']), facecolor="#A9A9A9", label='Train/Validate Cpx', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Na2O"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(cpx_georoc_err_tern, edgecolor="k", marker="^", c=cpx_georoc_err_sort['Na2O'],vmin=np.min(cpx_georoc_err_sort['Na2O']), vmax=np.max(cpx_georoc_err_sort['Na2O']), label='GEOROC Cpx Errors', s=75, alpha=0.5, cmap='hot', colormap='hot', colorbar=True, cb_kwargs={"shrink": 0.5, "label": "Na2O"})
# plt.legend(prop={'size': 10}, loc = (0.95, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)


# # %% 

# fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
# tax.scatter(opx_min_tern, edgecolor="k", marker="^", facecolor="#A9A9A9", label='Train/Validate Opx', s=75)
# tax.scatter(opx_georoc_err_tern, edgecolor="k", marker="^", facecolor="red", label='GEOROC Opx Errors', s=75, alpha=0.5)
# plt.legend(prop={'size': 10}, loc = (0.90, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# # %% 


# fig, (ax1) = plt.subplots(1, figsize=(10, 8), sharey=True)
# pt.add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=12, color=[0.3, 0.3, 0.3], linewidth=0.5, lower_text=0.3, upper_text=0.8, text_labels=True)

# # Now calculate the amphibole components
# cat_23ox = pt.calculate_Leake_Diagram_Class(amp_comps=amp_min_df.add_suffix('_Amp'))
# cat_23ox_lepr = pt.calculate_Leake_Diagram_Class(amp_comps=amp_lepr_err.add_suffix('_Amp'))
# cat_23ox_georoc = pt.calculate_Leake_Diagram_Class(amp_comps=amp_georoc_err.add_suffix('_Amp'))

# # Now add these components to the axis, you can change symbol size, plot multiple amphioble populations in different colors.
# ax1.scatter(cat_23ox['Si_Amp_cat_23ox'], cat_23ox['Mgno_Amp'], c='green', edgecolor="k", label ='Train/Validate Amp')
# ax1.scatter(cat_23ox_lepr['Si_Amp_cat_23ox'], cat_23ox_lepr['Mgno_Amp'], c='red', edgecolor="k", label ='LEPR Amp Errors')
# ax1.set_ylabel('Mg# Amphibole')
# ax1.set_xlabel('Si (apfu)')
# ax1.invert_xaxis()
# ax1.legend(prop={'size': 10}, loc=(1.0, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# # %% 

# fig, (ax1) = plt.subplots(1, figsize=(10, 8), sharey=True)
# pt.add_Leake_Amp_Fields_Fig3bot(ax1, fontsize=12, color=[0.3, 0.3, 0.3], linewidth=0.5, lower_text=0.3, upper_text=0.8, text_labels=True)

# # Now calculate the amphibole components
# cat_23ox = pt.calculate_Leake_Diagram_Class(amp_comps=amp_min_df.add_suffix('_Amp'))
# cat_23ox_lepr = pt.calculate_Leake_Diagram_Class(amp_comps=amp_lepr_err.add_suffix('_Amp'))
# cat_23ox_georoc = pt.calculate_Leake_Diagram_Class(amp_comps=amp_georoc_err.add_suffix('_Amp'))

# # Now add these components to the axis, you can change symbol size, plot multiple amphioble populations in different colors.
# ax1.scatter(cat_23ox['Si_Amp_cat_23ox'], cat_23ox['Mgno_Amp'], c='green', edgecolor="k", label ='Train/Validate Amp')
# ax1.scatter(cat_23ox_georoc['Si_Amp_cat_23ox'], cat_23ox_georoc['Mgno_Amp'], c='red', edgecolor="k", label ='GEOROC Amp Errors')
# ax1.set_ylabel('Mg# Amphibole')
# ax1.set_xlabel('Si (apfu)')
# ax1.invert_xaxis()
# ax1.legend(prop={'size': 10}, loc=(1.0, 0.95), labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)

# %%



# %% DSI PLOTTING

phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])

min_df = pd.read_csv('Training_Data/mindf_filt.csv')

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
label = ['Mineral']

min = min_df[label]
wt = min_df[oxides].fillna(0).to_numpy()
wt_scale = StandardScaler().fit_transform(wt)

start = time.time()
pca_for_wt = PCA(n_components = 3)
pca_for_z = PCA(n_components = 3)
wt_pca = pca_for_wt.fit_transform(wt)
wt_z_pca = pca_for_z.fit_transform(wt_scale)
end = time.time()
print(str(round(end-start, 2)) + ' seconds elapsed')

cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

fig, ax = plt.subplots(1, 3, figsize = (24, 8))
ax = ax.flatten()
for i in range(len(phase)):
    indx = min['Mineral'] == phase[i]
    ax[0].scatter(wt_z_pca[indx][:, 0], wt_z_pca[indx][:, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], edgecolor='k', linewidth=0.1, rasterized = True)

handles, labels = ax[0].get_legend_handles_labels()
order = [11, 18, 16, 14, 17, 8, 12, 9, 15, 3, 5, 2, 6, 0, 10, 1, 13] 
ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 10}, loc = 'lower right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
ax[0].set_ylabel('Latent Variable 2')
ax[0].annotate("PCA: Train/Validate Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].set_xlim([-4, 10])
ax[0].set_ylim([-3, 5])

array, params = feature_normalisation(z, return_params = True)
clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels, probs = clusterer.labels_, clusterer.probabilities_
array_filt = array[labels!=-1]
labels_filt = labels[labels!=-1]
probs_filt = probs[labels!=-1]
labels[labels==7] = 5 
labels[labels==4] = 3 

array_lepr, params_lepr = feature_normalisation(z_lepr, return_params = True)
labels_lepr, probs_lepr = approximate_predict_flat(clusterer, array_lepr, cluster_selection_epsilon=0.025)
array_lepr_filt = array_lepr[labels_lepr!=-1]
labels_lepr_filt = labels_lepr[labels_lepr!=-1]
z_lepr_filt = z_lepr[labels_lepr!=-1]
df_lepr_err = lepr_df[labels_lepr==-1]
labels_lepr_filt[labels_lepr_filt==7] = 5
labels_lepr_filt[labels_lepr_filt==4] = 3

oxides_sum = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3',  'P2O5']
df_lepr_err['Total'] = df_lepr_err[oxides_sum].sum(axis=1)
df_lepr_err.to_csv('lepr_err.csv')

tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_lepr_plot = list(set(labels_lepr_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}

for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas = probs[indx]
        ax[1].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], alpha=alphas, linewidth=0.1,edgecolor='k',rasterized = True)
ax[1].annotate("Autoencoder: Train/Validate Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].set_xlabel("Latent Variable 1")
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])

error = [0.0,0.5,1.0]
h = [plt.scatter([],[],s=55, c=(0,0,0,i), edgecolors='k') for i in error]
lg = ax[1].legend(h, error, prop={'size': 10}, loc = 'lower right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
lg.set_title('Cluster\nConfidence',prop={'size':10})


train_idx, test_idx = train_test_split(np.arange(len(georoc_df)), test_size=0.1, stratify = georoc_df['Mineral'], random_state=42)
georoc_df_lim = georoc_df.iloc[test_idx]
z_georoc_lim = z_georoc[test_idx]


for i in range(len(phase)):
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[2].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], linewidth=0.1, edgecolor='k', alpha = 0.8, rasterized=True)
ax[2].annotate("Autoencoder: Test GEOROC Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')

ax[2].set_xlim([-1.5, 2.0])
ax[2].set_ylim([-2.5, 2.5])
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)


plt.tight_layout()
# plt.savefig('dsi_pca_ae1.pdf', dpi = 300, transparent = True, bbox_inches='tight', pad_inches = 0.025)


# %%

# %% DSI PLOTTING

phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])

min_df = pd.read_csv('Training_Data/mindf_filt.csv')

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
label = ['Mineral']

min = min_df[label]
wt = min_df[oxides].fillna(0).to_numpy()
wt_scale = StandardScaler().fit_transform(wt)

start = time.time()
pca_for_wt = PCA(n_components = 3)
pca_for_z = PCA(n_components = 3)
wt_pca = pca_for_wt.fit_transform(wt)
wt_z_pca = pca_for_z.fit_transform(wt_scale)
end = time.time()
print(str(round(end-start, 2)) + ' seconds elapsed')

cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

fig, ax = plt.subplots(1, 2, figsize = (16, 8))
ax = ax.flatten()
for i in range(len(phase)):
    indx = min['Mineral'] == phase[i]
    ax[0].scatter(wt_z_pca[indx][:, 0], wt_z_pca[indx][:, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], edgecolor='k', linewidth=0.1, rasterized = True)

handles, labels = ax[0].get_legend_handles_labels()
order = [11, 18, 16, 14, 17, 8, 12, 9, 15, 3, 5, 2, 6, 0, 10, 1, 13] 
ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 12}, loc = 'lower right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
ax[0].set_xlabel('Latent Variable 1')
ax[0].set_ylabel('Latent Variable 2')
ax[0].annotate("PCA: Train/Validate Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].set_xlim([-4, 10])
ax[0].set_ylim([-3, 5])

array, params = feature_normalisation(z, return_params = True)
clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels, probs = clusterer.labels_, clusterer.probabilities_
array_filt = array[labels!=-1]
labels_filt = labels[labels!=-1]
probs_filt = probs[labels!=-1]
labels[labels==7] = 5 
labels[labels==4] = 3 

array_lepr, params_lepr = feature_normalisation(z_lepr, return_params = True)
labels_lepr, probs_lepr = approximate_predict_flat(clusterer, array_lepr, cluster_selection_epsilon=0.025)
array_lepr_filt = array_lepr[labels_lepr!=-1]
labels_lepr_filt = labels_lepr[labels_lepr!=-1]
z_lepr_filt = z_lepr[labels_lepr!=-1]
df_lepr_err = lepr_df[labels_lepr==-1]
labels_lepr_filt[labels_lepr_filt==7] = 5
labels_lepr_filt[labels_lepr_filt==4] = 3

# oxides_sum = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3',  'P2O5']
# df_lepr_err['Total'] = df_lepr_err[oxides_sum].sum(axis=1)
# df_lepr_err.to_csv('lepr_err.csv')

tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_lepr_plot = list(set(labels_lepr_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}

for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas = probs[indx]
        ax[1].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, alpha=alphas, linewidth=0.1,edgecolor='k',rasterized = True)
ax[1].annotate("Autoencoder: Test Cascades Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].set_xlabel("Latent Variable 1")
ax[1].set_ylabel("Latent Variable 2")
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])

ax[1].scatter(z_cpx[:, 0][S_Cpx], z_cpx[:, 1][S_Cpx], color='tab:red', edgecolor ='k', linewidth=0.2, label='Pub Cpx=NN Cpx', rasterized=True)
ax[1].scatter(z_amp[:, 0][SA_Amp], z_amp[:,1][SA_Amp], color='tab:blue', edgecolor ='k', linewidth=0.2, label='Pub Amp=NN Amp', rasterized=True)
# ax[1].scatter(z_amp[:,0][SA_Cpx], z_amp[:,1][SA_Cpx], color='cyan', marker = 'd', lw=1, edgecolor='black', label='Pub Amp; NN Cpx', s=80, rasterized=True)
# ax[1].scatter(z_cpx[:,0][S_Amp], z_cpx[:,1][S_Amp],  color='yellow', marker = 'd', lw=1, edgecolor='black', label='Pub Cpx; NN Amp', s=80, rasterized=True)
leg = ax[1].legend(prop={'size': 12}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
ax[1].add_artist(leg)


error = [0.0,0.5,1.0]
h = [plt.scatter([],[],s=55, c=(0,0,0,i), edgecolors='k') for i in error]
lg = ax[1].legend(h, error, prop={'size': 12}, loc = 'lower right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
lg.set_title('Cluster\nConfidence',prop={'size':12})

plt.tight_layout()
plt.savefig('egu3.pdf', dpi = 300, transparent = True, bbox_inches='tight', pad_inches = 0.025)



# %%


fig, ax = plt.subplots(1, 3, figsize = (24, 8))
ax = ax.flatten()

tab = plt.get_cmap('tab20')
label_plot = list(set(labels_filt))
label_lepr_plot = list(set(labels_lepr_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
phase = np.array(['Rutile', 'Tourmaline', 'Plagioclase', 'Olivine', 'Olivine1', 'Orthopyroxene', 'Quartz', 'Orthopyroxene1', 'Ilmenite', 'Magnetite', 'Spinel', 'Apatite', 'KFeldspar', 'Zircon', 'Clinopyroxene', 'Muscovite', 'Biotite', 'Garnet', 'Amphibole'])
label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
cluster_to_label = {i: phase[i] for i in label_plot}



for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas = probs[indx]
        ax[0].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, alpha=alphas, linewidth=0.1,edgecolor='k',label=cluster_to_label[label],rasterized = True)
ax[0].annotate("Autoencoder: Train/Validate Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].set_xlabel("Latent Variable 1")
ax[0].set_ylabel("Latent Variable 2")
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].set_xlim([-1.5, 2.0])
ax[0].set_ylim([-2.5, 2.5])
ax[0].legend(prop={'size': 12}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.4, handlelength = 1.0, frameon=False)



train_idx, test_idx = train_test_split(np.arange(len(georoc_df)), test_size=0.1, stratify = georoc_df['Mineral'], random_state=42)
georoc_df_lim = georoc_df.iloc[test_idx]
z_georoc_lim = z_georoc[test_idx]


for i in range(len(phase)):
    indx = georoc_df_lim['Mineral'] == phase[i]
    if np.any(indx):  # Add this condition
        ax[1].scatter(z_georoc_lim[indx, 0], z_georoc_lim[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], linewidth=0.1, edgecolor='k', alpha = 0.8, rasterized=True)
ax[1].annotate("Autoencoder: Test GEOROC Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].set_xlabel("Latent Variable 1")
ax[1].set_ylabel("Latent Variable 2")
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)


for label in label_georoc_plot:
    indx = labels_georoc_filt == label
    if np.any(indx):  # Add this condition
        alphas=probs_georoc
        ax[2].scatter(z_georoc_filt[indx, 0], z_georoc_filt[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, label=cluster_to_label[label], linewidth=0.1, edgecolor='k', alpha=alphas, rasterized=True)
ax[2].annotate("Autoencoder: Clustered GEOROC Data", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='medium')
ax[2].set_xlabel("Latent Variable 1")
ax[2].set_ylabel("Latent Variable 2")
ax[2].set_xlim([-1.5, 2.0])
ax[2].set_ylim([-2.5, 2.5])
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)

error = [0.0,0.5,1.0]
h = [plt.scatter([],[],s=55, c=(0,0,0,i), edgecolors='k') for i in error]
lg = ax[2].legend(h, error, prop={'size': 12}, loc = 'lower right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
lg.set_title('Cluster\nConfidence',prop={'size':12})



plt.tight_layout()
plt.savefig('egu_extra.pdf', dpi = 300, transparent = True, bbox_inches='tight', pad_inches = 0.025)

# %%
