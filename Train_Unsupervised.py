# %% 

""" Created on February 16, 2023 // @author: Sarah Shi """

import numpy as np
import pandas as pd
import scipy

import os
import sys
import copy
import time
import random
import warnings
warnings.simplefilter('ignore', category=(FutureWarning,UserWarning))

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from hdbscan.flat import (HDBSCAN_flat, approximate_predict_flat)

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

sys.path.append('src')
import mineralML as mm

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.colors as mcolors
import seaborn as sns

from matplotlib.gridspec import GridSpec
import matplotlib.cm as mcm

from pyrolite.comp.codata import ILR, CLR

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

# %% 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

min_df = pd.read_csv('Training_Data/mindf_filt_new.csv')

oxideslab = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3', 'Mineral']
oxides = oxideslab[:-1]

df = min_df[oxideslab].copy()
df = df.fillna(0)
df_ox = df[oxides]
mins = np.unique(df['Mineral'])

# fig, ax = plt.subplots(4, 5, figsize=(22, 22))
# ax = ax.flatten()
# for i in range(len(mins)): 
#     ax[i].violinplot(df[df['Mineral']==mins[i]][oxides], positions = np.linspace(0, 9, 10), showmeans = True, showextrema= False)
#     ax[i].set_title(mins[i])
#     ax[i].set_xticks(np.linspace(0, 9, 10))
#     ax[i].set_xticklabels(oxides, rotation = 45)
#     ax[i].set_ylim([-5, 105])
# plt.tight_layout()

# min_df = pd.read_csv('Training_Data/mindf_filt_new.csv')

# names = ["ae_256_64_16", "ae_64_16_4", "ae_128_32_8", 'ae_256_32_8']
# nodes_list = [(256, 64, 16), (64, 16, 4), (128, 32, 8), (256, 32, 8)]

# for i in range(len(names)): 
#     start_time = time.time()
#     print("starting " + str(names[i]))
#     z = mm.autoencode(min_df, names[i], mm.Tanh_Autoencoder, nodes_list[i], 50) # (512, 128, 32, 8)
#     print(names[i] + " done! Time: " + str(time.time() - start_time) + "s")

# %% 

clusterer, z_df = mm.load_clusterer()

phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 
    'Ilmenite', 'KFeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene', 
    'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])    
cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('tab20'))

fig = plt.figure(figsize = (14, 14))
gs = GridSpec(4, 4)
ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_x = fig.add_subplot(gs[0,0:3])
ax_hist_y = fig.add_subplot(gs[1:4, 3])
for i in range(len(phase)):
    indx = min_df['Mineral'] == phase[i]
    ax_scatter.scatter(z_df.LV1[indx], z_df.LV2[indx], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
ax_scatter.set_xlabel("Latent Variable 1")
ax_scatter.set_ylabel("Latent Variable 2")
ax_scatter.set_xlim([-1.5, 2.0])
ax_scatter.set_ylim([-2.5, 2.5])
ax_scatter.legend(prop={'size': 8})
pc1_sns = sns.kdeplot(data = z_df, x = 'LV1', color = 'k', ax = ax_hist_x)
pc1_sns.set_xlim([-1.5, 2.0])
pc1_sns.set(xlabel = None)
pc2_sns = sns.kdeplot(data = z_df, y = 'LV2', color = 'k')
pc2_sns.set_ylim([-2.5, 2.5])
pc2_sns.set(ylabel = None)
plt.tight_layout()
# plt.savefig(name + "_density.pdf")

# %% 

lepr = mm.load_df('Validation_Data/lepr_allphases_lim.csv')
lepr_df, lepr_df_ex = mm.prep_df_ae(lepr)
lepr_df_pred = mm.predict_class_prob_ae(lepr_df)

georoc = mm.load_df('Validation_Data/GEOROC_validationdata_Fe.csv')
georoc_df, georoc_df_ex = mm.prep_df_ae(georoc)
georoc_df_pred = mm.predict_class_prob_ae(georoc_df)

petdb = mm.load_df('Validation_Data/PetDB_validationdata_Fe.csv')
petdb_df, petdb_df_ex = mm.prep_df_ae(petdb)
petdb_pred = mm.predict_class_prob_ae(petdb_df)

# %% 

phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 
    'Ilmenite', 'KFeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene', 
    'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])    
oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']

wt = min_df[oxides].fillna(0).to_numpy()
wt_scale = StandardScaler().fit_transform(wt)
pca_for_z = PCA(n_components = 3)
wt_z_pca = pca_for_z.fit_transform(wt_scale)

fig, ax = plt.subplots(1, 3, figsize = (24, 8))
ax = ax.flatten()
for i in range(len(phase)):
    indx = min_df['Mineral'] == phase[i]
    ax[0].scatter(wt_z_pca[indx][:, 0], wt_z_pca[indx][:, 1], s=15, color=scalarMap.to_rgba(i), lw=0.1, ec='k', label=phase[i], rasterized = True)
    ax[1].scatter(z_df.LV1[indx], z_df.LV2[indx], s=15, color=scalarMap.to_rgba(i), lw=0.1, ec='k', label=phase[i], rasterized = True)
for i in range(len(phase)):
    lepr_indx = lepr_df_pred['Predict_Mineral'] == phase[i]
    ax[2].scatter(lepr_df_pred.LV1[lepr_indx], lepr_df_pred.LV2[lepr_indx], s=15, color=scalarMap.to_rgba(i), lw=0.1, ec='k', rasterized = True)
ax[0].legend(prop={'size': 8})
ax[0].set_ylabel('Latent Variable 2')
ax[0].annotate("PCA", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].set_xlabel('Latent Variable 1')
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].annotate("AE Test", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[2].set_xlim([-1.5, 2.0])
ax[2].set_ylim([-2.5, 2.5])
ax[2].annotate("AE LEPR", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('PCA_AE1_test.png', dpi = 300, transparent = False, bbox_inches='tight', pad_inches = 0.025)


# %% 

# %% 

# %% 

import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.lines as mlines

array, params = mm.feature_normalisation(z, return_params = True)
clusterer = HDBSCAN_flat(array, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
labels, probs = clusterer.labels_, clusterer.probabilities_
array_filt = array[labels!=-1]
labels_filt = labels[labels!=-1]
probs_filt = probs[labels!=-1]


df_cluster = pd.DataFrame(z, columns=['LV1', 'LV2'])
df_cluster['Predict_Code'] = labels
df_cluster['Predict_Probability'] = probs

label_plot = list(set(labels_filt))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

label_to_color_idx = {label: idx for idx, label in enumerate(label_plot)}
label_to_color_idx[6] = label_to_color_idx[7]  # Olivine
label_to_color_idx[11] = label_to_color_idx[12]  # Orthopyroxene


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
legend_handles = {}  # Dictionary to store custom legend handles

for label in label_plot:
    indx = labels == label
    if np.any(indx):
        alphas = probs[indx]
        ax.scatter(z[indx, 0], z[indx, 1], 
                   color=scalarMap.to_rgba(label_to_color_idx[label]),
                   alpha=alphas, marker = 'o', lw=0.1, ec = 'k',
                   label=label_dict[label] if label_dict[label] not in ax.get_legend_handles_labels()[1] else "")
        
        # Check if label is not already in legend
        if label_dict[label] not in legend_handles:
            # Create a custom legend handle with alpha=1
            legend_handles[label_dict[label]] = mlines.Line2D([], [], color=scalarMap.to_rgba(label_to_color_idx[label]),
                                                             marker='o', linestyle='None', mec='k', mew=0.1,
                                                             markersize=7, label=label_dict[label])
ax.set_title('Mineral Group Clustering in Latent Space')
ax.set_xlabel('Latent Dimension 1')
ax.set_ylabel('Latent Dimension 2')
ax.set_xlim([-1.5, 2.0])
ax.set_ylim([-2.5, 2.5])
ax.legend(handles=legend_handles.values())
plt.show()






# cluster_to_label = {i: phase[i] for i in label_plot}

for label in label_plot:
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))

    indx = labels == label
    if np.any(indx):  # Add this condition
        alphas = probs[indx]
        ax.scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, alpha=alphas, rasterized=True)
        ax.set_title(label)
    ax.set_xlim([-1.5, 2.0])
    ax.set_ylim([-2.5, 2.5])
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')



# %%
for label in label_plot:
    indx = labels == label
    if np.any(indx):  # Add this condition
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
        alphas = probs[indx]
        ax.scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(label_to_color_idx[label]), lw=1, alpha=alphas, rasterized=True)
        ax.set_title(label)
        ax.set_xlim([-1.5, 2.0])
        ax.set_ylim([-2.5, 2.5])

plt.scatter(z[:, 0], z[:, 1], s=15, c=labels, cmap='tab20', lw=1, rasterized=True)





# %% 


# %% 

# %%



z_lepr = mm.get_latent_space(lepr_df)

phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide',
    'Ilmenite', 'KFeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene',
    'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])

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


fig, ax = plt.subplots(1, 3, figsize = (24, 8))
ax = ax.flatten()
for i in range(len(phase)):
    indx = min_df['Mineral'] == phase[i]
    ax[0].scatter(wt_z_pca[indx][:, 0], wt_z_pca[indx][:, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized = True)
    ax[1].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized = True)

for i in range(len(phase)):
    lepr_indx = lepr_df['Mineral'] == phase[i]
    ax[2].scatter(z_lepr.Latent_Var1[lepr_indx], z_lepr.Latent_Var2[lepr_indx], s=15, color=scalarMap.to_rgba(i), lw=1, rasterized = True)

ax[0].legend(prop={'size': 8})
ax[0].set_ylabel('Latent Variable 2')
ax[0].annotate("PCA", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].set_xlabel('Latent Variable 1')
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].annotate("AE Test", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[2].set_xlim([-1.5, 2.0])
ax[2].set_ylim([-2.5, 2.5])
ax[2].annotate("AE LEPR", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()

# plt.savefig('PCA_AE1_test.png', dpi = 300, transparent = True, bbox_inches='tight', pad_inches = 0.025)
plt.savefig('PCA_AE1_test.png', dpi = 300, transparent = False, bbox_inches='tight', pad_inches = 0.025)

# %% 


# %% 

name = 'mindf_256_64_16'

z = np.load('autoencoder_parametermatrix/' + name + '_tanh.npz')['z']

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
ax_scatter.legend(prop={'size': 8}, loc='lower left')

pc1_sns = sns.kdeplot(data = z_scores_df, x = 'LV1', color = 'k', ax = ax_hist_x)
pc1_sns.set_xlim([-1.5, 2.0])
pc1_sns.set(xlabel = None)

pc2_sns = sns.kdeplot(data = z_scores_df, y = 'LV2', color = 'k')
pc2_sns.set_ylim([-2.5, 2.5])
pc2_sns.set(ylabel = None)

plt.tight_layout()
plt.savefig(name + "_density_dbscan.pdf")

# %% 

# %% 


array_lepr, params_lepr = feature_normalisation(z_lepr, return_params = True)
dbscan_lepr = DBSCAN(eps = 0.025, min_samples = 100).fit(array_lepr)
labels_lepr = dbscan_lepr.labels_



z_scores_df_lepr = pd.DataFrame(columns = ['LV1', 'LV2']) 
z_scores_df_lepr['LV1'] = z_lepr[:,0]
z_scores_df_lepr['LV2'] = z_lepr[:,1]


fig = plt.figure(figsize = (14, 14))
gs = GridSpec(4, 4)
ax_scatter = fig.add_subplot(gs[1:4, 0:3])
ax_hist_x = fig.add_subplot(gs[0,0:3])
ax_hist_y = fig.add_subplot(gs[1:4, 3])

phase = list(set(lepr_df['Mineral']))
tab = plt.get_cmap('tab20')
label_plot_lepr = list(set(labels_lepr))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(label_plot_lepr))
labelscalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

for i in range(len(label_plot_lepr)):
    indx = labels_lepr == i
    ax_scatter.scatter(z_lepr[indx, 0], z_lepr[indx, 1], s=15, color=labelscalarMap.to_rgba(i), lw=1, label=label_plot_lepr[i], rasterized = True)
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
# plt.savefig("lepr_density_dbscan.pdf")

# %% 

# %% 

wt = min_df[oxides].fillna(0)
wt = wt.to_numpy()

ss = StandardScaler()
array_norm = ss.fit_transform(wt)


idx_train, idx_test = train_test_split(np.arange(0, len(array_norm)), test_size=0.1, stratify = min_df['Mineral'], random_state=42)



min_lim = min
z_lim = z
labels_lim = labels
labels_lab = labels_lim

label_dict = {
    -1: 'False', 
    0: 'Amphibole',
    1: 'Garnet',
    2: 'Apatite',
    3: 'Biotite',
    4: 'Clinopyroxene',
    5: 'KFeldspar',
    6: 'Muscovite',
    7: 'Olivine',
    8: 'Olivine',
    9: 'Orthopyroxene',
    10: 'Orthopyroxene',
    11: 'Ilmenite',
    12: 'Magnetite',
    13: 'Plagioclase',
    14: 'Rutile',
    15: 'Spinel',
    16: 'Tourmaline',
    17: 'Zircon'
}

labels_str = np.vectorize(label_dict.get)(labels_lab)

labels_db_df = pd.DataFrame({'Mineral': labels_str})

labels_db_test = labels_db_df['Mineral'][idx_test]
min_test = min_lim['Mineral'][idx_test]

# %% 
idx_train, idx_test = train_test_split(np.arange(0, len(array_norm)), test_size=0.1, stratify = min_df['Mineral'], random_state=42)

disagree = np.sum(labels_db_test.values != min_test.values)
print(disagree)

# %% 

bool_lim = labels_db_test.values != min_test.values

# %% 

# train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(z, min_df['Mineral'], train_size=0.75, stratify = df['Mineral'], random_state=42)

# C = 2.0  # SVM regularization parameter
# models = (svm.SVC(kernel="linear", C=C),
#     svm.LinearSVC(C=C, max_iter=10000),
#     svm.SVC(kernel="rbf", gamma='auto', C=C),
#     svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),)

# titles = ("SVC with linear kernel",
#     "LinearSVC (linear kernel)",
#     "SVC with RBF kernel",
#     "SVC with polynomial (degree 3) kernel",)

# models = (clf.fit(train_data_x, train_data_y) for clf in models)

# fig, sub = plt.subplots(2, 2, figsize = (20, 20))
# x0, x1 = train_data_x[:, 0], train_data_x[:, 1]

# for clf, title, ax in zip(models, titles, sub.flatten()):
#     disp = DecisionBoundaryDisplay.from_estimator(
#         clf, train_data_x,response_method="predict",
#         cmap=plt.cm.coolwarm, alpha=0.8, ax=ax,
#         xlabel='LV1', ylabel='LV2',)
#     ax.scatter(x0, x1, c=train_data_y.astype('category').cat.codes, s=20, edgecolors="k", rasterized = True)
#     ax.set_title(title)
# plt.tight_layout()
# plt.show()

# # %% 

# test = pd.concat([z_scores_df, df['Mineral']], axis = 1)
# px.scatter(test, x='LV1', y='LV2', color='Mineral', hover_data=['Mineral'], width = 800, height = 800)

# %%