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

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

sys.path.append('src')
import MIN_ML as mm

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.colors as mcolors
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

# %% 

min_df = pd.read_csv('Training_Data/mindf_filt.csv')

oxideslab = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3', 'NiO', 'Mineral']
oxides = oxideslab[:-1]

df = min_df[oxideslab].copy()
df = df.fillna(0)
df_ox = df[oxides]
mins = np.unique(df['Mineral'])

fig, ax = plt.subplots(4, 5, figsize=(22, 22))
ax = ax.flatten()
for i in range(len(mins)): 
    ax[i].violinplot(df[df['Mineral']==mins[i]][oxides], positions = np.linspace(0, 10, 11), showmeans = True, showextrema= False)
    ax[i].set_title(mins[i])
    ax[i].set_xticks(np.linspace(0, 10, 11))
    ax[i].set_xticklabels(oxides, rotation = 45)
    ax[i].set_ylim([-5, 105])
plt.tight_layout()

# %% 

min_df = pd.read_csv('Training_Data/mindf_filt.csv')

names = ['mindf_128_64_25ep', 'mindf_256_128_25ep', "mindf_128_32_8_25ep", "mindf_256_64_16_25ep"]

nodes_list = [(128, 64), (256, 128), (128, 32, 8), (256, 64, 16), (512, 128, 32)]

for i in range(len(names)): 
    start_time = time.time()
    print("starting " + str(names[i]))
    z = mm.autoencode(min_df, names[i], mm.Tanh_Autoencoder, nodes_list[i], 25) # (512, 128, 32, 8)
    print(names[i] + " done! Time: " + str(time.time() - start_time) + "s")

# %% 

min_df = pd.read_csv('Training_Data/mindf_filt.csv')

names = ['mindf_128_64_40ep', 'mindf_256_128_40ep', "mindf_128_32_8_40ep", "mindf_256_64_16_40ep"]

nodes_list = [(128, 64), (256, 128), (128, 32, 8), (256, 64, 16), (512, 128, 32)]

for i in range(len(names)): 
    start_time = time.time()
    print("starting " + str(names[i]))
    z = mm.autoencode(min_df, names[i], mm.Tanh_Autoencoder, nodes_list[i], 40) # (512, 128, 32, 8)
    print(names[i] + " done! Time: " + str(time.time() - start_time) + "s")

# %% 

min_df = pd.read_csv('Training_Data/mindf_filt.csv')

names = ['mindf_128_64_45ep', 'mindf_256_128_45ep', "mindf_128_32_8_45ep", "mindf_256_64_16_45ep"]

nodes_list = [(128, 64), (256, 128), (128, 32, 8), (256, 64, 16), (512, 128, 32)]

for i in range(len(names)): 
    start_time = time.time()
    print("starting " + str(names[i]))
    z = mm.autoencode(min_df, names[i], mm.Tanh_Autoencoder, nodes_list[i], 45) # (512, 128, 32, 8)
    print(names[i] + " done! Time: " + str(time.time() - start_time) + "s")

# %% 

min_df = pd.read_csv('Training_Data/mindf_filt.csv')

names = ['mindf_128_64_50ep', 'mindf_256_128_50ep', "mindf_128_32_8_50ep", "mindf_256_64_16_50ep"]

nodes_list = [(128, 64), (256, 128), (128, 32, 8), (256, 64, 16), (512, 128, 32)]

for i in range(len(names)): 
    start_time = time.time()
    print("starting " + str(names[i]))
    z = mm.autoencode(min_df, names[i], mm.Tanh_Autoencoder, nodes_list[i], 50) # (512, 128, 32, 8)
    print(names[i] + " done! Time: " + str(time.time() - start_time) + "s")

# %% 


min_df = pd.read_csv('Training_Data/mindf_filt.csv')

names = ['mindf_128_64_55ep', 'mindf_256_128_55ep', "mindf_128_32_8_55ep", "mindf_256_64_16_55ep"]

nodes_list = [(128, 64), (256, 128), (128, 32, 8), (256, 64, 16), (512, 128, 32)]

for i in range(len(names)): 
    start_time = time.time()
    print("starting " + str(names[i]))
    z = mm.autoencode(min_df, names[i], mm.Tanh_Autoencoder, nodes_list[i], 55) # (512, 128, 32, 8)
    print(names[i] + " done! Time: " + str(time.time() - start_time) + "s")

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

min_df = pd.read_csv('TrainingData/mindf_filt.csv')

phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'FeTiOxide',
        'Garnet', 'KFeldspar', 'Muscovite', 'Olivine', 'Orthopyroxene',
        'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline',
        'Zircon'])

# phase = list(set(df['Mineral']))
tab = plt.get_cmap('tab20')
cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

for i in range(len(phase)):
    indx = min_df['Mineral'] == phase[i]
    ax_scatter.scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
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
# plt.savefig(name + "_density.pdf")


# %% 

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
lepr = pd.read_csv('ValidationData/lepr_allphases_lim.csv', index_col=0)
lepr_df = lepr.dropna(subset=oxides, thresh = 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name = 'mindf_256_64_16_noP'
path = 'autoencoder_parametermatrix/' + name + '_tanh_params.pt'
model = Tanh_Autoencoder(input_dim=10, hidden_layer_sizes=(256, 64, 16)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
load_model(model, optimizer, path)

lepr_wt = lepr_df[oxides].fillna(0)
lepr_wt = lepr_wt.to_numpy()
ss2 = StandardScaler()
lepr_norm_wt = ss2.fit_transform(lepr_wt)
z_lepr = getLatent(model, lepr_norm_wt)


phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide',
    'Ilmenite', 'KFeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene',
    'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])


min_df = pd.read_csv('TrainingData/mindf_filt.csv')

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
    indx = min['Mineral'] == phase[i]
    ax[0].scatter(wt_z_pca[indx][:, 0], wt_z_pca[indx][:, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized = True)
    ax[1].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized = True)

for i in range(len(phase)):
    indx = lepr_df['Mineral'] == phase[i]
    ax[2].scatter(z_lepr[indx, 0], z_lepr[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized = True)
ax[0].legend(prop={'size': 12}, loc = 'lower right')
# ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Latent Variable 2')
ax[0].annotate("PCA", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[1].set_xlabel('Latent Variable 1')
# ax[1].set_ylabel('Latent Variable 2')
ax[1].set_xlim([-1.5, 2.0])
ax[1].set_ylim([-2.5, 2.5])
ax[1].annotate("AE Test", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)

# ax[2].set_xlabel('Latent Variable 1')
# ax[2].set_ylabel('Latent Variable 2')
ax[2].set_xlim([-1.5, 2.0])
ax[2].set_ylim([-2.5, 2.5])
ax[2].annotate("AE LEPR", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)

plt.tight_layout()

plt.savefig('PCA_AE1.pdf', dpi = 300, transparent = True, bbox_inches='tight', pad_inches = 0.025)

# %% 


# %% 


def feature_normalisation(feature, return_params = False, mean_norm = True):
    """
    Function to perform mean normalisation on the dataset passed to it.
    
    Input
    ----------
    feature (numpy array) - features to be normalised
    return_params (boolean, optional) - set True if parameters used for mean normalisation
                            are to be returned for each feature
                            
    Returns
    ----------
    norm (numpy array) - mean normalised features
    params (list of numpy arrays) - only returned if set to True above; list of parameters
                            used for the mean normalisation as derived from the features
                            (ie. mean, min and max).
    """
    
    params = []
    
    norm = np.zeros_like(feature)
    
    if len(feature.shape) == 2:
        for i in range(feature.shape[1]):
            if mean_norm == True:
                temp_mean = feature[:,i].mean()
            elif mean_norm == False:
                temp_mean = 0
            else:
                raise ValueError("Mean_norm must be boolean")
            norm[:,i] = (feature[:,i] - temp_mean) / (feature[:,i].max() - feature[:,i].min())
            params.append(np.asarray([temp_mean,feature[:,i].min(),feature[:,i].max()]))
    
    elif len(feature.shape) == 1:
        if mean_norm == True:
            temp_mean = feature[:].mean()
        elif mean_norm == False:
                temp_mean = 0
        else:
            raise ValueError("Mean_norm must be boolean")
        norm[:] = (feature[:] - temp_mean) / (feature.max() - feature.min())
        params.append(np.asarray([temp_mean,feature[:].min(),feature[:].max()]))
        
    else:
        raise ValueError("Feature array must be either 1D or 2D numpy array.")
        
    
    if return_params == True:
        return norm, params
    else:
        return norm

def cluster(data, n_clusters, method, plot = False, plot_return = False, elements = None, df_shape = None, min_samples = None):
    
    """
    Function to perform clustering on the dataset passed using the selected clustering
    algorithm.
    
    Input
    ------------
    data (either 2D or 3D numpy array) - the dataset to perform clustering on.
    n_clusters (int) - number of clusters to find, default is 2.
    method (str) - clustering algorithm to be used ["k_means", "gmm"]; default is k_means.
    plot (bool) - Make True if results are to be plotted; default is false.
    plot_return (bool) - optional, if plot=true, make True to return fig and ax objects, default is false.
    elements (list/array) - optional, used when plotting results only, default is None.
    
    Return
    ------------
    labels (2D numpy array) - assigned labels for each cluster found within the passed dataset. 
        Shape is the same as first two dimensions of data if it's 3D, otherwise it's
        the shape parameter passed to the function.
    centers (2D numpy array of shape [n_clusters, n_features]) - list of the centres of clusters
        found in the dataset.
    fig, ax (matplotlib objects (both of length 2)) - only if both plot and plot_return are set True.
    """
    
    if len(data.shape) == 2:
        #assume it's in the right form
        array = data
    else:
        raise ValueError("Input array needs to have 2 dimensions or be Pandas dataframe.")

    array, params = feature_normalisation(array, return_params = True)

    start = time.time()
    
    if method.lower() == "gmm":
        from sklearn.mixture import GaussianMixture
        #perform GMM
        gmm = GaussianMixture(n_clusters)
        labels = gmm.fit_predict(array) + 1
        centers = gmm.means_

    elif method.lower() == "k_means":
        from sklearn.cluster import KMeans
        #perform k_means clustering
        kmeans = KMeans(n_clusters=n_clusters, init = 'k-means++').fit(array)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

    elif method.lower() == "b-gmm":
        from sklearn.mixture import BayesianGaussianMixture
        #perform b-gmm
        bgm = BayesianGaussianMixture(n_components = n_clusters, covariance_type = 'full', init_params = 'kmeans', max_iter = 500, random_state=42)
        labels = bgm.fit_predict(array)
        centers = bgm.means_

    elif method.lower() == "dbscan":
        from sklearn.cluster import DBSCAN
        #perform dbscan
        dbscan = DBSCAN(eps = 0.025, min_samples = min_samples).fit(array)
        labels = dbscan.labels_

    else:
        raise ValueError("Method " + str(method) + " is not recognised.")
        
    process_time = time.time() - start    
    print("Clustering processing time (s): " + str(process_time))
        
    if plot == True:
        fig, ax = plot_cluster(labels, centers, plot_return = True, elements=elements)
        if plot_return == True:
            return labels, centers, fig, ax
        else:
            pass
    else:
        pass
    
    return labels


# %% 

from sklearn.cluster import DBSCAN

array, params = feature_normalisation(z, return_params = True)
dbscan = DBSCAN(eps = 0.025, min_samples = 100).fit(array)
labels = dbscan.labels_


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
ax_scatter.legend(prop={'size': 8})

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