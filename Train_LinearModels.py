# %% 

""" Created on March 3, 2023 // @author: Sarah Shi """

import numpy as np
import pandas as pd
import scipy

import os
import time
import json
import random
import pickle
import pygmt
import warnings

from scipy.sparse import (SparseEfficiencyWarning)
warnings.simplefilter('ignore', category=(FutureWarning,SparseEfficiencyWarning))

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import TAS as tas
import Thermobar as pt
import stoichiometry as mm
import self_confusion_matrix as pcm

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import scale, normalize, StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

from imblearn.over_sampling import RandomOverSampler


%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

pt.__version__

# %% 

# %% 

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

def myplot(coeff,labels=None):
    n = coeff.shape[0]
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')


phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'FeTiOxide',
    'Garnet', 'KFeldspar', 'Muscovite', 'Olivine', 'Orthopyroxene',
    'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline',
    'Zircon'])
phasez = range(1,len(phase))

tab = plt.get_cmap('tab20')
cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

fig, ax = plt.subplots(2, 3, figsize = (21, 14))
ax = ax.flatten()
# PCA on wt. % 
for i in range(len(phase)):
    indx = min['Mineral'] == phase[i]
    ax[0].scatter(wt_pca[indx][:, 0], wt_pca[indx][:, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
    ax[1].scatter(wt_pca[indx][:, 0], wt_pca[indx][:, 2], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
    ax[2].scatter(wt_pca[indx][:, 1], wt_pca[indx][:, 2], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
myplot(pca_for_z.components_.T)
ax[0].legend(prop={'size': 8})
ax[0].set_title('No Normalization wt% - PCA')
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')

ax[1].legend(prop={'size': 8})
ax[1].set_title('No Normalization wt% - PCA')
ax[1].set_xlabel('PC2')
ax[1].set_ylabel('PC3')

ax[2].legend(prop={'size': 8})
ax[2].set_title('No Normalization wt% - PCA')
ax[2].set_xlabel('PC2')
ax[2].set_ylabel('PC3')

# My feature_normalisation function has the same function as normalization from sklearn.preprocessing
for i in range(len(phase)):
    indx = min['Mineral'] == phase[i]
    ax[3].scatter(wt_z_pca[indx][:, 0], wt_z_pca[indx][:, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
    ax[4].scatter(wt_z_pca[indx][:, 0], wt_z_pca[indx][:, 2], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
    ax[5].scatter(wt_z_pca[indx][:, 1], wt_z_pca[indx][:, 2], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
ax[3].legend(prop={'size': 8})
ax[3].set_title('Normalization wt% - PCA')
ax[3].set_xlabel('PC1')
ax[3].set_ylabel('PC2')

ax[4].legend(prop={'size': 8})
ax[4].set_title('Normalization wt% - PCA')
ax[4].set_xlabel('PC1')
ax[4].set_ylabel('PC3')

ax[5].legend(prop={'size': 8})
ax[5].set_title('Normalization wt% - PCA')
ax[5].set_xlabel('PC2')
ax[5].set_ylabel('PC3')

plt.tight_layout()
plt.show()

# %% 

phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'FeTiOxide',
    'Garnet', 'KFeldspar', 'Muscovite', 'Olivine', 'Orthopyroxene',
    'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline',
    'Zircon'])


fig, ax = plt.subplots(1, 1, figsize = (10, 10))
for i in range(len(phase)):
    indx = min['Mineral'] == phase[i]
    ax.scatter(wt_z_pca[indx][:, 0], wt_z_pca[indx][:, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
ax.legend(prop={'size': 12})
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.savefig('PCA_vis.png', dpi = 300, transparent = True, bbox_inches='tight', pad_inches = 0.025)


# %% 


# %% 

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(phase)):
    indx = min['Mineral'] == phase[i]
    ax.scatter(wt_pca[indx][:, 0], wt_pca[indx][:, 1], wt_pca[indx][:, 2], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA on Training Dataset")

xAxisLine = ((np.min(wt_pca[:, 0]), np.max(wt_pca[:, 0])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k')
yAxisLine = ((0, 0), (np.min(wt_pca[:, 1]), np.max(wt_pca[:, 1])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k')
zAxisLine = ((0, 0), (0,0), (np.min(wt_pca[:, 2]), np.max(wt_pca[:, 2])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k')

plt.tight_layout()
plt.show()


# %% 

n = 0.15
min_df = min_df.fillna(0)
train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(wt_scale, min_df['Mineral'], test_size=n, stratify = min_df['Mineral'], random_state=42)

def balance(train_data_x, train_data_y):

    oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)

    # Resample the dataset
    x_balanced, y_balanced = oversample.fit_resample(train_data_x, train_data_y)

    df_resampled = pd.DataFrame(x_balanced)
    df_resampled['Mineral'] = y_balanced

    df_balanced = pd.DataFrame()
    for class_label in df_resampled['Mineral'].unique():
        df_class = df_resampled[df_resampled['Mineral'] == class_label]
        df_balanced = pd.concat([df_balanced, df_class.sample(n=1000, replace = True, random_state=42)])

    # Reset the index of the balanced dataframe
    df_balanced = df_balanced.reset_index(drop=True)
    train_data_x = df_balanced.iloc[:, :-1].to_numpy()
    train_data_y = df_balanced.iloc[:, -1].to_numpy()

    return train_data_x, train_data_y

train_data_x, train_data_y = balance(train_data_x, train_data_y)

from sklearn.tree import DecisionTreeClassifier

max_depth_range = list(range(1, 15))
# List to store the accuracy for each value of max_depth:
accuracy_train = []
accuracy_test = []

for depth in max_depth_range:
    clf = DecisionTreeClassifier(max_depth = depth, 
        random_state = 0)
    clf.fit(train_data_x, train_data_y)
    score_train = clf.score(train_data_x, train_data_y)
    accuracy_train.append(score_train)
    score_test = clf.score(test_data_x, test_data_y)
    accuracy_test.append(score_test)

plt.figure(figsize = (8, 8))
plt.plot(list(range(1, 15)), accuracy_train, label = 'Train')
plt.plot(list(range(1, 15)), accuracy_test, label = 'Test')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


dtree_model = DecisionTreeClassifier(random_state=42).fit(train_data_x, train_data_y)
dtree_train_pred = dtree_model.predict(train_data_x)
dtree_test_pred = dtree_model.predict(test_data_x)

score = dtree_model.score(test_data_x, test_data_y)
print(str(round(score * 100)) + '% accuracy')

cm_train = confusion_matrix(train_data_y, dtree_train_pred)
cm_test = confusion_matrix(test_data_y, dtree_test_pred)

mapping = dict(zip(pd.Categorical(min_df['Mineral']).codes, pd.Categorical(min_df['Mineral'])))
sort_dictionary= dict(sorted(mapping.items(), key=lambda item: item[0])) 

df_train_cm = pd.DataFrame(cm_train, index=sort_dictionary.values(), columns=sort_dictionary.values())
cmap = 'viridis'
pcm.pp_matrix(df_train_cm, cmap=cmap, figsize = (12, 12), title = ' Train Confusion Matrix')

df_test_cm = pd.DataFrame(cm_test, index=sort_dictionary.values(), columns=sort_dictionary.values())
cmap = 'viridis'
pcm.pp_matrix(df_test_cm, cmap=cmap, figsize = (12, 12), title = ' Test Confusion Matrix')


# %% 

# train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(wt_scale, min_df['Mineral'], test_size=n, stratify = min_df['Mineral'], random_state=42)

from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(train_data_x, train_data_y)
svm_predictions = svm_model_linear.predict(test_data_x)
  
# model accuracy for X_test  
svm_accuracy = svm_model_linear.score(test_data_x, test_data_y)
print(svm_accuracy)

# creating a confusion matrix
cm = confusion_matrix(test_data_y, svm_predictions)

# %% 

# train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(wt_scale, min_df['Mineral'], test_size=n, stratify = min_df['Mineral'], random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier().fit(train_data_x, train_data_y)
  
# accuracy on test_data_x
knn_accuracy = knn.score(test_data_x, test_data_y)
print(knn_accuracy)

# creating a confusion matrix
knn_predictions = knn.predict(test_data_x) 
cm = confusion_matrix(test_data_y, knn_predictions)

# %% 

# train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(wt_scale, min_df['Mineral'], test_size=n, stratify = min_df['Mineral'], random_state=42)

# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(train_data_x, train_data_y)
gnb_predictions = gnb.predict(test_data_x)
  
# accuracy on test_data_x
gnb_accuracy = gnb.score(test_data_x, test_data_y)
print(gnb_accuracy)

# creating a confusion matrix
cm = confusion_matrix(test_data_y, gnb_predictions)

# %% 