# %% 

""" Created on February 16, 2023 // @author: Sarah Shi """

import numpy as np
import pandas as pd

import os
import sys
import pickle

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from pyrolite.plot import pyroplot

sys.path.append('src')
import mineralML as mm

from matplotlib import pyplot as plt
from matplotlib import rc
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42


# %% 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mm.same_seeds(42)

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']

min_df = pd.read_csv('Training_Data/mindf_filt_new.csv', dtype={'Mineral': 'category'})
min_df_lim = min_df[~min_df['Mineral'].isin(['Tourmaline', 'Quartz', 'Rutile', 'Apatite', 'Zircon'])]

lr = 5e-3 
wd = 1e-3 
dr = 0.1
n = 0.20
kl_weight_decay_list = [0.75] # [0.0, 0.25, 0.5, 0.75, 1.0]
hls_list = [[64, 32, 16]] # [[8], [16], [16, 8], [32, 16], [64, 32], [64, 32, 16]]
epochs = 1500 
# best_model_state = neuralnetwork(min_df_lim, hls_list, kl_weight_decay_list, lr, wd, dr, epochs, n, balanced=True) 

# %% 

opx = min_df_lim[min_df_lim.Mineral=='Orthopyroxene']
cpx = min_df_lim[min_df_lim.Mineral=='Clinopyroxene']
constants = ['Sample Name', 'Mineral']

opx_components = mm.calculate_clinopyroxene_components(opx.rename(columns={c: c+'_Cpx' for c in opx.columns if c not in constants}))
cpx_components = mm.calculate_clinopyroxene_components(cpx.rename(columns={c: c+'_Cpx' for c in cpx.columns if c not in constants}))

opx.loc[(opx_components['Ca_CaMgFe_Cpx']<0.05), 'Empirical_Mineral'] = 'Orthopyroxene'
opx.loc[(opx_components['Ca_CaMgFe_Cpx'].between(0.05, 0.2)), 'Empirical_Mineral'] = 'Pigeonite'
opx.loc[(opx_components['Ca_CaMgFe_Cpx']>0.2), 'Empirical_Mineral'] = 'Clinopyroxene'


cpx.loc[(cpx_components['Ca_CaMgFe_Cpx']<0.05), 'Empirical_Mineral'] = 'Orthopyroxene'
cpx.loc[(cpx_components['Ca_CaMgFe_Cpx'].between(0.05, 0.2)), 'Empirical_Mineral'] = 'Pigeonite'
cpx.loc[(cpx_components['Ca_CaMgFe_Cpx']>0.2), 'Empirical_Mineral'] = 'Clinopyroxene'

opx_bad = opx[opx.Mineral != opx.Empirical_Mineral]
cpx_bad = cpx[cpx.Mineral != cpx.Empirical_Mineral]

# %% 

# npz = np.load('parametermatrix_neuralnetwork/best_model_data.npz')
# hls = npz['best_hidden_layer_size']
# kl_weight = npz['best_kl_weight_decay']
# train_y = npz['train_y']
# train_pred_y = npz['train_pred_y']
# valid_y = npz['valid_y']
# valid_pred_y = npz['valid_pred_y']

# min_cat, mapping = mm.load_minclass()

# cm_train = confusion_matrix(train_y, train_pred_y)
# cm_test = confusion_matrix(valid_y, valid_pred_y)

# df_train_cm = pd.DataFrame(cm_train, index=mapping.values(), columns=mapping.values())
# cmap = 'BuGn'
# mm.pp_matrix(df_train_cm, cmap = cmap, savefig = 'train', figsize = (11.5, 11.5)) 
# plt.show()

# df_valid_cm = pd.DataFrame(cm_test, index=mapping.values(), columns=mapping.values())
# mm.pp_matrix(df_valid_cm, cmap = cmap, savefig = 'test', figsize = (11.5, 11.5))
# plt.show()


# %% 

# Step 2: Read in your DataFrame, drop rows with NaN in specific oxide columns, fill NaNs, and filter minerals
petrelli_df_load = mm.load_df('Petrelli_cpx.csv')
petrelli_df, petrelli_df_ex = mm.prep_df_nn(petrelli_df_load)
petrelli_df_pred, petrelli_probability_matrix = mm.predict_class_prob_nn(petrelli_df)

petrelli_bayes_valid_report = classification_report(
    petrelli_df_pred['Mineral'], petrelli_df_pred['Predict_Mineral'], zero_division=0
)
print("Petrelli Validation Report:\n", petrelli_bayes_valid_report)

petrelli_cm = mm.confusion_matrix_df(petrelli_df_pred['Mineral'], petrelli_df_pred['Predict_Mineral'])
print("Petrelli Confusion Matrix:\n", petrelli_cm)

petrelli_cm[petrelli_cm < len(petrelli_df_pred['Predict_Mineral'])*0.0005] = 0
mm.pp_matrix(petrelli_cm) # , savefig = 'none') 


# %%

# Step 2: Read in your DataFrame, drop rows with NaN in specific oxide columns, fill NaNs, and filter minerals
lepr_df_load = mm.load_df('Validation_Data/lepr_allphases_lim_sp.csv')
lepr_df, lepr_df_ex = mm.prep_df_nn(lepr_df_load)
lepr_df_pred, lepr_probability_matrix = mm.predict_class_prob_nn(lepr_df)

lepr_bayes_valid_report = classification_report(
    lepr_df_pred['Mineral'], lepr_df_pred['Predict_Mineral'], zero_division=0
)
print("LEPR Validation Report:\n", lepr_bayes_valid_report)

lepr_cm = mm.confusion_matrix_df(lepr_df_pred['Mineral'], lepr_df_pred['Predict_Mineral'])
print("LEPR Confusion Matrix:\n", lepr_cm)

lepr_cm[lepr_cm < len(lepr_df_pred['Predict_Mineral'])*0.0005] = 0
mm.pp_matrix(lepr_cm) 

# %% 


# Step 2: Read in your DataFrame, drop rows with NaN in specific oxide columns, fill NaNs, and filter minerals
petdb_df_load = mm.load_df('Validation_Data/PetDB_validationdata_Fe.csv')
petdb_df, petdb_df_ex = mm.prep_df_nn(petdb_df_load)
petdb_df_pred, petdb_probability_matrix = mm.predict_class_prob_nn(petdb_df)

petdb_bayes_valid_report = classification_report(
    petdb_df_pred['Mineral'], petdb_df_pred['Predict_Mineral'], zero_division=0
)
print("PetDB Validation Report:\n", petdb_bayes_valid_report)

petdb_cm = mm.confusion_matrix_df(petdb_df_pred['Mineral'], petdb_df_pred['Predict_Mineral'])
print("PetDB Confusion Matrix:\n", petdb_cm)

petdb_cm[petdb_cm < len(petdb_df_pred['Predict_Mineral'])*0.0005] = 0
mm.pp_matrix(petdb_cm)

# %% 

# %% 

georoc_df_load = mm.load_df('Validation_Data/GEOROC_validationdata_Fe.csv')
georoc_df_load['Mineral'] = georoc_df_load['Mineral'].replace('(Al)Kalifeldspar', 'KFeldspar')
georoc_df, georoc_df_ex = mm.prep_df_nn(georoc_df_load)

georoc_df_pred, georoc_probability_matrix = mm.predict_class_prob_nn(georoc_df)


georoc_bayes_valid_report = classification_report(
    georoc_df_pred['Mineral'], georoc_df_pred['Predict_Mineral'], zero_division=0
)
print("GEOROC Validation Report:\n", georoc_bayes_valid_report)

georoc_cm = mm.confusion_matrix_df(georoc_df_pred['Mineral'], georoc_df_pred['Predict_Mineral'])
print("GEOROC Confusion Matrix:\n", georoc_cm)

georoc_cm[georoc_cm < len(georoc_df_pred['Predict_Mineral'])*0.0005] = 0
mm.pp_matrix(georoc_cm, savefig = None) 


# %% 

cascades_df_load = mm.load_df('Validation_Data/Cascades_CpxAmp_NN.csv')
cascades_df, cascades_df_ex = mm.prep_df_nn(cascades_df_load)

cascades_df_pred, cascades_probability_matrix = mm.predict_class_prob_nn(cascades_df)


cascades_bayes_valid_report = classification_report(
    cascades_df_pred['Mineral'], cascades_df_pred['Predict_Mineral'], zero_division=0
)
print("Cascades Validation Report:\n", cascades_bayes_valid_report)


# %% 

cascades_df_pred.to_csv('Validation_Data/Cascades_CpxAmp_NN.csv')


# %% 

def confusion_matrix_df_test(given_min, pred_min):

    """

    Constructs a confusion matrix as a pandas DataFrame for easy visualization and 
    analysis. The function first finds the unique classes and maps them to their 
    corresponding mineral names. Then, it uses these mappings to construct the 
    confusion matrix, which compares the given and predicted classes.

    Parameters:
        given_class (array-like): The true class labels.
        pred_class (array-like): The predicted class labels.

    Returns:
        cm_df (DataFrame): A DataFrame representing the confusion matrix, with rows 
                           and columns labeled by the unique mineral names found in 
                           the given and predicted class arrays.

    """

    cm_matrix = confusion_matrix(given_min, pred_min)
    unique, valid_mapping  = mm.unique_mapping_nn(pred_min)
    cm_df = pd.DataFrame(cm_matrix, index=valid_mapping, columns=valid_mapping)

    return cm_df


# %% 


# %% 


def mineral_supergroup(df): 

    df['Supergroup'] = df['Predict_Mineral']

    pyroxene_condition = df['Predict_Mineral'].isin(['Orthopyroxene', 'Clinopyroxene'])
    feldspar_condition = df['Predict_Mineral'].isin(['KFeldspar', 'Plagioclase'])
    oxide_condition = df['Predict_Mineral'].isin(['Spinel', 'Ilmenite', 'Magnetite'])

    df.loc[pyroxene_condition, 'Supergroup'] = 'Pyroxene'
    df.loc[feldspar_condition, 'Supergroup'] = 'Feldspar'
    df.loc[oxide_condition, 'Supergroup'] = 'Oxide'

    return df

def empirical_classification(df): 

    constants = ['Mineral']
    df['Empirical_Mineral'] = df['Supergroup']
    
    pyroxene_condition = df['Supergroup']=='Pyroxene'
    pyroxene_components = mm.calculate_clinopyroxene_components(df[pyroxene_condition].rename(columns={c: c+'_Cpx' for c in df.columns if c not in constants}))
    df.loc[(pyroxene_condition & (pyroxene_components['Ca_CaMgFe_Cpx']<0.05)), 'Empirical_Mineral'] = 'Orthopyroxene'
    df.loc[(pyroxene_condition & (pyroxene_components['Ca_CaMgFe_Cpx'].between(0.05, 0.2))), 'Empirical_Mineral'] = 'Pigeonite'
    df.loc[(pyroxene_condition & (pyroxene_components['Ca_CaMgFe_Cpx']>0.2)), 'Empirical_Mineral'] = 'Clinopyroxene'

    feldspar_condition = df['Supergroup']=='Feldspar'
    feldspar_components = mm.calculate_cat_fractions_plagioclase(df[feldspar_condition].rename(columns={c: c+'_Plag' for c in df.columns if c not in constants}))
    df.loc[(feldspar_condition & (feldspar_components['An_Plag']>0.1) & (feldspar_components['Or_Plag']<0.1)), 'Empirical_Mineral'] = 'Plagioclase'
    df.loc[(feldspar_condition & (feldspar_components['An_Plag']<0.1) & (feldspar_components['Or_Plag']>0.1)), 'Empirical_Mineral'] = 'KFeldspar'
    df.loc[(feldspar_condition & (feldspar_components['An_Plag']<0.1) & (feldspar_components['Or_Plag']<0.1)), 'Empirical_Mineral'] = 'Albite'

    oxide_condition = df['Supergroup']=='Oxide' # all oxyspinels 


    return df


lepr_df_pred_super = mm.mineral_supergroup(lepr_df_pred)
lepr_df_emp = empirical_classification(lepr_df_pred_super)

# %% 
# %%


tlepr = lepr_df[lepr_df['Mineral'] == lepr_df['NN_Labels']]
flepr = lepr_df[lepr_df['Mineral'] != lepr_df['NN_Labels']]

import Thermobar as pt 

cpx_corr = tlepr[tlepr.Mineral=='Clinopyroxene']
cpx_incorr = tlepr[tlepr.Mineral=='Clinopyroxene']

opx_corr = tlepr[tlepr.Mineral=='Orthopyroxene']
opx_incorr = tlepr[tlepr.Mineral=='Orthopyroxene']

cpx_tern_corr = pt.tern_points_px(px_comps=cpx_corr.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
cpx_tern_incorr = pt.tern_points_px(px_comps=cpx_incorr.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
opx_tern_corr = pt.tern_points_px(opx_corr.rename(columns={'MgO':'MgO_Opx', 'FeOt':'FeOt_Opx', 'CaO':'CaO_Opx'}))
opx_tern_incorr = pt.tern_points_px(opx_incorr.rename(columns={'MgO':'MgO_Opx', 'FeOt':'FeOt_Opx', 'CaO':'CaO_Opx'}))

opx_comps_corr = pt.calculate_orthopyroxene_components(opx_corr.rename(columns={'MgO':'MgO_Opx', 'FeOt':'FeOt_Opx', 'CaO':'CaO_Opx'}))
opx_comps_incorr = pt.calculate_orthopyroxene_components(opx_incorr.rename(columns={'MgO':'MgO_Opx', 'FeOt':'FeOt_Opx', 'CaO':'CaO_Opx'}))

cpxpred_amplabel = lepr_df[(lepr_df.NN_Labels=='Clinopyroxene') & (lepr_df.Mineral=='Amphibole')]
amppred_cpxlabel = lepr_df[(lepr_df.NN_Labels=='Amphibole') & (lepr_df.Mineral=='Clinopyroxene')]

amp_tern_corr = pt.tern_points_px(px_comps=amppred_cpxlabel.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))

px_points_corr = pt.tern_points(opx_comps_corr["Fs_Simple_MgFeCa_Opx"],  opx_comps_corr["Wo_Simple_MgFeCa_Opx"],  opx_comps_corr["En_Simple_MgFeCa_Opx"])
px_points_incorr = pt.tern_points(opx_comps_incorr["Fs_Simple_MgFeCa_Opx"],  opx_comps_incorr["Wo_Simple_MgFeCa_Opx"],  opx_comps_incorr["En_Simple_MgFeCa_Opx"])

fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
tax.scatter(cpx_tern_corr, edgecolor="k", marker="^", facecolor="tab:blue", label='NN Predicted == LEPR Labeled Cpx', s=75, alpha = 0.25, rasterized=True)
tax.scatter(opx_tern_corr, edgecolor="k", marker="^", facecolor="tab:red", label='NN Predicted == LEPR Labeled Opx', s=75, alpha = 0.25, rasterized=True)

nn_cpx_lepr = lepr_df[lepr_df['NN_Labels']=='Clinopyroxene']
nn_opx_lepr = lepr_df[lepr_df['NN_Labels']=='Orthopyroxene']
nn_cpx_lepr_tern = pt.tern_points_px(px_comps=nn_cpx_lepr.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
nn_opx_lepr_tern = pt.tern_points_px(px_comps=nn_opx_lepr.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))

fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
tax.scatter(nn_cpx_lepr_tern, edgecolor="k", marker="^", facecolor="tab:blue", label='NN Predicted Cpx', s=75, alpha = 0.25, rasterized=True)
tax.scatter(nn_opx_lepr_tern, edgecolor="k", marker="^", facecolor="tab:red", label='NN Predicted Opx', s=75, alpha = 0.25, rasterized=True)
plt.legend()

# %%

with open('src/mineralML/scaler.pkl','rb') as f:
    scaler = pickle.load(f)

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']

georoc = pd.read_csv('Validation_Data/GEOROC_validationdata_Fe.csv', index_col=0)
georoc_df = georoc.dropna(subset=oxides, thresh = 6)

# georoc_df = georoc_df[georoc_df.Mineral.isin(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide', 'Ilmenite', '(Al)Kalifeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene','Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])]
georoc_df = georoc_df[georoc_df.Mineral.isin(['Amphibole', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide', 'Ilmenite', '(Al)Kalifeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene','Plagioclase', 'Spinel'])]
georoc_df['Mineral'] = georoc_df['Mineral'].replace('(Al)Kalifeldspar', 'KFeldspar')
georoc_df = georoc_df[~georoc_df['Mineral'].isin(['Tourmaline', 'Quartz', 'Rutile', 'Apatite', 'Zircon'])]

data_idx = np.arange(len(georoc_df))
train_idx, test_idx = train_test_split(data_idx, test_size=0.2, stratify=pd.Categorical(georoc_df['Mineral']).codes, random_state=42, shuffle=True)
georoc_df_lim = georoc_df.iloc[test_idx]

georoc_wt = georoc_df_lim[oxides].fillna(0)
georoc_wt = georoc_wt.to_numpy()
georoc_norm_wt = scaler.transform(georoc_wt)

min_df_lim['Mineral'] = min_df_lim['Mineral'].astype('category')
georoc_df_lim['Mineral'] = georoc_df_lim['Mineral'].astype(pd.CategoricalDtype(categories=min_df_lim['Mineral'].cat.categories))
new_validation_data_y_georoc = (georoc_df_lim['Mineral'].cat.codes).values

# Create a DataLoader for the new validation dataset
new_validation_dataset_georoc = LabelDataset(georoc_norm_wt, new_validation_data_y_georoc)
new_validation_loader_georoc = DataLoader(new_validation_dataset_georoc, batch_size=256, shuffle=False)

input_size = len(new_validation_dataset_georoc.__getitem__(0)[0])

path = 'parametermatrix_neuralnetwork/best_model.pt'

model = MultiClassClassifier(input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=hls).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

load_model(model, optimizer, path)

# Use the trained model to predict the classes for the new validation dataset
model.eval()
new_validation_pred_classes_georoc = []
with torch.no_grad():
    for data, labels in new_validation_loader_georoc:
        x = data.to(device)
        pred_classes = model.predict(x)
        new_validation_pred_classes_georoc.extend(pred_classes.tolist())


new_validation_pred_classes_georoc = np.array(new_validation_pred_classes_georoc)
unique_classes_georoc = np.unique(np.concatenate((new_validation_data_y_georoc[new_validation_data_y_georoc != -1], new_validation_pred_classes_georoc[new_validation_data_y_georoc != -1])))

sort_mapping = {key: value for key, value in sorted(mapping.items(), key=lambda item: item[0]) if key in unique_classes_georoc}

# Calculate classification metrics for the new validation dataset
new_validation_report = classification_report(new_validation_data_y_georoc[new_validation_data_y_georoc != -1], new_validation_pred_classes_georoc[new_validation_data_y_georoc!=-1], labels = unique_classes_georoc, target_names=[sort_mapping[x] for x in unique_classes_georoc], zero_division=0)
print("New validation report:\n", new_validation_report)

cm_valid = confusion_matrix(new_validation_data_y_georoc[new_validation_data_y_georoc!=-1], new_validation_pred_classes_georoc[new_validation_data_y_georoc!=-1])

df_valid_cm = pd.DataFrame(
    cm_valid,
    index=[sort_mapping[x] for x in unique_classes_georoc],
    columns=[sort_mapping[x] for x in unique_classes_georoc],
)

df_valid_cm[df_valid_cm < len(new_validation_pred_classes_georoc)*0.001] = 0

mm.pp_matrix(df_valid_cm, cmap = cmap, savefig = 'georoc_valid', figsize = (11.5, 11.5)) 
# mm.pp_matrix(df_valid_cm, cmap = cmap, figsize = (11.5, 11.5)) 

# # Convert the predicted integer labels to string labels using the sort_mapping dictionary
new_validation_pred_labels_georoc = np.array([sort_mapping[x] for x in new_validation_pred_classes_georoc])
georoc_df_lim['NN_Labels'] = new_validation_pred_labels_georoc

georoc_df_lim.to_csv('GEOROC_CpxAmp_NN_Variational.csv')

true_georoc = georoc_df_lim[georoc_df_lim['Mineral'] == georoc_df_lim['NN_Labels']]
false_georoc = georoc_df_lim[georoc_df_lim['Mineral'] != georoc_df_lim['NN_Labels']]

false_spinels_georoc = false_georoc[false_georoc['Mineral'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]
false_spinels_georoc = false_spinels_georoc[false_spinels_georoc['NN_Labels'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]

# %% 

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
petdb = pd.read_csv('Validation_Data/PetDB_validationdata_Fe.csv', index_col=0)
petdb_df = petdb.dropna(subset=oxides, thresh = 6)

petdb_df = petdb_df[petdb_df.Mineral.isin(['Amphibole','Apatite','Biotite','Clinopyroxene','Garnet','Ilmenite','K-Feldspar',
                                             'Magnetite','Muscovite','Olivine','Orthopyroxene','Plagioclase','Quartz','Rutile','Spinel','Zircon'])]
petdb_df['Mineral'] = petdb_df['Mineral'].replace('K-Feldspar', 'KFeldspar')
petdb_df = petdb_df[~petdb_df['Mineral'].isin(['Tourmaline', 'Quartz', 'Rutile', 'Apatite', 'Zircon'])]
petdb_wt = petdb_df[oxides].fillna(0).to_numpy()
petdb_norm_wt = ss.transform(petdb_wt)

min_df_lim['Mineral'] = min_df_lim['Mineral'].astype('category')
petdb_df['Mineral'] = petdb_df['Mineral'].astype(pd.CategoricalDtype(categories=min_df_lim['Mineral'].cat.categories))
new_validation_data_y_petdb = (petdb_df['Mineral'].cat.codes).values

# Create a DataLoader for the new validation dataset
new_validation_dataset_petdb = LabelDataset(petdb_norm_wt, new_validation_data_y_petdb)
new_validation_loader_petdb = DataLoader(new_validation_dataset_petdb, batch_size=256, shuffle=False)

input_size = len(new_validation_dataset_petdb.__getitem__(0)[0])

path = 'parametermatrix_neuralnetwork/best_model.pt'

model = MultiClassClassifier(input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=hls).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

load_model(model, optimizer, path)

# Use the trained model to predict the classes for the new validation dataset

model.eval()
new_validation_pred_classes_petdb = []
with torch.no_grad():
    for data, labels in new_validation_loader_petdb: 
        x = data.to(device)
        pred_classes = model.predict(x)
        new_validation_pred_classes_petdb.extend(pred_classes.tolist())

new_validation_pred_classes_petdb = np.array(new_validation_pred_classes_petdb)
unique_classes_petdb = np.unique(np.concatenate((new_validation_data_y_petdb[new_validation_data_y_petdb != -1], new_validation_pred_classes_petdb[new_validation_data_y_petdb != -1])))
sort_mapping_petdb = {key: value for key, value in sorted(mapping.items(), key=lambda item: item[0]) if key in unique_classes_petdb}

# Calculate classification metrics for the new validation dataset
new_validation_report = classification_report(new_validation_data_y_petdb[new_validation_data_y_petdb!=-1], new_validation_pred_classes_petdb[new_validation_data_y_petdb!=-1], labels = unique_classes_petdb, target_names=[sort_mapping_petdb[x] for x in unique_classes_petdb], zero_division=0)
print("New validation report:\n", new_validation_report)

cm_valid = confusion_matrix(new_validation_data_y_petdb[new_validation_data_y_petdb!=-1], new_validation_pred_classes_petdb[new_validation_data_y_petdb!=-1])

df_valid_cm_petdb = pd.DataFrame(
    cm_valid,
    index=[sort_mapping_petdb[x] for x in unique_classes_petdb],
    columns=[sort_mapping_petdb[x] for x in unique_classes_petdb],
)

df_valid_cm_petdb[df_valid_cm_petdb < len(new_validation_pred_classes_petdb)*0.001] = 0

mm.pp_matrix(df_valid_cm_petdb, cmap = cmap, savefig = 'petdb_valid', figsize = (11.5, 11.5)) 


new_validation_pred_labels_petdb = np.array([sort_mapping_petdb[x] for x in new_validation_pred_classes_petdb])
petdb_df['NN_Labels'] = new_validation_pred_labels_petdb

petdb_df.to_csv('PetDB_NN_Variational.csv')

true_petdb = petdb_df[petdb_df['Mineral'] == petdb_df['NN_Labels']]
false_petdb = petdb_df[petdb_df['Mineral'] != petdb_df['NN_Labels']]

false_spinels_petdb = false_petdb[false_petdb['Mineral'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]
false_spinels_petdb = false_spinels_petdb[false_spinels_petdb['NN_Labels'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]


# %%
# %%
# %%
# %% 
# %%

min_df = pd.read_csv('Training_Data/mindf_filt_new.csv')

sp_df = min_df[min_df.Mineral=='Spinel']
il_df = min_df[min_df.Mineral=='Ilmenite']
mt_df = min_df[min_df.Mineral=='Magnetite']

def spinels(data_oxides): 
    # Set up  mass and charge data
    molar_mass = {'SiO2':60.08,'TiO2':79.866,'Al2O3':101.96,'FeO':71.844,'Fe2O3':159.69,
        'MnO':70.9374,'MgO':40.3044,'CaO':56.0774,'Na2O':61.98,'K2O':94.2,'Cr2O3':151.99,'NiO':74.5928}
    oxygen_numbers = {'SiO2':2,'TiO2':2,'Al2O3':3,'FeO':1,'Fe2O3':3,'MnO':1,'MgO':1,
                      'Na2O':1,'K2O':1,'CaO':1,'Cr2O3':3,'NiO':1} # number of Os per mole of oxide
    cation_ratio = {'SiO2':0.5,'TiO2':0.5,'Al2O3':2/3,'FeO':1,'Fe2O3':2/3,'MnO':1,
                    'MgO':1,'Na2O':2,'K2O':2,'CaO':1,'Cr2O3':2/3,'NiO':1} # ratio of cations to Os per mole of oxide
    cation = {'SiO2':'Si','TiO2':'Ti','Al2O3':'Al','FeO':'Fe2','Fe2O3':'Fe3',
              'MnO':'Mn','MgO':'Mg','CaO':'Ca','Na2O':'Na','K2O':'K','Cr2O3':'Cr','NiO':'Ni'}
    # =============================================================================
    # Calculate cations assuming all Fe2, for ferric iron recalculation
    # =============================================================================
    O_prop = pd.DataFrame()
    cation_allFe2 = pd.DataFrame()
    cation_pfu = pd.DataFrame()

    # Calculate atomic proportion of O from each molecule
    for oxide in data_oxides.columns.to_list():
        O_prop[oxide+'_O_prop'] = data_oxides[oxide]/molar_mass[oxide]*oxygen_numbers[oxide]

    O_prop['O_sum'] = O_prop.sum(axis = 1)
    
    # What is the scaling factor for the mineral, based on desired numbers of O (4 for spinel)
    # and desired number of cations (3 for spinel)
    X = 4 
    T = 3

    for oxide in data_oxides.columns.to_list():
        # Calculate O and multiply by cation ratio to get to cations. Sum to get 'S' a la Droop
        cation_allFe2[cation[oxide]+'_pfu'] = O_prop[oxide+'_O_prop']/O_prop['O_sum']*X*cation_ratio[oxide]
    # =============================================================================
    # Use Droop equation for calculating stoichiometric Fe3: F=2X(1-T/S) 
    # T = 3 = O_prop['min_cat_sum']
    # X = 4 = O_prop['min_O_sum'] 
    # S = observed cation total, cation_allFe2.sum(axis=1)
    # =============================================================================
    mask = cation_allFe2.sum(axis=1) > T
    O_prop['Fe3'] = np.where(mask, (2*T)*(1-T/cation_allFe2.sum(axis=1)), 0)
    cation_pfu['Fe3'] = O_prop['Fe3']

    # Normalise cations to total expected cation number ('Fe2' is really total Fe)
    other_cations = ['Si','Ti','Al','Fe2','Mn','Mg','Na','K','Ca','Cr','Ni']
    for cation in other_cations: 
        cation_pfu[cation] = cation_allFe2[cation+'_pfu']/cation_allFe2.sum(axis=1)*T
    
    # Now replace the Fe2 which is currently actually the total Fe
    cation_pfu['Fe2'] = cation_pfu['Fe2'] - O_prop['Fe3'] # np.where(cation_pfu['Fe2'] > 0, cation_pfu['Fe2'] - O_prop['Fe3'], 0)

    return cation_pfu 

def spinels_nocr(data_oxides): 
    # Set up  mass and charge data
    molar_mass = {'SiO2':60.08,'TiO2':79.866,'Al2O3':101.96,'FeO':71.844,'Fe2O3':159.69,
        'MnO':70.9374,'MgO':40.3044,'CaO':56.0774,'Na2O':61.98,'K2O':94.2,'NiO':74.5928}
    oxygen_numbers = {'SiO2':2,'TiO2':2,'Al2O3':3,'FeO':1,'Fe2O3':3,'MnO':1,'MgO':1,
                      'Na2O':1,'K2O':1,'CaO':1,'NiO':1} # number of Os per mole of oxide
    cation_ratio = {'SiO2':0.5,'TiO2':0.5,'Al2O3':2/3,'FeO':1,'Fe2O3':2/3,'MnO':1,
                    'MgO':1,'Na2O':2,'K2O':2,'CaO':1,'NiO':1} # ratio of cations to Os per mole of oxide
    cation = {'SiO2':'Si','TiO2':'Ti','Al2O3':'Al','FeO':'Fe2','Fe2O3':'Fe3',
              'MnO':'Mn','MgO':'Mg','CaO':'Ca','Na2O':'Na','K2O':'K','NiO':'Ni'}
    # =============================================================================
    # Calculate cations assuming all Fe2, for ferric iron recalculation
    # =============================================================================
    O_prop = pd.DataFrame()
    cation_allFe2 = pd.DataFrame()
    cation_pfu = pd.DataFrame()

    # Calculate atomic proportion of O from each molecule
    for oxide in data_oxides.columns.to_list():
        O_prop[oxide+'_O_prop'] = data_oxides[oxide]/molar_mass[oxide]*oxygen_numbers[oxide]

    O_prop['O_sum'] = O_prop.sum(axis = 1)
    
    # What is the scaling factor for the mineral, based on desired numbers of O (4 for spinel)
    # and desired number of cations (3 for spinel)
    X = 4 
    T = 3

    for oxide in data_oxides.columns.to_list():
        # Calculate O and multiply by cation ratio to get to cations. Sum to get 'S' a la Droop
        cation_allFe2[cation[oxide]+'_pfu'] = O_prop[oxide+'_O_prop']/O_prop['O_sum']*X*cation_ratio[oxide]
    # =============================================================================
    # Use Droop equation for calculating stoichiometric Fe3: F=2X(1-T/S) 
    # T = 3 = O_prop['min_cat_sum']
    # X = 4 = O_prop['min_O_sum'] 
    # S = observed cation total, cation_allFe2.sum(axis=1)
    # =============================================================================
    mask = cation_allFe2.sum(axis=1) > T
    O_prop['Fe3'] = np.where(mask, (2*T)*(1-T/cation_allFe2.sum(axis=1)), 0)
    cation_pfu['Fe3'] = O_prop['Fe3']

    # Normalise cations to total expected cation number ('Fe2' is really total Fe)
    other_cations = ['Si','Ti','Al','Fe2','Mn','Mg','Na','K','Ca','Ni']
    for cation in other_cations: 
        cation_pfu[cation] = cation_allFe2[cation+'_pfu']/cation_allFe2.sum(axis=1)*T
    
    # Now replace the Fe2 which is currently actually the total Fe
    cation_pfu['Fe2'] = cation_pfu['Fe2'] - O_prop['Fe3'] # np.where(cation_pfu['Fe2'] > 0, cation_pfu['Fe2'] - O_prop['Fe3'], 0)

    return cation_pfu 

def ilmenites(data_oxides): 
    
    # Set up  mass and charge data
    mr = {'SiO2':60.08,'TiO2':79.88,'Al2O3':101.96,'FeO':71.85,'Fe2O3':159.69,
          'MnO':70.94,'MgO':40.3,'CaO':56.08,'Na2O':61.98,'K2O':94.2,'Cr2O3':151.99,'NiO':74.5928}
    oxygen_numbers = {'SiO2':2,'TiO2':2,'Al2O3':3,'Fe2O3':3,'FeO':1,'MnO':1,'MgO':1,
                      'CaO':1,'Na2O':1,'K2O':1,'Cr2O3':3,'NiO':1} # number of Os per mole of oxide
    cation_ratio = {'SiO2':0.5,'TiO2':0.5,'Al2O3':2/3,'Fe2O3':2/3,'FeO':1,'MnO':1,'MgO':1,
                    'Na2O':2,'K2O':2,'CaO':1,'Cr2O3':2/3,'NiO':1} # ratio of cations to Os per mole of oxide
    cation = {'SiO2':'Si','TiO2':'Ti','Al2O3':'Al','Fe2O3':'Fe3','FeO':'Fe2', 
            'MnO':'Mn','MgO':'Mg','CaO':'Ca','Na2O':'Na','K2O':'K','Cr2O3':'Cr','NiO':'Ni'}
    # =============================================================================
    # Calculate cations assuming all Fe2, for ferric iron recalculation
    # =============================================================================
    O_prop = pd.DataFrame()

    # Calculate atomic proportion of O from each molecule
    for oxide in data_oxides.columns.to_list():
        O_prop[oxide+'_O_prop'] = data_oxides[oxide]/mr[oxide]*oxygen_numbers[oxide]
            
    # What is the oxygen sum assuming all Fe2?    
    O_prop['O_sum'] = O_prop.sum(axis = 1)    

    # What is the scaling factor for the mineral, based on desired numbers of O (3 for ilmenite)
    O_prop['min_O_sum'] = 3

    cation_allFe2 = pd.DataFrame()
    for oxide in data_oxides.columns.to_list():
        cation_allFe2[cation[oxide]+'_pfu'] = O_prop[oxide+'_O_prop']*O_prop['min_O_sum']/O_prop['O_sum']*cation_ratio[oxide]

    # =============================================================================
    # Use Droop equation for calculating stoichiometric Fe3: F = 2X(1-T/S)
    # =============================================================================
    O_prop['min_cat_sum'] = 2 # 2 cations for ilmenite, rhombohedral 

    O_prop['Fe3'] = 2*O_prop['min_O_sum']*(1-O_prop['min_cat_sum']/cation_allFe2.sum(axis=1))

    cation_pfu = pd.DataFrame()
    cation_pfu['Fe3'] = O_prop['Fe3']

    # Normalise cations to total expected cation number ('Fe2' is really total Fe)
    other_cations = ['Si','Ti','Al','Fe2','Mn','Mg','Na','K','Ca','Cr','Ni']
    for cation in other_cations:
        cation_pfu[cation] = cation_allFe2[cation+'_pfu']*O_prop['min_cat_sum']/cation_allFe2.sum(axis=1)
    # Now replace the Fe2 which is currently actually the total Fe
    cation_pfu['Fe2'] = cation_pfu['Fe2']-O_prop['Fe3']

    return cation_pfu 

sp_df_calc = sp_df[['SiO2','TiO2','Al2O3','FeOt','MnO','MgO','CaO','Na2O','K2O','Cr2O3','NiO']]
sp_df_calc = sp_df_calc.rename(columns={'FeOt':'FeO'})

il_df_calc = il_df[['SiO2','TiO2','Al2O3','FeOt','MnO','MgO','CaO','Na2O','K2O','Cr2O3','NiO']]
il_df_calc = il_df_calc.rename(columns={'FeOt':'FeO'})

mt_df_calc = mt_df[['SiO2','TiO2','Al2O3','FeOt','MnO','MgO','CaO','Na2O','K2O','Cr2O3','NiO']]
mt_df_calc = mt_df_calc.rename(columns={'FeOt':'FeO'})

sp_cation_pfu = spinels(sp_df_calc) 
il_cation_pfu = ilmenites(il_df_calc)
mt_cation_pfu = spinels(mt_df_calc)
sp_cation_nocr_pfu = spinels_nocr(sp_df_calc[['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O','NiO']]) 

sp_cation_pfu['R3'] = sp_cation_pfu.Fe3 + sp_cation_pfu.Al + sp_cation_pfu.Cr
il_cation_pfu['R3'] = il_cation_pfu.Fe3 + il_cation_pfu.Al + il_cation_pfu.Cr
mt_cation_pfu['R3'] = mt_cation_pfu.Fe3 + mt_cation_pfu.Al + mt_cation_pfu.Cr
sp_cation_nocr_pfu['R3'] = sp_cation_nocr_pfu.Fe3 + sp_cation_nocr_pfu.Al

bool = (sp_cation_pfu.Fe3 / (sp_cation_pfu.Al+sp_cation_pfu.Fe3)<0.5) & (sp_cation_pfu.Fe2 / (sp_cation_pfu.Mg+sp_cation_pfu.Fe2)<0.5)
sp_cation_pfu_lim = sp_cation_pfu[bool]

x_sp = sp_cation_pfu_lim.Fe2 / (sp_cation_pfu_lim.Mg+sp_cation_pfu_lim.Fe2)
y_sp = sp_cation_pfu_lim.Fe3 / (sp_cation_pfu_lim.Al+sp_cation_pfu_lim.Fe3)
x1_sp = sp_cation_pfu_lim.Cr / (sp_cation_pfu_lim.Cr+sp_cation_pfu_lim.Al)
y1_sp = sp_cation_pfu_lim.Mg / (sp_cation_pfu_lim.Mg+sp_cation_pfu_lim.Fe2)
y2_sp = sp_cation_pfu_lim.Ti / (sp_cation_pfu_lim.Ti+sp_cation_pfu_lim.Cr)
y3_sp = sp_cation_pfu_lim.Al / (sp_cation_pfu_lim.Al+sp_cation_pfu_lim.Cr)
y4_sp = sp_cation_pfu_lim.Al / (sp_cation_pfu_lim.Al+sp_cation_pfu_lim.Ti)

bool_new = (sp_cation_nocr_pfu.Fe3 / (sp_cation_nocr_pfu.Al+sp_cation_nocr_pfu.Fe3)<0.5) & (sp_cation_nocr_pfu.Fe2 / (sp_cation_nocr_pfu.Mg+sp_cation_nocr_pfu.Fe2)<0.5)
sp_cation_nocr_pfu_lim = sp_cation_nocr_pfu[bool_new]

x_nocr_sp = sp_cation_nocr_pfu_lim.Fe2 / (sp_cation_nocr_pfu_lim.Mg+sp_cation_nocr_pfu_lim.Fe2)
y_nocr_sp = sp_cation_nocr_pfu_lim.Fe3 / (sp_cation_nocr_pfu_lim.Al+sp_cation_nocr_pfu_lim.Fe3)

x_mt = mt_cation_pfu.Fe2 / (mt_cation_pfu.Mg+mt_cation_pfu.Fe2)
y_mt = mt_cation_pfu.Fe3 / (mt_cation_pfu.Al+mt_cation_pfu.Fe3)
x1_mt = mt_cation_pfu.Cr / (mt_cation_pfu.Cr+mt_cation_pfu.Al)
y1_mt = mt_cation_pfu.Mg / (mt_cation_pfu.Mg+mt_cation_pfu.Fe2)
y2_mt = mt_cation_pfu.Ti / (mt_cation_pfu.Ti+mt_cation_pfu.Cr)
y3_mt = mt_cation_pfu.Al / (mt_cation_pfu.Al+mt_cation_pfu.Cr)
y4_mt = mt_cation_pfu.Al / (mt_cation_pfu.Al+mt_cation_pfu.Ti)


plt.figure(figsize=(8, 6))
plt.scatter(x_sp, y_sp)
plt.scatter(x_nocr_sp, y_nocr_sp)
plt.scatter(x_mt, y_mt)

plt.figure(figsize=(8, 6))
plt.scatter(x1_sp, y1_sp)
plt.scatter(x1_mt, y1_mt)

plt.figure(figsize=(8, 6))
plt.scatter(y2_sp, y1_sp)
plt.scatter(y2_mt, y1_mt)

plt.figure(figsize=(8, 6))
plt.scatter(y3_sp, y1_sp)
plt.scatter(y3_mt, y1_mt)

plt.figure(figsize=(8, 6))
plt.scatter(y4_sp, y1_sp)
plt.scatter(y4_mt, y1_mt)



# %% 

sp_df_ej = pd.read_excel('Training_Data/Mineral/Spinel.xlsx').tail(62)
sp_df_ej_calc = sp_df_ej[['SiO2','TiO2','Al2O3','FeOT','MnO','MgO','CaO','Na2O','K2O','Cr2O3','NiO']]
sp_df_ej_calc = sp_df_ej_calc.rename(columns={'FeOT':'FeO'})

sp_ej = pd.read_excel('EJ_data.xlsx')

sp_cation_ej_pfu = spinels(sp_df_ej_calc)
sp_cation_ej_pfu


# %% 

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sp_cation_pfu.loc[:, ["Ti", "Fe2", "R3"]].pyroplot.scatter(c="green", ax=ax, label='spinel')
sp_cation_nocr_pfu_lim.loc[:, ["Ti", "Fe2", "R3"]].pyroplot.scatter(c="red", ax=ax, label='spinel')
il_cation_pfu.loc[:, ["Ti", "Fe2", "R3"]].pyroplot.scatter(c="orange", ax=ax, label='ilmenite')
mt_cation_pfu.loc[:, ["Ti", "Fe2", "R3"]].pyroplot.scatter(c="b", ax=ax, label='magnetite')
plt.legend()
plt.show()


# %% 


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sp_cation_pfu.loc[:, ["Cr", "Al", "Fe3"]].pyroplot.scatter(c="green", ax=ax, label='spinel')
il_cation_pfu.loc[:, ["Cr", "Al", "Fe3"]].pyroplot.scatter(c="orange", ax=ax, label='ilmenite')
mt_cation_pfu.loc[:, ["Cr", "Al", "Fe3"]].pyroplot.scatter(c="b", ax=ax, label='magnetite')
plt.legend()
plt.show()


# %%



fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sp_cation_pfu_lim.loc[:, ["Mg", "Fe2", "Fe3"]].pyroplot.scatter(c="green", ax=ax, label='spinel')
il_cation_pfu.loc[:, ["Mg", "Fe2", "Fe3"]].pyroplot.scatter(c="orange", ax=ax, label='ilmenite')
mt_cation_pfu.loc[:, ["Mg", "Fe2", "Fe3"]].pyroplot.scatter(c="b", ax=ax, label='magnetite')
plt.legend()
plt.show()
# %%

# %%



data_oxides = sp_df_ej_calc

molar_mass = {'SiO2':60.08,'TiO2':79.866,'Al2O3':101.96,'FeO':71.844,'Fe2O3':159.69,
    'MnO':70.9374,'MgO':40.3044,'CaO':56.0774,'Cr2O3':151.99,'NiO':74.5928}
cation_numbers = {'SiO2':1,'TiO2':1,'Al2O3':2,'FeO':1,'Fe2O3':2,'MnO':1,
                'MgO':1,'CaO':1,'Cr2O3':2,'NiO':1} # numbers of cations per mole of oxide
oxygen_numbers = {'SiO2':2,'TiO2':2,'Al2O3':3,'FeO':1,'Fe2O3':3,'MnO':1,'MgO':1,
                'CaO':1,'Cr2O3':3,'NiO':1} # number of Os per mole of oxide
cation_ratio = {'SiO2':0.5,'TiO2':0.5,'Al2O3':2/3,'FeO':1,'Fe2O3':2/3,'MnO':1,
                'MgO':1,'CaO':1,'Cr2O3':2/3,'NiO':1} # ratio of cations to Os per mole of oxide
cation = {'SiO2':'Si','TiO2':'Ti','Al2O3':'Al','FeO':'Fe2','Fe2O3':'Fe3',
            'MnO':'Mn','MgO':'Mg','CaO':'Ca','Cr2O3':'Cr','NiO':'Ni'}
# =============================================================================
# Calculate cations assuming all Fe2, for ferric iron recalculation
# =============================================================================
O_prop = pd.DataFrame()
cat_prop = pd.DataFrame()
cation_pfu = pd.DataFrame()
cation_allFe2 = pd.DataFrame()
# Calculate atomic proportion of O from each molecule
for oxide in data_oxides.columns.to_list():
    O_prop[oxide+'_O_prop'] = data_oxides[oxide]/molar_mass[oxide]*oxygen_numbers[oxide]
    cat_prop[oxide] = data_oxides[oxide]/molar_mass[oxide]*cation_numbers[oxide]

# What is the oxygen sum assuming all Fe2?    
O_prop['O_sum'] = O_prop.sum(axis = 1)
orf = 4 / O_prop['O_sum'] 
cat_prop_norm = cat_prop.mul(orf, axis=0)

cat_prop_norm['cat_sum'] = cat_prop_norm.sum(axis = 1)
cat_prop_norm = cat_prop_norm.fillna(0)
cat_prop_norm['sum_charge'] = (2 * (cat_prop_norm["MgO"] + cat_prop_norm["MnO"] + cat_prop_norm["CaO"] + cat_prop_norm["NiO"])
                            + 3 * (cat_prop_norm["Al2O3"] + cat_prop_norm["Cr2O3"])
                            + 4 * (cat_prop_norm["TiO2"] + cat_prop_norm["SiO2"]))

cat_prop_norm['fe3'] = 0 
cat_prop_norm.loc[(8 * cat_prop_norm["cat_sum"] / 3 - cat_prop_norm["sum_charge"] - 2 * cat_prop_norm["FeO"]) > 0, "fe3",] = (8 * cat_prop_norm["cat_sum"] / 3 - cat_prop_norm["sum_charge"] - 2 * cat_prop_norm["FeO"])
cat_prop_norm["fe2"] = cat_prop_norm["FeO"] - cat_prop_norm["fe3"]


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
cat_prop_norm.loc[:, ["TiO2", "fe2", "fe3"]].pyroplot.scatter(c="red", ax=ax, label='EJ')
sp_cation_ej_pfu.loc[:, ["Ti", "Fe2", "Fe3"]].pyroplot.scatter(c="black", ax=ax, label='EJ')

# %%

