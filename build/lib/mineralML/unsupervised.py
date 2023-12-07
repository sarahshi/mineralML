# %%

import os
import math
import time
import copy

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.utils.data import DataLoader

from hdbscan.flat import (HDBSCAN_flat, approximate_predict_flat)

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

from mineralML.core import *

# %% 

def prep_df_ae(df):

    """

    Prepares a DataFrame for analysis by performing data cleaning specific to mineralogical data. 
    It filters the DataFrame for selected minerals, handles missing values, and separates the data 
    into two DataFrames: one that includes specified minerals and another that excludes them.
    The function defines a list of oxide column names and minerals to include and exclude. It drops 
    rows where the specified oxides and 'Mineral' column have fewer than six non-NaN values. 

    Parameters:
        df (DataFrame): The input DataFrame containing mineral composition data along with 'Mineral' column.

    Returns:
        df_in (DataFrame): A DataFrame with rows including only the specified minerals and 'NaN' filled with zero.
        df_ex (DataFrame): A DataFrame with rows excluding the specified minerals and 'NaN' filled with zero.
    
    """

    if "FeO" in df.columns and "FeOt" not in df.columns:
        raise ValueError("No 'FeOt' column found. You have a 'FeO' column. mineralML only recognizes 'FeOt' as a column. Please convert to FeOt.")
    if "Fe2O3" in df.columns and "FeOt" not in df.columns:
        raise ValueError("No 'FeOt' column found. You have a 'Fe2O3' column. mineralML only recognizes 'FeOt' as a column. Please convert to FeOt.")

    oxidesandmin = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3', 'Mineral']
    include_minerals = ['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 'Ilmenite', 
                        'KFeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene', 
                        'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon']
    # df.dropna(subset=oxidesandmin, thresh=5, inplace=True)
    df_in = df[df['Mineral'].isin(include_minerals)]
    df_ex = df[~df['Mineral'].isin(include_minerals)]

    df_in = df_in[oxidesandmin].fillna(0)

    df_in = df_in.reset_index(drop=True)
    df_ex = df_ex.reset_index(drop=True)

    return df_in, df_ex

def norm_data_ae(df):

    """

    Normalizes the oxide composition data in the input DataFrame using a predefined StandardScaler. 
    It ensures that the dataframe has been preprocessed accordingly before applying the transformation. 
    The function expects that the scaler is already fitted and available for use as defined in the 
    'load_scaler' function.

    Parameters:
        df (DataFrame): The input DataFrame containing the oxide composition data.

    Returns:
        array_x (ndarray): An array of the transformed oxide composition data.

    """

    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
    mean, std = load_scaler()

    if df[oxides].isnull().any().any():
        df, _ = prep_df_ae(df)
    else: 
        df = df 

    scaled_df = df[oxides].copy()

    for col in df[oxides].columns: 
        scaled_df[col] = (df[col] - mean[col]) / std[col]

    array_x = scaled_df.to_numpy()
    
    return array_x

def feature_normalisation(feature, return_params = False, mean_norm = True):

    """
    Function to perform mean normalisation on the dataset passed to it.
    
    Parameters: 
        feature (numpy array) - features to be normalised
        return_params (boolean, optional) - set True if parameters used for mean normalisation
                                are to be returned for each feature
                            
    Returns: 
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



class Autoencoder(nn.Module):
    def __init__(self,input_dim = 10, latent_dim = 2, hidden_layer_sizes=(512, 256, 128)):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hls = hidden_layer_sizes

        def element(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.LeakyReLU(0.02),
            ]

        encoder = element(self.input_dim, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += element(self.hls[i], self.hls[i + 1])
        encoder += [nn.Linear(self.hls[-1], latent_dim)]

        decoder = element(latent_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += element(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.input_dim)] # nn.Softmax()]

        self.encode = nn.Sequential(*encoder)
        self.decode = nn.Sequential(*decoder)

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de

class Tanh_Autoencoder(nn.Module):
    def __init__(self,input_dim = 10, latent_dim = 2, hidden_layer_sizes=(256, 64, 16)):
        super(Tanh_Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hls = hidden_layer_sizes

        def element(in_channel, out_channel):
            return [
                nn.Linear(in_channel, out_channel),
                nn.LayerNorm(out_channel),
                nn.Tanh(),
            ]

        encoder = element(self.input_dim, self.hls[0])
        for i in range(len(self.hls) - 1):
            encoder += element(self.hls[i], self.hls[i + 1])
        encoder += [nn.Linear(self.hls[-1], latent_dim)]

        decoder = element(latent_dim, self.hls[-1])
        for i in range(len(self.hls) - 1, 0, -1):
            decoder += element(self.hls[i], self.hls[i - 1])
        decoder += [nn.Linear(self.hls[0], self.input_dim)] # nn.Softmax()]

        self.encode = nn.Sequential(*encoder)
        self.decode = nn.Sequential(*decoder)

        self.apply(weights_init)

    def encoded(self, x):
        #encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        #decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de

def train(model, optimizer, train_loader, test_loader, n_epoch, criterion, patience=10, min_delta=0.0005):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    avg_train_loss = []
    avg_test_loss = []

    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epoch):
        # Training
        model.train()
        t = time.time()
        train_loss = []
        for i, data in enumerate(train_loader):
            x = data.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().item())
        
        # Testing
        model.eval()
        test_loss = []
        for i, test in enumerate(test_loader):
            x = test.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)
            test_loss.append(loss.detach().item())
            
        # Logging
        avg_loss = sum(train_loss) / len(train_loss)
        avg_test = sum(test_loss) / len(test_loss)
        avg_train_loss.append(avg_loss)
        avg_test_loss.append(avg_test)

        training_time = time.time() - t
        
        print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_loss:.6f}, test_loss: {avg_test:.6f}, time: {training_time:.2f} s')

        # Early stopping
        test_loss_improvement = best_test_loss - avg_test
        if test_loss_improvement > min_delta:
            best_test_loss = avg_test
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Test loss hasn't improved significantly for {patience} epochs. Stopping early.")
                break

    return avg_train_loss, avg_test_loss


def autoencode(df, name, AE_Model, hidden_layer_sizes, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']

    wt = df[oxides].fillna(0)
    wt = wt.to_numpy()

    #perform z-score normalisation
    ss = StandardScaler()
    array_norm = ss.fit_transform(wt)

    # #split the dataset into train and test sets
    train_data, test_data = train_test_split(array_norm, test_size=0.1, stratify = df['Mineral'], random_state=42)

    #define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = FeatureDataset(train_data)
    test_dataset = FeatureDataset(test_data)   

    #autoencoder params:
    lr = 5e-4
    wd = 0
    batch_size = 256
    epochs = epochs
    input_size = feature_dataset.__getitem__(0).size(0)

    #define data loaders
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    np.savez('parametermatrix_autoencoder/' + name + '_features.npz', feature_loader = feature_loader, test_loader = test_loader)

    #define model
    model = AE_Model(input_dim=input_size, hidden_layer_sizes = hidden_layer_sizes).to(device)

    #use ADAM optimizer with mean squared error loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) 
    criterion = nn.MSELoss()

    #train model using pre-defined function
    train_loss, test_loss = train(model, optimizer, feature_loader, test_loader, epochs, criterion)
    np.savez('parametermatrix_autoencoder/' + name + '_tanh_loss.npz', train_loss = train_loss, test_loss = test_loss)
    
    fig, ax = plt.subplots(1, 2, figsize = (16, 8))
    ax = ax.flatten()
    ax[0].plot(np.linspace(1, len(train_loss), len(train_loss)), train_loss, '.-', label = 'Train Loss')
    ax[0].plot(np.linspace(1, len(train_loss), len(train_loss)), test_loss, '.-', label = 'Test Loss')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend(prop={'size': 10})

    #transform entire dataset to latent space
    z = getLatent(model, array_norm)

    phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 
        'Garnet', 'Ilmenite', 'KFeldspar', 'Magnetite', 'Muscovite', 'Olivine', 
        'Orthopyroxene', 'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline',
        'Zircon'])    
    phasez = range(1,len(phase))
    tab = plt.get_cmap('tab20')
    cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
    scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

    # plot latent representation
    for i in range(len(phase)):
        indx = df['Mineral'] == phase[i]
        ax[1].scatter(z[indx, 0], z[indx, 1], s=15, color=scalarMap.to_rgba(i), lw=1, label=phase[i], rasterized = True)
    ax[1].set_xlabel("Latent Variable 1")
    ax[1].set_ylabel("Latent Variable 2")
    ax[1].set_title(name + " Latent Space Representation")
    ax[1].legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig('parametermatrix_autoencoder/' + name + '_loss_latentspace.pdf',)

    # save main model params
    model_path = 'parametermatrix_autoencoder/' + name + "_tanh_params.pt"
    save_model(model, optimizer, model_path)

    # save all other params
    conc_file = name + "_tanh.npz"
    np.savez('parametermatrix_autoencoder/' + name + "_tanh.npz", batch_size = batch_size, epochs = epochs, lr = lr, wd = wd, input_size = input_size, 
            conc_file = conc_file, z = z)

    return z 

def get_latent_space(df): 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'ae_best_model.pt')
    model = Tanh_Autoencoder(input_dim=10, hidden_layer_sizes=(256, 64, 16)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
    load_model(model, optimizer, model_path)

    # Set the model to evaluation mode
    model.eval()

    norm_wt = norm_data_ae(df)
    z = getLatent(model, norm_wt)
    z_df = pd.DataFrame(z, columns=['LV1', 'LV2'])

    return z_df

def load_minclass_ae():

    """
    Loads the label dictionary for the autoencoder from a .npz file.
    The file is expected to contain a dictionary under the 'label_dict' key. 
    This function reads the dictionary and returns it.

    Returns:
        mapping (dict): A dictionary mapping integer codes to their corresponding
                           class names.
    """

    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, 'mineral_classes_ae.npz')

    with np.load(filepath, allow_pickle=True) as data:
        mapping = data['mapping'].item()  # .item() is used to get the dictionary from numpy array

    return mapping


def load_clusterer(): 

    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, 'ae_tanh.npz')
    z = np.load(filepath)['z']

    array_train, params_train = feature_normalisation(z, return_params = True)
    clusterer = HDBSCAN_flat(array_train, min_cluster_size=30, cluster_selection_epsilon=0.025, prediction_data=True)
    labels, probs = clusterer.labels_, clusterer.probabilities_
    minerals = class2mineral_ae(labels)

    z_df = pd.DataFrame(z, columns=['LV1', 'LV2'])
    z_df['Label'] = labels
    z_df['Predict_Mineral'] = minerals
    z_df['Predict_Probability'] = probs

    return clusterer, z_df


def predict_class_prob_ae(df): 

    """

    Predicts the class probabilities, corresponding mineral names, and the maximum 
    probability for each class using a predefined Tanh_Autoencoder model. This 
    function loads a pre-trained model and its optimizer state, normalizes input 
    data, and performs multiple inference iterations to compute the prediction probabilities.

    Parameters:
        df (DataFrame): The input DataFrame containing the oxide composition data.

    Returns:
        df (DataFrame): The input DataFrame with columns predict_mineral (predicted mineral names) 
        and predict_prob (clustering probability of predicted class). 

    """

    clusterer, z_df = load_clusterer()

    df_pred = df.copy()

    z_df = get_latent_space(df)
    df_pred['LV1'] = z_df['LV1']
    df_pred['LV2'] = z_df['LV2']

    array_valid, params_valid = feature_normalisation(z_df.to_numpy(), return_params = True)
    labels_valid, probs_valid = approximate_predict_flat(clusterer, array_valid, cluster_selection_epsilon=0.025)
    predict_mineral = class2mineral_ae(labels_valid)
    df_pred['Predict_Code'] = labels_valid
    df_pred['Predict_Mineral'] = predict_mineral
    df_pred['Predict_Probability'] = probs_valid

    return df_pred

def unique_mapping_ae(pred_class): 

    """

    Generates a mapping of unique class codes from given and predicted class labels, 
    considering only the classes present in both input arrays. It loads a predefined 
    category list and mapping, encodes the 'given_class' labels into categorical codes, 
    and creates a subset mapping for the unique classes found. It also handles unknown 
    classes by assigning them a code of -1 and mapping the 'Unknown' label to them.

    Parameters:
        given_class (array-like): The array of actual class labels.
        pred_class (array-like): The array of predicted class labels.

    Returns:
        unique (ndarray): Array of unique class codes found in both given and predicted classes.
        valid_mapping (dict): Dictionary mapping class codes to their corresponding labels, 
                              including 'Unknown' for any class code of -1.

    """

    mapping = load_minclass_ae()
    unique = np.unique(pred_class)
    valid_mapping = {key: mapping[key] for key in unique}
    # if -1 in unique:
    #     valid_mapping[-1] = "Outlier" 

    return unique, valid_mapping


def class2mineral_ae(pred_class): 

    """

    Translates predicted class codes into mineral names using a mapping obtained from the
    unique classes present in the 'pred_class' array. It utilizes the 'unique_mapping_ae' 
    function to establish the relevant class-to-mineral name mapping.

    Parameters:
        pred_class (array-like): The array of predicted class codes to be translated into mineral names.

    Returns:
        pred_mineral (ndarray): An array of mineral names corresponding to the predicted class codes.
        
    """

    _, valid_mapping = unique_mapping_ae(pred_class)
    pred_mineral = np.array([valid_mapping[x] for x in pred_class])

    return pred_mineral


def plot_latent_space(df_pred):

    clusterer, z_df = load_clusterer()

    phase = np.array(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 
        'Ilmenite', 'KFeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene', 
        'Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])    
    cNorm  = mcolors.Normalize(vmin=0, vmax=len(phase))
    scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('tab20'))

    fig, ax = plt.subplots(1, 2, figsize = (16, 8))
    ax = ax.flatten()
    for i in range(len(phase)):
        indx_z = z_df['Predict_Mineral'] == phase[i]
        if not z_df[indx_z].empty:
            ax[0].scatter(z_df.LV1[indx_z], z_df.LV2[indx_z], marker='o', s=15, color=scalarMap.to_rgba(i), lw=0.1, ec='k', alpha=z_df.Predict_Probability[indx_z], label=phase[i], rasterized = True)
        indx_pred = df_pred['Predict_Mineral'] == phase[i]
        if not df_pred[indx_pred].empty:
            ax[1].scatter(df_pred.LV1[indx_pred], df_pred.LV2[indx_pred], marker='o', s=15, color=scalarMap.to_rgba(i), lw=0.1, ec='k', alpha=df_pred.Predict_Probability[indx_pred], label=phase[i], rasterized = True)
    leg = ax[0].legend(prop={'size': 8}, loc = 'upper right', labelspacing = 0.4, handletextpad = 0.8, handlelength = 1.0, frameon=False)
    for handle in leg.legendHandles: 
        colors = handle.get_facecolor()
        # Set alpha value for face color
        colors[:, 3] = 1  # Set the alpha value of the RGBA color to 1
        handle.set_facecolor(colors)

        # If the handles also have edge colors, set their alpha values as well
        edge_colors = handle.get_edgecolor()
        edge_colors[:, 3] = 1
        handle.set_edgecolor(edge_colors)
    ax[0].set_xlabel('Latent Variable 1')
    ax[0].set_ylabel('Latent Variable 2')
    ax[0].set_xlim([-1.5, 2.0])
    ax[0].set_ylim([-2.5, 2.5])
    ax[0].annotate("Training Latent Space", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
    ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
    ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
    ax[1].set_xlabel('Latent Variable 1')
    ax[1].set_xlabel('Latent Variable 2')
    ax[1].set_xlim([-1.5, 2.0])
    ax[1].set_ylim([-2.5, 2.5])
    ax[1].annotate("Validation Latent Space", xy=(0.03, 0.94), xycoords="axes fraction", fontsize=20, weight='medium')
    ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5) 
    ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
    plt.tight_layout()

