# %%

import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from hdbscan.flat import (HDBSCAN_flat, approximate_predict_flat)

# %% 


class FeatureDataset(Dataset):

    """

    A PyTorch Dataset subclass for wrapping input features for easy batching and loading by a DataLoader. 
    It ensures the input feature tensor is in the correct two-dimensional shape (samples, features).

    Parameters:
        x (ndarray): The input feature data. Can be a 1D or 2D array which will be reshaped to 2D if necessary.

    The class provides methods to retrieve the length of the dataset and individual items by index, 
    which are required for PyTorch's DataLoader functionality.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(n): Retrieves the n-th sample from the dataset as a torch Tensor.

    """

    def __init__(self, x):
        if len(x.shape)==2:
            self.x = x
        else:
            self.x = x.reshape(-1, x.shape[-1]) #dataset keeps the right shape for training

    def __len__(self):
        return self.x.shape[0] 
    
    def __getitem__(self, n): 
        return torch.Tensor(self.x[n])


class LabelDataset(Dataset):

    """

    A PyTorch Dataset subclass designed to contain features and labels for supervised learning. It verifies 
    and maintains the input features as a float tensor and labels as a long tensor in a shape that's 
    compatible with model training requirements. If the input data are not already in a 2D shape, 
    the data are reshaped to ensure compatibility with PyTorch's batch processing.

    Parameters:
        x (ndarray): The array of input features, expected to be a 2D array (samples by features).
        labels (ndarray): The array of labels corresponding to the input data, expected to be a 1D array.

    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(n): Retrieves the nth sample and its corresponding label as a tuple of tensors.

    """

    def __init__(self, x, labels):
        if len(x.shape)==2:
            self.x = torch.from_numpy(x).type(torch.FloatTensor)
            self.labels = torch.from_numpy(labels.copy()).type(torch.LongTensor)
        else:
            self.x = x.reshape(-1, x.shape[-1]) #dataset keeps the right shape for training
            self.labels = labels

    def __len__(self):
        return len(self.x) 
    
    def __getitem__(self, n): 
        return self.x[n], self.labels[n]
    

def load_df(filepath):

    """

    Loads a DataFrame from a CSV file specified by the given file path. The first column 
    of the CSV is set as the index of the DataFrame.

    Parameters:
        filepath (str): The path to the CSV file to be loaded.

    Returns:
        df (DataFrame): Pandas DataFrame containing the data from the CSV file.

    """

    df = pd.read_csv(filepath, index_col=0)

    return df


def load_scaler(scaler_path):
    
    """

    Loads a pre-fitted scaler's mean and std from a .npz file. This scaler is a StandardScaler
    for normalizing or standardizing input data before passing it to a machine learning model. 

    Returns:
        mean, std (pandas Series): The mean and std from the scaler object 'scaler_ae/nn.npz'.

    Raises:
        FileNotFoundError: If 'scaler_ae/nn.npz' is not found in the expected directory.
        Exception: Propagates any exception raised during the scaler loading process.

    """

    # Define the path to the scaler relative to this file's location.
    current_dir = os.path.dirname(__file__)
    scaler_path = os.path.join(current_dir, scaler_path)  # Note the .joblib extension

    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3', 'P2O5']

    # Attempt to load the scaler and handle exceptions if the loading fails.
    try:
        # Load the mean and standard deviation using numpy's load function
        npz = np.load(scaler_path)
        mean = pd.Series(npz['mean'], index=oxides)
        std = pd.Series(npz['scale'], index=oxides)

    except FileNotFoundError:
        raise FileNotFoundError(f"The scaler file was not found at {scaler_path}.")
    except Exception as e:
        raise e  # Propagate other exceptions up

    return mean, std


def weights_init(m):

    """

    Applies an initialization scheme to the weights and biases of a Batch Normalization layer 
    in a neural network. If the module 'm' is of the class 'BatchNorm', it initializes the layer's 
    weights with a normal distribution centered around 1.0 with a standard deviation of 0.02, and 
    sets the biases to 0.

    Parameters:
        m (nn.Module): The module to initialize.

    This function is typically used as an argument to `apply` method of `nn.Module` when 
    initializing the weights of a neural network.

    """

    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def same_seeds(seed):

    """

    Sets the seed for generating random numbers to the provided value for various libraries including 
    PyTorch, NumPy, and Python's random module to ensure reproducibility across multiple runs. It also 
    sets the CuDNN backend to operate in a deterministic mode. This function is helpful for debugging 
    and to ensure that experimental runs are repeatable with the same sequence of random numbers being 
    generated each time. It is particularly useful when working with stochastic processes in machine 
    learning experiments where reproducibility is crucial.

    Parameters:
        seed (int): The seed value to use for all random number generators.

    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    


def save_model_ae(model, optimizer, path):

    """

    Saves the current state of a model and its optimizer to a file at the specified path. 
    The state is saved as a dictionary with keys 'params' and 'optimizer', which store 
    the state dictionaries of the model and the optimizer, respectively.

    Parameters:
        model (nn.Module): The model whose parameters are to be saved.
        optimizer (Optimizer): The optimizer whose state is to be saved.
        path (str): The filepath where the checkpoint will be saved.
    
    """

    check_point = {'params': model.state_dict(),                            
                   'optimizer': optimizer.state_dict()}
    torch.save(check_point, path)


def save_model_nn(optimizer, best_model_state, path):

    """

    Saves the state dictionary of a neural network's best model along with the state of its optimizer to a file. 
    The checkpoint is saved as a dictionary with 'params' holding the model state and 'optimizer' holding the 
    optimizer state. The saved file can be used to load the model and continue training or for evaluation 
    without the need to retrain the model from scratch.

    Parameters:
        optimizer (Optimizer): The optimizer associated with the best model.
        best_model_state (dict): The state dictionary of the best performing model.
        path (str): The path to the file where the checkpoint will be saved.

    """

    check_point = {'params': best_model_state,                            
                   'optimizer': optimizer.state_dict()}
    torch.save(check_point, path)


def load_model(model, optimizer=None, path=''):

    """

    Loads a model's state and optionally an optimizer's state from a saved checkpoint file. 
    The function updates the model's parameters with those found in the checkpoint and, if an 
    optimizer is provided, also updates the optimizer's state.

    Parameters:
        model (nn.Module): The PyTorch model to which the saved state will be loaded.
        optimizer (torch.optim.Optimizer, optional): The optimizer for which the state is to be 
                                                     loaded. If None, only the model state is loaded.
                                                     Defaults to None.
        path (str): The path to the file containing the saved checkpoint. The checkpoint file 
                    should have a dictionary containing 'params' and 'optimizer' keys.

    It is assumed that the checkpoint file at the specified 'path' is accessible and contains 
    a valid state dictionary for the model and, optionally, the optimizer.

    """

    check_point = torch.load(path, weights_only=True)
    model.load_state_dict(check_point['params'])
    if optimizer is not None:
        optimizer.load_state_dict(check_point['optimizer'])


def mineral_supergroup(df): 

    """

    Assigns minerals to their supergroup: pyroxene, feldspar, oxide. 

    Parameters:
        df (DataFrame): Dataframe with the classification of 'Predict_Mineral'. 

    Returns:
        df (DataFrame): Dataframe with the new classification of 'Supergroup'.

    """

    df['Supergroup'] = df['Predict_Mineral']

    pyroxene_condition = df['Predict_Mineral'].isin(['Orthopyroxene', 'Clinopyroxene'])
    feldspar_condition = df['Predict_Mineral'].isin(['KFeldspar', 'Plagioclase'])
    oxide_condition = df['Predict_Mineral'].isin(['Spinel', 'Ilmenite', 'Magnetite'])

    df.loc[pyroxene_condition, 'Supergroup'] = 'Pyroxene'
    df.loc[feldspar_condition, 'Supergroup'] = 'Feldspar'
    df.loc[oxide_condition, 'Supergroup'] = 'Oxide'

    return df


