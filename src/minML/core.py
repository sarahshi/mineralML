# %%

import os
import math
import copy
import random
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from joblib import load

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# %% 

def load_scaler():
    
    """

    Loads a pre-fitted scaler from a .joblib file. This scaler is a StandardScaler
    for normalizing or standardizing input data before passing it to a machine learning model. 

    Returns:
        scaler (sklearn.preprocessing): The scaler object loaded from 'scaler.joblib'.

    Raises:
        FileNotFoundError: If 'scaler.joblib' is not found in the expected directory.
        Exception: Propagates any exception raised during the scaler loading process.

    """

    # Define the path to the scaler relative to this file's location.
    current_dir = os.path.dirname(__file__)
    scaler_path = os.path.join(current_dir, 'scaler.joblib')  # Note the .joblib extension

    # Attempt to load the scaler and handle exceptions if the loading fails.
    try:
        # Load the scaler using joblib's load function
        scaler = load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The scaler file was not found at {scaler_path}.")
    except Exception as e:
        raise e  # Propagate other exceptions up

    return scaler

def load_minclass():

    """

    Loads mineral classes and their corresponding mappings from a .npz file. 
    The file is expected to contain an array of class names under the 'classes' key. 
    This function creates a dictionary that maps an integer code to each class name.

    Returns:
        min_cat (list): A list of mineral class names.
        mapping (dict): A dictionary that maps each integer code to its corresponding 
                        class name in the 'min_cat' list.

    """

    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, 'mineral_classes.npz')

    with np.load(filepath, allow_pickle=True) as data:
        min_cat = data['classes'].tolist()
    mapping = {code: cat for code, cat in enumerate(min_cat)}

    return min_cat, mapping

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

def prep_df(df):

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
        raise ValueError("No 'FeOt' column found. You have a 'FeO' column. minML only recognizes 'FeOt' as a column. Please convert to FeOt.")
    if "Fe2O3" in df.columns and "FeOt" not in df.columns:
        raise ValueError("No 'FeOt' column found. You have a 'Fe2O3' column. minML only recognizes 'FeOt' as a column. Please convert to FeOt.")

    oxidesandmin = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3', 'Mineral']
    include_minerals = ['Amphibole', 'Biotite', 'Clinopyroxene', 'Garnet', 'Ilmenite', 
                        'KFeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene', 
                        'Plagioclase', 'Spinel']
    exclude_minerals = ['Tourmaline', 'Quartz', 'Rutile', 'Apatite', 'Zircon']
    df.dropna(subset=oxidesandmin, thresh=6, inplace=True)
    df_in = df[df['Mineral'].isin(include_minerals)]
    df_ex = df[df['Mineral'].isin(exclude_minerals)]

    df_in = df_in[oxidesandmin].fillna(0)
    df_ex = df_ex[oxidesandmin].fillna(0)

    return df_in, df_ex

def norm_data(df):

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
    scaler = load_scaler()

    if df[oxides].isnull().any().any():
        df, _ = prep_df(df)
    else: 
        df = df 

    array_x = scaler.transform(df[oxides])

    return array_x

def create_dataloader(df, batch_size=128, shuffle=False):

    """

    Creates a DataLoader for the given DataFrame. It normalizes the input features and converts 
    the 'Mineral' column to categorical codes based on predefined mineral classes. The resulting 
    DataLoader can be used to iterate over the dataset in batches during model training or evaluation.
    The function relies on the 'load_minclass' function to obtain the list of category names 
    for the 'Mineral' column and the 'norm_data' function to normalize the feature columns 
    before creating the DataLoader.

    Parameters:
        df (DataFrame): The DataFrame containing features and mineral labels to load into the DataLoader.
        batch_size (int): The number of samples to load per batch. Defaults to 128.
        shuffle (bool): Whether to shuffle the data before loading it. Defaults to False.

    Returns:
        dataloader (DataLoader): A PyTorch DataLoader object ready for model consumption.
    
    """

    min_cat, _ = load_minclass()
    data_x = norm_data(df)
    df['Mineral'] = pd.Categorical(df['Mineral'], categories=min_cat)
    data_y = df['Mineral'].cat.codes.values

    label_dataset = LabelDataset(data_x, data_y)
    dataloader = DataLoader(label_dataset, batch_size, shuffle)

    return dataloader

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

def save_model(model, optimizer, path):

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

    check_point = torch.load(path)
    model.load_state_dict(check_point['params'])
    if optimizer is not None:
        optimizer.load_state_dict(check_point['optimizer'])

def getLatent(model, dataset:np):

    """

    Processes a given dataset through the encoder part of a trained model to obtain and return 
    the latent space representations. The dataset is wrapped in a FeatureDataset and passed 
    through the model in evaluation mode to generate latent vectors without computing gradients.

    Parameters:
        model (nn.Module): The trained model with an 'encoded' method that projects input to latent space.
        dataset (numpy.ndarray): The input data to be transformed into latent space representations.

    Returns:
        numpy.ndarray: An array of the latent space representations for the input dataset.

    The function uses a DataLoader to batch process the dataset for efficiency and assumes the 
    model has been moved to the appropriate device (CPU or GPU) before calling this function.

    """

    #transform real data to latent space using the trained model
    latents=[]
    model.to(device)

    dataset_ = FeatureDataset(dataset)
    loader = DataLoader(dataset_,batch_size=20,shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.to(device)
            z = model.encoded(x)
            latents.append(z.detach().cpu().numpy())
    
    return np.concatenate(latents, axis=0)

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

def balance(train_x, train_y, n=1000):

    """

    Balances the training dataset by oversampling the minority class using the RandomOverSampler method. 
    It aims to equalize the number of samples for each class in the dataset to prevent the model from 
    being biased towards the majority class.

    Parameters:
        train_x (numpy.ndarray): The feature matrix for the training data.
        train_y (numpy.ndarray): The corresponding label vector for the training data.

    Returns:
        train_x (numpy.ndarray): The feature matrix after oversampling the minority class.
        train_y (numpy.ndarray): The label vector after oversampling the minority class.

    The function creates a new balanced DataFrame with an equal number of samples for each class. 
    Classes are oversampled to reach a count of 1000 samples per class, with the random state set 
    for reproducibility. The function returns the resampled feature matrix and label vector suitable 
    for training a machine learning model.

    """


    oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)

    # Resample the dataset
    x_balanced, y_balanced = oversample.fit_resample(train_x, train_y)

    df_resampled = pd.DataFrame(x_balanced)
    df_resampled['Mineral'] = y_balanced

    df_balanced = pd.DataFrame()
    for class_label in df_resampled['Mineral'].unique():
        df_class = df_resampled[df_resampled['Mineral'] == class_label]
        df_balanced = pd.concat([df_balanced, df_class.sample(n=n, replace = True, random_state=42)])

    # Reset the index of the balanced dataframe
    df_balanced = df_balanced.reset_index(drop=True)
    train_x = df_balanced.iloc[:, :-1].to_numpy()
    train_y = df_balanced.iloc[:, -1].to_numpy()

    return train_x, train_y
