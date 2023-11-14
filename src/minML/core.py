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

    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, 'mineral_classes.npz')

    with np.load(filepath, allow_pickle=True) as data:
        min_cat = data['classes'].tolist()
    mapping = {code: cat for code, cat in enumerate(min_cat)}

    return min_cat, mapping

def load_df(filepath):

    df = pd.read_csv(filepath, index_col=0)

    return df

def prep_df(df):

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

    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
    scaler = load_scaler()

    if df[oxides].isnull().any().any():
        df, _ = prep_df(df)
    else: 
        df = df 

    array_x = scaler.transform(df[oxides])

    return array_x

def create_dataloader(df, batch_size=128, shuffle=False):
    
    min_cat, _ = load_minclass()
    data_x = norm_data(df)
    df['Mineral'] = pd.Categorical(df['Mineral'], categories=min_cat)
    data_y = df['Mineral'].cat.codes.values

    label_dataset = LabelDataset(data_x, data_y)
    dataloader = DataLoader(label_dataset, batch_size, shuffle)

    return dataloader

class FeatureDataset(Dataset):

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
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_model(model, optimizer, path):
    check_point = {'params': model.state_dict(),                            
                   'optimizer': optimizer.state_dict()}
    torch.save(check_point, path)

def save_model_nn(optimizer, best_model_state, path):
    check_point = {'params': best_model_state,                            
                   'optimizer': optimizer.state_dict()}
    torch.save(check_point, path)

def load_model(model, optimizer=None, path=''):
    check_point = torch.load(path)
    model.load_state_dict(check_point['params'])
    if optimizer is not None:
        optimizer.load_state_dict(check_point['optimizer'])

def getLatent(model, dataset:np):
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True    

def balance(train_x, train_y):

    oversample = RandomOverSampler(sampling_strategy='minority', random_state=42)

    # Resample the dataset
    x_balanced, y_balanced = oversample.fit_resample(train_x, train_y)

    df_resampled = pd.DataFrame(x_balanced)
    df_resampled['Mineral'] = y_balanced

    df_balanced = pd.DataFrame()
    for class_label in df_resampled['Mineral'].unique():
        df_class = df_resampled[df_resampled['Mineral'] == class_label]
        df_balanced = pd.concat([df_balanced, df_class.sample(n=1000, replace = True, random_state=42)])

    # Reset the index of the balanced dataframe
    df_balanced = df_balanced.reset_index(drop=True)
    train_x = df_balanced.iloc[:, :-1].to_numpy()
    train_y = df_balanced.iloc[:, -1].to_numpy()

    return train_x, train_y
