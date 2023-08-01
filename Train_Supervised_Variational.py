# %% 

""" Created on February 16, 2023 // @author: Sarah Shi """

import numpy as np
import pandas as pd
import scipy
import math 

import os
import sys
import copy
import time
import random
import warnings

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
import torch.nn.functional as F

from pyrolite.plot import pyroplot


sys.path.append('src')
import MIN_ML as mm

import concurrent.futures
from multiprocessing import freeze_support
import itertools 

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42


# %% 


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_model(model, optimizer, path):
    check_point = {'params': model.state_dict(),                            
                   'optimizer': optimizer.state_dict()}
    torch.save(check_point, path)

def save_model_nn(model, optimizer, path, best_model_state):
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

# %% 



def predict_with_uncertainty(model, input_data, n_iterations=100):
    model.eval()
    output_list = []
    for i in range(n_iterations):
        with torch.no_grad():
            output = model(input_data)
            output_list.append(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy())

    output_list = np.array(output_list)
    
    # Calculate mean and standard deviation
    prediction_mean = output_list.mean(axis=0)
    prediction_stddev = output_list.std(axis=0)
    return prediction_mean, prediction_stddev


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


class LabelDataset(Dataset):
    def __init__(self, x, labels):
        if len(x.shape)==2:
            self.x = torch.from_numpy(x).type(torch.FloatTensor)
            self.labels = torch.from_numpy(labels).type(torch.LongTensor)
        else:
            self.x = x.reshape(-1, x.shape[-1]) #dataset keeps the right shape for training
            self.labels = labels

    def __len__(self):
        return len(self.x) 
    
    def __getitem__(self, n): 
        return self.x[n], self.labels[n]


class VariationalLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super(VariationalLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.softplus = nn.Softplus()
        self.reset_parameters()
        
    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.uniform_(-stdv, stdv)
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_rho.data.uniform_(-stdv, stdv)
        
    def forward(self, input):

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        weight_epsilon = torch.normal(mean=0., std=1., size=weight_sigma.size(), device=input.device)
        bias_epsilon = torch.normal(mean=0., std=1., size=bias_sigma.size(), device=input.device)
        
        weight_sample = self.weight_mu + weight_epsilon * weight_sigma
        bias_sample = self.bias_mu + bias_epsilon * bias_sigma
        
        output = F.linear(input, weight_sample, bias_sample)
        return output

    def kl_divergence(self):

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        kl_div = -0.5 * torch.sum(1 + torch.log(weight_sigma.pow(2)) - self.weight_mu.pow(2) - weight_sigma.pow(2))
        kl_div += -0.5 * torch.sum(1 + torch.log(bias_sigma.pow(2)) - self.bias_mu.pow(2) - bias_sigma.pow(2))

        return kl_div



class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim=10, classes=12, dropout_rate=0.1, hidden_layer_sizes=[8]):
        super(MultiClassClassifier, self).__init__()
        self.input_dim = input_dim
        self.classes = classes
        self.dropout_rate = dropout_rate
        self.hls = hidden_layer_sizes

        def element(in_channel, out_channel, is_last=False):
            if not is_last:
                layers = [
                    nn.Linear(in_channel, out_channel),
                    nn.BatchNorm1d(out_channel),  # Add batch normalization
                    nn.LeakyReLU(0.02),
                    nn.Dropout(self.dropout_rate),  # Add dropout
                ]
            else:
                layers = [VariationalLayer(in_channel, out_channel)]
            return layers

        encoder = []
        for i, size in enumerate(self.hls):
            if i == 0:
                encoder += element(self.input_dim, size, is_last=(i==len(self.hls)-1))
            else:
                encoder += element(self.hls[i-1], size, is_last=(i==len(self.hls)-1))

        encoder += [nn.Linear(size, self.classes)]  # Add this line

        self.encode = nn.Sequential(*encoder)
        self.apply(weights_init)

    def encoded(self, x):
        return self.encode(x)

    def forward(self, x):
        en = self.encoded(x)
        return en

    def predict(self, x):
        # Get predicted scores
        scores = self.forward(x)
        # Get predicted class indices
        class_indices = scores.argmax(dim=1)
        return class_indices


def train_nn(model, optimizer, label, train_loader, test_loader, n_epoch, criterion, patience=50, kl_weight_decay=0.2, kl_decay_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    avg_train_loss = []
    avg_test_loss = []
    best_test_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    kl_weight = 1.0  # Initial kl_weight

    for epoch in range(n_epoch):
        model.train()
        t = time.time()
        train_loss = []
        for i, (data, labels) in enumerate(train_loader):
            x = data.to(device)
            y = labels.to(device)
            train_output = model(x)
            loss = criterion(train_output, y)

            # Add KL divergence with weight decay
            kl_div = 0.
            kl_weight = min(kl_weight + (kl_weight_decay * (epoch // kl_decay_epochs)), 1)
            for module in model.modules():
                if isinstance(module, VariationalLayer):
                    kl_div += module.kl_divergence()
            loss += kl_weight * kl_div / len(train_loader.dataset)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().item())
        
        # Validation
        model.eval()
        test_loss = []
        with torch.no_grad():
            for i, (data, labels) in enumerate(test_loader):
                x = data.to(device)
                y = labels.to(device)
                test_output = model(x)
                loss = criterion(test_output, y)
                test_loss.append(loss.detach().item())

        # Logging
        avg_train = sum(train_loss) / len(train_loss)
        avg_test = sum(test_loss) / len(test_loss)
        avg_train_loss.append(avg_train)
        avg_test_loss.append(avg_test)
        
        training_time = time.time() - t
 
        print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_train:.6f}, test_loss: {avg_test:.6f}, time: {training_time:.2f} s')

        # Early stopping
        if avg_test < best_test_loss:
            best_test_loss = avg_test
            best_epoch = epoch
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())  # Save the best model weights

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Validation loss hasn't improved for {patience} epochs. Stopping early.")
                break

    return train_output, test_output, avg_train_loss, avg_test_loss, best_test_loss, best_model_state



def neuralnetwork(df, name, hls, lr, wd, dr, ep, n, balanced):
    path_beg = os.getcwd() + '/'
    output_dir = ["nn_parametermatrix", "autoencoder_parametermatrix"] 
    for ii in range(len(output_dir)):
        if not os.path.exists(path_beg + output_dir[ii]):
            os.makedirs(path_beg + output_dir[ii], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
    label = ['Mineral']

    min_df = df[label]
    wt = df[oxides].fillna(0).to_numpy()

    ss = StandardScaler()
    array_norm = ss.fit_transform(wt)

    code = pd.Categorical(df['Mineral']).codes
    cat_lab = pd.Categorical(df['Mineral'])

    # Split the dataset into train and test sets
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(array_norm, code, test_size=n, stratify=code, random_state=42)

    if balanced == True: 
        train_data_x, train_data_y = balance(train_data_x, train_data_y)

    # Define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = LabelDataset(train_data_x, train_data_y)
    test_dataset = LabelDataset(test_data_x, test_data_y)

    mapping = dict(zip(code, cat_lab))
    sort_mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))

    # Autoencoder params:
    lr = lr
    wd = wd 
    dr = dr
    epochs = ep
    batch_size = 256
    input_size = len(feature_dataset.__getitem__(0)[0])

    kl_weight_decay_values = [0.0, 0.25, 0.5, 0.75, 1.0]# [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_kl_weight_decay = None
    best_test_loss = float('inf')
    best_model_state = None

    # Define data loaders
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    np.savez('nn_parametermatrix/' + name + '_nn_features.npz', feature_loader=feature_loader, test_loader=test_loader)

    train_losses_dict = {}
    test_losses_dict = {}

    for kl_weight_decay in kl_weight_decay_values:
        print(f"Training with kl_weight_decay={kl_weight_decay}")

        # Initialize model
        model = MultiClassClassifier(input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=hls).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

        # Train model and get the best test loss and model state
        train_output, test_output, avg_train_loss, avg_test_loss, current_best_test_loss, current_best_model_state = train_nn(model, optimizer, label, feature_loader, test_loader, epochs, criterion, kl_weight_decay=kl_weight_decay)

        if avg_test_loss[-1] < best_test_loss:
            best_test_loss = avg_test_loss[-1]
            best_kl_weight_decay = kl_weight_decay
            best_model_state = current_best_model_state

        train_losses_dict[kl_weight_decay] = avg_train_loss
        test_losses_dict[kl_weight_decay] = avg_test_loss

    # Create a new model with the best model state
    best_model = MultiClassClassifier(input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=hls)
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    # Perform predictions on the test dataset using the best model
    with torch.no_grad():
        test_predictions = best_model(torch.Tensor(test_data_x))
        train_predictions = best_model(torch.Tensor(train_data_x))
        test_pred_classes = test_predictions.argmax(dim=1).cpu().numpy()
        train_pred_classes = train_predictions.argmax(dim=1).cpu().numpy()

    # Calculate classification metrics for the test dataset
    test_report = classification_report(test_data_y, test_pred_classes, target_names=list(sort_mapping.values()), zero_division=0, output_dict=True)
    train_report = classification_report(train_data_y, train_pred_classes, target_names=list(sort_mapping.values()), zero_division=0, output_dict=True) # output_dict=True

    # Print the best kl_weight_decay value and test report
    print("Best kl_weight_decay:", best_kl_weight_decay)
    print("Test report:")
    print(test_report)

    # Save the best model and other relevant information
    model_path = 'best_model.pt'
    save_model_nn(model, optimizer, model_path, best_model_state)
    np.savez('best_model_info.npz', best_kl_weight_decay=best_kl_weight_decay, test_report=test_report, train_report=train_report)

    train_pred_mean, train_pred_stddev = predict_with_uncertainty(model, feature_dataset.x)
    test_pred_mean, test_pred_stddev = predict_with_uncertainty(model, test_dataset.x)

    # Get the most probable classes
    train_pred_classes = np.argmax(train_pred_mean, axis=1)
    test_pred_classes = np.argmax(test_pred_mean, axis=1)

    # Save the train and test losses
    # np.savez('losses.npz', train_losses=train_losses_dict, test_losses=np.array(test_losses))

    return train_pred_classes, test_pred_classes, train_data_y, test_data_y, train_pred_mean, train_pred_stddev, test_pred_mean, test_pred_stddev, train_report, test_report, best_model_state, best_kl_weight_decay, train_losses_dict, test_losses_dict


# %% 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
same_seeds(42)

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']

min_df = pd.read_csv('Training_Data/mindf_filt.csv')
min_df_lim = min_df[~min_df['Mineral'].isin(['Tourmaline', 'Quartz', 'Rutile', 'Apatite', 'Zircon'])]

name = "test_variational_lim"

lr = 5e-3 
wd = 1e-3 
dr = 0.1 # 0.25
n = 0.20
hls = [64, 32]
epochs = 1500 

train_pred_classes, test_pred_classes, train_data_y, test_data_y, train_pred_mean, train_pred_stddev, test_pred_mean, test_pred_stddev, train_report, test_report, best_model_state, best_kl_weight_decay, train_losses_dict, test_losses_dict = neuralnetwork(min_df_lim, name, hls, lr, wd, dr, epochs, n, balanced = True, ) 

ss = StandardScaler()
array_norm = ss.fit_transform(min_df_lim[oxides])


 # %% 

cm_train = confusion_matrix(train_data_y, train_pred_classes)
cm_test = confusion_matrix(test_data_y, test_pred_classes)

mapping = dict(zip(pd.Categorical(min_df_lim['Mineral']).codes, pd.Categorical(min_df_lim['Mineral'])))
sort_dictionary= dict(sorted(mapping.items(), key=lambda item: item[0])) 

df_train_cm = pd.DataFrame(cm_train, index=sort_dictionary.values(), columns=sort_dictionary.values())
cmap = 'BuGn'
mm.pp_matrix(df_train_cm, cmap = cmap, savefig = 'none', figsize = (11.5, 11.5)) 
plt.show()

df_test_cm = pd.DataFrame(cm_test, index=sort_dictionary.values(), columns=sort_dictionary.values())
mm.pp_matrix(df_test_cm, cmap = cmap, savefig = 'none', figsize = (11.5, 11.5))
plt.show()




# %%
# %% 
# %% 
# %%

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
lepr = pd.read_csv('Validation_Data/lepr_allphases_lim_sp.csv', index_col=0)
lepr_df = lepr.dropna(subset=oxides, thresh = 5)

lepr_df = lepr_df[~lepr_df['Mineral'].isin(['Tourmaline', 'Quartz', 'Rutile', 'Apatite', 'Zircon'])]
lepr_wt = lepr_df[oxides].fillna(0).to_numpy()
lepr_norm_wt = ss.transform(lepr_wt)

min_df_lim['Mineral'] = min_df_lim['Mineral'].astype('category')
lepr_df['Mineral'] = lepr_df['Mineral'].astype(pd.CategoricalDtype(categories=min_df_lim['Mineral'].cat.categories))
new_validation_data_y_lepr = (lepr_df['Mineral'].cat.codes).values

# Create a DataLoader for the new validation dataset
new_validation_dataset_lepr = LabelDataset(lepr_norm_wt, new_validation_data_y_lepr)
new_validation_loader_lepr = DataLoader(new_validation_dataset_lepr, batch_size=256, shuffle=False)

input_size = len(new_validation_dataset_lepr.__getitem__(0)[0])

name = 'best_model'
path = name+'.pt'

model = MultiClassClassifier(input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=hls).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

load_model(model, optimizer, path)

# Use the trained model to predict the classes for the new validation dataset

model.eval()
new_validation_pred_classes_lepr = []
with torch.no_grad():
    for data, labels in new_validation_loader_lepr:
        x = data.to(device)
        pred_classes = model.predict(x)
        new_validation_pred_classes_lepr.extend(pred_classes.tolist())

new_validation_pred_classes_lepr = np.array(new_validation_pred_classes_lepr)
unique_classes_lepr = np.unique(np.concatenate((new_validation_data_y_lepr[new_validation_data_y_lepr != -1], new_validation_pred_classes_lepr[new_validation_data_y_lepr != -1])))


sort_mapping_lepr = {key: value for key, value in sorted(mapping.items(), key=lambda item: item[0]) if key in unique_classes_lepr}

# Calculate classification metrics for the new validation dataset
new_validation_report = classification_report(new_validation_data_y_lepr[new_validation_data_y_lepr!=-1], new_validation_pred_classes_lepr[new_validation_data_y_lepr!=-1], labels = unique_classes_lepr, target_names=[sort_mapping_lepr[x] for x in unique_classes_lepr], zero_division=0)
print("New validation report:\n", new_validation_report)

cm_valid = confusion_matrix(new_validation_data_y_lepr[new_validation_data_y_lepr!=-1], new_validation_pred_classes_lepr[new_validation_data_y_lepr!=-1])

df_valid_cm_lepr = pd.DataFrame(
    cm_valid,
    index=[sort_mapping_lepr[x] for x in unique_classes_lepr],
    columns=[sort_mapping_lepr[x] for x in unique_classes_lepr],
)

mm.pp_matrix(df_valid_cm_lepr, cmap = cmap, savefig = 'lepr_valid', figsize = (11.5, 11.5)) 

# %%
# %%

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']

georoc = pd.read_csv('Validation_Data/GEOROC_validationdata_Fe.csv', index_col=0)
georoc_df = georoc.dropna(subset=oxides, thresh = 6)

# georoc_df = georoc_df[georoc_df.Mineral.isin(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide', 'Ilmenite', '(Al)Kalifeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene','Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])]
georoc_df = georoc_df[georoc_df.Mineral.isin(['Amphibole', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide', 'Ilmenite', '(Al)Kalifeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene','Plagioclase', 'Spinel'])]

georoc_df['Mineral'] = georoc_df['Mineral'].replace('(Al)Kalifeldspar', 'KFeldspar')

data_idx = np.arange(len(georoc_df))
train_idx, test_idx = train_test_split(data_idx, test_size=0.2, stratify=pd.Categorical(georoc_df['Mineral']).codes, random_state=42, shuffle=True)
georoc_df_lim = georoc_df.iloc[test_idx]

georoc_wt = georoc_df_lim[oxides].fillna(0)
georoc_wt = georoc_wt.to_numpy()
georoc_norm_wt = ss.transform(georoc_wt)

min_df_lim['Mineral'] = min_df_lim['Mineral'].astype('category')
georoc_df_lim['Mineral'] = georoc_df_lim['Mineral'].astype(pd.CategoricalDtype(categories=min_df_lim['Mineral'].cat.categories))
new_validation_data_y_georoc = (georoc_df_lim['Mineral'].cat.codes).values


# min_df_lim['Mineral'] = min_df_lim['Mineral'].astype('category')
# lepr_df['Mineral'] = lepr_df['Mineral'].astype(pd.CategoricalDtype(categories=min_df_lim['Mineral'].cat.categories))
# new_validation_data_y_lepr = (lepr_df['Mineral'].cat.codes).values



# Create a DataLoader for the new validation dataset
new_validation_dataset_georoc = LabelDataset(georoc_norm_wt, new_validation_data_y_georoc)
new_validation_loader_georoc = DataLoader(new_validation_dataset_georoc, batch_size=256, shuffle=False)

input_size = len(new_validation_dataset_georoc.__getitem__(0)[0])

name = 'best_model'
path = name+'.pt'

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

mm.pp_matrix(df_valid_cm, cmap = cmap, savefig = 'georoc_valid', figsize = (11.5, 11.5)) 

# # Convert the predicted integer labels to string labels using the sort_mapping dictionary
new_validation_pred_labels_georoc = np.array([sort_mapping[x] for x in new_validation_pred_classes_georoc])
georoc_df_lim['NN_Labels'] = new_validation_pred_labels_georoc

georoc_df_lim.to_csv('GEOROC_CpxAmp_NN_Variational.csv')

# %%


true = georoc_df_lim[georoc_df_lim['Mineral'] == georoc_df_lim['NN_Labels']]
false = georoc_df_lim[georoc_df_lim['Mineral'] != georoc_df_lim['NN_Labels']]

# %% 

false_spinels = false[false['Mineral'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]
false_spinels = false_spinels[false_spinels['NN_Labels'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]


# %% 

import MIN_ML as mm


false_spinel = false_spinels[false_spinels.Mineral=='Spinel']
false_spinel_ox = false_spinel[['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOt', 'MnO', 'MgO', 'NiO', 'CaO', 'Na2O', 'K2O', 'P2O5']].add_suffix('_Sp')
false_magnetite = false_spinels[false_spinels.Mineral=='Magnetite']
false_magnetite_ox = false_magnetite[['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOt', 'MnO', 'MgO', 'NiO', 'CaO', 'Na2O', 'K2O', 'P2O5']].add_suffix('_Sp')
false_ilmenite = false_spinels[false_spinels.Mineral=='Ilmenite']
false_ilmenite_ox = false_ilmenite[['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOt', 'MnO', 'MgO', 'NiO', 'CaO', 'Na2O', 'K2O', 'P2O5']].add_suffix('_Ox')

spinel_nn = false_spinels[false_spinels.NN_Labels=='Spinel']
spinel_ox = spinel_nn[['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOt', 'MnO', 'MgO', 'NiO', 'CaO', 'Na2O', 'K2O', 'P2O5']].add_suffix('_Sp')
magnetite_nn = false_spinels[false_spinels.NN_Labels=='Magnetite']
magnetite_ox = magnetite_nn[['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOt', 'MnO', 'MgO', 'NiO', 'CaO', 'Na2O', 'K2O', 'P2O5']].add_suffix('_Sp')
ilmenite_nn = false_spinels[false_spinels.NN_Labels=='Ilmenite']
ilmenite_ox = ilmenite_nn[['SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOt', 'MnO', 'MgO', 'NiO', 'CaO', 'Na2O', 'K2O', 'P2O5']].add_suffix('_Ox')




false_sp_cat_4o = mm.calculate_4oxygens_spinel(false_spinel_ox) 
false_sp_cat_4o['Fe2_Sp_cat_4ox'] = 1 - (false_sp_cat_4o['Mg_Sp_cat_4ox'] + false_sp_cat_4o['Mn_Sp_cat_4ox'] + false_sp_cat_4o['Ca_Sp_cat_4ox'])
false_sp_cat_4o['Fe3_Sp_cat_4ox'] = 2 - (false_sp_cat_4o['Si_Sp_cat_4ox'] + false_sp_cat_4o['Al_Sp_cat_4ox'] + false_sp_cat_4o['Cr_Sp_cat_4ox'] + false_sp_cat_4o['Ti_Sp_cat_4ox'])


false_sp_mol = false_sp_cat_4o[['Si_Sp_cat_4ox', 'Mg_Sp_cat_4ox', 'Fe2_Sp_cat_4ox', 'Fe3_Sp_cat_4ox', 'Ca_Sp_cat_4ox', 'Al_Sp_cat_4ox', 'Na_Sp_cat_4ox', 'K_Sp_cat_4ox', 'Mn_Sp_cat_4ox', 'Ti_Sp_cat_4ox', 'Cr_Sp_cat_4ox', 'P_Sp_cat_4ox']]
false_sp_mol['Al_Sp_cat_4ox'] = false_sp_mol['Al_Sp_cat_4ox']/2
false_sp_mol['Cr_Sp_cat_4ox'] = false_sp_mol['Cr_Sp_cat_4ox']/2
false_sp_mol['Na_Sp_cat_4ox'] = false_sp_mol['Na_Sp_cat_4ox']/2
false_sp_mol['K_Sp_cat_4ox'] = false_sp_mol['K_Sp_cat_4ox']/2
false_sp_mol['P_Sp_cat_4ox'] = false_sp_mol['P_Sp_cat_4ox']/2
false_sp_mol = false_sp_mol.div(false_sp_mol.sum(axis=1), axis=0)


sp_cat_4o = mm.calculate_4oxygens_spinel(spinel_ox)
sp_cat_4o['Fe2_Sp_cat_4ox'] = 1 - (sp_cat_4o['Mg_Sp_cat_4ox'] + sp_cat_4o['Mn_Sp_cat_4ox'] + sp_cat_4o['Ca_Sp_cat_4ox'])
sp_cat_4o['Fe3_Sp_cat_4ox'] = 2 - (sp_cat_4o['Si_Sp_cat_4ox'] + sp_cat_4o['Al_Sp_cat_4ox'] + sp_cat_4o['Cr_Sp_cat_4ox'] + sp_cat_4o['Ti_Sp_cat_4ox'])


sp_mol = sp_cat_4o[['Si_Sp_cat_4ox', 'Mg_Sp_cat_4ox', 'Fe2_Sp_cat_4ox', 'Fe3_Sp_cat_4ox', 'Ca_Sp_cat_4ox', 'Al_Sp_cat_4ox', 'Na_Sp_cat_4ox', 'K_Sp_cat_4ox', 'Mn_Sp_cat_4ox', 'Ti_Sp_cat_4ox', 'Cr_Sp_cat_4ox', 'P_Sp_cat_4ox']]
sp_mol['Al_Sp_cat_4ox'] = sp_mol['Al_Sp_cat_4ox']/2
sp_mol['Cr_Sp_cat_4ox'] = sp_mol['Cr_Sp_cat_4ox']/2
sp_mol['Na_Sp_cat_4ox'] = sp_mol['Na_Sp_cat_4ox']/2
sp_mol['K_Sp_cat_4ox'] = sp_mol['K_Sp_cat_4ox']/2
sp_mol['P_Sp_cat_4ox'] = sp_mol['P_Sp_cat_4ox']/2
sp_mol = sp_mol.div(sp_mol.sum(axis=1), axis=0)


false_mt_cat_4o = mm.calculate_4oxygens_spinel(false_magnetite_ox) 
false_mt_cat_4o['Fe2_Sp_cat_4ox'] = 1 - (false_mt_cat_4o['Mg_Sp_cat_4ox'] + false_mt_cat_4o['Mn_Sp_cat_4ox'] + false_mt_cat_4o['Ca_Sp_cat_4ox'])
false_mt_cat_4o['Fe3_Sp_cat_4ox'] = 2 - (false_mt_cat_4o['Si_Sp_cat_4ox'] + false_mt_cat_4o['Al_Sp_cat_4ox'] + false_mt_cat_4o['Cr_Sp_cat_4ox'] + false_mt_cat_4o['Ti_Sp_cat_4ox'])

false_mt_mol = false_mt_cat_4o[['Si_Sp_cat_4ox', 'Mg_Sp_cat_4ox', 'Fe2_Sp_cat_4ox', 'Fe3_Sp_cat_4ox', 'Ca_Sp_cat_4ox', 'Al_Sp_cat_4ox', 'Na_Sp_cat_4ox', 'K_Sp_cat_4ox', 'Mn_Sp_cat_4ox', 'Ti_Sp_cat_4ox', 'Cr_Sp_cat_4ox', 'P_Sp_cat_4ox']]
false_mt_mol['Al_Sp_cat_4ox'] = false_mt_mol['Al_Sp_cat_4ox']/2
false_mt_mol['Cr_Sp_cat_4ox'] = false_mt_mol['Cr_Sp_cat_4ox']/2
false_mt_mol['Na_Sp_cat_4ox'] = false_mt_mol['Na_Sp_cat_4ox']/2
false_mt_mol['K_Sp_cat_4ox'] = false_mt_mol['K_Sp_cat_4ox']/2
false_mt_mol['P_Sp_cat_4ox'] = false_mt_mol['P_Sp_cat_4ox']/2
false_mt_mol = false_mt_mol.div(false_mt_mol.sum(axis=1), axis=0)




mt_cat_4o = mm.calculate_4oxygens_spinel(magnetite_ox)
mt_cat_4o['Fe2_Sp_cat_4ox'] = 1 - (mt_cat_4o['Mg_Sp_cat_4ox'] + mt_cat_4o['Mn_Sp_cat_4ox'] + mt_cat_4o['Ca_Sp_cat_4ox'])
mt_cat_4o['Fe3_Sp_cat_4ox'] = 2 - (mt_cat_4o['Si_Sp_cat_4ox'] + mt_cat_4o['Al_Sp_cat_4ox'] + mt_cat_4o['Cr_Sp_cat_4ox'] + mt_cat_4o['Ti_Sp_cat_4ox'])



mt_mol = mt_cat_4o[['Si_Sp_cat_4ox', 'Mg_Sp_cat_4ox', 'Fe2_Sp_cat_4ox', 'Fe3_Sp_cat_4ox', 'Ca_Sp_cat_4ox', 'Al_Sp_cat_4ox', 'Na_Sp_cat_4ox', 'K_Sp_cat_4ox', 'Mn_Sp_cat_4ox', 'Ti_Sp_cat_4ox', 'Cr_Sp_cat_4ox', 'P_Sp_cat_4ox']]
mt_mol['Al_Sp_cat_4ox'] = mt_mol['Al_Sp_cat_4ox']/2
mt_mol['Cr_Sp_cat_4ox'] = mt_mol['Cr_Sp_cat_4ox']/2
mt_mol['Na_Sp_cat_4ox'] = mt_mol['Na_Sp_cat_4ox']/2
mt_mol['K_Sp_cat_4ox'] = mt_mol['K_Sp_cat_4ox']/2
mt_mol['P_Sp_cat_4ox'] = mt_mol['P_Sp_cat_4ox']/2
mt_mol = mt_mol.div(mt_mol.sum(axis=1), axis=0)



# %% 


plt.scatter(spinel_nn['FeOt'], spinel_nn['TiO2'], alpha=0.5) #, color=false_spinels['NN_Labels'])
plt.scatter(magnetite_nn['FeOt'], magnetite_nn['TiO2'], alpha=0.5) #, color=false_spinels['NN_Labels'])



# %%


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
false_sp_mol.loc[:, ["Ti_Sp_cat_4ox", "Fe2_Sp_cat_4ox", "Fe3_Sp_cat_4ox"]].pyroplot.scatter(c="k", ax=ax)
sp_mol.loc[:, ["Ti_Sp_cat_4ox", "Fe2_Sp_cat_4ox", "Fe3_Sp_cat_4ox"]].pyroplot.scatter(c="r", ax=ax)

false_mt_mol.loc[:, ["Ti_Sp_cat_4ox", "Fe2_Sp_cat_4ox", "Fe3_Sp_cat_4ox"]].pyroplot.scatter(c="blue", ax=ax)
mt_mol.loc[:, ["Ti_Sp_cat_4ox", "Fe2_Sp_cat_4ox", "Fe3_Sp_cat_4ox"]].pyroplot.scatter(c="green", ax=ax)


plt.show()

# %%
