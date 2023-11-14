# %%

import os
import math
import copy
import pickle

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
import torch.nn.functional as F

from minML.core import *

# %%

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

        std = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-std, std)
        self.weight_rho.data.uniform_(-std, std)
        self.bias_mu.data.uniform_(-std, std)
        self.bias_rho.data.uniform_(-std, std)
        
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

def predict_class_prob_train(model, input_data, n_iterations=100):

    model.eval()
    output_list = []
    for i in range(n_iterations):
        with torch.no_grad():
            output = model(input_data)
            output_list.append(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy())

    output_list = np.array(output_list)
    
    # Calculate mean and standard deviation
    prediction_mean = output_list.mean(axis=0)
    prediction_std = output_list.std(axis=0)

    return prediction_mean, prediction_std

def load_model(model, optimizer=None, path=''):
    check_point = torch.load(path)
    model.load_state_dict(check_point['params'])
    if optimizer is not None:
        optimizer.load_state_dict(check_point['optimizer'])

def predict_class_prob(df, n_iterations=100): 

    lr = 5e-3 
    wd = 1e-3 
    dr = 0.1
    hls = [64, 32, 16]

    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiClassClassifier(input_dim=len(oxides), dropout_rate=dr, hidden_layer_sizes=hls).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'best_model.pt')  # Note the .joblib extension

    load_model(model, optimizer, model_path)

    norm_wt = norm_data(df)
    input_data = torch.Tensor(norm_wt).to(device)

    model.eval()
    output_list = []
    for i in range(n_iterations):
        with torch.no_grad():
            output = model(input_data)
            output_list.append(torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy())

    output_list = np.array(output_list)
    
    predict_probability = output_list.mean(axis=0)
    predict_class = np.argmax(predict_probability, axis=1)
    predict_prob = np.max(predict_probability, axis=1)

    return predict_class, predict_prob

def unique_mapping(df, pred_class): 

    min_cat, mapping = load_minclass()
    df['Mineral'] = pd.Categorical(df['Mineral'], categories=min_cat)
    given_class = df['Mineral'].cat.codes.values
    unique = np.unique(np.concatenate((given_class, pred_class)))
    valid_mapping = {key: mapping[key] for key in unique}
    if -1 in unique:
        valid_mapping[-1] = "Unknown" 

    return unique, valid_mapping

def class2mineral(df, pred_class): 

    unique, valid_mapping = unique_mapping(df, pred_class)
    pred_mineral = np.array([valid_mapping[x] for x in pred_class])

    return pred_mineral


def confusion_matrix_df(df, pred_class):
    # Use a list comprehension to prepare the index and columns once,
    # as they are the same for both indices and columns.

    unique, valid_mapping = unique_mapping(df, pred_class)
    pred_mineral = class2mineral(df, pred_class)
    cm_matrix = confusion_matrix(df.Mineral, pred_mineral)

    labels = [valid_mapping[x] for x in unique]
    cm_df = pd.DataFrame(cm_matrix, index=labels, columns=labels)

    return cm_df



def train_nn(model, optimizer, train_loader, valid_loader, n_epoch, criterion, kl_weight_decay, kl_decay_epochs=750, patience=50):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    avg_train_loss = []
    avg_valid_loss = []
    best_valid_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    kl_weight = 0.0  # Initial kl_weight

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
            # kl_weight = min(kl_weight + (kl_weight_decay * (epoch // kl_decay_epochs)), 1)
            kl_weight_increment = kl_weight_decay / kl_decay_epochs
            kl_weight = min(kl_weight + kl_weight_increment, 1)

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
        valid_loss = []
        with torch.no_grad():
            for i, (data, labels) in enumerate(valid_loader):
                x = data.to(device)
                y = labels.to(device)
                valid_output = model(x)
                loss = criterion(valid_output, y)
                valid_loss.append(loss.detach().item())

        # Logging
        avg_train = sum(train_loss) / len(train_loss)
        avg_valid = sum(valid_loss) / len(valid_loss)
        avg_train_loss.append(avg_train)
        avg_valid_loss.append(avg_valid)
        
        training_time = time.time() - t
 
        print(f'[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_train:.6f}, valid_loss: {avg_valid:.6f}, time: {training_time:.2f} s')

        # Early stopping
        if avg_valid < best_valid_loss:
            best_valid_loss = avg_valid
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())  # Save the best model weights

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Validation loss hasn't improved for {patience} epochs. Stopping early.")
                break

    return train_output, valid_output, avg_train_loss, avg_valid_loss, best_valid_loss, best_model_state

def neuralnetwork(df, hls_list, kl_weight_decay_list, lr, wd, dr, ep, n, balanced):

    path_beg = os.getcwd() + '/'
    output_dir = ["parametermatrix_neuralnetwork"] 
    for ii in range(len(output_dir)):
        if not os.path.exists(path_beg + output_dir[ii]):
            os.makedirs(path_beg + output_dir[ii], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wt = df[['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']].fillna(0)

    ss = StandardScaler()
    array_norm = ss.fit_transform(wt)

    code = pd.Categorical(df['Mineral']).codes
    cat_lab = pd.Categorical(df['Mineral'])

    # Split the dataset into train and test sets
    train_x, valid_x, train_y, valid_y = train_test_split(array_norm, code, test_size=n, stratify=code, random_state=42)

    if balanced == True: 
        train_x, train_y = balance(train_x, train_y)

    # Define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = LabelDataset(train_x, train_y)
    valid_dataset = LabelDataset(valid_x, valid_y)

    mapping = dict(zip(code, cat_lab))
    sort_mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))

    # Autoencoder params:
    lr = lr
    wd = wd 
    dr = dr
    epochs = ep
    batch_size = 256
    input_size = len(feature_dataset.__getitem__(0)[0])

    best_hidden_layer_size = None
    best_kl_weight_decay = None
    best_model_state = None
    best_valid_loss = float('inf')

    # Define data loaders
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    np.savez('parametermatrix_neuralnetwork/' + 'best_model_nn_features.npz', feature_loader=feature_loader, valid_loader=valid_loader)

    train_losses_dict = {}
    valid_losses_dict = {}

    for hls in hls_list:
        for kl_weight_decay in kl_weight_decay_list:

            print(f"Training with KL weight decay: {kl_weight_decay} and hidden layer sizes: {hls}")

            # Initialize model
            model = MultiClassClassifier(input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=hls).to(device)

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

            # Train model and get the best test loss and model state
            train_output, valid_output, avg_train_loss, avg_valid_loss, current_best_valid_loss, current_best_model_state = train_nn(model, optimizer, feature_loader, valid_loader, epochs, criterion, kl_weight_decay=kl_weight_decay)

            if current_best_valid_loss < best_valid_loss:
                best_valid_loss = current_best_valid_loss
                best_kl_weight_decay = kl_weight_decay
                best_hidden_layer_size = hls
                best_model_state = current_best_model_state

            train_losses_dict[(kl_weight_decay, tuple(hls))] = avg_train_loss
            valid_losses_dict[(kl_weight_decay, tuple(hls))] = avg_valid_loss

    # Create a new model with the best model state
    best_model = MultiClassClassifier(input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=best_hidden_layer_size)
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    # Perform predictions on the test dataset using the best model
    with torch.no_grad():
        valid_predictions = best_model(torch.Tensor(valid_x))
        train_predictions = best_model(torch.Tensor(train_x))
        valid_pred_y = valid_predictions.argmax(dim=1).cpu().numpy()
        train_pred_y = train_predictions.argmax(dim=1).cpu().numpy()

    # Calculate classification metrics for the test dataset
    valid_report = classification_report(valid_y, valid_pred_y, target_names=list(sort_mapping.values()), zero_division=0, output_dict=True)
    train_report = classification_report(train_y, train_pred_y, target_names=list(sort_mapping.values()), zero_division=0, output_dict=True) # output_dict=True

    # Print the best kl_weight_decay value and test report
    print("Best kl_weight_decay:", best_kl_weight_decay)
    print("Best best_hidden_layer_size:", best_hidden_layer_size)

    # Save the best model and other relevant information
    model_path = 'parametermatrix_neuralnetwork/best_model.pt'
    save_model_nn(optimizer, best_model_state, model_path)

    train_pred_mean, train_pred_std = predict_class_prob_train(model, feature_dataset.x)
    valid_pred_mean, valid_pred_std = predict_class_prob_train(model, valid_dataset.x)

    # Get the most probable classes
    train_pred_y = np.argmax(train_pred_mean, axis=1)
    valid_pred_y = np.argmax(valid_pred_mean, axis=1)

    np.savez('parametermatrix_neuralnetwork/' + 'best_model_data.npz', best_hidden_layer_size=best_hidden_layer_size, best_kl_weight_decay=best_kl_weight_decay, 
             valid_report=valid_report, train_report=train_report,
             train_y=train_y, valid_y=valid_y, train_pred_y=train_pred_y, valid_pred_y=valid_pred_y, 
             train_pred_mean=train_pred_mean, train_pred_std=train_pred_std, valid_pred_mean=valid_pred_mean, valid_pred_std=valid_pred_std)

    # Save the train and test losses
    np.savez('parametermatrix_neuralnetwork/'+'best_model_losses.npz', train_losses=train_losses_dict, valid_losses=valid_losses_dict)

    return best_model_state


