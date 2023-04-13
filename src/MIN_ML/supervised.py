# %%

import numpy as np
import pandas as pd
import scipy
import random
import time
import copy
from sklearn.preprocessing import scale, normalize, StandardScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU, Sigmoid
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from MIN_ML.core import *

# %% 


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

class MultiClassClassifier(nn.Module):
    def __init__(self, input_dim=10, classes=17, dropout_rate=0.1, hidden_layer_sizes=[8]):
        super(MultiClassClassifier, self).__init__()
        self.input_dim = input_dim
        self.classes = classes
        self.dropout_rate = dropout_rate
        self.hls = hidden_layer_sizes

        def element(in_channel, out_channel, is_last=False):
            layers = [
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),  # Add batch normalization
                nn.LeakyReLU(0.02),
            ]
            if not is_last:
                layers.append(nn.Dropout(self.dropout_rate))  # Add dropout
            return layers

        if len(self.hls) == 1:
            encoder = element(self.input_dim, self.hls[0], is_last=True)
            encoder += [nn.Linear(self.hls[-1], classes)]
        else:
            encoder = element(self.input_dim, self.hls[0])
            for i in range(len(self.hls) - 1):
                is_last = (i == len(self.hls) - 2)
                encoder += element(self.hls[i], self.hls[i + 1], is_last)
            encoder += [nn.Linear(self.hls[-1], classes)]

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

def train_nn(model, optimizer, label, train_loader, test_loader, n_epoch, criterion, patience=10):
    """
    Train a neural network using early stopping.

    Args:
        model: a PyTorch model
        optimizer: a PyTorch optimizer
        label: a string representing the type of labels being used
        train_loader: a PyTorch DataLoader for the training set
        test_loader: a PyTorch DataLoader for the validation set
        n_epoch: the maximum number of epochs to train for
        criterion: the loss function to optimize
        patience: the number of epochs to wait before stopping if the validation loss does not improve

    Returns:
        train_output: the final output of the model on the training set
        test_output: the final output of the model on the validation set
        avg_train_loss: a list of average training losses for each epoch
        avg_test_loss: a list of average validation losses for each epoch
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    avg_train_loss = []
    avg_test_loss = []
    best_test_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(n_epoch):
        # Training
        model.train()
        t = time.time()
        train_loss = []
        for i, (data, labels) in enumerate(train_loader):
            x = data.to(device)
            y = labels.to(device)
            train_output = model(x)
            loss = criterion(train_output, y)

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

    return train_output, test_output, avg_train_loss, avg_test_loss, best_model_state


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


def neuralnetwork(df, name, WhichClassifier, hidden_layer_sizes, lr, wd, dr, epochs, n, balanced): 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
    label = ['Mineral']

    min = df[label]
    wt = df[oxides].fillna(0).to_numpy()

    ss = StandardScaler()
    array_norm = ss.fit_transform(wt)

    #split the dataset into train and test sets
    train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(array_norm, pd.Categorical(df['Mineral']).codes, test_size=n, stratify = pd.Categorical(df['Mineral']).codes, random_state=42)

    if balanced == True: 
        train_data_x, train_data_y = balance(train_data_x, train_data_y)

    #define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = LabelDataset(train_data_x, train_data_y)
    test_dataset = LabelDataset(test_data_x, test_data_y)

    mapping = dict(zip(pd.Categorical(df['Mineral']).codes, pd.Categorical(df['Mineral'])))
    sort_mapping= dict(sorted(mapping.items(), key=lambda item: item[0])) 

    #autoencoder params:
    lr = lr
    wd = wd 
    batch_size = 256
    epochs = epochs
    input_size = len(feature_dataset.__getitem__(0)[0])
    dr = dr

    #define data loaders
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    np.savez('nn_parametermatrix/' + name + '_nn_features.npz', feature_loader = feature_loader, test_loader = test_loader)

    # Initialize model
    if WhichClassifier == MultiClassClassifier:
        model = WhichClassifier(input_dim=input_size, dropout_rate = dr, hidden_layer_sizes=hidden_layer_sizes).to(device) # dropout_rate = dr
    elif WhichClassifier == OneLayerClassifier: 
        model = WhichClassifier(input_dim=input_size).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=lr, weight_decay = wd)

    #train model using pre-defined function
    train_output, test_output, train_loss, test_loss, best_model_state = train_nn(model, optimizer, label, feature_loader, test_loader, epochs, criterion)
    np.savez('nn_parametermatrix/' + name + '_nn_loss.npz', train_loss = train_loss, test_loss = test_loss)
    
    model_path = 'nn_parametermatrix/' + name + "_nn_params.pt"
    save_model_nn(model, optimizer, model_path, best_model_state)

    # predict classes for entire training and test datasets
    train_pred_classes = model.predict(feature_dataset.x)
    test_pred_classes = model.predict(test_dataset.x)

    # calculate classification metrics
    train_report = classification_report(train_data_y, train_pred_classes, target_names = sort_mapping.values(), zero_division=0)
    test_report = classification_report(test_data_y, test_pred_classes, target_names = sort_mapping.values(), zero_division=0)

    return train_pred_classes, test_pred_classes, train_report, test_report, best_model_state

