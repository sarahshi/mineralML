# %%

import os
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.utils.data import DataLoader
from scipy.special import softmax

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.mlls.deep_approximate_mll import DeepApproximateMLL

from hdbscan.flat import HDBSCAN_flat, approximate_predict_flat

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

from .core import *
from .supervised import *
from .constants import OXIDES


# %%


# def prep_df_ae(df):
#     """

#     Prepares a DataFrame for analysis by performing data cleaning specific to mineralogical data.
#     It filters the DataFrame for selected minerals, handles missing values, and separates the data
#     into two DataFrames: one that includes specified minerals and another that excludes them.
#     The function defines a list of oxide column names and minerals to include and exclude. It drops
#     rows where the specified oxides and 'Mineral' column have fewer than six non-NaN values.

#     Parameters:
#         df (DataFrame): The input DataFrame containing mineral composition data along with 'Mineral' column.

#     Returns:
#         df_in (DataFrame): A DataFrame with rows including only the specified minerals and 'NaN' filled with zero.
#         df_ex (DataFrame): A DataFrame with rows excluding the specified minerals and 'NaN' filled with zero.

#     """

#     if "FeO" in df.columns and "FeOt" not in df.columns:
#         raise ValueError(
#             "No 'FeOt' column found. You have a 'FeO' column. mineralML only recognizes 'FeOt' as a column. Please convert to FeOt."
#         )
#     if "Fe2O3" in df.columns and "FeOt" not in df.columns:
#         raise ValueError(
#             "No 'FeOt' column found. You have a 'Fe2O3' column. mineralML only recognizes 'FeOt' as a column. Please convert to FeOt."
#         )

#     oxidesandmin = [
#         "SiO2",
#         "TiO2",
#         "Al2O3",
#         "FeOt",
#         "MnO",
#         "MgO",
#         "CaO",
#         "Na2O",
#         "K2O",
#         "Cr2O3",
#         'P2O5',
#         "Mineral",
#     ]
#     include_minerals = [
#         "Amphibole",
#         "Apatite",
#         "Biotite",
#         "Clinopyroxene",
#         "Garnet",
#         "Glass",
#         "Ilmenite",
#         "KFeldspar",
#         "Magnetite",
#         "Muscovite",
#         "Olivine",
#         "Orthopyroxene",
#         "Plagioclase",
#         "Quartz",
#         "Rutile",
#         "Spinel",
#         "Tourmaline",
#         "Zircon",
#     ]
#     # df.dropna(subset=oxidesandmin, thresh=5, inplace=True)
#     df_in = df[df["Mineral"].isin(include_minerals)]
#     df_ex = df[~df["Mineral"].isin(include_minerals)]

#     df_in = df_in[oxidesandmin].fillna(0)

#     df_in = df_in.reset_index(drop=True)
#     df_ex = df_ex.reset_index(drop=True)

#     return df_in, df_ex


# def norm_data_ae(df):
#     """

#     Normalizes the oxide composition data in the input DataFrame using a predefined StandardScaler.
#     It ensures that the dataframe has been preprocessed accordingly before applying the transformation.
#     The function expects that the scaler is already fitted and available for use as defined in the
#     'load_scaler' function.

#     Parameters:
#         df (DataFrame): The input DataFrame containing the oxide composition data.

#     Returns:
#         array_x (ndarray): An array of the transformed oxide composition data.

#     """

#     oxides = [
#         "SiO2",
#         "TiO2",
#         "Al2O3",
#         "FeOt",
#         "MnO",
#         "MgO",
#         "CaO",
#         "Na2O",
#         "K2O",
#         "Cr2O3",
#         'P2O5',
#     ]
#     mean, std = load_scaler("scaler_ae.npz")

#     if df[oxides].isnull().any().any():
#         df, _ = prep_df_nn(df)
#     else:
#         df = df

#     scaled_df = df[oxides].copy()

#     for col in df[oxides].columns:
#         scaled_df[col] = (df[col] - mean[col]) / std[col]

#     array_x = scaled_df.to_numpy()

#     return array_x


def feature_normalisation(feature, return_params=False, mean_norm=True):
    """
    Function to perform mean normalisation on the dataset passed to it.

    Parameters:
        feature (numpy array): Features to be normalised.
        return_params (boolean, optional): Set to True if parameters used for mean normalisation
        are to be returned for each feature.

    Returns:
        norm (numpy array): Mean normalised features.
        params (list of numpy arrays): Only returned if set to True above; list of parameters
        used for the mean normalisation as derived from the features (i.e., mean, min, and max).
    """

    params = []

    norm = np.zeros_like(feature)

    if len(feature.shape) == 2:
        for i in range(feature.shape[1]):
            if mean_norm == True:
                temp_mean = feature[:, i].mean()
            elif mean_norm == False:
                temp_mean = 0
            else:
                raise ValueError("Mean_norm must be boolean")
            norm[:, i] = (feature[:, i] - temp_mean) / (
                feature[:, i].max() - feature[:, i].min()
            )
            params.append(
                np.asarray([temp_mean, feature[:, i].min(), feature[:, i].max()])
            )

    elif len(feature.shape) == 1:
        if mean_norm == True:
            temp_mean = feature[:].mean()
        elif mean_norm == False:
            temp_mean = 0
        else:
            raise ValueError("Mean_norm must be boolean")
        norm[:] = (feature[:] - temp_mean) / (feature.max() - feature.min())
        params.append(np.asarray([temp_mean, feature[:].min(), feature[:].max()]))

    else:
        raise ValueError("Feature array must be either 1D or 2D numpy array.")

    if return_params == True:
        return norm, params
    else:
        return norm


class Autoencoder(nn.Module):

    """

    A neural network module for dimensionality reduction and feature learning,
    implementing an autoencoder architecture. It compresses input data into a
    lower-dimensional latent space and then reconstructs it back.

    Parameters:
        input_dim (int): Dimension of the input data. Default: 10.
        latent_dim (int): Dimension of the latent space. Default: 2.
        hidden_layer_sizes (tuple of ints): Sizes of the hidden layers in the encoder
                                            and decoder. Default: (256, 64, 16).

    Attributes:
        input_dim (int): The input dimension.
        latent_dim (int): The latent space dimension.
        hls (tuple): Tuple representing the sizes of hidden layers.

    Methods:
        encoded(self, x): Encodes input data `x` to the latent space.
        decoded(self, x): Decodes data `x` from the latent space to the input space.
        forward(self, x): Defines the forward pass of the autoencoder.

    """

    def __init__(self, input_dim=10, latent_dim=2, hidden_layer_sizes=(256, 64, 16)):
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
        decoder += [nn.Linear(self.hls[0], self.input_dim)]  # nn.Softmax()]

        self.encode = nn.Sequential(*encoder)
        self.decode = nn.Sequential(*decoder)

        self.apply(weights_init)

    def encoded(self, x):
        # encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        # decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de


class Tanh_Autoencoder(nn.Module):

    """

    A neural network module implementing an autoencoder with Tanh activation functions.
    This class is designed for dimensionality reduction and feature learning,
    encoding input data into a lower-dimensional latent space and reconstructing it.

    The architecture uses Tanh activation functions in each layer, which can provide
    smoother gradients and potentially better convergence in some cases.

    Parameters:
        input_dim (int): Dimension of the input data. Default: 10.
        latent_dim (int): Dimension of the latent space. Default: 2.
        hidden_layer_sizes (tuple of ints): Sizes of the hidden layers in the encoder
                                            and decoder. Default: (256, 64, 16).

    Attributes:
        input_dim (int): The input dimension.
        latent_dim (int): The latent space dimension.
        hls (tuple): Tuple representing the sizes of hidden layers.

    Methods:
        encoded(self, x): Encodes input data `x` to the latent space.
        decoded(self, x): Decodes data `x` from the latent space back to the input space.
        forward(self, x): Defines the forward pass of the autoencoder.

    Note:
        This autoencoder uses a Tanh activation function in each layer
        for both the encoder and the decoder.

    """

    def __init__(self, input_dim=10, latent_dim=2, hidden_layer_sizes=(256, 64, 16)):
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
        decoder += [nn.Linear(self.hls[0], self.input_dim)]  # nn.Softmax()]

        self.encode = nn.Sequential(*encoder)
        self.decode = nn.Sequential(*decoder)

        self.apply(weights_init)

    def encoded(self, x):
        # encodes data to latent space
        return self.encode(x)

    def decoded(self, x):
        # decodes latent space data to 'real' space
        return self.decode(x)

    def forward(self, x):
        en = self.encoded(x)
        de = self.decoded(en)
        return de


def train(
    model,
    optimizer,
    train_loader,
    valid_loader,
    n_epoch,
    criterion,
    patience=200,
    min_delta=0.00005,
):
    """

    Trains a given model using specified data loaders, optimizer, and loss criterion.

    The function performs training for a fixed number of epochs and evaluates the model
    on validation data at the end of each epoch. It also implements early stopping based
    on validation loss improvement.

    Parameters:
        model (nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        n_epoch (int): Number of epochs to train the model.
        criterion (function): Loss function to evaluate model performance.
        patience (int, optional): Number of epochs to wait for improvement in validation loss
                                  before stopping early. Default: 10.
        min_delta (float, optional): Minimum change in validation loss to qualify as an improvement.
                                     Default: 0.0005.

    Returns:
        tuple: A tuple containing two lists: average training loss per epoch and average validation loss per epoch.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_train_loss = []
    avg_valid_loss = []

    best_valid_loss = float("inf")
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
        valid_loss = []
        for i, test in enumerate(valid_loader):
            x = test.to(device)
            x_recon = model(x)
            loss = criterion(x_recon, x)
            valid_loss.append(loss.detach().item())

        # Logging
        avg_loss = sum(train_loss) / len(train_loss)
        avg_valid = sum(valid_loss) / len(valid_loss)
        avg_train_loss.append(avg_loss)
        avg_valid_loss.append(avg_valid)

        training_time = time.time() - t

        print(
            f"[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_loss:.6f}, valid_loss: {avg_valid:.6f}, time: {training_time:.2f} s"
        )

        # Early stopping
        valid_loss_improvement = best_valid_loss - avg_valid
        if valid_loss_improvement > min_delta:
            best_valid_loss = avg_valid
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Test loss hasn't improved significantly for {patience} epochs. Stopping early."
                )
                break

    return avg_train_loss, avg_valid_loss


def autoencode(df, name, AE_Model, hls, lr, wd, ep, n, balanced):
    """

    Trains an autoencoder on a given dataset and visualizes the latent space representation.

    This function preprocesses the dataset, splits it into training and validation sets,
    initializes an autoencoder model, and trains it. The latent space representation of the
    entire dataset is then visualized, and the model parameters are saved.

    Parameters:
        df (pd.DataFrame): Input dataset containing the features and mineral labels.
        name (str): Name to be used for saving outputs (plots, model parameters, etc.).
        AE_Model (nn.Module): Autoencoder model class to be instantiated.
        hidden_layer_sizes (tuple of ints): Sizes of the hidden layers in the autoencoder.
        ep (int): Number of epochs to train the model.

    Returns:
        np.ndarray: Latent space representation of the entire dataset.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oxides = OXIDES


    all_cats_init = pd.Categorical(df["Mineral"])
    df["_code"] = all_cats_init.codes

    # Split the dataset into train and test sets
    train_df, valid_df = train_test_split(
        df, test_size=n, stratify=df["_code"], random_state=42
    )
    if balanced == True:
        train_df = balance(train_df, n=1000)

    all_cats = pd.Categorical(train_df["Mineral"])
    mapping = dict(enumerate(all_cats.categories))
    inv_mapping = {cat: idx for idx, cat in mapping.items()}
    sort_mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))

    valid_df['Mineral'] = (
      valid_df['Mineral'].astype(str)\
        .replace(['Clinopyroxene', 'Orthopyroxene'], 'Pyroxene')
        .replace(['Plagioclase', 'KFeldspar'], 'Feldspar')
        .replace(['Hematite', 'Ilmenite'], 'Rhombohedral_Oxides')
        .replace(['Magnetite', 'Spinel'], 'Spinels')
    )

    train_df["_code"] = train_df["Mineral"].map(inv_mapping).astype(int)
    valid_df["_code"] = valid_df["Mineral"].map(inv_mapping).astype(int)

    ss = StandardScaler().fit(train_df[OXIDES])
    train_x = ss.transform(train_df[OXIDES].fillna(0))
    valid_x = ss.transform(valid_df[OXIDES].fillna(0))


    # # perform z-score normalisation
    wt = df[oxides].fillna(0) # 1e-4
    wt = wt.to_numpy()
    ss = StandardScaler()
    array_norm = ss.fit_transform(wt)

    # # #split the dataset into train and test sets
    # # train_data, valid_data = train_test_split(array_norm, test_size=0.1, stratify = df['Mineral'], random_state=42)
    # train_data, valid_data = train_test_split(
    #     array_norm, test_size=0.2, stratify=df["Mineral"], random_state=42
    # )

    # define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = FeatureDataset(train_x)
    valid_dataset = FeatureDataset(valid_x)

    # autoencoder params:
    lr = lr # 5e-4
    wd = wd # 0.005
    ep = ep
    batch_size = 256
    input_size = feature_dataset.__getitem__(0).size(0)

    # define data loaders
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    np.savez(
        "parametermatrix_autoencoder/" + name + "_features.npz",
        feature_loader=feature_loader,
        valid_loader=valid_loader,
    )

    # define model
    model = AE_Model(input_dim=input_size, hidden_layer_sizes=hls).to(
        device
    )

    # use ADAM optimizer with mean squared error loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    # train model using pre-defined function
    # train_loss, valid_loss = train(
    #     model, optimizer, feature_loader, valid_loader, ep, criterion
    # )

    # testing
    def kl_anneal(epoch):
        warm, ramp, beta_max = 20, 150, 0.3
        if epoch+1 <= warm: return 0.0
        e = min(1.0, (epoch+1 - warm)/ramp)
        return beta_max * e
    
    train_loss, valid_loss = train_vae(
        model,
        optimizer,
        train_loader=feature_loader,
        valid_loader=valid_loader,
        n_epoch=ep,
        beta=1.0,
        recon='mse',
        kl_anneal=kl_anneal,
        patience=50, start_es_beta=1.0
    )


    np.savez(
        "parametermatrix_autoencoder/" + name + "_tanh_loss.npz",
        train_loss=train_loss,
        valid_loss=valid_loss,
    )

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.flatten()
    ax[0].plot(
        np.linspace(1, len(train_loss), len(train_loss)),
        train_loss,
        ".-",
        label="Train Loss",
    )
    ax[0].plot(
        np.linspace(1, len(train_loss), len(train_loss)),
        valid_loss,
        ".-",
        label="Test Loss",
    )
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend(prop={"size": 10})

    # #transform entire dataset to latent space
    z = getLatent(model, array_norm)

    phase = np.array([
        "Amphibole",
        "Apatite",
        "Biotite",
        "Calcite",
        "Chlorite",
        "Epidote",
        "Feldspar",
        "Garnet",
        "Glass",
        "Kalsilite",
        "Leucite",
        "Melilite",
        "Muscovite",
        "Nepheline",
        "Olivine",
        "Pyroxene",
        "Quartz",
        "Rhombohedral_Oxides",
        "Rutile",
        "Serpentine",
        "Spinels",
        "Titanite",
        "Tourmaline",
        "Zircon",
    ])

    phasez = range(1, len(phase))
    tab = plt.get_cmap("tab20")
    cNorm = mcolors.Normalize(vmin=0, vmax=len(phase))
    scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)

    # plot latent representation
    for i in range(len(phase)):
        indx = df["Mineral"] == phase[i]
        ax[1].scatter(
            z[indx, 0],
            z[indx, 1],
            s=15,
            color=scalarMap.to_rgba(i),
            lw=1,
            label=phase[i],
            rasterized=True,
        )
    ax[1].set_xlabel("Latent Variable 1")
    ax[1].set_ylabel("Latent Variable 2")
    ax[1].set_title(name + " Latent Space Representation")
    ax[1].legend(prop={"size": 8})
    plt.tight_layout()
    plt.savefig(
        "parametermatrix_autoencoder/" + name + "_loss_latentspace.pdf",
    )

    # save main model params
    model_path = "parametermatrix_autoencoder/" + name + "_tanh_params.pt"
    save_model_ae(model, optimizer, model_path)

    # save all other params
    conc_file = name + "_tanh.npz"
    np.savez(
        "parametermatrix_autoencoder/" + name + "_tanh.npz",
        batch_size=batch_size,
        lr=lr,
        wd=wd,
        ep=ep,
        input_size=input_size,
        conc_file=conc_file,
        z=z,
    )

    return z


def getLatent(model, dataset: np):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform real data to latent space using the trained model
    latents = []
    model.to(device)

    dataset_ = FeatureDataset(dataset)
    loader = DataLoader(dataset_, batch_size=20, shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.to(device)
            z = model.encoded(x)
            latents.append(z.detach().cpu().numpy())

    return np.concatenate(latents, axis=0)


def get_latent_space(df):
    """

    Loads a pre-trained autoencoder model and computes the latent space representations
    for the input dataframe. This function loads a model from a predefined path, normalizes
    the input data, and then uses the model to generate latent space representations for
    each data point.

    Parameters:
        df (pd.DataFrame): Input data frame to be processed by the autoencoder.

    Returns:
        pd.DataFrame: DataFrame containing the latent space representations.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "ae_best_model.pt")
    model = Tanh_Autoencoder(input_dim=11, hidden_layer_sizes=(256, 64, 16)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0)
    load_model(model, optimizer, model_path)

    # # Set the model to evaluation mode
    # model.eval()

    norm_wt = norm_data_nn(df)
    z = getLatent(model, norm_wt)
    z_df = pd.DataFrame(z, columns=["LV1", "LV2"])

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
    filepath = os.path.join(current_dir, "mineral_classes_ae.npz")

    with np.load(filepath, allow_pickle=True) as data:
        mapping = data[
            "mapping"
        ].item()  # .item() is used to get the dictionary from numpy array

    return mapping


def load_clusterer():
    """

    Loads latent space data from a saved file, performs normalization, and applies clustering.

    This function reads latent space representations from a saved .npz file, normalizes them,
    and then uses HDBSCAN to cluster the data. It returns the clusterer object and a DataFrame
    containing the latent space data along with clustering results.

    Returns:
        tuple: A tuple containing the HDBSCAN clusterer object and a DataFrame with latent variables,
               cluster labels, predicted minerals, and prediction probabilities.

    """

    current_dir = os.path.dirname(__file__)
    filepath = os.path.join(current_dir, "ae_tanh.npz")
    z = np.load(filepath)["z"]

    array_train, params_train = feature_normalisation(z, return_params=True)
    clusterer = HDBSCAN_flat(
        array_train,
        min_cluster_size=30,
        cluster_selection_epsilon=0.025,
        prediction_data=True,
    )
    labels, probs = clusterer.labels_, clusterer.probabilities_
    minerals = class2mineral_ae(labels)

    z_df = pd.DataFrame(z, columns=["LV1", "LV2"])
    z_df["Label"] = labels
    z_df["Predict_Mineral"] = minerals
    z_df["Predict_Probability"] = probs

    return clusterer, z_df


def predict_class_prob_ae(df):
    """

    Predicts the class probabilities, corresponding mineral names, and the maximum
    probability for each class using a predefined Autoencoder model. This
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
    df_pred["LV1"] = z_df["LV1"]
    df_pred["LV2"] = z_df["LV2"]

    array_valid, params_valid = feature_normalisation(
        z_df.to_numpy(), return_params=True
    )
    labels_valid, probs_valid = approximate_predict_flat(
        clusterer, array_valid, cluster_selection_epsilon=0.025
    )
    predict_mineral = class2mineral_ae(labels_valid)
    df_pred["Predict_Code"] = labels_valid
    df_pred["Predict_Mineral"] = predict_mineral
    df_pred["Predict_Probability"] = probs_valid

    return df_pred


def unique_mapping_ae(pred_class):
    """
    Generates a mapping of unique class codes from given and predicted class labels,
    considering only the classes present in both input arrays. It loads a predefined
    category list and mapping, encodes the 'given_class' labels into categorical codes,
    and creates a subset mapping for the unique classes found. It also handles unknown
    classes by assigning them a code of -1 and mapping the 'Unknown' label to them.

    Parameters:
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
    """

    Plots the latent space representations of the training and validation datasets.
    This function loads the latent space data using the `load_clusterer` function
    and visualizes it in a scatter plot. It compares the latent space representations
    of the training set with the predictions on the validation set, using different colors
    for each predicted mineral class.

    Parameters:
        df_pred (pd.DataFrame): DataFrame containing latent space representations and predictions
                                for the validation dataset.

    """

    clusterer, z_df = load_clusterer()

    phase = np.array(
        [
            "Amphibole",
            "Apatite",
            "Biotite",
            "Clinopyroxene",
            "Garnet",
            "Ilmenite",
            "KFeldspar",
            "Magnetite",
            "Muscovite",
            "Olivine",
            "Orthopyroxene",
            "Plagioclase",
            "Quartz",
            "Rutile",
            "Spinel",
            "Tourmaline",
            "Zircon",
        ]
    )
    cNorm = mcolors.Normalize(vmin=0, vmax=len(phase))
    scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap("tab20"))

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.flatten()
    for i in range(len(phase)):
        indx_z = z_df["Predict_Mineral"] == phase[i]
        if not z_df[indx_z].empty:
            ax[0].scatter(
                z_df.LV1[indx_z],
                z_df.LV2[indx_z],
                marker="o",
                s=15,
                color=scalarMap.to_rgba(i),
                lw=0.1,
                ec="k",
                alpha=z_df.Predict_Probability[indx_z],
                label=phase[i],
                rasterized=True,
            )
        indx_pred = df_pred["Predict_Mineral"] == phase[i]
        if not df_pred[indx_pred].empty:
            ax[1].scatter(
                df_pred.LV1[indx_pred],
                df_pred.LV2[indx_pred],
                marker="o",
                s=15,
                color=scalarMap.to_rgba(i),
                lw=0.1,
                ec="k",
                alpha=df_pred.Predict_Probability[indx_pred],
                label=phase[i],
                rasterized=True,
            )
    leg = ax[0].legend(
        prop={"size": 8},
        loc="upper right",
        labelspacing=0.4,
        handletextpad=0.8,
        handlelength=1.0,
        frameon=False,
    )
    for handle in leg.legendHandles:
        colors = handle.get_facecolor()
        # Set alpha value for face color
        colors[:, 3] = 1  # Set the alpha value of the RGBA color to 1
        handle.set_facecolor(colors)

        # If the handles also have edge colors, set their alpha values as well
        edge_colors = handle.get_edgecolor()
        edge_colors[:, 3] = 1
        handle.set_edgecolor(edge_colors)
    ax[0].set_xlabel("Latent Variable 1")
    ax[0].set_ylabel("Latent Variable 2")
    ax[0].set_xlim([-1.5, 2.0])
    ax[0].set_ylim([-2.5, 2.5])
    ax[0].annotate(
        "Training Latent Space",
        xy=(0.03, 0.94),
        xycoords="axes fraction",
        fontsize=20,
        weight="medium",
    )
    ax[0].tick_params(axis="x", direction="in", length=5, pad=6.5)
    ax[0].tick_params(axis="y", direction="in", length=5, pad=6.5)
    ax[1].set_xlabel("Latent Variable 1")
    ax[1].set_xlabel("Latent Variable 2")
    ax[1].set_xlim([-1.5, 2.0])
    ax[1].set_ylim([-2.5, 2.5])
    ax[1].annotate(
        "Validation Latent Space",
        xy=(0.03, 0.94),
        xycoords="axes fraction",
        fontsize=20,
        weight="medium",
    )
    ax[1].tick_params(axis="x", direction="in", length=5, pad=6.5)
    ax[1].tick_params(axis="y", direction="in", length=5, pad=6.5)
    plt.tight_layout()


# %%

class VAE(nn.Module):
    def __init__(self, input_dim=10, latent_dim=2, hidden_layer_sizes=(256, 64, 16), use_tanh=False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hls = hidden_layer_sizes
        Act = nn.Tanh if use_tanh else lambda: nn.LeakyReLU(0.02)

        def block(in_c, out_c):
            return [nn.Linear(in_c, out_c), nn.LayerNorm(out_c), Act()]

        # Encoder trunk
        enc = block(input_dim, self.hls[0])
        for i in range(len(self.hls) - 1):
            enc += block(self.hls[i], self.hls[i+1])
        self.encoder = nn.Sequential(*enc)

        # Heads for mean & log-variance of q(z|x)
        self.fc_mu = nn.Linear(self.hls[-1], latent_dim)
        self.fc_logvar = nn.Linear(self.hls[-1], latent_dim)

        # Decoder
        dec = block(latent_dim, self.hls[-1])
        for i in range(len(self.hls)-1, 0, -1):
            dec += block(self.hls[i], self.hls[i-1])
        dec += [nn.Linear(self.hls[0], input_dim)]
        self.decoder = nn.Sequential(*dec)

        self.apply(weights_init)

    # ---- encoder helpers
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # during eval you can choose to return mu only
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---- decoder helpers
    def decode(self, z):
        return self.decoder(z)

    # ---- convenience
    def forward(self, x, sample_latent=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    # used by your getLatent() to export deterministic embeddings
    def encoded(self, x):
        mu, _ = self.encode(x)
        return mu

# %%

def kl_capacity(epoch, start=0, end=200, C_max=2.0):
    # grow target KL capacity C(t) from 0 to C_max nats over 'end' epochs
    if epoch < start: return 0.0
    e = min(1.0, (epoch - start) / max(1, end - start))
    return C_max * e

def beta_sigmoid(epoch, mid=100, sharp=10, beta_max=1.0):
    # smooth 0->1 sigmoid
    import math
    s = 1 / (1 + math.exp(-(epoch - mid)/sharp))
    return beta_max * s

def vae_loss(x_hat, x, mu, logvar, beta, recon='mse', C_t=0.0):
    rec = (nn.functional.mse_loss if recon=='mse' else nn.functional.l1_loss)(x_hat, x, reduction='mean')
    logvar = logvar.clamp(-15.0, 15.0)
    kl_pd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # [B,D]
    kl = kl_pd.sum(dim=1).mean()  # sum over dims, mean over batch (nats)
    # capacity loss: penalize only the excess above C_t
    kl_term = beta * torch.relu(kl - C_t)
    total = rec + kl_term
    return total, rec, kl

def train_vae(model, optimizer, train_loader, valid_loader, n_epoch, beta=1.0,
              patience=20, min_delta=5e-5, recon='mse', kl_anneal=None,
              earlystop_on='auto', start_es_beta=1.0, start_es_epoch=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def pick_metric(beta_t, val_rec, val_total):
        if earlystop_on == 'recon':
            return val_rec
        if earlystop_on == 'total':
            return val_total
        return val_rec if beta_t < 1.0 else val_total  # auto

    best_valid, patience_ctr = float('inf'), 0
    avg_train, avg_valid = [], []

    for epoch in range(n_epoch):
        model.train()
        beta_t = kl_anneal(epoch) if kl_anneal else beta
        C_t = kl_capacity(epoch, start=20, end=200, C_max=5.0)  # tune C_max ~ latent_dim * 0.5–3.0

        tr_rec, tr_kl, tr_tot = [], [], []
        for x in train_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x, sample_latent=True)
            total, rec, kl = vae_loss(x_hat, x, mu, logvar, beta=beta_t, recon=recon, C_t=C_t)
            optimizer.zero_grad(); total.backward(); optimizer.step()
            tr_rec.append(rec.item()); tr_kl.append(kl.item()); tr_tot.append(total.item())

        model.eval()
        va_rec, va_kl, va_tot = [], [], []
        with torch.no_grad():
            for x in valid_loader:
                x = x.to(device)
                # use deterministic latents for validation if you want
                x_hat, mu, logvar = model(x, sample_latent=False)
                total, rec, kl = vae_loss(x_hat, x, mu, logvar, beta=beta_t, recon=recon)
                va_rec.append(rec.item()); va_kl.append(kl.item()); va_tot.append(total.item())

        tr_rec_m, tr_kl_m, tr_tot_m = np.mean(tr_rec), np.mean(tr_kl), np.mean(tr_tot)
        va_rec_m, va_kl_m, va_tot_m = np.mean(va_rec), np.mean(va_kl), np.mean(va_tot)
        val_metric = pick_metric(beta_t, va_rec_m, va_tot_m)

        print(f"[{epoch+1:03}/{n_epoch:03}] β={beta_t:.3f} | "
              f"train rec:{tr_rec_m:.4f} kl:{tr_kl_m:.4f} tot:{tr_tot_m:.4f} || "
              f"valid rec:{va_rec_m:.4f} kl:{va_kl_m:.4f} tot:{va_tot_m:.4f}")

        avg_train.append(tr_tot_m); avg_valid.append(va_tot_m)

        es_enabled = (beta_t >= start_es_beta) and (epoch+1 >= start_es_epoch)
        if es_enabled:
            if best_valid - val_metric > min_delta:
                best_valid, patience_ctr = val_metric, 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"Early stop on metric={val_metric:.4f} (patience {patience}).")
                    break
        else:
            best_valid, patience_ctr = float('inf'), 0

    return avg_train, avg_valid
