# %%

import os
import math
import time
import copy
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from mineralML.core import *
from .constants import OXIDES


# %%


def load_minclass_nn():
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
    filepath = os.path.join(current_dir, "mineral_classes_nn.npz")

    with np.load(filepath, allow_pickle=True) as data:
        min_cat = data["classes"].tolist()
    mapping = {code: cat for code, cat in enumerate(min_cat)}

    return min_cat, mapping


def prep_df_nn(df):
    """
    Prepares a DataFrame for analysis by performing data cleaning specific to mineralogical data.
    It handles missing values and ensures the presence of required oxide columns.
    The function defines a list of oxide column names and drops rows where the specified oxides
    have fewer than six non-NaN values.

    Parameters:
        df (DataFrame): The input DataFrame containing mineral composition data along with 'Mineral' column.

    Returns:
        df (DataFrame): The cleaned DataFrame with 'NaN' filled with zero for oxides.
    """

    if "FeO" in df.columns:
        if "FeOt" not in df.columns:
            raise ValueError(
                "No 'FeOt' column found. You have a 'FeO' column. "
                "mineralML only recognizes 'FeOt' as a column. Please convert to FeOt."
            )
    if "Fe2O3" in df.columns:
        if "FeOt" not in df.columns:
            raise ValueError(
                "No 'FeOt' column found. You have a 'Fe2O3' column. "
                "mineralML only recognizes 'FeOt' as a column. Please convert to FeOt."
            )

    oxides = OXIDES

    # Ensure all required columns are present in the DataFrame
    for col in oxides + ["Mineral", "SampleID"]:
        if col not in df.columns:
            df[col] = np.nan
            warnings.warn(
                f"The column '{col}' was missing and has been filled with NaN.",
                UserWarning,
                stacklevel=2,
            )

    # Convert columns to numeric, coercing errors to NaN
    df[oxides] = df[oxides].apply(pd.to_numeric, errors='coerce')
    
    # Warn if any non-numeric values were coerced to NaN
    if df[oxides].isnull().any().any():
        warnings.warn(
            "Some non-numeric values were found in the oxides columns and have been coerced to NaN.",
            UserWarning,
            stacklevel=2,
        )

    # Drop rows with fewer than 3 non-NaN values in the oxides columns
    df.dropna(subset=oxides, thresh=3, inplace=True)

    # Fill remaining NaN values with 0 for oxides, keep NaN for 'Mineral'
    df.loc[:, oxides] = df.loc[:, oxides].fillna(0)

    # Ensure only oxides, 'Mineral', and 'SampleID' columns are kept
    df = df.loc[:, oxides + ["Mineral", "SampleID"]]

    # Ensure SampleID is the index
    df.set_index("SampleID", inplace=True)

    return df


def norm_data_nn(df):
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

    oxides = OXIDES
    
    mean, std = load_scaler("scaler_nn.npz")

    # Ensure that mean and std are Series objects with indices matching the columns
    if not isinstance(mean, pd.Series) or not isinstance(std, pd.Series):
        raise ValueError("mean and std should be Series")

    for col in oxides:
        if col not in mean.index or col not in std.index:
            raise ValueError(f"Missing mean or std for column: {col}")

    df = df.reset_index(drop=False)
    scaled_df = df[oxides].copy()

    # scaled_df = df[oxides].reset_index(drop=True).copy()

    if df[oxides].isnull().any().any():
        df = prep_df_nn(df)
    else:
        df = df

    for col in df[oxides].columns:
        scaled_df[col] = (df[col] - mean[col]) / std[col]

    array_x = scaled_df.to_numpy()

    return array_x


def balance(df, n=1000):
    """

    Balances the training dataset with special handling for:
    - Pyroxene group (clinopyroxene + orthopyroxene -> 'pyroxene')
    - Feldspar group (plagioclase + k-feldspar -> 'feldspar')
    - Glass (separate group with 2000 samples)
    - All other classes get standard n samples (default 1000)
    
    Uses RandomOverSampler for minority classes before grouping.

    Parameters:
        train_x (numpy.ndarray): The feature matrix for the training data.
        train_y (numpy.ndarray): The corresponding label vector for the training data.
        n (int): Number of samples to take from each standard class (default: 1000)

    Returns:
        train_x (numpy.ndarray): The feature matrix after oversampling and group combination.
        train_y (numpy.ndarray): The label vector after oversampling and group combination.

    """
    from pyrolite.util.classification import TAS

    try:
        from imblearn.over_sampling import RandomOverSampler
    except ImportError:
        raise RuntimeError(
            "You have not installed imblearn, which is required to balance the datasets used for training the neural networks. "
            "If you use conda, run conda install -c conda-forge imbalanced-learn. If you use pip, run pip install -U imbalanced-learn."
        )

    # Resample the dataset
    oversample = RandomOverSampler(sampling_strategy="minority", random_state=42)
    # x_balanced, y_balanced = oversample.fit_resample(train_x, train_y)
    Xo, yo = oversample.fit_resample(df[OXIDES], df["Mineral"])
    df = pd.DataFrame(Xo, columns=OXIDES)
    df["Mineral"] = yo

    # Define special groups
    pyroxene_classes = ['Clinopyroxene', 'Orthopyroxene']
    feldspar_classes = ['Plagioclase', 'KFeldspar']
    glass_class = ['Glass']
    
    # Get other classes (not in special groups)
    other_classes = [
        c for c in df["Mineral"].unique() 
        if c not in pyroxene_classes + feldspar_classes + glass_class
    ]

    def process_group(group_classes, group_name, samples_per_member):
        present = [c for c in group_classes if c in df["Mineral"].unique()]
        if not present:
            return pd.DataFrame(columns=OXIDES + ["Mineral"])
        frames = []
        for member in present:
            member_df = df[df["Mineral"] == member]
            replace = len(member_df) < samples_per_member
            frames.append(
                member_df.sample(
                    n=samples_per_member,
                    replace=replace,
                    random_state=42
                )
            )
        out = pd.concat(frames, ignore_index=True)
        out["Mineral"] = group_name
        return out

    # Process special groups
    pyroxene_df = process_group(pyroxene_classes, 'Pyroxene', n)  # 1000 each
    feldspar_df = process_group(feldspar_classes, 'Feldspar', n)  # 1000 each
    
    # Process glass (2000 samples)
    gl_df = df[df.Mineral == 'Glass']
    if len(gl_df):
        gl_df = gl_df[gl_df.SiO2 > 40].copy()
        from pyrolite.util.classification import TAS
        cm = TAS()
        gl_df['Na2O + K2O'] = gl_df['Na2O'] + gl_df['K2O']
        gl_df['TAS'] = cm.predict(gl_df)
        min_per = max(1, (2*n) // gl_df['TAS'].nunique())
        resampled = (
            gl_df
            .groupby('TAS', group_keys=False)
            .apply(lambda x: x.sample(
                n = max(min_per, int(2*n * len(x) / len(gl_df))),
                replace = True,
                random_state = 42
            ))
            .sample(n=2*n, random_state=42)
            .reset_index(drop=True)
        )
        resampled = resampled.drop(columns=['Na2O + K2O', 'TAS'])
        resampled['Mineral'] = 'Glass'
        glass_df = resampled
    else:
        glass_df = pd.DataFrame(columns=OXIDES + ['Mineral'])

    # Process other classes (n samples each)
    other_dfs = []
    for cls in other_classes:
        grp    = df[df.Mineral == cls]
        replace = len(grp) < n
        other_dfs.append(grp.sample(n=n, replace=replace, random_state=42))
    other_df = pd.concat(other_dfs, ignore_index=True)

    # Combine all groups
    df_balanced = pd.concat(
        [pyroxene_df, feldspar_df, glass_df, other_df],
        ignore_index=True
    )

    return df_balanced


class VariationalLayer(nn.Module):

    """

    The VariationalLayer class implements a Bayesian approach to linear layers
    in neural networks, which allows for the incorporation
    of uncertainty in the weights and biases. This is achieved by modeling the
    parameters as distributions rather than point estimates. The layer utilizes
    variational inference to learn the parameters of these distributions.

    Parameters:
        in_features (int): The number of input features to the layer.
        out_features (int): The number of output features from the layer.

    Attributes:
        weight_mu (Parameter): The mean of the Gaussian distributions of the weights.
        weight_rho (Parameter): The rho parameters (unconstrained) for the standard
                                deviations of the Gaussian distributions of the weights.
        bias_mu (Parameter): The mean of the Gaussian distributions of the biases.
        bias_rho (Parameter): The rho parameters (unconstrained) for the standard
                              deviations of the Gaussian distributions of the biases.
        softplus (nn.Softplus): A Softplus activation function used for ensuring the
                                standard deviation is positive.

    Methods:
        reset_parameters(): Initializes the parameters based on the number of input features.
        forward(input): Performs the forward pass using a sampled weight and bias according
                        to their respective distributions.
        kl_divergence(): Computes the Kullback-Leibler divergence of the layer's
                         parameters, which can be used as a part of the loss function
                         to regulate the learning of the distribution parameters.

    The forward computation of this layer is equivalent to a standard linear layer
    with sampled weights and biases. The KL divergence method returns a value that
    quantifies the difference between the prior and variational distributions of the
    layer's parameters, which encourages the learning of plausible weights and biases
    while controlling complexity.

    """

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
        std = 1.0 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-std, std)
        self.weight_rho.data.uniform_(-std, std)
        self.bias_mu.data.uniform_(-std, std)
        self.bias_rho.data.uniform_(-std, std)

    def forward(self, input):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        weight_epsilon = torch.normal(
            mean=0.0, std=1.0, size=weight_sigma.size(), device=input.device
        )
        bias_epsilon = torch.normal(
            mean=0.0, std=1.0, size=bias_sigma.size(), device=input.device
        )

        weight_sample = self.weight_mu + weight_epsilon * weight_sigma
        bias_sample = self.bias_mu + bias_epsilon * bias_sigma

        output = F.linear(input, weight_sample, bias_sample)
        return output

    def kl_divergence(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        kl_div = -0.5 * torch.sum(
            1
            + torch.log(weight_sigma.pow(2))
            - self.weight_mu.pow(2)
            - weight_sigma.pow(2)
        )
        kl_div += -0.5 * torch.sum(
            1 + torch.log(bias_sigma.pow(2)) - self.bias_mu.pow(2) - bias_sigma.pow(2)
        )

        return kl_div


class MultiClassClassifier(nn.Module):

    """
    A neural network module for multi-class classification tasks. It
    consists of a sequence of layers defined by the input dimensions, number
    of classes, dropout rate, and sizes of hidden layers. It can be
    customized with different numbers and sizes of hidden layers, as well as
    varying dropout rates to prevent overfitting. The final output layer is
    designed for classification among a fixed number of classes.

    Parameters:
        input_dim (int): Dimensionality of the input features. Defaults to 10.
        classes (int): The number of output classes for classification. Defaults to 12.
        dropout_rate (float): The dropout rate applied after each hidden layer. Defaults to 0.1.
        hidden_layer_sizes (list of int): The sizes of each hidden layer. Defaults to a single
                                          hidden layer with 8 units.

    Attributes:
        input_dim (int): Internal storage of the input dimensionality.
        classes (int): Internal storage of the number of classes.
        dropout_rate (float): Internal storage of the dropout rate.
        hls (list of int): Internal storage of the hidden layer sizes.
        encode (nn.Sequential): The sequential container of layers making up the encoder part
                                of the classifier, including linear, batch normalization,
                                leaky ReLU, and dropout layers.

    Methods:
        encoded(x): Encodes input `x` through the sequence of layers defined in `encode`.
        forward(x): Implements the forward pass of the network, returning raw scores for each class.
        predict(x): Provides class predictions for input `x` based on the scores from the forward pass.

    The class utilizes a helper function `element` to create each hidden layer or the variational
    layer if it is the last one. The `weights_init` function is applied to initialize weights
    after the model is constructed.

    """

    def __init__(
        self,
        input_dim=10,
        # classes=18,
        classes=26,
        dropout_rate=0.1,
        hidden_layer_sizes=[64, 32, 16],
    ):
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
                encoder += element(
                    self.input_dim, size, is_last=(i == len(self.hls) - 1)
                )
            else:
                encoder += element(
                    self.hls[i - 1], size, is_last=(i == len(self.hls) - 1)
                )

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


def predict_class_prob_nn_train(model, input_data, n_iterations=500):
    """

    Computes the predicted class probabilities for the given input data using the model by
    performing multiple forward passes. The function operates in evaluation mode and does not
    track gradients. It returns the mean and standard deviation of the softmax probabilities
    across all iterations, providing a measure of model uncertainty.

    Parameters:
        model (nn.Module): The model to be used for prediction, which should already be trained.
        input_data (Tensor): The input data to be passed to the model for prediction.
        n_iterations (int): The number of forward passes to perform for prediction. Defaults to 250.

    Returns:
        prediction_mean (ndarray): The mean class probabilities across all iterations.
        prediction_std (ndarray): The standard deviation of class probabilities, indicating uncertainty.

    """

    model.eval()
    output_list = []
    for i in range(n_iterations):
        with torch.no_grad():
            output = model(input_data)
            output_list.append(
                torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
            )

    output_list = np.array(output_list)

    # Calculate mean and standard deviation
    prediction_mean = output_list.mean(axis=0)
    prediction_std = output_list.std(axis=0)

    return prediction_mean, prediction_std


def predict_class_prob_nn(df, n_iterations=500):
    """

    Predicts the class probabilities, corresponding mineral names, and the maximum
    probability for each class using a predefined MultiClassClassifier model. This
    function loads a pre-trained model and its optimizer state, normalizes input
    data, and performs multiple inference iterations to compute the prediction probabilities.

    Parameters:
        df (DataFrame): The input DataFrame containing the oxide composition data.
        n_iterations (int): The number of inference iterations to average over for predictions.

    Returns:
        df (DataFrame): The input DataFrame with columns predict_mineral (predicted mineral names)
        and predict_prob (maximum probability of predicted class).
        probability_matrix (ndarray): The matrix of class probabilities for each sample.

    """

    lr = 5e-3
    wd = 1e-3
    dr = 0.1
    hls = [64, 32, 16]

    oxides = OXIDES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiClassClassifier(
        input_dim=len(oxides), dropout_rate=dr, hidden_layer_sizes=hls
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "nn_best_model.pt")

    load_model(model, optimizer, model_path)

    norm_wt = norm_data_nn(df)
    input_data = torch.Tensor(norm_wt).to(device)

    model.eval()
    output_list = []
    for i in range(n_iterations):
        with torch.no_grad():
            output = model(input_data)
            output_list.append(
                torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
            )

    output_arr = np.array(output_list)
    probability_matrix = output_arr.mean(axis=0)

    min_cat, _ = load_minclass_nn()
    probability_df = pd.DataFrame(probability_matrix, columns=min_cat)

    top_two_indices = np.argsort(probability_matrix, axis=1)[:, -2:]
    first_predict_prob = probability_matrix[
        np.arange(probability_matrix.shape[0]), top_two_indices[:, 1]
    ]
    second_predict_prob = probability_matrix[
        np.arange(probability_matrix.shape[0]), top_two_indices[:, 0]
    ]
    first_predict_mineral = class2mineral_nn(top_two_indices[:, 1])
    second_predict_mineral = class2mineral_nn(top_two_indices[:, 0])

    df.loc[:, "Predict_Mineral"] = first_predict_mineral
    df.loc[:, "Predict_Probability"] = first_predict_prob
    df.loc[:, "Second_Predict_Mineral"] = second_predict_mineral
    df.loc[:, "Second_Predict_Probability"] = second_predict_prob

    return df, probability_df


def unique_mapping_nn(pred_class):
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

    _, mapping = load_minclass_nn()
    unique = np.unique(pred_class)
    valid_mapping = {key: mapping[key] for key in unique}
    if -1 in unique:
        valid_mapping[-1] = "Unknown"

    return unique, valid_mapping


def class2mineral_nn(pred_class):
    """

    Translates predicted class codes into mineral names using a mapping obtained from the
    unique classes present in the 'pred_class' array. It utilizes the 'unique_mapping_nn'
    function to establish the relevant class-to-mineral name mapping.

    Parameters:
        pred_class (array-like): The array of predicted class codes to be translated into mineral names.

    Returns:
        pred_mineral (ndarray): An array of mineral names corresponding to the predicted class codes.

    """

    _, valid_mapping = unique_mapping_nn(pred_class)
    pred_mineral = np.array([valid_mapping[x] for x in pred_class])

    return pred_mineral


def confusion_matrix_df(given_min, pred_min):
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

    minerals = [
        "Amphibole",
        "Apatite",
        "Biotite",
        "Calcite",
        "Chlorite",
        "Epidote",
        "Feldspar"
        "Garnet",
        "Glass",
        "Hematite",
        "Ilmenite",
        "Kalsilite",
        "Leucite",
        "Magnetite",
        "Melilite",
        "Muscovite",
        "Nepheline",
        "Olivine",
        "Pyroxene",
        "Quartz",
        "Rutile",
        "Serpentine",
        "Spinel",
        "Titanite"
        "Tourmaline",
        "Zircon"
    ]

    # Create a confusion matrix with labels as all possible minerals
    cm_matrix = confusion_matrix(given_min, pred_min, labels=minerals)

    # Create a DataFrame from the confusion matrix
    cm_df = pd.DataFrame(cm_matrix, index=minerals, columns=minerals)

    # Adjust DataFrame to handle missing minerals
    # Ensure all minerals are included as rows and columns, filling missing ones with zeros
    for mineral in minerals:
        if mineral not in cm_df:
            cm_df[mineral] = 0
        if mineral not in cm_df.index:
            cm_df.loc[mineral] = 0

    # Reorder rows and columns based on the predefined minerals list
    cm_df = cm_df.reindex(index=minerals, columns=minerals)

    # cm_matrix = confusion_matrix(given_min, pred_min, labels=minerals)

    # min_cat, _ = load_minclass_nn()
    # cm_df = pd.DataFrame(cm_matrix, index=min_cat, columns=min_cat)

    return cm_df


def train_nn(
    model,
    optimizer,
    train_loader,
    valid_loader,
    n_epoch,
    criterion,
    kl_weight_decay,
    kl_decay_epochs=750,
    patience=50,
):
    """

    Trains a neural network model using the provided data loaders, optimizer, and loss criterion. It incorporates KL divergence
    into the loss to enable learning in a variational framework, with the KL weight increasing each epoch until a maximum value
    is reached. The function includes an early stopping mechanism that terminates training if validation loss does not improve
    for a specified number of consecutive epochs.

    Parameters:
        model (nn.Module): The neural network model to train.
        optimizer (Optimizer): The optimization algorithm used to update model weights.
        train_loader (DataLoader): The DataLoader containing the training data.
        valid_loader (DataLoader): The DataLoader containing the validation data.
        n_epoch (int): The total number of training epochs.
        criterion (Loss): The loss function used for training.
        kl_weight_decay (float): The increment to the KL divergence weight per epoch.
        kl_decay_epochs (int): The number of epochs over which to increment the KL weight. Defaults to 750.
        patience (int): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 50.

    Returns:
        train_output (Tensor): The output from the model for the last training batch.
        valid_output (Tensor): The output from the model for the last validation batch.
        avg_train_loss (list): The list of average training losses per epoch.
        avg_valid_loss (list): The list of average validation losses per epoch.
        best_valid_loss (float): The best validation loss observed during training.
        best_model_state (dict): The state dictionary of the model at the point of the best validation loss.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avg_train_loss = []
    avg_valid_loss = []
    best_valid_loss = float("inf")
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
            kl_div = 0.0
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

        print(
            f"[{epoch+1:03}/{n_epoch:03}] train_loss: {avg_train:.6f}, valid_loss: {avg_valid:.6f}, time: {training_time:.2f} s"
        )

        # Early stopping
        if avg_valid < best_valid_loss:
            best_valid_loss = avg_valid
            patience_counter = 0
            best_model_state = copy.deepcopy(
                model.state_dict()
            )  # Save the best model weights

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Validation loss hasn't improved for {patience} epochs. Stopping early."
                )
                break

    return (
        train_output,
        valid_output,
        avg_train_loss,
        avg_valid_loss,
        best_valid_loss,
        best_model_state,
    )


def neuralnetwork(df, hls_list, kl_weight_decay_list, lr, wd, dr, ep, n, balanced):
    """

    Trains a neural network with various configurations of hidden layer sizes and KL weight
    decay parameters to find the best model for classifying minerals based on their oxide
    composition. It normalizes input data, balances the dataset if required, initializes
    the model and optimizer, and performs training and validation. The best performing
    model's parameters are saved, along with training and validation losses, and prediction
    reports.

    Parameters:
        df (DataFrame): The input DataFrame with mineral composition data and labels.
        hls_list (list of list of int): List of configurations for hidden layer sizes.
        kl_weight_decay_list (list of float): List of KL weight decay values to try during training.
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay factor for regularization.
        dr (float): Dropout rate for the model.
        ep (int): Number of epochs to train.
        n (float): Test size fraction or absolute number for splitting the dataset.
        balanced (bool): Whether to balance the dataset or not.

    Returns:
        best_model_state (dict): The state dictionary of the best performing model.

    """

    path_beg = os.getcwd() + "/"
    output_dir = ["parametermatrix_neuralnetwork"]
    for ii in range(len(output_dir)):
        if not os.path.exists(path_beg + output_dir[ii]):
            os.makedirs(path_beg + output_dir[ii], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    )

    train_df["_code"] = train_df["Mineral"].map(inv_mapping).astype(int)
    valid_df["_code"] = valid_df["Mineral"].map(inv_mapping).astype(int)

    # scale
    ss = StandardScaler().fit(train_df[OXIDES])
    train_x = ss.transform(train_df[OXIDES].fillna(0))
    valid_x = ss.transform(valid_df[OXIDES].fillna(0))

    # encode labels
    train_y = train_df["_code"].to_numpy()
    valid_y = valid_df["_code"].to_numpy()

    # Define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = LabelDataset(train_x, train_y)
    valid_dataset = LabelDataset(valid_x, valid_y)

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
    best_valid_loss = float("inf")

    # Define data loaders
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    np.savez(
        "parametermatrix_neuralnetwork/" + str(lr) + "_" + str(wd) + "_" + str(dr) + "_" + str(ep) + "_best_model_nn_features.npz",
        feature_loader=feature_loader,
        valid_loader=valid_loader,
    )

    train_losses_dict = {}
    valid_losses_dict = {}

    for hls in hls_list:
        for kl_weight_decay in kl_weight_decay_list:
            print(
                f"Training with KL weight decay: {kl_weight_decay} and hidden layer sizes: {hls}"
            )

            # Initialize model
            model = MultiClassClassifier(
                input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=hls
            ).to(device)

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

            # Train model and get the best test loss and model state
            (
                train_output,
                valid_output,
                avg_train_loss,
                avg_valid_loss,
                current_best_valid_loss,
                current_best_model_state,
            ) = train_nn(
                model,
                optimizer,
                feature_loader,
                valid_loader,
                epochs,
                criterion,
                kl_weight_decay=kl_weight_decay,
            )

            if current_best_valid_loss < best_valid_loss:
                best_valid_loss = current_best_valid_loss
                best_kl_weight_decay = kl_weight_decay
                best_hidden_layer_size = hls
                best_model_state = current_best_model_state

            train_losses_dict[(kl_weight_decay, tuple(hls))] = avg_train_loss
            valid_losses_dict[(kl_weight_decay, tuple(hls))] = avg_valid_loss

    # Create a new model with the best model state
    best_model = MultiClassClassifier(
        input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=best_hidden_layer_size
    )
    best_model.load_state_dict(best_model_state)
    best_model.eval()

    # Perform predictions on the test dataset using the best model
    with torch.no_grad():
        valid_predictions = best_model(torch.Tensor(valid_x))
        train_predictions = best_model(torch.Tensor(train_x))
        valid_pred_y = valid_predictions.argmax(dim=1).cpu().numpy()
        train_pred_y = train_predictions.argmax(dim=1).cpu().numpy()

    # Calculate classification metrics for the test dataset
    valid_report = classification_report(
        valid_y,
        valid_pred_y,
        target_names=list(sort_mapping.values()),
        zero_division=0,
        output_dict=True,
    )
    train_report = classification_report(
        train_y,
        train_pred_y,
        target_names=list(sort_mapping.values()),
        zero_division=0,
        output_dict=True,
    )  # output_dict=True

    # Print the best kl_weight_decay value and test report
    print("Best kl_weight_decay:", best_kl_weight_decay)
    print("Best best_hidden_layer_size:", best_hidden_layer_size)

    # Save the best model and other relevant information
    model_path = "parametermatrix_neuralnetwork/" + str(lr) + "_" + str(wd) + "_" + str(dr) + "_" + str(ep) + "_" + str(kl_weight_decay) + "_" + str(hls) + "_best_model.pt"
    save_model_nn(optimizer, best_model_state, model_path)

    train_pred_mean, train_pred_std = predict_class_prob_nn_train(
        model, feature_dataset.x
    )
    valid_pred_mean, valid_pred_std = predict_class_prob_nn_train(
        model, valid_dataset.x
    )

    # Get the most probable classes
    train_pred_y = np.argmax(train_pred_mean, axis=1)
    valid_pred_y = np.argmax(valid_pred_mean, axis=1)

    np.savez(
        "parametermatrix_neuralnetwork/" + str(lr) + "_" + str(wd) + "_" + str(dr) + "_" + str(ep) + "_" + str(kl_weight_decay) + "_" + str(hls) + "_best_model_data.npz",
        best_hidden_layer_size=best_hidden_layer_size,
        best_kl_weight_decay=best_kl_weight_decay,
        valid_report=valid_report,
        train_report=train_report,
        train_y=train_y,
        valid_y=valid_y,
        train_pred_y=train_pred_y,
        valid_pred_y=valid_pred_y,
        train_pred_mean=train_pred_mean,
        train_pred_std=train_pred_std,
        valid_pred_mean=valid_pred_mean,
        valid_pred_std=valid_pred_std,
    )

    # Save the train and test losses
    np.savez(
        "parametermatrix_neuralnetwork/" + str(lr) + "_" + str(wd) + "_" + str(dr) + "_" + str(ep) + "_best_model_losses.npz",
        train_losses=train_losses_dict,
        valid_losses=valid_losses_dict,
    )

    return best_model_state

# %%