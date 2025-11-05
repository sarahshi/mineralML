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

from .core import *
from .stoichiometry import *
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
    filepath = os.path.join(current_dir, "mineral_classes_nn_v0019.npz")

    with np.load(filepath, allow_pickle=True) as data:
        min_cat = data["classes"].tolist()
    mapping = {code: cat for code, cat in enumerate(min_cat)}

    return min_cat, mapping


def prep_df_nn(df):
    """
    Prepares a DataFrame for analysis by performing data cleaning specific to mineralogical data.
    It handles missing values and ensures the presence of required oxide columns.
    The function defines a list of oxide column names and drops rows where the specified oxides
    have fewer than three non-NaN values.

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
    oxides_plus_zr = oxides + ["ZrO2"]

    sample_cols = ["SampleID", "Sample", "Sample Name"]
    present_sample_cols = [c for c in sample_cols if c in df.columns]
    sample_col = present_sample_cols[0] if present_sample_cols else None

    # ensure required columns exist
    for col in oxides_plus_zr + ["Mineral"] + present_sample_cols:
        if col not in df.columns:
            df[col] = np.nan
            warnings.warn(
                f"The column '{col}' was missing and has been filled with NaN.",
                UserWarning,
                stacklevel=2,
            )

    # Convert columns to numeric, coercing errors to NaN
    df[oxides_plus_zr] = df[oxides_plus_zr].apply(pd.to_numeric, errors="coerce")
    if df[oxides_plus_zr].isnull().any().any():
        warnings.warn(
            "Some non-numeric values were found in the oxides columns and have been coerced to NaN.",
            UserWarning,
            stacklevel=2,
        )

    # Drop rows with fewer than 3 non-NaN values in the oxides columns
    df.dropna(subset=oxides, thresh=3, inplace=True)

    # Fill remaining NaN values with 0 for oxides, keep NaN for 'Mineral'
    df.loc[:, oxides_plus_zr] = df.loc[:, oxides_plus_zr].fillna(0)

    # Ensure only oxides, 'Mineral', and 'SampleID' columns are kept
    keep_cols = oxides_plus_zr + ["Mineral"] + present_sample_cols
    df = df.loc[:, keep_cols]

    # if sample_col:
    #     df.set_index(sample_col, inplace=True)

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
    # mean, std = load_scaler("scaler_nn_v0013.npz")
    mean, std = load_scaler("scaler_nn_v0019.npz")

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

    Groups to 2000 total:
    - Pyroxene group (clinopyroxene + orthopyroxene -> 'pyroxene'), kmeans for representative sampling
    - Feldspar group (plagioclase + k-feldspar -> 'feldspar'), kmeans for representative sampling
    - Olivine, kmeans for representative sampling
    - Amphibole, kmeans for representative sampling to capture tremolite and actinolite
    - Rhombohedral oxide group (hematite + ilmenite -> 'rhombohedral oxide')
    - Spinel group (magnetite + spinel -> 'spinel')
    - Glass (separate group with 2000 samples), TAS stratified sampling
    - All other classes get standard n samples (default 1000). If count <1250, shuffle+oversample. 

    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    olivine_classes = ['Olivine']
    pyroxene_classes = ['Clinopyroxene', 'Orthopyroxene']
    feldspar_classes = ['Plagioclase', 'KFeldspar']
    rhombohedral_oxide_classes = ['Hematite', 'Ilmenite']
    spinel_classes = ['Magnetite', 'Spinel']
    amphibole_class = ['Amphibole']
    glass_class = ['Glass']
    lower_threshold = 1250
    random_seed = 42

    oxides = OXIDES
    oxides_plus_zr = oxides + ["ZrO2"]

    sample_cols = ["SampleID", "Sample", "Sample Name"]
    present_sample_cols = [c for c in sample_cols if c in df.columns]

    # ensure required columns exist
    for col in oxides_plus_zr + ["Mineral"] + present_sample_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Helpers
    def kmeans_multi_sample(member_df,
                            n_target,
                            n_clusters=5,
                            min_per_cluster=1):
        """Cluster into 10 groups and sample multiple per cluster to total n_target."""
        n_rows = len(member_df)
        if n_rows == 0:
            return member_df.iloc[[]].copy()
        if n_rows <= n_target:
            return member_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        C = int(max(2, min(n_clusters, n_rows, n_target)))
        X = member_df[oxides_plus_zr].fillna(0.0).to_numpy()
        Xs = StandardScaler().fit_transform(X)

        km = KMeans(n_clusters=C, random_state=random_seed, n_init=10)
        labels = km.fit_predict(Xs)

        tmp = member_df.copy()
        tmp["_cluster"] = labels

        # cluster sizes
        sizes = tmp["_cluster"].value_counts().sort_index().to_numpy()

        # start uniform then distribute remainder to the largest clusters
        base = n_target // C
        alloc = np.full(C, base, dtype=int)
        remainder = n_target - alloc.sum()
        if remainder > 0:
            give = np.argsort(-sizes)[:remainder]
            alloc[give] += 1

        # enforce floor
        alloc = np.maximum(alloc, min_per_cluster)
        # trim if we overshot
        over = alloc.sum() - n_target
        if over > 0:
            order = np.argsort(-(alloc - min_per_cluster))
            for idx in order:
                if over <= 0:
                    break
                can_trim = alloc[idx] - min_per_cluster
                if can_trim > 0:
                    t = min(can_trim, over)
                    alloc[idx] -= t
                    over -= t

        parts = []
        for c_idx, n_c in enumerate(alloc):
            g = tmp[tmp["_cluster"] == c_idx]
            replace = len(g) < n_c
            parts.append(g.sample(n=n_c, replace=replace, random_state=random_seed))

        out = (pd.concat(parts, ignore_index=True)
                 .drop(columns=["_cluster"])
                 .sample(frac=1, random_state=random_seed)
                 .reset_index(drop=True))
        return out

    def random_cap(member_df, n_target):
        """Random cap to n_target (no replacement if enough)."""
        if len(member_df) <= n_target:
            return member_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        return member_df.sample(n=n_target, replace=False, random_state=random_seed).reset_index(drop=True)

    def shuffle_oversample_to(member_df, n_target):
        """Shuffle, then oversample with replacement up to n_target."""
        if len(member_df) == 0:
            return member_df.iloc[[]].copy()
        if len(member_df) >= n_target:
            return random_cap(member_df, n_target)
        # oversample with replacement
        return (member_df.sample(frac=1, random_state=random_seed)
                          .sample(n=n_target, replace=True, random_state=random_seed)
                          .reset_index(drop=True))

    def process_group_per_member(group_classes, sampler_fn, n_per_member, relabel_as):
        """Apply sampler per member class to n_per_member, then relabel to group name."""
        present = [c for c in group_classes if c in df["Mineral"].unique()]
        if not present:
            return pd.DataFrame(columns=OXIDES + ["Mineral"])
        pieces = []
        for m in present:
            sub = df[df["Mineral"] == m]
            pieces.append(sampler_fn(sub, n_per_member))
        out = pd.concat(pieces, ignore_index=True)
        out["Mineral"] = relabel_as
        return out

    # Build groups
    # Olivine: kmeans to n
    oli_df = pd.DataFrame(columns=OXIDES + ["Mineral"])
    if 'Olivine' in df["Mineral"].unique():
        oli_df = kmeans_multi_sample(df[df.Mineral == 'Olivine'], n_target=n,
                                     n_clusters=2)
        oli_df["Mineral"] = 'Olivine'

    amph_df = pd.DataFrame(columns=OXIDES + ["Mineral"])
    if 'Amphibole' in df["Mineral"].unique():
        amph_df = kmeans_multi_sample(df[df.Mineral == 'Amphibole'], n_target=n)
        amph_df["Mineral"] = 'Amphibole'

    # Pyroxene: Cpx kmeans->n + Opx kmeans->n  => total 2n
    pyroxene_df = process_group_per_member(pyroxene_classes, 
                                           lambda d, k: kmeans_multi_sample(d, n_target=k),
                                           n_per_member=n, relabel_as='Pyroxene')

    # Feldspar: Plag kmeans->n + KFspar kmeans->n => total 2n
    feldspar_df = process_group_per_member(feldspar_classes,
                                           lambda d, k: kmeans_multi_sample(d, n_target=k),
                                           n_per_member=n, relabel_as='Feldspar')

    # Rhombohedral oxides: per member random->n (=> 2n total)
    rhombohedral_oxide_df = process_group_per_member(rhombohedral_oxide_classes,
                                                     lambda d, k: random_cap(d, k),
                                                     n_per_member=n, relabel_as='Rhombohedral_Oxides')

    # Spinels: per member random->n (=> 2n total)
    spinel_df = process_group_per_member(spinel_classes,
                                         lambda d, k: random_cap(d, k),
                                         n_per_member=n, relabel_as='Spinels')

    # Glass: TAS-stratified to 2n
    from pyrolite.util.classification import TAS
    gl_df = df[df.Mineral == 'Glass']
    if len(gl_df):
        gl_df = gl_df[gl_df.SiO2 > 40].copy()
        cm = TAS()
        gl_df['Na2O + K2O'] = gl_df['Na2O'] + gl_df['K2O']
        gl_df['TAS'] = cm.predict(gl_df)
        min_per = max(1, (2*n) // gl_df['TAS'].nunique())
        resampled = (
            gl_df
            .groupby('TAS', group_keys=False)
            .apply(lambda x: x.sample(
                n=max(min_per, int(2*n * len(x) / len(gl_df))),
                replace=True, random_state=random_seed))
            .sample(n=2*n, random_state=random_seed)
            .reset_index(drop=True)
        )
        resampled = resampled.drop(columns=['Na2O + K2O', 'TAS'])
        resampled['Mineral'] = 'Glass'
        glass_df = resampled
    else:
        glass_df = pd.DataFrame(columns=OXIDES + ['Mineral'])

    # Other classes: if <1250 -> shuffle+oversample to n; else cap to n
    special_flat = set(olivine_classes + pyroxene_classes + feldspar_classes +
                       rhombohedral_oxide_classes + spinel_classes + amphibole_class + glass_class)
    other_classes = [c for c in df["Mineral"].unique() if c not in special_flat]

    other_dfs = []
    for cls in other_classes:
        grp = df[df.Mineral == cls]
        if len(grp) < lower_threshold:
            other_dfs.append(shuffle_oversample_to(grp, n))
        else:
            other_dfs.append(random_cap(grp, n))
    other_df = pd.concat(other_dfs, ignore_index=True) if other_dfs else pd.DataFrame(columns=OXIDES+["Mineral"])

    df_balanced = pd.concat(
        [oli_df, pyroxene_df, feldspar_df, rhombohedral_oxide_df, spinel_df, amph_df, glass_df, other_df],
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
        input_dim (int): Dimensionality of the input features. Defaults to 11.
        classes (int): The number of output classes for classification. Defaults to 24.
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
        input_dim=11,
        classes=23, #24,
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


def predict_class_prob_nn_train(model, input_data, n_iterations=100):
    """

    Computes the predicted class probabilities for the given input data using the model by
    performing multiple forward passes. The function operates in evaluation mode and does not
    track gradients. It returns the mean and standard deviation of the softmax probabilities
    across all iterations, providing a measure of model uncertainty.

    Parameters:
        model (nn.Module): The model to be used for prediction, which should already be trained.
        input_data (Tensor): The input data to be passed to the model for prediction.
        n_iterations (int): The number of forward passes to perform for prediction. Defaults to 100.

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


def predict_class_prob_nn(df, n_iterations=50):
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
    
    # Create output DataFrame with original indices and structure
    result_df = df.copy()
    pred_cols = ["Predict_Mineral", "Predict_Probability", 
                 "Second_Predict_Mineral", "Second_Predict_Probability"]
    
    for col in pred_cols:
        result_df[col] = np.nan
    
    # Initialize with proper dtypes
    result_df["Predict_Mineral"] = pd.Series(index=df.index, dtype="object")
    result_df["Second_Predict_Mineral"] = pd.Series(index=df.index, dtype="object")
    result_df["Predict_Probability"] = pd.Series(index=df.index, dtype="float64")
    result_df["Second_Predict_Probability"] = pd.Series(index=df.index, dtype="float64")

    # Identify and classify zircons
    zircon_mask = (df['ZrO2'] > 50) if 'ZrO2' in df.columns else pd.Series(False, index=df.index)
    result_df.loc[zircon_mask, "Predict_Mineral"] = "Zircon"
    result_df.loc[zircon_mask, "Predict_Probability"] = 1.0
    result_df.loc[zircon_mask, "Second_Predict_Mineral"] = np.nan
    result_df.loc[zircon_mask, "Second_Predict_Probability"] = np.nan

    # Process non-zircon samples
    non_zircon_mask = ~zircon_mask
    probability_matrix = np.array([])

    if non_zircon_mask.any():
        non_za_df = df[non_zircon_mask]
        
        # Neural network setup — model and device
        oxides = OXIDES
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = MultiClassClassifier(
            input_dim=len(oxides),
            dropout_rate=0.1,
            hidden_layer_sizes=[64, 32, 16]
        ).to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, weight_decay=1e-3)
        # model_path = os.path.join(os.path.dirname(__file__), "nn_best_model_v0013.pt")
        model_path = os.path.join(os.path.dirname(__file__), "nn_best_model_v0019.pt")
        load_model(model, optimizer, model_path)
        
        # Normalize data
        norm_wt = norm_data_nn(non_za_df).astype(np.float32, copy=False)
        input_data = torch.Tensor(norm_wt).to(device)

        # print("=== PREDICTION FUNCTION DEBUG ===")
        # print(f"Model classes: {model.classes}")
        # print(f"Input data shape: {input_data.shape}")
        # print(f"Probability matrix shape: {probability_matrix.shape}")

        # # Check if the mapping matches
        # min_cat, mapping = load_minclass_nn()
        # print(f"Loaded classes: {len(min_cat)}")

        # print("=== NORMALIZATION CHECK ===")
        # print(f"Input data stats - min: {input_data.min()}, max: {input_data.max()}")
        # print(f"Input data mean: {input_data.mean()}, std: {input_data.std()}")

        # # Compare with what was used in training
        # print("Expected range should be roughly: mean ~0, std ~1 (StandardScaler output)")


        model.eval()
        with torch.inference_mode():
            device = next(model.parameters()).device
            N = len(input_data)
            C = model.classes
            BATCH = 2**13 # bump this as high as memory allows
            K = 8 # MC passes per batch chunk (grouped to cut Python overhead)

            probs_mean = torch.empty((N, C), device=device, dtype=torch.float32)

            for start in range(0, N, BATCH):
                end = min(start + BATCH, N)
                x = input_data[start:end] 
                b = x.shape[0]

                done = 0
                acc = torch.zeros((b, C), device=device, dtype=torch.float32)

                while done < n_iterations:
                    kk = min(K, n_iterations - done)

                    # run kk independent forwards and average on the device
                    outs = []
                    for _ in range(kk):
                        logits = model(x)
                        outs.append(torch.softmax(logits, dim=1))
                    acc += torch.stack(outs, dim=0).mean(dim=0) * kk
                    done += kk

                probs_mean[start:end] = acc / float(n_iterations)

            probability_matrix = probs_mean.detach().cpu().numpy()

        # Get top predictions efficiently
        top_two_indices = np.argsort(probability_matrix, axis=1)[:, -2:]
        first_probs = probability_matrix[np.arange(len(probability_matrix)), top_two_indices[:, 1]]
        second_probs = probability_matrix[np.arange(len(probability_matrix)), top_two_indices[:, 0]]
        first_mins = class2mineral_nn(top_two_indices[:, 1])
        second_mins = class2mineral_nn(top_two_indices[:, 0])

        # Update result dataframe
        result_df.loc[non_zircon_mask, "Predict_Mineral"] = first_mins
        result_df.loc[non_zircon_mask, "Predict_Probability"] = first_probs
        result_df.loc[non_zircon_mask, "Second_Predict_Mineral"] = second_mins
        result_df.loc[non_zircon_mask, "Second_Predict_Probability"] = second_probs

    # Process specialized classifiers (unchanged, but could also be optimized)
    oxide_cols = [c for c in result_df.columns if c in OXIDES]
    mineral_col = "Predict_Mineral" if "Predict_Mineral" in result_df.columns else None
    cols = oxide_cols + ([mineral_col] if mineral_col else [])

    def _merge_subclass(mask, Classifier, want_sub=True):
        if not mask.any():
            return
        sub = result_df.loc[mask, cols]
        clf = Classifier(sub)
        # Ensure subclass=True to get submineral back by default
        out = clf.classify(subclass=want_sub)
        # Expect out to have "Mineral" and (if want_sub) "Submineral"
        if "Mineral" in out.columns:
            result_df.loc[mask, "Predict_Mineral"] = out["Mineral"].values
        if want_sub and "Submineral" in out.columns:
            result_df.loc[mask, "Submineral"] = out["Submineral"].values

    # Pyroxene classification
    px_mask = result_df["Predict_Mineral"] == "Pyroxene"
    _merge_subclass(px_mask, PyroxeneClassifier, want_sub=True)

    # Feldspar classification
    fspar_mask = result_df["Predict_Mineral"] == "Feldspar"
    _merge_subclass(fspar_mask, FeldsparClassifier, want_sub=True)
    
    # Oxide classification
    ox_mask = result_df["Predict_Mineral"].isin(["Rhombohedral_Oxides", "Spinels"])
    _merge_subclass(ox_mask, OxideClassifier, want_sub=True)

    if "Submineral" not in result_df.columns:
        result_df["Submineral"] = pd.Series(index=result_df.index, dtype="object")

    cols = list(result_df.columns)
    if "Predict_Mineral" in cols and "Submineral" in cols:
        # remove and re-insert Submineral right after Predict_Mineral
        cols.remove("Submineral")
        insert_at = cols.index("Predict_Mineral") + 1
        cols.insert(insert_at, "Submineral")
        result_df = result_df[cols]

    return result_df, probability_matrix


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
    # print("Sample predictions:", pred_class[:10])
    # print("Mapping used:", valid_mapping)
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
        "Clinopyroxene",
        "Epidote",
        "Garnet",
        "Glass",
        "KFeldspar",
        "Kalsilite",
        "Leucite",
        "Melilite",
        "Muscovite",
        "Nepheline",
        "Olivine",
        'Orthopyroxene',
        "Plagioclase",
        "Quartz",
        "Rhombohedral_Oxides",
        "Rutile",
        "Serpentine",
        "Spinels",
        "Titanite",
        "Tourmaline",
        "Zircon",
    ]

    given = pd.Series(given_min)
    pred = pd.Series(pred_min)

    # case-insensitive group merges
    def _merge_magnetite_to_spinels(x):
        if pd.isna(x):
            return x
        s = str(x).strip().lower()
        if s in {"magnetite",}:
            return "Spinels"
        return x
    def _merge_ulvospinel_to_spinels(x):
        if pd.isna(x):
            return x
        s = str(x).strip().lower()
        if s in {"ulvöspinel",}:
            return "Spinels"
        return x
    def _merge_hematite_to_rhomb_oxide(x):
        if pd.isna(x):
            return x
        s = str(x).strip().lower()
        if s in {"hematite",}:
            return "Rhombohedral_Oxides"
        return x
    given_min_merged = given.map(_merge_magnetite_to_spinels)
    def _merge_ilmenite_to_spinels(x):
        if pd.isna(x):
            return x
        s = str(x).strip().lower()
        if s in {"ilmenite",}:
            return "Rhombohedral_Oxides"
        return x

    given_min_merged = given.map(_merge_magnetite_to_spinels)
    given_min_merged = given_min_merged.map(_merge_ulvospinel_to_spinels)
    given_min_merged = given_min_merged.map(_merge_hematite_to_rhomb_oxide)
    given_min_merged = given_min_merged.map(_merge_ilmenite_to_spinels)
    pred_min_merged = pred.map(_merge_magnetite_to_spinels)
    pred_min_merged = pred_min_merged.map(_merge_ulvospinel_to_spinels)
    pred_min_merged = pred_min_merged.map(_merge_hematite_to_rhomb_oxide)
    pred_min_merged = pred_min_merged.map(_merge_ilmenite_to_spinels)

    # Create a confusion matrix with labels as all possible minerals
    cm_matrix = confusion_matrix(given_min_merged, pred_min_merged, labels=minerals)

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

    # Split the dataset into train and test sets
    train_df, valid_df = train_test_split(df, test_size=n, random_state=42)
    if balanced == True:
        train_df = balance(train_df, n=1000)
    train_df.to_csv('train_df_nn.csv', index=False)

    for _df in (train_df, valid_df):
        _df['Mineral'] = (
            _df['Mineral'].astype(str)
            .replace(['Clinopyroxene', 'Orthopyroxene'], 'Pyroxene')
            .replace(['Plagioclase', 'KFeldspar'], 'Feldspar')
            .replace(['Hematite', 'Ilmenite'], 'Rhombohedral_Oxides')
            .replace(['Magnetite', 'Spinel'], 'Spinels')
    )
    
    train_df_nozirc = train_df[train_df['Mineral'] != 'Zircon'].copy()
    valid_df_nozirc = valid_df[valid_df['Mineral'] != 'Zircon'].copy()

    # print("=== TRAINING DATA CLASS DISTRIBUTION ===")
    # class_dist = train_df_nozirc["Mineral"].value_counts()
    # print(class_dist)
    # print(f"Total classes: {len(class_dist)}")
            
    all_cats = pd.Categorical(train_df_nozirc["Mineral"])
    mapping = dict(enumerate(all_cats.categories))

    inv_mapping = {cat: idx for idx, cat in mapping.items()}
    sort_mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))
    # min_cat = list(all_cats.categories)
    # np.savez(os.path.join(path_beg, "parametermatrix_neuralnetwork", "mineral_classes_nn_v001.npz"),
    #          classes=np.array(min_cat, dtype=object))
    # print("Saved class list:", min_cat)

    train_df_nozirc["_code"] = train_df_nozirc["Mineral"].map(inv_mapping).astype(int)
    valid_df_nozirc["_code"] = valid_df_nozirc["Mineral"].map(inv_mapping).astype(int)

    # scale
    ss = StandardScaler().fit(train_df_nozirc[OXIDES].fillna(0))
    train_x = ss.transform(train_df_nozirc[OXIDES].fillna(0))
    valid_x = ss.transform(valid_df_nozirc[OXIDES].fillna(0))
    scaler_path = os.path.join(path_beg, "parametermatrix_neuralnetwork", "scaler_nn_v0019.npz")
    np.savez(scaler_path,
             mean=pd.Series(ss.mean_, index=OXIDES),
             scale=pd.Series(np.sqrt(ss.var_), index=OXIDES))

    # encode labels
    train_y = train_df_nozirc["_code"].to_numpy()
    valid_y = valid_df_nozirc["_code"].to_numpy()

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


def export_predictions_to_excel(results_df, filename="prediction_results.xlsx"):
    """
    Export prediction results to an Excel workbook with one sheet called "All"
    containing all rows, and additional sheets for each predicted mineral.

    Parameters:
        results_df (pd.DataFrame): The results DataFrame returned by predict_class_prob_nn.
        filename (str): The name of the Excel file to write.

    Returns:
        str: Path to the saved Excel file.
    """
    # check if Predict_Mineral column exists
    if "Predict_Mineral" not in results_df.columns:
        raise ValueError("results_df must contain a 'Predict_Mineral' column")

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Write all results
        results_df.to_excel(writer, sheet_name="All", index=False)

        # write separate sheets for each mineral
        for mineral, group in results_df.groupby("Predict_Mineral"):
            sheet_name = str(mineral)[:31].replace("/", "-").replace("\\", "-")
            group.to_excel(writer, sheet_name=sheet_name, index=False)

    return filename


# %%