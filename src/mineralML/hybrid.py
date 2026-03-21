# %%

__author__ = "Sarah Shi"

import os
import re
import math
import copy
import random

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .core import *
from .core import same_seeds
from .stoichiometry import *
from .constants import OXIDES
from .supervised import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from sklearn.decomposition import PCA


# %%


def load_minclass_nn(minclass_path="mineral_classes_nn_v0030.npz"):
    """
    Loads mineral classes and their corresponding mappings from a .npz file.
    The file is expected to contain an array of class names under the 'classes' key.
    This function creates a dictionary that maps an integer code to each class name.
 
    Parameters:
        minclass_path (str): Filename or relative path (relative to this module).
 
    Returns:
        min_cat (list): A list of mineral class names.
        mapping (dict): A dictionary that maps each integer code to its
            corresponding class name.
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, minclass_path)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Class file not found: {filepath}")

    with np.load(filepath, allow_pickle=True) as data:
        if "classes" not in data:
            raise KeyError(f'"classes" not found in {filepath}. Keys: {list(data.keys())}')
        min_cat = data["classes"].tolist()

    mapping = dict(enumerate(min_cat))
    return min_cat, mapping


def prep_df_nn(df):
    """
    Prepares a DataFrame for analysis by performing data cleaning specific
    to mineralogical data. Handles missing values and ensures the presence
    of required oxide columns. Fills missing oxide values with zero while
    preserving all original columns in the dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing mineral composition data.
            Metadata columns ('Mineral', 'Source', 'SampleID', 'Sample',
            'Sample Name', 'Sample ID') are preserved in the output when present.
 
    Returns:
        df (pd.DataFrame): Cleaned DataFrame with NaN filled with zero for oxides.
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

    sample_cols = ["SampleID", "Sample", "Sample Name", "Sample ID"]
    present_sample_cols = [c for c in sample_cols if c in df.columns]
    
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

    # Fill remaining NaN values with 0 for oxides only
    df.loc[:, oxides_plus_zr] = df.loc[:, oxides_plus_zr].fillna(0)

    # Keep all columns, just reset the index
    df = df.reset_index(drop=True)

    return df


def norm_data_nn(df, scaler_path="scaler_nn_v0030.npz"):
    """
    Normalizes oxide composition data using a predefined StandardScaler.
    Ensures that the DataFrame has been preprocessed before applying the
    transformation.
 
    Parameters:
        df (pd.DataFrame): Input DataFrame containing oxide composition data.
        scaler_path (str): Filename or relative path to the saved scaler .npz file.
 
    Returns:
        array_x (ndarray): Transformed oxide composition data.
    """

    oxides = OXIDES    
    mean, std = load_scaler(scaler_path)

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

    Groups to 1000 total:
    - Garnet group
    - All other classes get standard n samples (default 1000). If count <1250, shuffle+oversample. 

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'Mineral' column and oxide columns.
        n (int): Base target sample count per member class (default 1000).
 
    Returns:
        df_balanced (pd.DataFrame): Resampled DataFrame with balanced class counts.

    """

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    olivine_class = ['Olivine']
    pyroxene_classes = ['Clinopyroxene', 'Orthopyroxene']
    feldspar_classes = ['Plagioclase', 'Alkali_Feldspar']
    rhombohedral_oxide_classes = ['Hematite', 'Ilmenite']
    spinel_classes = ['Magnetite', 'Spinel']
    amphibole_class = ['Amphibole']
    garnet_class = ['Garnet']
    glass_class = ['Glass']
    lower_threshold = 1000
    random_seed = 42

    oxides = OXIDES
    oxides_plus_zr = oxides + ["ZrO2"]

    sample_cols = ["SampleID", "Sample", "Sample Name", "Sample ID"]
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
                                     n_clusters=3)
        oli_df["Mineral"] = 'Olivine'

    amph_df = pd.DataFrame(columns=OXIDES + ["Mineral"])
    if 'Amphibole' in df["Mineral"].unique():
        amph_df = kmeans_multi_sample(df[df.Mineral == 'Amphibole'], n_target=2*n, n_clusters=8)
        amph_df["Mineral"] = 'Amphibole'

    gt_df = pd.DataFrame(columns=OXIDES + ["Mineral"])
    if 'Garnet' in df["Mineral"].unique():
        gt_df = kmeans_multi_sample(df[df.Mineral == 'Garnet'], n_target=n, n_clusters=8)
        gt_df["Mineral"] = 'Garnet'

    # Pyroxene: Cpx kmeans->n + Opx kmeans->n  => total 2n
    pyroxene_df = process_group_per_member(pyroxene_classes, 
                                           lambda d, k: kmeans_multi_sample(d, n_target=k),
                                           n_per_member=n, relabel_as='Pyroxene')

    # Feldspar: Plag kmeans->n + KFspar kmeans->n => total 2n
    feldspar_df = process_group_per_member(feldspar_classes,
                                           lambda d, k: kmeans_multi_sample(d, n_target=k, n_clusters=8),
                                           n_per_member=n, relabel_as='Feldspar')

    # Rhombohedral oxides: per member random->n (=> 2n total)
    rhombohedral_oxide_df = process_group_per_member(rhombohedral_oxide_classes,
                                                     lambda d, k: random_cap(d, k),
                                                     n_per_member=n, relabel_as='Rhombohedral_Oxides')

    # Spinels: per member random->n (=> 2n total)
    spinel_df = process_group_per_member(spinel_classes,
                                         lambda d, k: random_cap(d, k),
                                         n_per_member=n, relabel_as='Spinel_Group')

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
        to_drop = [c for c in ["Na2O + K2O", "TAS"] if c in resampled.columns]
        resampled = resampled.drop(columns=to_drop)
        resampled = resampled.copy()
        resampled['Mineral'] = 'Glass'
        glass_df = resampled
    else:
        glass_df = pd.DataFrame(columns=OXIDES + ['Mineral'])

    # Other classes: if <1250 -> shuffle+oversample to n; else cap to n
    special_flat = set(olivine_class + pyroxene_classes + feldspar_classes +
                       rhombohedral_oxide_classes + spinel_classes + amphibole_class + 
                       garnet_class + glass_class)
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
        [oli_df, pyroxene_df, feldspar_df, rhombohedral_oxide_df, spinel_df, amph_df, gt_df, glass_df, other_df],
        ignore_index=True
    )
    return df_balanced


class VariationalLayer(nn.Module):
    """
    Bayesian linear layer using variational inference. Models weights and
    biases as Gaussian distributions rather than point estimates, enabling
    uncertainty quantification through weight sampling

    Parameters:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Attributes:
        weight_mu (Parameter): Mean of the weight distributions.
        weight_rho (Parameter): Unconstrained std parameters for weight distributions.
        bias_mu (Parameter): Mean of the bias distributions.
        bias_rho (Parameter): Unconstrained std parameters for bias distributions.
        softplus (nn.Softplus): Softplus activation ensuring positive standard deviations.

    Methods:
        reset_parameters(): Initializes parameters based on the number of input features.
        forward(input): Performs a forward pass using sampled weights and biases.
        kl_divergence(): Computes KL divergence between the variational posterior
            and a standard normal prior, used as a regularization term in the loss.
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


def unique_mapping_nn(pred_class):
    """
    Generates a mapping of unique class codes from predicted class labels.
    Loads a predefined category list and creates a subset mapping for the
    unique classes found. Unknown classes are assigned a code of -1.

    Parameters:
        pred_class (array-like): Array of predicted class labels (integer codes).

    Returns:
        unique (ndarray): Array of unique class codes found in pred_class.
        valid_mapping (dict): Dictionary mapping class codes to their corresponding
            mineral names, including 'Unknown' for code -1.

        
    Generates a mapping of unique class codes from predicted class labels.
    Loads a predefined category list and creates a subset mapping for the
    unique classes found. Unknown classes are assigned a code of -1.
 
    """


    _, mapping = load_minclass_nn()
    unique = np.unique(pred_class)
    valid_mapping = {key: mapping[key] for key in unique}
    if -1 in unique:
        valid_mapping[-1] = "Unknown"

    return unique, valid_mapping


def class2mineral_nn(pred_class):
    """
    Translates predicted class codes into mineral names using a mapping from the
    trained neural network.

    Parameters:
        pred_class (array-like): Array of predicted class codes (integers).
 
    Returns:
        pred_mineral (ndarray): Array of mineral names corresponding to the
            predicted class codes.
    """

    _, valid_mapping = unique_mapping_nn(pred_class)
    pred_mineral = np.array([valid_mapping[x] for x in pred_class])
    return pred_mineral


def format_oxide_label(label):
    """
    Converts an oxide string to a matplotlib mathregular label with subscripts
    (e.g., 'SiO2' -> '$\\mathregular{SiO_2}$', 'FeOt' -> '$\\mathregular{FeO_t}$').
 
    Parameters:
        label (str): Oxide name string.
 
    Returns:
        formatted (str): LaTeX-formatted label for matplotlib.
    """

    if label == "Total":
        return label

    # Subscript digits and the lowercase 't'
    # The pattern (\d+|t) matches one or more digits OR the letter 't'
    formatted = re.sub(r"(\d+|t)", r"_\1", label)

    return f"$\mathregular{{{formatted}}}$"


# %% 

class NNWRFeatureExtractor(nn.Module):
    """
    Stage A classifier: extracts features and returns logits, optionally
    with the intermediate feature embedding h.
 
    Parameters:
        input_dim (int): Number of input oxide features.
        classes (int): Number of output mineral classes.
        hidden_layer_sizes (list[int]): Sizes of hidden layers.
        dropout_rate (float): Dropout probability.
        use_bayesian_feature_layer (bool): If True, the final feature layer
            is a VariationalLayer instead of a standard Linear layer.
        use_bayesian_classifier (bool): If True, the classification head
            is a VariationalLayer.
    """

    def __init__(
        self,
        input_dim=11,
        classes=23,
        hidden_layer_sizes=[64, 32, 16],
        dropout_rate=0.1,
        use_bayesian_feature_layer=True,
        use_bayesian_classifier=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.classes = classes
        self.hls = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.use_bayesian_feature_layer = use_bayesian_feature_layer
        self.use_bayesian_classifier = use_bayesian_classifier
        
        def enc_element(in_channel, out_channel, is_last=False):
            if not is_last:
                return [
                    nn.Linear(in_channel, out_channel),
                    nn.BatchNorm1d(out_channel),
                    nn.LeakyReLU(0.02),
                    nn.Dropout(self.dropout_rate),
                ]
            return [VariationalLayer(in_channel, out_channel)]

        feat = []
        for i, size in enumerate(self.hls):
            is_last = (i == len(self.hls) - 1) and self.use_bayesian_feature_layer
            in_ch = self.input_dim if i == 0 else self.hls[i - 1]
            feat += enc_element(in_ch, size, is_last=is_last)

        self.feature_extractor = nn.Sequential(*feat)
        self.feat_dim = int(self.hls[-1])

        if self.use_bayesian_classifier:
            self.classifier = VariationalLayer(self.feat_dim, self.classes)
        else:
            self.classifier = nn.Linear(self.feat_dim, self.classes)

        self.apply(weights_init)

    def forward(self, x, return_features=False):
        h = self.feature_extractor(x)
        logits = self.classifier(h)
        if return_features:
            return logits, h
        return logits


class NNWRLatentProjector(nn.Module):
    """
    Stage B: trainable mapper from feature embedding h to a 2D latent space z2.
 
    Parameters:
        feat_dim (int): Dimensionality of the input feature embedding.
        hidden (int): Hidden layer size for the nonlinear projection.
        dropout_rate (float): Dropout probability (0.0 = no dropout).
        nonlinear (bool): If True, uses a two-layer MLP; otherwise a single linear map.
    """

    def __init__(self, feat_dim, hidden=32, dropout_rate=0.0, nonlinear=True):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.hidden = int(hidden)
        self.nonlinear = bool(nonlinear)
        self.dropout_rate = float(dropout_rate)

        if nonlinear:
            layers = [
                nn.Linear(self.feat_dim, self.hidden),
                nn.LayerNorm(self.hidden),
                nn.LeakyReLU(0.02),
            ]
            if self.dropout_rate > 0:
                layers += [nn.Dropout(self.dropout_rate)]
            layers += [nn.Linear(self.hidden, 2)]
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Linear(self.feat_dim, 2)

        self.apply(weights_init)

    def forward(self, h):
        return self.net(h)


class NNWRReconstructionDecoder(nn.Module):
    """
    Stage B: trainable decoder from 2D latent z2 back to oxide space x.
 
    Parameters:
        z_dim (int): Dimensionality of the latent input (typically 2).
        output_dim (int): Number of output features (number of oxides).
        decoder_hidden_sizes (list[int]): Sizes of hidden layers in the decoder.
        dropout_rate (float): Dropout probability (0.0 = no dropout).
    """

    def __init__(
        self, z_dim, output_dim, decoder_hidden_sizes=[64, 32], dropout_rate=0.0
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.output_dim = int(output_dim)
        self.decoder_hidden_sizes = list(decoder_hidden_sizes)
        self.dropout_rate = float(dropout_rate)

        layers = []
        in_ch = self.z_dim
        for size in self.decoder_hidden_sizes:
            layers += [
                nn.Linear(in_ch, size),
                nn.LayerNorm(size),
                nn.LeakyReLU(0.02),
            ]
            if self.dropout_rate > 0:
                layers += [nn.Dropout(self.dropout_rate)]
            in_ch = size

        layers += [nn.Linear(in_ch, self.output_dim)]
        self.decode = nn.Sequential(*layers)

        self.apply(weights_init)

    def forward(self, z2):
        return self.decode(z2)


class NNWRReconstructionWrapper(nn.Module):
    """
    Inference wrapper combining classifier, mapper, and decoder.
    Returns (logits, reconstructed oxides, z2) on forward pass.
 
    Parameters:
        classifier (NNWRFeatureExtractor): Trained Stage A classifier.
        mapper2d (NNWRLatentProjector): Trained Stage B latent projector.
        decoder (NNWRReconstructionDecoder): Trained Stage B decoder.
    """

    def __init__(
        self,
        classifier: NNWRFeatureExtractor,
        mapper2d: NNWRLatentProjector,
        decoder: NNWRReconstructionDecoder,
    ):
        super().__init__()
        self.classifier = classifier
        self.mapper2d = mapper2d
        self.decoder = decoder
        self.latent_dim = 2  # makes plotting gate happy

    def forward(self, x):
        logits, h = self.classifier(x, return_features=True)
        z2 = self.mapper2d(h)
        recon = self.decoder(z2)
        return logits, recon, z2


def train_nn_hybrid_bottleneck(
    classifier,
    mapper2d,
    decoder,
    optimizer,
    train_loader,
    valid_loader,
    n_epoch,
    criterion_recon=None,
    patience=50,
    plot_latent=False,
    plot_every=100,
    max_plot_points=10000,
    plot_on="valid",
):
    """
    Stage B training: freezes the classifier and trains the mapper (h -> z2)
    and decoder (z2 -> x) with MSE reconstruction loss.
 
    Parameters:
        classifier (NNWRFeatureExtractor): Frozen Stage A classifier.
        mapper2d (NNWRLatentProjector): Trainable latent projector.
        decoder (NNWRReconstructionDecoder): Trainable reconstruction decoder.
        optimizer (torch.optim.Optimizer): Optimizer for mapper + decoder parameters.
        train_loader (DataLoader): Training data loader.
        valid_loader (DataLoader): Validation data loader.
        n_epoch (int): Maximum number of training epochs.
        criterion_recon (nn.Module|None): Reconstruction loss function. Defaults to MSELoss.
        patience (int): Early stopping patience (epochs without improvement).
        plot_latent (bool): If True, periodically plot the 2D latent space.
        plot_every (int): Plot interval in epochs.
        max_plot_points (int): Maximum number of points to plot.
        plot_on (str): 'valid' or 'train' — which dataset to plot.
 
    Returns:
        train_losses (dict): Training reconstruction loss history.
        valid_losses (dict): Validation reconstruction loss history.
        best_valid (float): Best validation reconstruction loss achieved.
        best_mapper_state (dict): State dict of the best mapper.
        best_decoder_state (dict): State dict of the best decoder.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_cat, _ = load_minclass_nn(minclass_path="mineral_classes_nn_v0030.npz")

    tab20 = plt.get_cmap("tab20")
    tab20b = plt.get_cmap("tab20b")
    combined_colors = [tab20(i) for i in range(20)] + [tab20b(i) for i in range(20)]
    random.shuffle(combined_colors)
    custom_cmap = mcolors.ListedColormap(combined_colors)
    custom_norm = mcolors.Normalize(vmin=0, vmax=max(len(min_cat), 1))

    classifier.to(device).eval()
    mapper2d.to(device).train()
    decoder.to(device).train()

    # freeze classifier forever in Stage B
    for p in classifier.parameters():
        p.requires_grad = False
    for p in mapper2d.parameters():
        p.requires_grad = True
    for p in decoder.parameters():
        p.requires_grad = True

    if criterion_recon is None:
        criterion_recon = nn.MSELoss()

    train_losses = {"reconstruction": []}
    valid_losses = {"reconstruction": []}

    best_valid = float("inf")
    best_mapper_state = None
    best_decoder_state = None
    patience_counter = 0

    def avg(l):
        return float(np.mean(l)) if len(l) else float("nan")

    for epoch in range(n_epoch):
        mapper2d.train()
        decoder.train()
        tr = []

        for x, _y in train_loader:
            x = x.to(device)

            with torch.no_grad():
                _logits, h = classifier(x, return_features=True)

            z2 = mapper2d(h)  # (B, 2)
            recon = decoder(z2)  # (B, D)
            loss = criterion_recon(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr.append(loss.item())

        mapper2d.eval()
        decoder.eval()
        va = []
        with torch.no_grad():
            for x, _y in valid_loader:
                x = x.to(device)
                _logits, h = classifier(x, return_features=True)
                z2 = mapper2d(h)
                recon = decoder(z2)
                va.append(criterion_recon(recon, x).item())

        train_losses["reconstruction"].append(avg(tr))
        valid_losses["reconstruction"].append(avg(va))

        print(
            f"[BOTTLE {epoch + 1:03}/{n_epoch:03}] train_rec={avg(tr):.6f} valid_rec={avg(va):.6f}"
        )

        # optional plotting of 2D latent (z2)
        do_plot = plot_latent and (
            epoch == 0 or (epoch + 1) % plot_every == 0 or epoch == n_epoch - 1
        )
        if do_plot:
            loader = valid_loader if plot_on == "valid" else train_loader
            z_list, y_list, n_seen = [], [], 0

            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    _, h = classifier(xb, return_features=True)
                    z2 = mapper2d(h)
                    z_list.append(z2.cpu().numpy())
                    y_list.append(yb.cpu().numpy())
                    n_seen += len(yb)
                    if n_seen >= max_plot_points:
                        break

            if len(z_list) > 0:
                Z = np.concatenate(z_list, axis=0)[:max_plot_points]
                Y = np.concatenate(y_list, axis=0)[:max_plot_points]
                classes_uniq = np.unique(Y)

                plt.figure(figsize=(8, 7))
                for k in classes_uniq:
                    mask = Y == k
                    # Use label_names if available, else fall back to ID
                    lab = (
                        min_cat[int(k)]
                        if (min_cat is not None and int(k) < len(min_cat))
                        else f"class_{int(k)}"
                    )

                    # Apply your custom color logic
                    class_color = custom_cmap(custom_norm(int(k)))

                    plt.scatter(
                        Z[mask, 0],
                        Z[mask, 1],
                        c=[class_color],
                        s=15,
                        ec="k",
                        lw=0.25,
                        marker="o",
                        alpha=0.9,
                        label=lab,
                    )

                plt.xlabel("z2_1")
                plt.ylabel("z2_2")
                plt.title(f"Bottleneck z2 ({plot_on}) - epoch {epoch + 1}")
                plt.legend(
                    bbox_to_anchor=(1.0175, 1.0),
                    loc="upper left",
                    frameon=False,
                    markerscale=1,
                    prop={"size": 8},
                )
                plt.tight_layout()
                plt.show()

        # early stopping
        metric = avg(va)
        if metric < best_valid:
            best_valid = metric
            best_mapper_state = copy.deepcopy(mapper2d.state_dict())
            best_decoder_state = copy.deepcopy(decoder.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= (patience):
                print(
                    f"[BOTTLE] Early stopping after {patience * 2} epochs w/o improvement."
                )
                break

    if best_mapper_state is not None:
        mapper2d.load_state_dict(best_mapper_state)
    if best_decoder_state is not None:
        decoder.load_state_dict(best_decoder_state)

    return train_losses, valid_losses, best_valid, best_mapper_state, best_decoder_state


def plot_loss_curves(train_losses, valid_losses, filename):
    """
    Plots Stage A (classification, KL, total) and Stage B (reconstruction)
    loss curves, then saves to disk.
 
    Parameters:
        train_losses (dict): Training loss histories keyed by loss component.
        valid_losses (dict): Validation loss histories keyed by loss component.
        filename (str): Output filepath for the saved figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # 0: classification loss
    axes[0].plot(
        train_losses.get("cls_classification", []), label="Train CE", alpha=0.8
    )
    axes[0].plot(
        valid_losses.get("cls_classification", []), label="Valid CE", alpha=0.8
    )
    axes[0].set_title("Stage A - Classification Loss (CE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 1: KL
    axes[1].plot(train_losses.get("cls_kl", []), label="Train KL", alpha=0.8)
    axes[1].plot(valid_losses.get("cls_kl", []), label="Valid KL", alpha=0.8)
    axes[1].set_title("Stage A - KL Term")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 2: total
    axes[2].plot(train_losses.get("cls_total", []), label="Train Total", alpha=0.8)
    axes[2].plot(valid_losses.get("cls_total", []), label="Valid Total", alpha=0.8)
    axes[2].set_title("Stage A - Total (CE + KL)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 3: reconstruction
    axes[3].plot(
        train_losses.get("dec_reconstruction", []), label="Train Recon", alpha=0.8
    )
    axes[3].plot(
        valid_losses.get("dec_reconstruction", []), label="Valid Recon", alpha=0.8
    )
    axes[3].set_title("Stage B - Reconstruction (from z2)")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("MSE")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def kl_divergence_sum(model):
    """
    Sums KL divergences across all VariationalLayer modules in a model.
 
    Parameters:
        model (nn.Module): PyTorch model containing VariationalLayer submodules.
 
    Returns:
        kl_div (float): Total KL divergence.
    """

    kl_div = 0.0
    for module in model.modules():
        if isinstance(module, VariationalLayer):
            kl_div += module.kl_divergence()
    return kl_div


def train_nn_hybrid_classifier(
    model,
    optimizer,
    train_loader,
    valid_loader,
    n_epoch,
    criterion_cls=None,
    kl_weight_decay=1e-3,
    kl_decay_epochs=750,
    patience=50,
):
    """
    Stage A training: trains the classifier with cross-entropy loss and
    annealed KL divergence regularization from VariationalLayers.
 
    Parameters:
        model (NNWRFeatureExtractor): Classifier model to train.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        train_loader (DataLoader): Training data loader.
        valid_loader (DataLoader): Validation data loader.
        n_epoch (int): Maximum number of training epochs.
        criterion_cls (nn.Module|None): Classification loss. Defaults to CrossEntropyLoss.
        kl_weight_decay (float): Maximum KL weight after annealing.
        kl_decay_epochs (int): Number of epochs over which to anneal the KL weight.
        patience (int): Early stopping patience (epochs without improvement).
 
    Returns:
        train_losses (dict): Training loss histories ('total', 'classification', 'kl').
        valid_losses (dict): Validation loss histories.
        best_valid (float): Best validation classification loss achieved.
        best_state (dict): State dict of the best model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if criterion_cls is None:
        criterion_cls = nn.CrossEntropyLoss()

    train_losses = {"total": [], "classification": [], "kl": []}
    valid_losses = {"total": [], "classification": [], "kl": []}

    best_valid = float("inf")
    best_state = None
    patience_counter = 0

    steps_per_epoch = len(train_loader)
    kl_max = float(kl_weight_decay)
    total_ramp_steps = max(1, int(kl_decay_epochs) * steps_per_epoch)
    global_step = 0

    def avg(l):
        return float(np.mean(l)) if len(l) else float("nan")

    for epoch in range(n_epoch):
        model.train()
        tr = {"tot": [], "cls": [], "kl": []}

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # classifier-only
            loss_cls = criterion_cls(logits, y)

            kl_weight = kl_max * min(1.0, global_step / total_ramp_steps)
            global_step += 1
            kl_div = kl_divergence_sum(model)
            loss_kl = kl_weight * kl_div / x.size(0)

            loss = loss_cls + loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr["tot"].append(loss.item())
            tr["cls"].append(loss_cls.item())
            tr["kl"].append(loss_kl.item())

        model.eval()
        va = {"tot": [], "cls": [], "kl": []}
        kl_weight_eval = kl_max * min(1.0, global_step / total_ramp_steps)

        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss_cls = criterion_cls(logits, y)
                kl_div = kl_divergence_sum(model)
                loss_kl = kl_weight_eval * kl_div / x.size(0)
                loss = loss_cls + loss_kl

                va["tot"].append(loss.item())
                va["cls"].append(loss_cls.item())
                va["kl"].append(loss_kl.item())

        train_losses["total"].append(avg(tr["tot"]))
        train_losses["classification"].append(avg(tr["cls"]))
        train_losses["kl"].append(avg(tr["kl"]))

        valid_losses["total"].append(avg(va["tot"]))
        valid_losses["classification"].append(avg(va["cls"]))
        valid_losses["kl"].append(avg(va["kl"]))

        metric = avg(va["cls"])  # track classification val loss
        print(
            f"[CLS {epoch + 1:03}/{n_epoch:03}] train_cls={avg(tr['cls']):.6f} valid_cls={avg(va['cls']):.6f} (tot={avg(va['tot']):.6f})"
        )

        if metric < best_valid:
            best_valid = metric
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[CLS] Early stopping after {patience} epochs w/o improvement.")
                break

    return train_losses, valid_losses, best_valid, best_state


def plot_latent_space_hybrid(
    model,
    dataset,
    title,
    filename,
    batch_size=256,
):
    """
    Plots latent space representations (z) for a dataset. If latent_dim > 2,
    uses PCA to reduce to 2D for visualization. Saves a PDF to filename.
 
    Parameters:
        model (nn.Module): Trained NNWRReconstructionWrapper model.
        dataset (TensorDataset): Dataset of (features, labels) tensors.
        title (str): Plot title.
        filename (str): Output filepath for the saved figure.
        batch_size (int): Batch size for inference.
 
    Returns:
        latents (ndarray): Latent vectors with shape (N, latent_dim).
        labels (ndarray): Integer labels with shape (N,).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    latents = []
    labels = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            x = batch_data.to(device)
            _, _, z = model(x)

            latents.append(z.detach().cpu().numpy())
            labels.append(batch_labels.detach().cpu().numpy())

    latents = (
        np.concatenate(latents, axis=0)
        if len(latents)
        else np.zeros((0, getattr(model, "latent_dim", 2)))
    )
    labels = (
        np.concatenate(labels, axis=0) if len(labels) else np.zeros((0,), dtype=int)
    )

    # Reduce to 2D for plotting (PCA only if needed)
    used_pca = False
    latents_2d = latents

    if latents.shape[0] > 0 and latents.shape[1] > 2:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        used_pca = True
        try:
            print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        except Exception:
            pass

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    min_cat, _ = load_minclass_nn(minclass_path="mineral_classes_nn_v0030.npz")

    tab20 = plt.get_cmap("tab20")
    tab20b = plt.get_cmap("tab20b")
    colors_tab20 = [tab20(i) for i in range(20)]
    colors_tab20b = [tab20b(i) for i in range(20)]
    combined_colors = colors_tab20 + colors_tab20b

    c_norm = mcolors.Normalize(vmin=0, vmax=max(len(min_cat), 1))
    scalar_map = mcm.ScalarMappable(
        norm=c_norm,
        cmap=mcolors.ListedColormap(combined_colors),
    )

    for i, mineral in enumerate(min_cat):
        mask = labels == i
        if mask.sum() == 0:
            continue

        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            marker="o",
            s=20,
            ec="k",
            lw=0.5,
            color=scalar_map.to_rgba(i),
            label=mineral,
            rasterized=True,
        )

    ax.set_xlabel("PC1" if used_pca else "Latent Dimension 1")
    ax.set_ylabel("PC2" if used_pca else "Latent Dimension 2")
    ax.set_title(title)
    ax.legend(prop={"size": 8}, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return latents, labels


def neuralnetwork_wr(
    df,
    hls_list,
    kl_weight_decay_list,
    lr,
    wd,
    dr,
    ep,
    n,
    balanced,
    # ---- second stage ----
    ep_bottle=1000,  # second
    lr_bottle=1e-4,
    wd_bottle=1e-4,
    mapper_hidden=16,
    mapper_nonlinear=True,
    decoder_hidden_sizes=(64, 32),
    kl_decay_epochs=750,
    use_bayesian_feature_layer=True,
    use_bayesian_classifier=False,
    name="nn2d",
    plot_latent_during_training=False,
    plot_every=50,
    plot_on="valid",
):
    """
    Full training pipeline for the neural network classifier with reconstruction.
 
    Stage A trains the classifier (CE + KL annealing), picking the best model
    by validation CE. Stage B freezes the classifier and trains a mapper
    (h -> z2) plus decoder (z2 -> x) with MSE reconstruction loss.
 
    Parameters:
        df (pd.DataFrame): Training DataFrame with 'Mineral' column and oxide columns.
        hls_list (list[list[int]]): Hidden layer size configurations to sweep.
        kl_weight_decay_list (list[float]): KL weight decay values to sweep.
        lr (float): Learning rate for Stage A.
        wd (float): Weight decay for Stage A.
        dr (float): Dropout rate.
        ep (int): Number of epochs for Stage A.
        n (int): Validation split size (number of samples).
        balanced (bool): If True, balance the training set via ``balance()``.
        ep_bottle (int): Number of epochs for Stage B.
        lr_bottle (float): Learning rate for Stage B.
        wd_bottle (float): Weight decay for Stage B.
        mapper_hidden (int): Hidden layer size for the latent projector.
        mapper_nonlinear (bool): If True, use a nonlinear mapper.
        decoder_hidden_sizes (tuple[int]): Hidden layer sizes for the decoder.
        kl_decay_epochs (int): Number of epochs over which to anneal the KL weight.
        use_bayesian_feature_layer (bool): If True, use VariationalLayer for features.
        use_bayesian_classifier (bool): If True, use VariationalLayer for the classifier head.
        name (str): Run name used for output filenames.
        plot_latent_during_training (bool): If True, plot z2 during Stage B training.
        plot_every (int): Plot interval in epochs during Stage B.
        plot_on (str): 'valid' or 'train' — which dataset to plot during training.
 
    Returns:
        best_model_state (dict): State dict of the best NNWRReconstructionWrapper.
    """

    path_beg = os.getcwd() + "/"
    output_dir = ["parametermatrix_neuralnetwork"]
    for ii in range(len(output_dir)):
        if not os.path.exists(path_beg + output_dir[ii]):
            os.makedirs(path_beg + output_dir[ii], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Data prep (reuse your supervised pipeline)
    # -----------------------------
    train_df, valid_df = train_test_split(df, test_size=n, random_state=42)
    if balanced is True:
        train_df = balance(train_df, n=1000)
    train_df.to_csv("nnwithrecon_train_df.csv")

    for _df in (train_df, valid_df):
        _df["Mineral"] = (
            _df["Mineral"]
            .astype(str)
            .replace(["Clinopyroxene", "Orthopyroxene"], "Pyroxene")
            .replace(["Plagioclase", "KFeldspar"], "Feldspar")
            .replace(["Hematite", "Ilmenite"], "Rhombohedral_Oxides")
            .replace(["Magnetite", "Spinel"], "Spinel_Group")
        )

    exclude = {"Zircon", "Carbonate", "SiO2_Polymorph"}

    train_df_nonempirical = train_df.loc[~train_df["Mineral"].isin(exclude)].copy()

    valid_df_nonempirical = valid_df.loc[~valid_df["Mineral"].isin(exclude)].copy()

    all_cats = pd.Categorical(train_df_nonempirical["Mineral"])
    mapping = dict(enumerate(all_cats.categories))
    inv_mapping = {cat: idx for idx, cat in mapping.items()}

    classes_path = os.path.join(
        path_beg, "parametermatrix_neuralnetwork", "mineral_classes_nn_v0030.npz"
    )
    classes = np.asarray(all_cats.categories.tolist(), dtype=object)
    np.savez_compressed(classes_path, classes=classes)

    # min_names = pd.Categorical(train_df_nozirc["Mineral"]).categories.to_list()
    # sort_mapping = dict(sorted(mapping.items(), key=lambda item: item[0]))

    train_df_nonempirical["_code"] = (
        train_df_nonempirical["Mineral"].map(inv_mapping).astype(int)
    )
    valid_df_nonempirical["_code"] = (
        valid_df_nonempirical["Mineral"].map(inv_mapping).astype(int)
    )

    ss = StandardScaler().fit(train_df_nonempirical[OXIDES].fillna(0))
    train_x = ss.transform(train_df_nonempirical[OXIDES].fillna(0))
    valid_x = ss.transform(valid_df_nonempirical[OXIDES].fillna(0))
    scaler_path = os.path.join(
        path_beg, "parametermatrix_neuralnetwork", "scaler_nn_v0030.npz"
    )
    np.savez(
        scaler_path,
        mean=pd.Series(ss.mean_, index=OXIDES),
        scale=pd.Series(np.sqrt(ss.var_), index=OXIDES),
    )

    # encode labels
    train_y = train_df_nonempirical["_code"].to_numpy()
    valid_y = valid_df_nonempirical["_code"].to_numpy()

    # Define datasets to be used with PyTorch - see autoencoder file for details
    feature_dataset = LabelDataset(train_x, train_y)
    valid_dataset = LabelDataset(valid_x, valid_y)

    batch_size = 128
    input_size = len(feature_dataset.__getitem__(0)[0])

    feature_loader = DataLoader(
        feature_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    # num_workers=4, pin_memory=True, prefetch_factor=2,
    # persistent_workers=True)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    #   num_workers=4, pin_memory=True, prefetch_factor=2,
    #   persistent_workers=True)
    np.savez(
        "parametermatrix_neuralnetwork/"
        + str(lr)
        + "_"
        + str(wd)
        + "_"
        + str(dr)
        + "_"
        + str(ep)
        + "_best_model_nnwithrecon_features.npz",
        feature_loader=feature_loader,
        valid_loader=valid_loader,
    )

    # -----------------------------
    # sweep + pick best classifier
    # -----------------------------
    best_valid_cls = float("inf")
    best_combo = None
    best_classifier_state = None
    best_cls_train_losses = None
    best_cls_valid_losses = None

    cls_train_losses_dict = {}
    cls_valid_losses_dict = {}

    for kl_weight_decay in kl_weight_decay_list:
        for hls in hls_list:
            classifier = NNWRFeatureExtractor(
                input_dim=input_size,
                hidden_layer_sizes=hls,
                dropout_rate=dr,
                use_bayesian_feature_layer=use_bayesian_feature_layer,
                use_bayesian_classifier=use_bayesian_classifier,
            ).to(device)

            optimizer_cls = torch.optim.AdamW(
                classifier.parameters(), lr=lr, weight_decay=wd
            )

            cls_train, cls_valid, cls_best, cls_state = train_nn_hybrid_classifier(
                classifier,
                optimizer_cls,
                feature_loader,
                valid_loader,
                n_epoch=ep,
                criterion_cls=nn.CrossEntropyLoss(),
                kl_weight_decay=kl_weight_decay,
                kl_decay_epochs=kl_decay_epochs,
                patience=50,
            )

            cls_train_losses_dict[(kl_weight_decay, tuple(hls))] = cls_train
            cls_valid_losses_dict[(kl_weight_decay, tuple(hls))] = cls_valid

            if cls_best < best_valid_cls:
                best_valid_cls = cls_best
                best_combo = (kl_weight_decay, hls)
                best_classifier_state = cls_state
                best_cls_train_losses = cls_train
                best_cls_valid_losses = cls_valid

    best_kl_weight_decay, best_hidden_layer_size = best_combo

    # -----------------------------
    # build mapper + decoder and train (classifier frozen)
    # -----------------------------
    classifier = NNWRFeatureExtractor(
        input_dim=input_size,
        hidden_layer_sizes=best_hidden_layer_size,
        dropout_rate=dr,
        use_bayesian_feature_layer=use_bayesian_feature_layer,
        use_bayesian_classifier=use_bayesian_classifier,
    ).to(device)
    classifier.load_state_dict(best_classifier_state)
    classifier.eval()

    mapper2d = NNWRLatentProjector(
        feat_dim=classifier.feat_dim,
        hidden=mapper_hidden,
        dropout_rate=0.0,
        nonlinear=mapper_nonlinear,
    ).to(device)

    decoder = NNWRReconstructionDecoder(
        z_dim=2,
        output_dim=input_size,
        decoder_hidden_sizes=list(decoder_hidden_sizes),
        dropout_rate=0.0,
    ).to(device)

    optimizer_bottle = torch.optim.AdamW(
        list(mapper2d.parameters()) + list(decoder.parameters()),
        lr=lr_bottle,
        weight_decay=wd_bottle,
    )

    dec_train, dec_valid, best_valid_rec, best_mapper_state, best_decoder_state = (
        train_nn_hybrid_bottleneck(
            classifier,
            mapper2d,
            decoder,
            optimizer_bottle,
            feature_loader,
            valid_loader,
            n_epoch=ep_bottle,
            criterion_recon=nn.MSELoss(),
            patience=50,
            plot_latent=plot_latent_during_training,
            plot_every=plot_every,
            max_plot_points=10000,
            plot_on=plot_on,
        )
    )

    # load best Stage B
    if best_mapper_state is not None:
        mapper2d.load_state_dict(best_mapper_state)
    if best_decoder_state is not None:
        decoder.load_state_dict(best_decoder_state)

    best_model = NNWRReconstructionWrapper(classifier, mapper2d, decoder).to(device)
    best_model.eval()

    best_model_state = best_model.state_dict()

    # -----------------------------
    # Save outputs + plots
    # -----------------------------
    # save sweep losses
    np.savez(
        "parametermatrix_neuralnetwork/"
        + str(lr)
        + "_"
        + str(wd)
        + "_"
        + str(dr)
        + "_"
        + str(ep)
        + "_kl"
        + str(best_kl_weight_decay)
        + "_hls"
        + str(best_hidden_layer_size)
        + "_best_model_nnwithrecon_losses.npz",
        cls_train_losses=cls_train_losses_dict,
        cls_valid_losses=cls_valid_losses_dict,
        dec_train_losses=dec_train,
        dec_valid_losses=dec_valid,
    )

    # latent space plots (train + valid) using TRUE z2 (2D)
    print("Generating MTL z2 (2D) latent space visualizations...")

    train_latents, train_labels = plot_latent_space_hybrid(
        model=best_model,
        dataset=feature_dataset,
        title=f"{name} - Training Set z2 (2D)",
        filename=f"parametermatrix_neuralnetwork/{name}_train_z2_space.pdf",
    )

    valid_latents, valid_labels = plot_latent_space_hybrid(
        model=best_model,
        dataset=valid_dataset,
        title=f"{name} - Validation Set z2 (2D)",
        filename=f"parametermatrix_neuralnetwork/{name}_valid_z2_space.pdf",
    )

    # loss curves PDF (Stage A + Stage B)
    train_losses = {
        "cls_total": best_cls_train_losses["total"] if best_cls_train_losses else [],
        "cls_classification": best_cls_train_losses["classification"]
        if best_cls_train_losses
        else [],
        "cls_kl": best_cls_train_losses["kl"] if best_cls_train_losses else [],
        "dec_reconstruction": dec_train["reconstruction"],
    }
    valid_losses = {
        "cls_total": best_cls_valid_losses["total"] if best_cls_valid_losses else [],
        "cls_classification": best_cls_valid_losses["classification"]
        if best_cls_valid_losses
        else [],
        "cls_kl": best_cls_valid_losses["kl"] if best_cls_valid_losses else [],
        "dec_reconstruction": dec_valid["reconstruction"],
    }

    plot_loss_curves(
        train_losses,
        valid_losses,
        f"parametermatrix_neuralnetwork/{name}_loss_curves.pdf",
    )

    # save model + config
    model_path = f"parametermatrix_neuralnetwork/{name}_best_model.pt"
    model_config = {
        "arch": "MTL-2D",
        "input_dim": input_size,
        "oxides": list(OXIDES),
        "hidden_layer_sizes": best_hidden_layer_size,
        "feat_dim": int(classifier.feat_dim),
        "dropout_rate": dr,
        "use_bayesian_feature_layer": use_bayesian_feature_layer,
        "use_bayesian_classifier": use_bayesian_classifier,
        "best_kl_weight_decay": best_kl_weight_decay,
        "kl_decay_epochs": int(kl_decay_epochs),
        "epochs_cls": int(ep),
        "epochs_bottle": int(ep_bottle),
        "lr_cls": float(lr),
        "wd_cls": float(wd),
        "lr_bottle": float(lr_bottle),
        "wd_bottle": float(wd_bottle),
        "mapper_hidden": int(mapper_hidden),
        "mapper_nonlinear": bool(mapper_nonlinear),
        "decoder_hidden_sizes": list(decoder_hidden_sizes),
        "balanced": bool(balanced),
        "valid_fraction": float(n),
        "name": name,
    }

    torch.save(
        {
            "model_state_dict": best_model_state,
            "best_valid_cls_loss": best_valid_cls,
            "best_valid_rec_loss": best_valid_rec,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "model_config": model_config,
        },
        model_path,
    )

    # save preprocessing (name-specific, so you can load reliably later)
    preprocess_path = f"parametermatrix_neuralnetwork/{name}_preprocess.npz"
    np.savez(
        preprocess_path,
        mean=ss.mean_.astype(np.float32),
        scale=ss.scale_.astype(np.float32),
        feature_names=np.array(OXIDES, dtype=object),
    )

    # save latent arrays
    np.savez(
        f"parametermatrix_neuralnetwork/{name}_latent_data.npz",
        train_latents=train_latents,
        train_labels=train_labels,
        valid_latents=valid_latents,
        valid_labels=valid_labels,
        best_valid_cls_loss=best_valid_cls,
        best_valid_rec_loss=best_valid_rec,
    )

    print(f"MTL-2D completed! Results saved to parametermatrix_neuralnetwork/{name}_*")
    print(f"Best validation classification loss (Stage A): {best_valid_cls:.6f}")
    print(f"Best validation reconstruction loss (Stage B): {best_valid_rec:.6f}")

    return best_model_state


def predict_class_prob_nnwr(
    df,
    n_iterations=50,
    *,
    model_path=None,
    hidden_layer_sizes=None,
    mc_dropout=True,
    return_recon_oxides=False,
    scaler_path="scaler_nn_v0030.npz",
):
    """
    Predicts mineral classes with Monte Carlo Bayesian averaging using the
    neural network with reconstruction (NNWR) classifier.
 
    Parameters:
        df (pd.DataFrame): Input oxide compositions. Metadata columns ('Mineral',
            'Source', 'SampleID', 'Sample', 'Sample Name', 'Sample ID') are
            preserved in the output when present.
        n_iterations (int): Number of MC forward passes for probability averaging.
        model_path (str|None): Path to the .pt checkpoint. If None, defaults to the
            bundled model in the same directory as this module.
        hidden_layer_sizes (list[int]|None): Override hidden layer sizes. Usually
            leave None to use the checkpoint configuration.
        mc_dropout (bool): If True, enables dropout during inference for MC sampling.
        return_recon_oxides (bool): If True, appends reconstructed oxide columns
            to the output DataFrame.
        scaler_path (str): Filename or relative path to the saved scaler .npz file.
 
    Returns:
        result_df (pd.DataFrame): Predictions including 'Predict_Mineral',
            'Prediction_Score', 'Prediction_Score_Sigma', 'Second_Predict_Mineral',
            and 'Second_Prediction_Score'.
        probability_matrix (ndarray): (N, C) array of class probabilities for
            non-empirical rows; empty array if all rows are empirical.
    """

    # ---- set up result DataFrame  ----
    oxides = OXIDES
    oxides_plus_zr = oxides + ["ZrO2"]
    metadata = ["Mineral", "Source", "SampleID", "Sample", "Sample Name", "Sample ID"]
    cols = oxides_plus_zr + [c for c in metadata if c in df.columns]
    result_df = df[cols].copy()

    pred_cols = [
        "Predict_Mineral",
        "Prediction_Score",
        "Prediction_Score_Sigma",
        "Second_Predict_Mineral",
        "Second_Prediction_Score",
    ]
    for col in pred_cols:
        result_df[col] = np.nan

    result_df["Predict_Mineral"] = pd.Series(index=df.index, dtype="object")
    result_df["Second_Predict_Mineral"] = pd.Series(index=df.index, dtype="object")
    result_df["Prediction_Score"] = pd.Series(index=df.index, dtype="float64")
    result_df["Prediction_Score_Sigma"] = pd.Series(index=df.index, dtype="float64")
    result_df["Second_Prediction_Score"] = pd.Series(index=df.index, dtype="float64")

    # --- detect rows with fewer than 1 valid (non-zero) oxide values ---
    oxide_cols_in_df = [c for c in OXIDES if c in df.columns]
    if oxide_cols_in_df:
        # count how many oxides are not zero (treating NaN as 0)
        valid_oxide_count = (df[oxide_cols_in_df].fillna(0) != 0).sum(axis=1)
        invalid_mask = valid_oxide_count < 1
    else:
        # if no oxide columns exist, all rows are invalid
        invalid_mask = pd.Series(True, index=df.index) 

    # explicitly set outputs to nan for invalid rows
    result_df.loc[invalid_mask, "Predict_Mineral"] = np.nan
    result_df.loc[invalid_mask, "Second_Predict_Mineral"] = np.nan
    result_df.loc[invalid_mask, "Prediction_Score"] = np.nan
    result_df.loc[invalid_mask, "Prediction_Score_Sigma"] = np.nan
    result_df.loc[invalid_mask, "Second_Prediction_Score"] = np.nan
    # ------------------------------------------------------------

    if "Total" not in df.columns:
        df["Total"] = (
            df[oxides_plus_zr].sum(axis=1, skipna=True)
            if all(c in df.columns for c in oxides_plus_zr)
            else pd.Series(0, index=df.index)
        )

    invalid_mask = invalid_mask | (df["Total"] < 50)
    result_df.loc[invalid_mask, "Predict_Mineral"] = np.nan
    result_df.loc[invalid_mask, "Second_Predict_Mineral"] = np.nan
    result_df.loc[invalid_mask, "Prediction_Score"] = np.nan
    result_df.loc[invalid_mask, "Prediction_Score_Sigma"] = np.nan
    result_df.loc[invalid_mask, "Second_Prediction_Score"] = np.nan

    # ------------------------------------------------------------

    # Identify and classify zircons
    zircon_mask = (
        (df["ZrO2"] > 50) if "ZrO2" in df.columns else pd.Series(False, index=df.index)
    )
    zircon_mask = zircon_mask & ~invalid_mask
    result_df.loc[zircon_mask, "Predict_Mineral"] = "Zircon"
    result_df.loc[zircon_mask, "Prediction_Score"] = np.nan
    result_df.loc[zircon_mask, "Prediction_Score_Sigma"] = np.nan
    result_df.loc[zircon_mask, "Second_Predict_Mineral"] = np.nan
    result_df.loc[zircon_mask, "Second_Prediction_Score"] = np.nan

    # Identify and classify si-polymorphs (quartz + coesite + stishovite + tridymite + cristobalite)
    quartz_mask = (
        (df["SiO2"] > 90) if "SiO2" in df.columns else pd.Series(False, index=df.index)
    )
    quartz_mask = quartz_mask & ~invalid_mask
    result_df.loc[quartz_mask, "Predict_Mineral"] = "SiO2_Polymorph"
    result_df.loc[quartz_mask, "Prediction_Score"] = np.nan
    result_df.loc[quartz_mask, "Prediction_Score_Sigma"] = np.nan
    result_df.loc[quartz_mask, "Second_Predict_Mineral"] = np.nan
    result_df.loc[quartz_mask, "Second_Prediction_Score"] = np.nan

    # Identify and classify carbonates (calcite + dolomite + magnesite + siderite)
    carbonate_mask = (
        (df["SiO2"] < 5) & (df["Total"] < 70)
        if "CaO" in df.columns
        else pd.Series(False, index=df.index)
    )
    carbonate_mask = carbonate_mask & ~invalid_mask
    result_df.loc[carbonate_mask, "Predict_Mineral"] = "Carbonate"
    result_df.loc[carbonate_mask, "Prediction_Score"] = np.nan
    result_df.loc[carbonate_mask, "Prediction_Score_Sigma"] = np.nan
    result_df.loc[carbonate_mask, "Second_Predict_Mineral"] = np.nan
    result_df.loc[quartz_mask, "Second_Prediction_Score"] = np.nan

    non_zircon_mask = (
        (~zircon_mask) & (~quartz_mask) & (~carbonate_mask) & (~invalid_mask)
    )
    probability_matrix = np.array([])

    if non_zircon_mask.any():
        non_za_df = df.loc[non_zircon_mask].copy()

        # neural network setup — model and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # default paths
        model_path = os.path.join(
            os.path.dirname(__file__), "nnwr_best_model_v0030.pt"
        )

        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get("model_config", {})

        # architecture knobs (prefer checkpoint)
        hidden_layer_sizes = model_config.get("hidden_layer_sizes", [64, 32, 16])
        dropout_rate = float(model_config.get("dropout_rate", 0.1))
        use_bayesian_feature_layer = bool(
            model_config.get("use_bayesian_feature_layer", True)
        )
        use_bayesian_classifier = bool(
            model_config.get("use_bayesian_classifier", False)
        )

        # ---- build classifier ----
        # run the classifier only for prediction; mapper/decoder are irrelevant for logits.
        model = NNWRFeatureExtractor(
            input_dim=len(OXIDES),
            hidden_layer_sizes=hidden_layer_sizes,
            dropout_rate=dropout_rate,
            use_bayesian_feature_layer=use_bayesian_feature_layer,
            use_bayesian_classifier=use_bayesian_classifier,
        ).to(device)

        # ---- load weights ----
        state_dict = checkpoint.get("model_state_dict", {})
        if not state_dict:
            raise KeyError("Checkpoint missing 'model_state_dict'.")

        cls_sd = {}
        for k, v in state_dict.items():
            if k.startswith("classifier."):
                cls_sd[k[len("classifier.") :]] = v

        if len(cls_sd) == 0:
            cls_sd = state_dict

        missing, unexpected = model.load_state_dict(cls_sd, strict=False)
        if len(unexpected) > 0:
            pass
        if len(missing) > 0:
            pass

        # eval mode by default (with Bayesian weight sampling in VariationalLayer)
        classifier = enable_mc_sampling(model, enable_dropout=mc_dropout)

        # ---- preprocess inputs to match training ----
        norm_wt = norm_data_nn(non_za_df, scaler_path=scaler_path).astype(
            np.float32, copy=False
        )
        input_data = torch.from_numpy(norm_wt).to(device)

        recon_df = None
        if return_recon_oxides:
            # --- build full wrapper to compute recon = decoder(z2) ---
            feat_dim = int(model_config.get("feat_dim", hidden_layer_sizes[-1]))
            mapper_hidden = int(model_config.get("mapper_hidden", 32))
            mapper_nonlinear = bool(model_config.get("mapper_nonlinear", True))
            decoder_hidden_sizes = model_config.get("decoder_hidden_sizes", [64, 32, 16])

            mapper2d = NNWRLatentProjector(
                feat_dim=feat_dim,
                hidden=mapper_hidden,
                dropout_rate=0.0,
                nonlinear=mapper_nonlinear,
            ).to(device)

            decoder = NNWRReconstructionDecoder(
                z_dim=2,
                output_dim=len(OXIDES),
                decoder_hidden_sizes=list(decoder_hidden_sizes),
                dropout_rate=0.0,
            ).to(device)

            wrapper = NNWRReconstructionWrapper(model, mapper2d, decoder).to(device)
            wrapper.load_state_dict(state_dict, strict=False)
            wrapper.eval()

            # one deterministic forward pass is usually what you want for recon
            # (MC sampling for recon is possible, but start simple)
            with torch.inference_mode():
                N = input_data.shape[0]
                BATCH_R = 2**13
                recon_acc = []

                for start in range(0, N, BATCH_R):
                    end = min(start + BATCH_R, N)
                    xb = input_data[start:end]
                    _logits, recon_norm, _z2 = wrapper(xb)
                    recon_acc.append(recon_norm.detach().cpu().numpy())
                recon_norm = np.concatenate(
                    recon_acc, axis=0
                )  # (N, D) in normalized space

                # mean, std = load_scaler(scaler_path="scaler_nn_v0019.npz")
                mean, std = load_scaler(scaler_path="scaler_nn_v0030.npz")
                mean_vec = mean[OXIDES].to_numpy(dtype=np.float32)
                std_vec = std[OXIDES].to_numpy(dtype=np.float32)
                recon_out = recon_norm * std_vec[None, :] + mean_vec[None, :]

            # df aligned to original df index (including zircons/all-zero)
            recon_cols = [f"{c}_recon" for c in OXIDES]

            # recon DF on the non-zircon index (guaranteed aligned to recon_out row order)
            recon_nonza = pd.DataFrame(
                recon_out,
                index=non_za_df.index,
                columns=recon_cols,
                dtype=float,
            )

            # Expand to full df index (zircons/all-zero become NaN)
            recon_df = recon_nonza.reindex(df.index)

        # ---- Monte Carlo Bayesian averaging ----
        with torch.inference_mode():
            device = next(model.parameters()).device
            N = len(input_data)
            C = model.classes
            BATCH = 2**13
            K = 10 # MC passes per chunk

            probs_mean = torch.empty((N, C), device=device, dtype=torch.float32)
            probs_var  = torch.empty((N, C), device=device, dtype=torch.float32)

            for start in range(0, N, BATCH):
                end = min(start + BATCH, N)
                x = input_data[start:end]
                b = x.shape[0]

                done = 0
                acc = torch.zeros((b, C), device=device, dtype=torch.float32)
                acc2 = torch.zeros((b, C), device=device, dtype=torch.float32)

                while done < n_iterations:
                    kk = min(K, n_iterations - done)

                    # p = 0.0
                    for _ in range(kk):
                        logits = classifier(x)  # classifier-only forward
                        # p = p + torch.softmax(logits, dim=1)
                        s = torch.softmax(logits, dim=1)
                        acc += s
                        acc2 += s**2

                    done += kk

                probs_mean[start:end] = acc / float(n_iterations)
                probs_var[start:end] = (acc2 / float(n_iterations)) - probs_mean[start:end]**2

            probability_matrix = probs_mean.detach().cpu().numpy()
            std_matrix = np.sqrt(probs_var.detach().cpu().numpy())

        # ---- top-2 predictions ----
        top_two_indices = np.argsort(probability_matrix, axis=1)[:, -2:]
        first_probs = probability_matrix[
            np.arange(len(probability_matrix)), top_two_indices[:, 1]
        ]
        first_uncertainty = std_matrix[
        np.arange(len(std_matrix)), top_two_indices[:, 1]
        ]
        second_probs = probability_matrix[
            np.arange(len(probability_matrix)), top_two_indices[:, 0]
        ]
        first_mins = class2mineral_nn(top_two_indices[:, 1])
        second_mins = class2mineral_nn(top_two_indices[:, 0])

        result_df.loc[non_zircon_mask, "Predict_Mineral"] = first_mins
        result_df.loc[non_zircon_mask, "Prediction_Score"] = first_probs
        result_df.loc[non_zircon_mask, "Prediction_Score_Sigma"] = first_uncertainty
        result_df.loc[non_zircon_mask, "Second_Predict_Mineral"] = second_mins
        result_df.loc[non_zircon_mask, "Second_Prediction_Score"] = second_probs

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
    # ox_mask = result_df["Predict_Mineral"].isin(["Rhombohedral_Oxides", "Spinel_Group", "Oxide"])
    # # ox_mask = result_df["Predict_Mineral"].isin(["Oxide"])
    # _merge_subclass(ox_mask, OxideClassifier, want_sub=True)
    ox_mask = result_df["Predict_Mineral"].isin(["Rhombohedral_Oxides", "Spinel_Group", "Oxide"])
    if ox_mask.any():
        # Preserve the original NN label as the Submineral
        result_df.loc[ox_mask, "Submineral"] = result_df.loc[ox_mask, "Predict_Mineral"].values
        # Collapse Predict_Mineral to "Oxide"
        result_df.loc[ox_mask, "Predict_Mineral"] = "Oxide"

    if "Submineral" not in result_df.columns:
        result_df["Submineral"] = pd.Series(index=result_df.index, dtype="object")

    cols = list(result_df.columns)
    for col, after in [
        ("Prediction_Score_Sigma", "Prediction_Score"),
        ("Submineral", "Predict_Mineral"),
    ]:
        if col in cols and after in cols:
            cols.remove(col)
            cols.insert(cols.index(after) + 1, col)
    result_df = result_df[cols]

    if return_recon_oxides and recon_df is not None:
        result_df = pd.concat([result_df, recon_df], axis=1)

    return result_df, probability_matrix


def enable_mc_sampling(model, *, enable_dropout: bool):
    """
    Enables stochasticity for MC inference without breaking BatchNorm.
    Keeps BatchNorm in eval() mode, enables VariationalLayer sampling,
    and optionally enables Dropout.
 
    Parameters:
        model (nn.Module): The model to configure for MC sampling.
        enable_dropout (bool): If True, sets Dropout layers to train() mode.
 
    Returns:
        model (nn.Module): The configured model (modified in-place).
    """
    model.eval()  # baseline

    for m in model.modules():
        # keep BN deterministic
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

        # turn on variational sampling
        if isinstance(m, VariationalLayer):
            m.train()

        # optional dropout noise
        if enable_dropout and isinstance(m, nn.Dropout):
            m.train()

        # if dropout disabled, force dropout off
        if (not enable_dropout) and isinstance(m, nn.Dropout):
            m.eval()

    return model


# %%


def _downsample(Z, labels=None, max_points=250_000):
    """
    Randomly downsamples arrays to at most max_points rows.
 
    Parameters:
        Z (ndarray|None): 2D array to downsample.
        labels (ndarray|None): Corresponding label array.
        max_points (int): Maximum number of rows to keep.
 
    Returns:
        Z (ndarray|None): Downsampled array (or original if already small enough).
        labels (ndarray|None): Downsampled labels (or None).
    """

    if Z is None or Z.shape[0] <= max_points:
        return Z, labels
    idx = np.random.choice(Z.shape[0], size=max_points, replace=False)
    return Z[idx], labels[idx] if labels is not None else None


def load_nnwr_wrapper_for_latents(model_path=None, device=None):
    """
    Loads the trained NNWRReconstructionWrapper from a checkpoint.
 
    Parameters:
        model_path (str|None): Path to the .pt checkpoint. If None, defaults
            to the bundled model in the same directory as this module.
        device (str|None): Device string (e.g., 'cpu', 'cuda'). If None,
            auto-detects GPU availability.
 
    Returns:
        wrapper (nn.Module): The loaded model in evaluation mode.
        model_config (dict): Configuration dictionary from the checkpoint.
    """

    # Device Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load Checkpoint
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "nnwr_best_model_v0030.pt"
        )

    ckpt = torch.load(model_path, map_location=device)
    sd = ckpt["model_state_dict"]
    cfg = ckpt.get("model_config", {})

    # Extract Configuration with Fallbacks
    input_dim = int(cfg["input_dim"])

    # Initialize Submodules
    classifier = NNWRFeatureExtractor(
        input_dim=input_dim,
        classes=int(cfg.get("classes", 23)),
        hidden_layer_sizes=cfg["hidden_layer_sizes"],
        dropout_rate=float(cfg.get("dropout_rate", 0.0)),
        use_bayesian_feature_layer=bool(cfg.get("use_bayesian_feature_layer", True)),
        use_bayesian_classifier=bool(cfg.get("use_bayesian_classifier", False)),
    )

    mapper2d = NNWRLatentProjector(
        feat_dim=int(cfg["feat_dim"]),
        hidden=int(cfg.get("mapper_hidden", 16)),
        dropout_rate=0.0,
        nonlinear=bool(cfg.get("mapper_nonlinear", True)),
    )

    decoder = NNWRReconstructionDecoder(
        z_dim=2,
        output_dim=input_dim,
        decoder_hidden_sizes=list(cfg.get("decoder_hidden_sizes", [64, 32])),
        dropout_rate=0.0,
    )

    # Wrap, Load Weights, Transfer to Device, and set to Eval Mode
    wrapper = NNWRReconstructionWrapper(classifier, mapper2d, decoder).to(device)
    wrapper.load_state_dict(sd, strict=False)
    wrapper.eval()

    return wrapper, cfg


@torch.inference_mode()
def compute_z2_from_df(df, wrapper, batch_size=256, device=None):
    """
    Computes 2D latent representations (z2) and predicted class labels for a DataFrame.
 
    Parameters:
        df (pd.DataFrame): Input DataFrame with oxide columns.
        wrapper (NNWRReconstructionWrapper): Loaded NNWR model wrapper.
        batch_size (int): Batch size for inference.
        device (str|None): Device string. If None, uses the wrapper's current device.
 
    Returns:
        Z2_out (ndarray): (N, 2) array of 2D latent coordinates.
        Preds_out (ndarray): (N,) array of predicted class indices.
    """

    device = torch.device(device) if device else next(wrapper.parameters()).device
    wrapper.eval()

    X_df = df[OXIDES].fillna(0.0)
    X_norm = norm_data_nn(X_df)

    if isinstance(X_norm, pd.DataFrame):
        X_norm = X_norm.to_numpy(dtype=np.float32)
    else:
        X_norm = X_norm.astype(np.float32, copy=False)

    dataset = TensorDataset(torch.from_numpy(X_norm))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    N = len(dataset)
    Z2_out = np.empty((N, 2), dtype=np.float32)
    Preds_out = np.empty(N, dtype=np.int32)  # <-- Pre-allocate predictions array

    idx = 0
    for (x_batch,) in dataloader:
        x_batch = x_batch.to(device)

        # Grab logits as well as z2
        logits, _, z2 = wrapper(x_batch)

        # Get the predicted class index (highest logit)
        preds = logits.argmax(dim=1)

        batch_len = x_batch.size(0)
        Z2_out[idx : idx + batch_len] = z2.cpu().numpy()
        Preds_out[idx : idx + batch_len] = preds.cpu().numpy()  # <-- Save predictions
        idx += batch_len

    return Z2_out, Preds_out


def plot_z2_overlay(
    df,
    label_column="Predict_Mineral",
    title="Latent Space (z2) Overlay",
    ref_kws=None,
    new_kws=None,
    max_points=250_000,
    filename=None,
):
    """
    Plots a 2D latent space overlaying new data on top of reference (training) data.
    Loads pre-computed training latents as a background and projects the provided
    DataFrame samples as a foreground overlay.
 
    Parameters:
        df (pd.DataFrame): Input data to be projected into the latent space.
        label_column (str): Column name in df representing pre-computed labels.
        title (str): Title displayed at the top of the plot.
        ref_kws (dict|None): Keyword arguments for the background (training) scatter.
            Defaults to {"s": 10, "alpha": 0.10, "marker": "x", "edgecolors": "none"}.
        new_kws (dict|None): Keyword arguments for the foreground (new data) scatter.
        max_points (int): Maximum number of points to plot per layer.
        filename (str|None): Path to save the figure. If None, displays interactively.
    """
 
    # Load Training Background
    latent_path = os.path.join(
        os.path.dirname(__file__), "nnwr_latent_data_v0030.npz"
    )
    if os.path.exists(latent_path):
        with np.load(latent_path) as data:
            Z_ref = data["valid_latents"]
            labels_ref = data["valid_labels"]
    else:
        raise FileNotFoundError(
            f"Could not find {latent_path}. Please provide Z_ref manually."
        )

    # Compute Foreground Z2 and predictions
    wrapper, _ = load_nnwr_wrapper_for_latents()
    Z_new, _ = compute_z2_from_df(df, wrapper)

    if label_column not in df.columns:
        raise KeyError(
            f"Dataframe must contain '{label_column}' column for pre-classified plotting."
        )
    labels_new = df[label_column].values

    # Build mapping strictly from the REFERENCE labels
    _, label_names = unique_mapping_nn(labels_ref)
    name_to_id = {v: k for k, v in label_names.items()}

    mineral_rollup = {
        "Plagioclase": "Feldspar",
        "KFeldspar": "Feldspar",
        "Feldspar_Miscibility_Gap": "Feldspar",
        "Clinopyroxene": "Pyroxene",
        "Orthopyroxene": "Pyroxene",
        "Na-Pyroxene": "Pyroxene",
    }

    # Convert df labels from strings to ints if they aren't already
    if isinstance(labels_new[0], str):
        yn_ints = []
        for label in labels_new:
            clean_label = label.strip()

            # Translate granular labels to broad labels if they exist in the dictionary
            mapped_label = mineral_rollup.get(clean_label, clean_label)

            # Fetch the ID, defaulting to -1 if STILL not found
            yn_ints.append(name_to_id.get(mapped_label, -1))

        yn_ints = np.array(yn_ints)

        # Optional but highly recommended: warn if anything is STILL unmapped
        unmapped_mask = yn_ints == -1
        if unmapped_mask.any():
            unmapped_labels = set(np.array(labels_new)[unmapped_mask])
            print(
                f"WARNING: Skipping {unmapped_mask.sum()} points. Unrecognized labels: {unmapped_labels}"
            )
    else:
        yn_ints = labels_new

    Zr, yr = _downsample(np.asarray(Z_ref), labels_ref, max_points)
    Zn, yn = _downsample(np.asarray(Z_new), yn_ints, max_points)

    # Set up axes
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # Configure style parameters (removed 'cmap' from default_new)
    default_train = {
        "s": 10,
        "alpha": 0.10,
        "marker": "x",
        "edgecolors":
        "none"}
    default_df = {
        "s": 25,
        "alpha": 0.85,
        "marker": "o",
        "edgecolors": "black",
        "linewidths": 0.4,
    }

    ref_kws = {**default_train, **(ref_kws or {})}
    new_kws = {**default_df, **(new_kws or {})}

    # Create Custom Combined Colormap
    all_labels = np.concatenate([y for y in (yr, yn) if y is not None])
    valid_labels = all_labels[all_labels >= 0].astype(int)

    tab20 = plt.get_cmap("tab20")
    tab20b = plt.get_cmap("tab20b")
    combined_colors = [tab20(i) for i in range(20)] + [tab20b(i) for i in range(1)]

    # Shuffle with a fixed seed to maximize visual distance
    # random.seed(42)
    # random.shuffle(combined_colors)

    # Create the cmap and set a fixed normalization range
    cmap = mcolors.ListedColormap(combined_colors)
    norm = mcolors.Normalize(vmin=0, vmax=20)

    # Plot Reference Data (Background) and create dummy legend markers
    uniq_ref_classes = np.unique(yr).astype(int)
    ref_kws.pop("c", None)
    for cls in uniq_ref_classes:
        if cls < 0:
            continue  # Skip unmapped
        mask = yr == cls
        name = label_names.get(cls, f"Class {cls}")
        color = cmap(norm(cls))
        ax.scatter(Zr[mask, 0], Zr[mask, 1], color=color, **ref_kws)
        ax.scatter([], [], s=40, marker="o", color=color, ec="k", lw=0.5, label=name)

    # Plot new data in foreground
    uniq_new_classes = np.unique(yn).astype(int)
    for cls in uniq_new_classes:
        if cls < 0:
            continue
        mask = yn == cls
        color = cmap(norm(cls))
        ax.scatter(Zn[mask, 0], Zn[mask, 1], color=color, **new_kws)

    # Final formatting and legend
    if len(uniq_ref_classes) <= 30:
        ax.legend(
            bbox_to_anchor=(1.015, 1.01),
            loc="upper left",
            frameon=False,
            prop={"size": 9},
        )
    ax.set_xlabel("z2_1")
    ax.set_ylabel("z2_2")
    ax.set_title(title)
    ax.grid(True, alpha=0.15, linestyle="--")

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# %%


def plot_harker(
    df_train=None,
    train_minerals=None,
    overlay_datasets=None,
    oxides=OXIDES,
    x_oxide="SiO2",
    extra_pairs=None,
    plot_totals=False,
    title=None,
    train_mineral_col="Mineral",
    train_kws=None,
    new_kws=None,
):
    """
    Plots Harker diagrams for training data with optional study dataset overlays.
 
    Parameters:
        df_train (pd.DataFrame|None): Primary dataset containing training geochemical data.
        train_minerals (list|None): Mineral phases to filter and plot as background points.
        overlay_datasets (dict|None): {study_name: DataFrame} or
            {study_name: (DataFrame, kws_dict)} for datasets plotted at full opacity.
        oxides (list): Oxide names to plot on the Y-axes against x_oxide.
        x_oxide (str): Independent variable on the X-axis.
        extra_pairs (list[tuple]|None): Additional specific plots, e.g., [("CaO", "Na2O")].
        plot_totals (bool): If True, calculates and plots x_oxide vs. oxide sum.
        title (str|None): Figure suptitle.
        train_mineral_col (str): Column name for mineral labels in df_train.
        train_kws (dict|None): Scatter keywords for training data.
            Defaults to {"s": 20, "alpha": 0.1, "ec": "k", "lw": 0.25}.
        new_kws (dict|None): Default scatter keywords for overlay datasets.
            Defaults to {"s": 60, "alpha": 1.0, "ec": "k", "lw": 1}.
    """

    overlay_datasets = overlay_datasets or {}
    train_minerals = train_minerals or []

    # Configure style parameters to match your consistent coding style
    default_train = {"s": 20, "alpha": 0.1, "ec": "k", "lw": 0.25}
    default_new = {"s": 60, "alpha": 1.0, "ec": "k", "lw": 1}

    train_kws = {**default_train, **(train_kws or {})}
    new_kws = {**default_new, **(new_kws or {})}

    # Setup Colors/Markers for overlays
    overlay_colors = ["magenta", "cyan", "lime", "yellow", "orange"]
    overlay_markers = ["s", "^", "D", "o", "v"]

    # Build plotting pairs
    pairs = [(x_oxide, ox) for ox in oxides if ox != x_oxide]
    if extra_pairs:
        pairs.extend(extra_pairs)

    if plot_totals:
        target_y = "Total"
        pairs.append((x_oxide, target_y))

        # Safely calculate Total for df_train
        if df_train is not None:
            df_train = df_train.copy()
            available_ox = [ox for ox in OXIDES if ox in df_train.columns]
            df_train[target_y] = df_train[available_ox].sum(axis=1)

        # Safely calculate Total for overlay_datasets
        updated_overlays = {}
        for name, item in overlay_datasets.items():
            if isinstance(item, (tuple, list)) and len(item) == 2:
                df_ov, ind_kws = item
                df_ov = df_ov.copy()
                available_ox = [ox for ox in oxides if ox in df_ov.columns]
                df_ov[target_y] = df_ov[available_ox].sum(axis=1)
                updated_overlays[name] = (df_ov, ind_kws)
            else:
                df_ov = item.copy()
                available_ox = [ox for ox in oxides if ox in df_ov.columns]
                df_ov[target_y] = df_ov[available_ox].sum(axis=1)
                updated_overlays[name] = df_ov
        overlay_datasets = updated_overlays

    # Setup grid
    cols = 4
    n = len(pairs)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for ax, (x, y) in zip(axes, pairs):
        # 1. Plot Background Training Data
        if df_train is not None:
            for min_name in train_minerals:
                df_sub = df_train[df_train[train_mineral_col] == min_name]
                if not df_sub.empty and x in df_sub.columns and y in df_sub.columns:
                    ax.scatter(
                        df_sub[x], df_sub[y], label=f"Train: {min_name}", **train_kws
                    )

        # Plot Overlay Datasets
        for i, (name, item) in enumerate(overlay_datasets.items()):
            # Allow individual dataset overrides: {'Name': (df, {custom_kws})}
            if isinstance(item, (tuple, list)) and len(item) == 2:
                df_ov, individual_kws = item
            else:
                df_ov, individual_kws = item, {}

            # Combine: Default < New_Kws < Individual_Kws
            style = {
                "c": overlay_colors[i % len(overlay_colors)],
                "marker": overlay_markers[i % len(overlay_markers)],
                **new_kws,
                **individual_kws,
            }

            if x in df_ov.columns and y in df_ov.columns:
                ax.scatter(df_ov[x], df_ov[y], label=name, **style)

        ax.set_xlabel(format_oxide_label(x))
        ax.set_ylabel(format_oxide_label(y))

    # Legend Logic (Identical to your current implementation)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        has_empty_slot = (n % cols) != 0
        leg_loc, leg_bbox = (
            ("center left", (0.775, 0.2))
            if has_empty_slot
            else ("center left", (1.0, 0.5))
        )
        leg = fig.legend(
            by_label.values(),
            by_label.keys(),
            loc=leg_loc,
            bbox_to_anchor=leg_bbox,
            frameon=True,
        )
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)  # Ensure legend is opaque

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

