# %%

import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from .constants import OXIDES

# %% 

class LabelDataset(Dataset):
    """
    A PyTorch Dataset subclass designed to contain features and labels for machine learning. It verifies 
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


def load_df(filepath, index_col=0, **kwargs):
    """
    Loads a DataFrame from a CSV/Excel file specified by the given file path. The first 
    column of the CSV is set as the index of the DataFrame.

    Parameters:
        filepath (str): The path to the CSV file to be loaded.
        index_col : int | str | None, default 0
            Column to use as the row labels of the DataFrame.
        **kwargs
            Passed through to pandas reader:
            - pd.read_csv for CSV
            - pd.read_excel for Excel

    Returns:
        df (DataFrame): Pandas DataFrame containing the data from the CSV file.
    """

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath, index_col=index_col, **kwargs)

    if ext in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        return pd.read_excel(filepath, index_col=index_col, **kwargs)
    
    raise ValueError(
        f"Unsupported file extension '{ext}'. Expected .csv or an Excel file (.xlsx/.xls/.xlsm/.xlsb)."
    )


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

    # Attempt to load the scaler and handle exceptions if the loading fails.
    try:
        # Load the mean and standard deviation using numpy's load function
        npz = np.load(scaler_path)
        mean = pd.Series(npz['mean'], index=OXIDES)
        std = pd.Series(npz['scale'], index=OXIDES)

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
 
