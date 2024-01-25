import unittest
from unittest.mock import patch, mock_open
from tempfile import TemporaryDirectory

import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import mineralML as mm


class test_FeatureDataset(unittest.TestCase):

    def test_initialization(self):
        # Test with 1D array
        x1d = np.random.rand(10)
        dataset1d = mm.FeatureDataset(x1d)
        self.assertEqual(dataset1d.x.ndim, 2)

        # Test with 2D array
        x2d = np.random.rand(10, 5)
        dataset2d = mm.FeatureDataset(x2d)
        self.assertTrue(np.array_equal(dataset2d.x, x2d))

    def test_len(self):
        x = np.random.rand(10, 5)
        dataset = mm.FeatureDataset(x)
        self.assertEqual(len(dataset), 10)

    def test_getitem(self):
        x = np.random.rand(10, 5)
        dataset = mm.FeatureDataset(x)
        self.assertTrue(torch.equal(dataset[0], torch.Tensor(x[0])))


class test_LabelDataset(unittest.TestCase):

    def test_initialization(self):
        x = np.random.rand(10, 5)
        labels = np.random.randint(0, 2, 10)
        dataset = mm.LabelDataset(x, labels)

        self.assertTrue(torch.equal(dataset.x, torch.from_numpy(x).type(torch.FloatTensor)))
        self.assertTrue(torch.equal(dataset.labels, torch.from_numpy(labels).type(torch.LongTensor)))

    def test_len(self):
        x = np.random.rand(10, 5)
        labels = np.random.randint(0, 2, 10)
        dataset = mm.LabelDataset(x, labels)
        self.assertEqual(len(dataset), 10)

    def test_getitem(self):
        x = np.random.rand(10, 5)
        labels = np.random.randint(0, 2, 10)
        dataset = mm.LabelDataset(x, labels)
        sample, label = dataset[0]
        self.assertTrue(torch.equal(sample, torch.FloatTensor(x[0])))
        self.assertEqual(label.item(), labels[0])


class test_load_functions(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load_df(self, mock_read_csv):
        # Create a dummy DataFrame
        mock_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_read_csv.return_value = mock_df

        # Call the function
        df = mm.load_df('dummy_path.csv')

        # Assert read_csv was called correctly
        mock_read_csv.assert_called_with('dummy_path.csv', index_col=0)

        # Assert the returned DataFrame is correct
        pd.testing.assert_frame_equal(df, mock_df)

    @patch('numpy.load')
    @patch('os.path.dirname')
    def test_load_scaler(self, mock_dirname, mock_np_load):
        # Mock the current directory
        mock_dirname.return_value = '/path/to/current/dir'

        # Set up the mock return values for numpy.load
        mean_array = np.random.rand(10)  # Create an array with 10 elements
        std_array = np.random.rand(10)   # Create an array with 10 elements
        mock_np_load.return_value = {'mean': mean_array, 'scale': std_array}

        # Update the function call to include scaler_path
        mean, std = mm.load_scaler('scaler_ae.npz')

        expected_index = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
        self.assertTrue((mean == pd.Series(mean_array, index=expected_index)).all())
        self.assertTrue((std == pd.Series(std_array, index=expected_index)).all())

        # Assert numpy.load was called with the correct full file path
        full_scaler_path = '/path/to/current/dir/scaler_ae.npz'
        mock_np_load.assert_called_with(full_scaler_path)

        # Test for FileNotFoundError
        mock_np_load.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            mm.load_scaler('non_existing_path.npz')


class test_CreateDataLoader(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'Mineral': ['Mineral1', 'Mineral2', 'Mineral1', 'Mineral2', 'Mineral1', 
                        'Mineral2', 'Mineral1', 'Mineral2', 'Mineral1', 'Mineral2']
        })

    @patch('mm.load_minclass_nn')
    @patch('mm.norm_data')
    def test_create_dataloader(self, mock_norm_data, mock_load_minclass_nn):
        # Mock return values for dependencies
        mock_norm_data.return_value = self.df
        mock_load_minclass_nn.return_value = (['Mineral1', 'Mineral2'], None)

        # Create DataLoader
        dataloader = mm.create_dataloader(self.df, batch_size=2, shuffle=False)

        # Check if DataLoader is created and has correct properties
        self.assertIsNotNone(dataloader, "DataLoader not created")
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader, "Returned object is not a DataLoader")
        
        # Check DataLoader's batch size
        for batch in dataloader:
            self.assertEqual(len(batch), 2)  # Assuming LabelDataset returns a tuple
            break  # We just need to check the first batch

        # Check if dependent functions are called
        mock_norm_data.assert_called_once_with(self.df)
        mock_load_minclass_nn.assert_called_once()


class MockNetwork(nn.Module):
    def __init__(self):
        super(MockNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.bn1 = nn.BatchNorm2d(20)

def is_normal_tensor(tensor, mean, std):
    return torch.all(torch.abs(tensor - mean) < 3 * std)

class test_weights(unittest.TestCase):

    def test_weights_init(self):
        # Create a mock network
        net = MockNetwork()

        # Apply the weights_init function
        net.apply(mm.weights_init)

        # Check if weights and biases of BatchNorm layers are initialized correctly
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Check weights
                self.assertTrue(is_normal_tensor(module.weight.data, 1.0, 0.02), 
                                "Weights of BatchNorm layer are not properly initialized")
                # Check biases
                self.assertTrue(torch.all(module.bias.data == 0), 
                                "Biases of BatchNorm layer are not initialized to 0")

class test_same_seeds(unittest.TestCase):

    def test_reproducibility(self):
        seed = 42

        # Set seeds and generate random numbers
        mm.same_seeds(seed)
        torch_rand = torch.rand(5).tolist()
        np_rand = np.random.rand(5).tolist()
        py_rand = [random.random() for _ in range(5)]

        # Set seeds again and generate another set of random numbers
        mm.same_seeds(seed)
        torch_rand_repeat = torch.rand(5).tolist()
        np_rand_repeat = np.random.rand(5).tolist()
        py_rand_repeat = [random.random() for _ in range(5)]

        # Check if the generated numbers are the same in both instances
        self.assertEqual(torch_rand, torch_rand_repeat, "PyTorch random numbers do not match")
        self.assertEqual(np_rand, np_rand_repeat, "NumPy random numbers do not match")
        self.assertEqual(py_rand, py_rand_repeat, "Python random numbers do not match")



class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.conv = nn.Conv2d(1, 20, 5)


class test_SaveModel(unittest.TestCase):

    def setUp(self):
        self.model = MockModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def test_save_model_ae(self):
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "model_ae.pth")
            mm.save_model_ae(self.model, self.optimizer, filepath)

            # Check if file exists
            self.assertTrue(os.path.exists(filepath))

            # Load and check the content
            checkpoint = torch.load(filepath)
            self.assertIn('params', checkpoint)
            self.assertIn('optimizer', checkpoint)
            self.assertDictEqual(checkpoint['params'], self.model.state_dict())

    def test_save_model_nn(self):
        best_model_state = self.model.state_dict()
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "model_nn.pth")
            mm.save_model_nn(self.optimizer, best_model_state, filepath)

            # Check if file exists
            self.assertTrue(os.path.exists(filepath))

            # Load and check the content
            checkpoint = torch.load(filepath)
            self.assertIn('params', checkpoint)
            self.assertIn('optimizer', checkpoint)
            self.assertDictEqual(checkpoint['params'], best_model_state)


def save_checkpoint(model, optimizer, path):
    check_point = {'params': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(check_point, path)

class test_LoadModel(unittest.TestCase):

    def setUp(self):
        self.model = MockModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def test_load_model(self):
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "model_checkpoint.pth")
            save_checkpoint(self.model, self.optimizer, filepath)

            # Create new model and optimizer for loading
            loaded_model = MockModel()
            loaded_optimizer = optim.SGD(loaded_model.parameters(), lr=0.001, momentum=0.9)

            # Load the saved states
            mm.load_model(loaded_model, loaded_optimizer, filepath)

            # Check if model state is correctly loaded
            for param, loaded_param in zip(self.model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(param.data, loaded_param.data))

            # Check if optimizer state is correctly loaded
            for original_group, loaded_group in zip(self.optimizer.param_groups, loaded_optimizer.param_groups):
                self.assertEqual(original_group['lr'], loaded_group['lr'])
                self.assertEqual(original_group['momentum'], loaded_group['momentum'])


if __name__ == '__main__':
    unittest.main()
