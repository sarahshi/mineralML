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


class test_NetworkWeights(unittest.TestCase):

    class MockNetwork(nn.Module):
        def __init__(self):
            super(test_NetworkWeights.MockNetwork, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.bn1 = nn.BatchNorm2d(20)

    def is_normal_distribution(self, tensor, mean, std, num_std=3):
        # Calculate Z-score
        z_scores = (tensor - mean) / std
        # Check if values are within num_std standard deviations
        return torch.all(torch.abs(z_scores) < num_std).item()

    def setUp(self):
        self.net = test_NetworkWeights.MockNetwork()

    def test_weights_init(self):
        # Apply the weights_init function
        self.net.apply(mm.weights_init)

        # Check if weights and biases of BatchNorm layers are initialized correctly
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d):
                # Check weights
                self.assertTrue(self.is_normal_distribution(module.weight.data, 1.0, 0.02),
                                "Weights of BatchNorm layer are not properly initialized")
                # Check biases
                self.assertTrue(torch.all(module.bias.data == 0).item(), 
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

    def compare_state_dicts(self, dict1, dict2):
        self.assertEqual(set(dict1.keys()), set(dict2.keys()))
        for key in dict1:
            self.assertTrue(torch.equal(dict1[key], dict2[key]), f"Mismatch in tensors for key: {key}")

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
            self.compare_state_dicts(checkpoint['params'], self.model.state_dict())

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
            self.compare_state_dicts(checkpoint['params'], best_model_state)


class test_LoadModel(unittest.TestCase):

    def setUp(self):
        self.model = MockModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def save_checkpoint(self, model, optimizer, path):
        check_point = {'params': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(check_point, path)

    def test_load_model(self):
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "model_checkpoint.pth")
            self.save_checkpoint(self.model, self.optimizer, filepath)  # Use self here

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


class test_MineralSupergroup(unittest.TestCase):

    def test_mineral_supergroup(self):
        # Create a sample DataFrame
        data = {
            'Predict_Mineral': ['Orthopyroxene', 'Clinopyroxene', 'KFeldspar', 
                                'Plagioclase', 'Spinel', 'Ilmenite', 'Magnetite', 'Other']
        }
        df = pd.DataFrame(data)

        # Call the function
        result_df = mm.mineral_supergroup(df)

        # Expected supergroups
        expected_supergroups = ['Pyroxene', 'Pyroxene', 'Feldspar', 
                                'Feldspar', 'Oxide', 'Oxide', 'Oxide', 'Other']

        # Check if the Supergroup column is added and correctly classified
        self.assertIn('Supergroup', result_df.columns, "Supergroup column not added")
        self.assertListEqual(result_df['Supergroup'].tolist(), expected_supergroups,
                             "Supergroup classification is incorrect")
    

if __name__ == '__main__':
    unittest.main()
