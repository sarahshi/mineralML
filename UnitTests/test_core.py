import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import mineralML as mm


class test_LabelDataset(unittest.TestCase):

    def test_initialization(self):
        x = np.random.rand(11, 5)
        labels = np.random.randint(0, 2, 11)
        dataset = mm.LabelDataset(x, labels)

        self.assertTrue(torch.equal(dataset.x, torch.from_numpy(x).type(torch.FloatTensor)))
        self.assertTrue(torch.equal(dataset.labels, torch.from_numpy(labels).type(torch.LongTensor)))

    def test_initialization_non2d(self):
        x = np.random.rand(3, 4, 5)
        labels = np.random.randint(0, 2, 12)
        dataset = mm.LabelDataset(x, labels)
 
        expected_x = x.reshape(-1, x.shape[-1])
        np.testing.assert_array_equal(dataset.x, expected_x)
        np.testing.assert_array_equal(dataset.labels, labels)

    def test_len(self):
        x = np.random.rand(11, 5)
        labels = np.random.randint(0, 2, 11)
        dataset = mm.LabelDataset(x, labels)
        self.assertEqual(len(dataset), 11)

    def test_len_non2d(self):
        x = np.random.rand(3, 4, 5)
        labels = np.random.randint(0, 2, 12)
        dataset = mm.LabelDataset(x, labels)
        self.assertEqual(len(dataset), 12)
 
    def test_getitem(self):
        x = np.random.rand(11, 5)
        labels = np.random.randint(0, 2, 11)
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
 
    @patch('pandas.read_excel')
    def test_load_df_excel(self, mock_read_excel):
        # Create a dummy DataFrame
        mock_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_read_excel.return_value = mock_df
 
        # Test each supported Excel extension
        for ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
            mock_read_excel.reset_mock()
            filepath = f'dummy_path{ext}'
            df = mm.load_df(filepath)
 
            # Assert read_excel was called correctly
            mock_read_excel.assert_called_with(filepath, index_col=0)
 
            # Assert the returned DataFrame is correct
            pd.testing.assert_frame_equal(df, mock_df)
 
    def test_load_df_unsupported_extension(self):
        # Assert ValueError is raised for unsupported file types
        with self.assertRaises(ValueError):
            mm.load_df('dummy_path.json')
 
        with self.assertRaises(ValueError):
            mm.load_df('dummy_path.parquet')
 
    @patch('pandas.read_csv')
    def test_load_df_kwargs(self, mock_read_csv):
        # Create a dummy DataFrame
        mock_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mock_read_csv.return_value = mock_df
 
        # Call with extra kwargs
        df = mm.load_df('dummy_path.csv', index_col=None, sep='\t')
 
        # Assert read_csv was called with forwarded kwargs
        mock_read_csv.assert_called_with('dummy_path.csv', index_col=None, sep='\t')
        pd.testing.assert_frame_equal(df, mock_df)
 
    @patch('numpy.load')
    @patch('os.path.dirname')
    def test_load_scaler(self, mock_dirname, mock_np_load):
        # Mock the current directory
        mock_dirname.return_value = '/path/to/current/dir'
 
        # Set up the mock return values for numpy.load
        mean_array = np.random.rand(11)  # Create an array with 11 elements
        std_array = np.random.rand(11)   # Create an array with 11 elements
        mock_np_load.return_value = {'mean': mean_array, 'scale': std_array}
 
        # Update the function call to include scaler_path
        mean, std = mm.load_scaler('scaler_nn_v001.npz')
 
        expected_index = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3', 'P2O5']
        self.assertTrue((mean == pd.Series(mean_array, index=expected_index)).all())
        self.assertTrue((std == pd.Series(std_array, index=expected_index)).all())
 
        # Assert numpy.load was called with the correct full file path
        full_scaler_path = '/path/to/current/dir/scaler_nn_v001.npz'
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

    def is_normal_distribution(self, tensor, mean, std, tolerance=0.05):
        # Check if the mean of the tensor is close to the expected mean
        mean_close = math.isclose(tensor.mean().item(), mean, abs_tol=tolerance)
        # Check if the standard deviation of the tensor is close to the expected std
        std_close = math.isclose(tensor.std().item(), std, abs_tol=tolerance)
        return mean_close and std_close

    def setUp(self):
        self.net = test_NetworkWeights.MockNetwork()

    def test_weights_init(self):
        # Apply the weights_init function
        self.net.apply(mm.weights_init)

        # Check if weights and biases of BatchNorm layers are initialized correctly
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm2d):
                print("BatchNorm2d weights: ", module.weight.data)
                print("BatchNorm2d weights mean: ", module.weight.data.mean().item())
                print("BatchNorm2d weights std: ", module.weight.data.std().item())

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
 
    def test_load_model_without_optimizer(self):
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "model_checkpoint.pth")
            self.save_checkpoint(self.model, self.optimizer, filepath)
 
            # Create new model and load without optimizer
            loaded_model = MockModel()
            mm.load_model(loaded_model, optimizer=None, path=filepath)
 
            # Check if model state is correctly loaded
            for param, loaded_param in zip(self.model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(param.data, loaded_param.data))
 

class test_ExportPredictionsToExcel(unittest.TestCase):
 
    def setUp(self):
        self.results_df = pd.DataFrame({
            'SiO2': [50.1, 40.2, 55.3, 38.7, 52.0],
            'MgO': [10.0, 45.0, 1.5, 42.0, 8.0],
            'Predict_Mineral': ['Plagioclase', 'Olivine', 'Plagioclase', 'Olivine', 'Clinopyroxene'],
            'Predict_Score': [0.95, 0.88, 0.91, 0.97, 0.85],
        })
 
    def test_export_creates_file(self):
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "test_results.xlsx")
            returned_path = mm.export_predictions_to_excel(self.results_df, filename=filepath)
 
            # Check if file exists and return value matches
            self.assertTrue(os.path.exists(filepath))
            self.assertEqual(returned_path, filepath)
 
    def test_export_all_sheet(self):
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "test_results.xlsx")
            mm.export_predictions_to_excel(self.results_df, filename=filepath)
 
            # Read back the "All" sheet and verify contents
            all_df = pd.read_excel(filepath, sheet_name="All")
            self.assertEqual(len(all_df), len(self.results_df))
            self.assertListEqual(list(all_df.columns), list(self.results_df.columns))
 
    def test_export_mineral_sheets(self):
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "test_results.xlsx")
            mm.export_predictions_to_excel(self.results_df, filename=filepath)
 
            # Read back each mineral sheet and verify row counts
            expected_minerals = self.results_df['Predict_Mineral'].unique()
            xl = pd.ExcelFile(filepath)
            for mineral in expected_minerals:
                self.assertIn(mineral, xl.sheet_names)
                mineral_df = pd.read_excel(filepath, sheet_name=mineral)
                expected_count = len(self.results_df[self.results_df['Predict_Mineral'] == mineral])
                self.assertEqual(len(mineral_df), expected_count)
 
    def test_export_missing_column_raises(self):
        # DataFrame without Predict_Mineral should raise ValueError
        bad_df = pd.DataFrame({'SiO2': [50.1], 'MgO': [10.0]})
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "test_results.xlsx")
            with self.assertRaises(ValueError):
                mm.export_predictions_to_excel(bad_df, filename=filepath)
 
    def test_export_long_mineral_name_truncated(self):
        # Sheet names are capped at 31 characters
        long_name = "A" * 50
        df = pd.DataFrame({
            'SiO2': [50.1],
            'Predict_Mineral': [long_name],
        })
        with TemporaryDirectory() as tmp_dir:
            filepath = os.path.join(tmp_dir, "test_results.xlsx")
            mm.export_predictions_to_excel(df, filename=filepath)
 
            xl = pd.ExcelFile(filepath)
            # "All" sheet plus the truncated mineral sheet
            self.assertEqual(len(xl.sheet_names), 2)
            mineral_sheet = [s for s in xl.sheet_names if s != "All"][0]
            self.assertLessEqual(len(mineral_sheet), 31)


if __name__ == '__main__':
    unittest.main()
