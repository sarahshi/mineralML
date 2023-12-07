import unittest
from unittest.mock import patch, mock_open

import os
import glob
import numpy as np
import pandas as pd
import torch
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
    def test_load_scaler(self, mock_np_load):
        # Set up the mock return values
        mean_array = np.random.rand(10)  # Create an array with 10 elements
        std_array = np.random.rand(10)   # Create an array with 10 elements
        mock_np_load.return_value = {'mean': mean_array, 'scale': std_array}

        mean, std = mm.load_scaler()

        expected_index = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
        self.assertTrue((mean == pd.Series(mean_array, index=expected_index)).all())
        self.assertTrue((std == pd.Series(std_array, index=expected_index)).all())

        # Test for FileNotFoundError
        mock_np_load.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            mm.load_scaler()


if __name__ == '__main__':
    unittest.main()
