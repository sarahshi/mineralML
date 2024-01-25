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


class test_InsertTotals(unittest.TestCase):

    def test_insert_totals(self):
        # Create a sample DataFrame
        df = pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]})
        mm.insert_totals(df)

        # Expected DataFrame after insert_totals
        # Ensure the index is of the same type as df's index
        expected_df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [3.0, 4.0, 7.0], 'sum_row': [4.0, 6.0, 10.0]}, 
                                   index=[0, 1, 'sum_col'])

        # Check if DataFrame is modified correctly
        pd.testing.assert_frame_equal(df, expected_df)

    def test_insert_totals(self):
        # Create a sample DataFrame
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        mm.insert_totals(df)

        # Expected DataFrame after insert_totals
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'B': [3, 4, 7], 'sum_row': [4, 6, 10]}, 
                                   index=['0', '1', 'sum_col'])

        # Check if DataFrame is modified correctly
        pd.testing.assert_frame_equal(df, expected_df)


class test_ConfigCellTextAndColors(unittest.TestCase):

    @patch('matplotlib.text.Text')
    def test_config_cell_text_and_colors(self, mock_text):
        # Mock parameters
        array_df = np.array([[1, 2], [3, 4]])
        lin, col = 0, 0
        oText = mock_text()
        facecolors = np.array([[1, 2, 3, 4]])
        posi, fz, fmt = 0, 12, ".2f"
        show_null_values = 0

        # Call the function
        text_add, text_del = mm.config_cell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values)

        self.assertIsInstance(text_add, list, "text_add should be a list")
        self.assertIsInstance(text_del, list, "text_del should be a list")


class test_ppMatrix(unittest.TestCase):

    @patch('matplotlib.pyplot.figure')
    @patch('seaborn.heatmap')
    def test_pp_matrix(self, mock_heatmap, mock_figure):
        # Create a sample DataFrame
        df_cm = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

        # Call the function
        mm.pp_matrix(df_cm)

        # Check if seaborn's heatmap is called
        mock_heatmap.assert_called()


if __name__ == '__main__':
    unittest.main()
