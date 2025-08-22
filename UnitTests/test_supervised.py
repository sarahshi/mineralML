import unittest
import numpy as np
import pandas as pd

import torch
from math import sqrt

import mineralML as mm


def _get_oxides():
    # Be tolerant to where OXIDES lives
    if hasattr(mm, "constants") and hasattr(mm.constants, "OXIDES"):
        return mm.constants.OXIDES
    if hasattr(mm, "OXIDES"):
        return mm.OXIDES
    raise AttributeError("Could not find OXIDES in mineralML.")

class mineralML_supervised(unittest.TestCase):
    def setUp(self):
        self.data = {
            "SampleID": [72065, 72066, 31890, 31891, 59237, 59238, 37643, 37644],
            "Mineral": [
                "Amphibole", "Amphibole",
                "Pyroxene",  "Pyroxene",
                "Garnet",    "Garnet",
                "Olivine",   "Olivine",
            ],
            "SiO2":  [40, 39.7, 51.49, 51.15, 39.8, 40.2, 40.31, 38.99],
            "TiO2":  [3.1, 3.2, 0.6,   0.53,  0.6,  0.6,  0.01,  0.08],
            "Al2O3": [16.1, 16,  2.57,  2.57, 22.5, 23.4, 0.01,  np.nan],
            "Cr2O3": [0.08, 0.06, 0.24, 0.19, np.nan, np.nan, 0.06, 0.05],
            "FeOt":  [12,   13,   6.98, 5.55, 16.1, 15.1, 11.88, 19.2],
            "MnO":   [0.16, 0.17, 0.22, 0.16, 0.5,  0.5,  0.18,  0.25],
            "MgO":   [10.2, 9.5,  16.77,16.37,9.8, 12.3, 47.02, 40.73],
            "CaO":   [10,   10.7, 19.42,21.36,10.7, 7.9,  0.08,  0.26],
            "Na2O":  [3.1,  2.9,  0.25, 0.32, np.nan, np.nan, np.nan, np.nan],
            "K2O":   [1.9,  1.7,  np.nan,np.nan, np.nan, np.nan, np.nan, np.nan],
            "P2O5":  [np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan],
        }
        self.df = pd.DataFrame(self.data)

    def test_load_minclass_nn(self):
        min_cat, mapping = mm.load_minclass_nn()

        # Robust checks (less brittle than fully hard-coding)
        self.assertIsInstance(min_cat, list)
        self.assertIsInstance(mapping, dict)
        # Keys are 0..N-1 and values match min_cat order
        self.assertEqual(sorted(mapping.keys()), list(range(len(min_cat))))
        self.assertEqual([mapping[i] for i in range(len(min_cat))], min_cat)

        # Sanity: core classes exist (names from your current mapping)
        required = {"Amphibole", "Pyroxene", "Garnet", "Olivine", "Spinels"}
        self.assertTrue(required.issubset(set(min_cat)))

        # If you want to **freeze** the exact mapping, uncomment and paste your mapping here:
        # expected_mapping = {
        #     0: 'Amphibole', 1: 'Apatite', 2: 'Biotite', 3: 'Calcite', 4: 'Chlorite',
        #     5: 'Epidote', 6: 'Feldspar', 7: 'Garnet', 8: 'Glass', 9: 'Kalsilite',
        #     10: 'Leucite', 11: 'Melilite', 12: 'Muscovite', 13: 'Nepheline',
        #     14: 'Olivine', 15: 'Pyroxene', 16: 'Quartz', 17: 'Rhombohedral_Oxides',
        #     18: 'Rutile', 19: 'Serpentine', 20: 'Spinels', 21: 'Titanite',
        #     22: 'Tourmaline', 23: 'Zircon'
        # }
        # self.assertEqual(mapping, expected_mapping)

    def test_prep_df_nn(self):
        df_cleaned = mm.prep_df_nn(self.df.copy())

        # No NaNs after cleaning (oxides filled with 0, Mineral preserved)
        self.assertEqual(int(df_cleaned.isnull().sum().sum()), 0)
        self.assertEqual(df_cleaned.index.name, "SampleID")

        oxides = set(_get_oxides())
        # Required columns: all oxides + ZrO2 + Mineral
        expected_cols = oxides.union({"ZrO2", "Mineral"})
        self.assertTrue(expected_cols.issubset(set(df_cleaned.columns)))

    def test_norm_data_nn(self):
        # Prepare
        df_cleaned = mm.prep_df_nn(self.df.copy())
        oxides = _get_oxides()

        # Under test
        normalized_data = mm.norm_data_nn(df_cleaned)
        self.assertEqual(normalized_data.shape, (len(df_cleaned), len(oxides)))

        # Compute expected normalization directly from the scaler
        mean, std = mm.load_scaler("scaler_nn_v001.npz")
        # Ensure Series aligned to oxides
        mean = mean.reindex(oxides)
        std = std.reindex(oxides)

        expected = (df_cleaned[oxides] - mean.values) / std.values
        np.testing.assert_allclose(
            normalized_data, expected.to_numpy(), rtol=1e-6, atol=1e-6
        )

    def test_unique_mapping_nn(self):
        # Use indices that exist in the CURRENT mapping (no more Clinopyroxene/Spinel)
        # Choose a small set: Amphibole(0), Pyroxene(15), Garnet(7), Olivine(14), Spinels(20)
        pred_class = np.array([0, 15, 7, 14, 20, 0, 7, 20])

        unique, valid_mapping = mm.unique_mapping_nn(pred_class)

        # Expected unique set (order not guaranteed)
        expected_unique = np.array(sorted({0, 7, 14, 15, 20}))
        np.testing.assert_array_equal(np.sort(unique), expected_unique)

        # Names from the **loaded** mapping (robust to future reorderings)
        _, mapping = mm.load_minclass_nn()
        expected_valid_mapping = {i: mapping[i] for i in expected_unique}
        self.assertEqual(valid_mapping, expected_valid_mapping)

    def test_class2mineral_nn(self):
        pred_class = np.array([0, 15, 7, 14, 20, 0, 7, 20])
        pred_mineral = mm.class2mineral_nn(pred_class)

        # Build expected labels from the current mapping
        _, mapping = mm.load_minclass_nn()
        expected_pred_mineral = np.array([mapping[i] for i in pred_class])

        np.testing.assert_array_equal(pred_mineral, expected_pred_mineral)


class test_variational_layer(unittest.TestCase):
    def setUp(self):
        self.input_features = 11
        self.output_features = 3
        self.layer = mm.VariationalLayer(self.input_features, self.output_features)
        self.input = torch.randn(self.batch_size, self.input_features)

    def test_initialization(self):
        self.assertEqual(
            self.layer.weight_mu.size(), (self.output_features, self.input_features)
        )
        self.assertEqual(
            self.layer.weight_rho.size(), (self.output_features, self.input_features)
        )
        self.assertEqual(self.layer.bias_mu.size(), (self.output_features,))
        self.assertEqual(self.layer.bias_rho.size(), (self.output_features,))

        std = 1.0 / sqrt(self.input_features)
        self.assertTrue(
            torch.all(self.layer.weight_mu.data <= std)
            and torch.all(self.layer.weight_mu.data >= -std)
        )
        self.assertTrue(
            torch.all(self.layer.weight_rho.data <= std)
            and torch.all(self.layer.weight_rho.data >= -std)
        )

    def test_forward_pass(self):
        output = self.layer(self.input)
        self.assertEqual(output.size(), (self.input_features, self.output_features))

    def test_kl_divergence(self):
        kl_div = self.layer.kl_divergence()
        self.assertIsInstance(kl_div, torch.Tensor)
        self.assertGreaterEqual(kl_div.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
