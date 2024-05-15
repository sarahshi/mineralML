import unittest
import numpy as np
import pandas as pd
import mineralML as mm


class mineralML_supervised(unittest.TestCase):

    def setUp(self):
        self.data = {
            "SampleID": [72065, 72066, 31890, 31891, 59237, 59238, 37643, 37644],
            "Mineral": [
                "Amphibole",
                "Amphibole",
                "Clinopyroxene",
                "Clinopyroxene",
                "Garnet",
                "Garnet",
                "Olivine",
                "Olivine",
            ],
            "SiO2": [40, 39.7, 51.49, 51.15, 39.8, 40.2, 40.31, 38.99],
            "TiO2": [3.1, 3.2, 0.6, 0.53, 0.6, 0.6, 0.01, 0.08],
            "Al2O3": [16.1, 16, 2.57, 2.57, 22.5, 23.4, 0.01, np.nan],
            "Cr2O3": [0.08, 0.06, 0.24, 0.19, np.nan, np.nan, 0.06, 0.05],
            "FeOt": [12, 13, 6.98, 5.55, 16.1, 15.1, 11.88, 19.2],
            "MnO": [0.16, 0.17, 0.22, 0.16, 0.5, 0.5, 0.18, 0.25],
            "MgO": [10.2, 9.5, 16.77, 16.37, 9.8, 12.3, 47.02, 40.73],
            "CaO": [10, 10.7, 19.42, 21.36, 10.7, 7.9, 0.08, 0.26],
            "Na2O": [3.1, 2.9, 0.25, 0.32, np.nan, np.nan, np.nan, np.nan],
            "K2O": [1.9, 1.7, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }

        self.df = pd.DataFrame(self.data)


    def test_load_minclass_nn(self):
        # Load actual mineral classes
        min_cat, mapping = mm.load_minclass_nn()

        expected_mapping = {
            0: 'Amphibole',
            1: 'Biotite',
            2: 'Clinopyroxene',
            3: 'Garnet',
            4: 'Ilmenite',
            5: 'KFeldspar',
            6: 'Magnetite',
            7: 'Muscovite',
            8: 'Olivine',
            9: 'Orthopyroxene',
            10: 'Plagioclase',
            11: 'Spinel'
        }

        expected_min_cat = list(expected_mapping.values())

        # Verify expected output
        self.assertEqual(min_cat, expected_min_cat)
        self.assertEqual(mapping, expected_mapping)


    def test_prep_df_nn(self):
        df_cleaned = mm.prep_df_nn(self.df.copy())

        # Verify that NaN values were replaced correctly and DataFrame is cleaned
        self.assertEqual(
            df_cleaned.isnull().sum().sum(), 0
        )  # No NaN values should be present
        self.assertEqual(df_cleaned.index.name, "SampleID")
        self.assertEqual(
            set(df_cleaned.columns),
            set(
                [
                    "SiO2",
                    "TiO2",
                    "Al2O3",
                    "Cr2O3",
                    "FeOt",
                    "MnO",
                    "MgO",
                    "CaO",
                    "Na2O",
                    "K2O",
                    "Mineral",
                ]
            ),
        )

    def test_norm_data_nn(self):
  
        df_cleaned = mm.prep_df_nn(self.df.copy())
        normalized_data = mm.norm_data_nn(df_cleaned)

        # Check the shape of the output
        self.assertEqual(normalized_data.shape, (8, 10))

        # Expected normalized data
        expected_normalized_data = np.array([
            [-0.22643322,  0.08201123,  0.37556234, -0.09862359, -0.32940736,
             -0.46334456,  0.40043244,  0.33010893, -0.22369372, -0.17032885],
            [-0.24528221,  0.0942531 ,  0.36763161, -0.04468734, -0.29926058,
             -0.50116598,  0.48428333,  0.25318957, -0.26727423, -0.17443251],
            [ 0.49548345, -0.22403538, -0.69746528, -0.36938356, -0.14852665,
             -0.10836359,  1.52882587, -0.765992  , -0.63770857, -0.13749953],
            [ 0.47412125, -0.23260468, -0.69746528, -0.4465124 , -0.32940736,
             -0.12997583,  1.76121262, -0.73907022, -0.63770857, -0.14775869],
            [-0.23899921, -0.22403538,  0.88312898,  0.12251504,  0.69558336,
             -0.4849568 ,  0.48428333, -0.8621412 , -0.63770857, -0.18674351],
            [-0.21386722, -0.22403538,  0.95450554,  0.06857879,  0.69558336,
             -0.34988033,  0.14887976, -0.8621412 , -0.63770857, -0.18674351],
            [-0.20695592, -0.29626238, -0.90049194, -0.10509594, -0.26911379,
              1.52606173, -0.78785449, -0.8621412 , -0.63770857, -0.17443251],
            [-0.28989151, -0.28769307, -0.90128501,  0.28971741, -0.05808629,
              1.18620932, -0.76629283, -0.8621412 , -0.63770857, -0.17648434]
        ])

        np.testing.assert_almost_equal(normalized_data, expected_normalized_data, decimal=4)


if __name__ == "__main__":
    unittest.main()
