import unittest
import numpy as np
import pandas as pd

import mineralML as mm


class TestBaseMineralCalculator(unittest.TestCase):

    def setUp(self):
        # Minimal olivine-like test composition
        self.df = pd.DataFrame({
            'SiO2': [39.84],
            'MgO': [43.12],
            'FeOt': [17.39]
        })

    def test_invalid_base_instantiation(self):
        with self.assertRaises(NotImplementedError):
            mm.BaseMineralCalculator(self.df)

    def test_invalid_fe_combinations(self):
        bad_df = self.df.copy()
        bad_df["Fe2O3"] = 1.0
        with self.assertRaises(ValueError):
            mm.OlivineCalculator(bad_df)

    def test_valid_instantiation(self):
        calc = mm.OlivineCalculator(self.df)
        self.assertIsInstance(calc, mm.OlivineCalculator)

    def test_calculate_moles(self):
        calc = mm.OlivineCalculator(self.df)
        moles = calc.calculate_moles()
        self.assertIn("SiO2_mols", moles.columns)
        self.assertAlmostEqual(
            moles["SiO2_mols"].iloc[0],
            39.84 / mm.OlivineCalculator.OXIDE_MASSES["SiO2"],
            places=4
        )

    def test_calculate_oxygens(self):
        calc = mm.OlivineCalculator(self.df)
        oxy = calc.calculate_oxygens()
        self.assertIn("MgO_ox", oxy.columns)
        self.assertGreater(oxy.sum(axis=1).iloc[0], 0)

    def test_calculate_cations(self):
        calc = mm.OlivineCalculator(self.df)
        cats = calc.calculate_cations()
        self.assertIn("Mg_cat_4ox", cats.columns)
        self.assertAlmostEqual(cats.sum(axis=1).iloc[0], 3.0, places=1)

    def test_calculate_all(self):
        calc = mm.OlivineCalculator(self.df)
        full = calc.calculate_all()
        self.assertIn("Mg_cat_4ox", full.columns)
        self.assertIn("SiO2_mols", full.columns)


class TestOlivineCalculator(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'SiO2': [39.84],
            'MgO': [43.12],
            'FeOt': [17.39]
        })

    def test_calculate_components(self):
        calc = mm.OlivineCalculator(self.df)
        result = calc.calculate_components()
        self.assertIn("XFo", result.columns)
        self.assertTrue(np.isfinite(result["XFo"].iloc[0]))

    def test_fo_range(self):
        calc = mm.OlivineCalculator(self.df)
        result = calc.calculate_components()
        fo = result["XFo"].iloc[0]
        self.assertGreaterEqual(fo, 0)
        self.assertLessEqual(fo, 1)


if __name__ == '__main__':
    unittest.main()
