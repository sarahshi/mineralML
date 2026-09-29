import unittest
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mineralML as mm
from mineralML.stoichiometry import BaseMineralCalculator


R = {
    "Alkali_Feldspar": {
        "Sample Name": "DG-44",
        "SiO2": 65.0797, "TiO2": np.nan, "Al2O3": 18.8768, "FeOt": 0.0, "MnO": np.nan,
        "MgO": np.nan, "CaO": 0.0664, "Na2O": 2.4138, "K2O": 13.194, "P2O5": np.nan, "Cr2O3": np.nan
    },
    "Amphibole": {
        "Sample Name": "Z2099",
        "SiO2": 42.96, "TiO2": 1.8, "Al2O3": 14.33, "FeOt": 4.07, "MnO": 0.07,
        "MgO": 17.39, "CaO": 12.03, "Na2O": 3.1, "K2O": 0.03, "P2O5": np.nan, "Cr2O3": 0.65
    },
    "Apatite": {
        "Sample Name": "SG-09-32_12",
        "SiO2": 0.17, "TiO2": np.nan, "Al2O3": np.nan, "FeOt": 0.7, "MnO": np.nan,
        "MgO": 0.13, "CaO": 54.32, "Na2O": np.nan, "K2O": np.nan, "P2O5": 40.97, "Cr2O3": np.nan
    },
    "Biotite": {
        "Sample Name": "IgnA-2",
        "SiO2": 36.9, "TiO2": 2.31, "Al2O3": 16.4, "FeOt": 8.2, "MnO": 0.08,
        "MgO": 20.6, "CaO": 0.03, "Na2O": 0.71, "K2O": 8.79, "P2O5": np.nan, "Cr2O3": 0.09
    },
    "Calcite": {
        "Sample Name": "REG55-calcite-1",
        "SiO2": 0.0, "TiO2": 0.0, "Al2O3": 0.0, "FeOt": 0.07, "MnO": 0.0457,
        "MgO": 0.0526, "CaO": 57.0312, "Na2O": 0.0046, "K2O": 0.0216, "P2O5": np.nan, "Cr2O3": 0.1757
    },
    "Chlorite": {
        "Sample Name": "zk803-31-01-01",
        "SiO2": 29.41, "TiO2": 0.03, "Al2O3": 18.6, "FeOt": 33.22, "MnO": 0.08,
        "MgO": 6.7, "CaO": 0.1, "Na2O": 0.03, "K2O": 0.26, "P2O5": np.nan, "Cr2O3": np.nan
    },
    "Clinopyroxene": {
        "Sample Name": "17MMSG37_cpx4-1",
        "SiO2": 46.8911, "TiO2": 2.8722, "Al2O3": 6.5948, "FeOt": 8.4685, "MnO": 0.1886,
        "MgO": 13.884, "CaO": 20.2569, "Na2O": 0.3403, "K2O": np.nan, "P2O5": np.nan, "Cr2O3": 0.0908
    },
    "Epidote": {
        "Sample Name": "CL09MB009 C2 ep 20",
        "SiO2": 37.38, "TiO2": 0.01, "Al2O3": 21.29, "FeOt": 15.3507, "MnO": 0.29,
        "MgO": 0.06, "CaO": 23.19, "Na2O": 0.0, "K2O": 0.0, "P2O5": np.nan, "Cr2O3": np.nan
    },
    "Garnet": {
        "Sample Name": "Lw-Ec_GR1_core",
        "SiO2": 38.52, "TiO2": 0.04, "Al2O3": 21.84, "FeOt": 27.07, "MnO": 1.49,
        "MgO": 5.58, "CaO": 5.82, "Na2O": 0.03, "K2O": 0.0, "P2O5": np.nan, "Cr2O3": 0.0
    },
    "Glass": {
        "Sample Name": "20B-04",
        "SiO2": 49.96, "TiO2": 1.59, "Al2O3": 13.73, "FeOt": 11.8, "MnO": 0.21,
        "MgO": 7.27, "CaO": 11.88, "Na2O": 2.27, "K2O": 0.25, "P2O5": 0.18, "Cr2O3": np.nan
    },
    "Hematite": {
        "Sample Name": "LS-13_C1_Hem-1-1",
        "SiO2": 0.396, "TiO2": 0.026, "Al2O3": 0.218, "FeOt": 86.9907, "MnO": np.nan,
        "MgO": 0.016, "CaO": 0.096, "Na2O": np.nan, "K2O": 0.003, "P2O5": 0.145, "Cr2O3": np.nan
    },
    "Ilmenite": {
        "Sample Name": "UC1250",
        "SiO2": 0.0424, "TiO2": 49.1927, "Al2O3": 0.0757, "FeOt": 43.9452, "MnO": 1.6045,
        "MgO": 2.8229, "CaO": 0.0139, "Na2O": np.nan, "K2O": np.nan, "P2O5": np.nan, "Cr2O3": 0.0148
    },
    "Kalsilite": {
        "Sample Name": "REG-18_kalsilite-1",
        "SiO2": 37.02, "TiO2": 0.0, "Al2O3": 31.43, "FeOt": 0.35, "MnO": 0.0,
        "MgO": 0.0, "CaO": 0.0, "Na2O": 0.04, "K2O": 29.61, "P2O5": np.nan, "Cr2O3": 0.0069
    },
    "Leucite": {
        "Sample Name": "VS219_129 / 1 . Leu",
        "SiO2": 54.7008, "TiO2": 0.113, "Al2O3": 23.0378, "FeOt": 0.5208, "MnO": 0.0064,
        "MgO": 0.0493, "CaO": np.nan, "Na2O": 0.0205, "K2O": 21.7093, "P2O5": np.nan, "Cr2O3": np.nan
    },
    "Magnetite": {
        "Sample Name": "UC1080",
        "SiO2": 0.16, "TiO2": 16.572, "Al2O3": 0.941, "FeOt": 74.515, "MnO": 0.565,
        "MgO": 0.851, "CaO": 0.037, "Na2O": np.nan, "K2O": np.nan, "P2O5": np.nan, "Cr2O3": 0.012
    },
    "Melilite": {
        "Sample Name": "S80_7 / 2 .",
        "SiO2": 41.0155, "TiO2": 0.0521, "Al2O3": 5.6907, "FeOt": 5.2198, "MnO": 0.131,
        "MgO": 6.6045, "CaO": 28.6936, "Na2O": 3.7814, "K2O": 0.1842, "P2O5": np.nan, "Cr2O3": np.nan
    },
    "Muscovite": {
        "Sample Name": "WS5_37",
        "SiO2": 44.8, "TiO2": 0.24, "Al2O3": 35.12, "FeOt": 0.269, "MnO": 0.04,
        "MgO": 3.22, "CaO": 0.16, "Na2O": 0.15, "K2O": 10.57, "P2O5": np.nan, "Cr2O3": 0.0
    },
    "Na-Pyroxene": {
        "Sample Name": "samp. B 61",
        "SiO2": 53.76, "TiO2": 0.78, "Al2O3": 1.44, "FeOt": 26.68, "MnO": 0.26,
        "MgO": 1.27, "CaO": 2.27, "Na2O": 11.93, "K2O": 0.0, "P2O5": 0.0, "Cr2O3": 0.0
    },
    "Nepheline": {
        "Sample Name": "10_N_1",
        "SiO2": 42.033, "TiO2": np.nan, "Al2O3": 32.705, "FeOt": 0.0, "MnO": np.nan,
        "MgO": np.nan, "CaO": 0.105, "Na2O": 15.507, "K2O": 7.859, "P2O5": np.nan, "Cr2O3": np.nan
    },
    "Olivine": {
        "Sample Name": "CN_C_Ol1",
        "SiO2": 39.846, "TiO2": 2e-05, "Al2O3": 0.01915, "FeOt": 17.3987, "MnO": 0.243865,
        "MgO": 43.1267, "CaO": 0.21963, "Na2O": 0.01495, "K2O": 0.007775, "P2O5": 0.013685, "Cr2O3": np.nan
    },
    "Orthopyroxene": {
        "Sample Name": "L04_N1_1",
        "SiO2": 56.08, "TiO2": 0.2769, "Al2O3": 1.868, "FeOt": 7.21, "MnO": 0.1732,
        "MgO": 34.17, "CaO": 0.517, "Na2O": np.nan, "K2O": np.nan, "P2O5": np.nan, "Cr2O3": 0.4681
    },
    "Plagioclase": {
        "Sample Name": "K8_plag1_rtoc",
        "SiO2": 46.6657, "TiO2": 0.0297, "Al2O3": 32.4782, "FeOt": 0.5769, "MnO": 0.0013,
        "MgO": 0.2178, "CaO": 16.8384, "Na2O": 1.7939, "K2O": 0.0031, "P2O5": np.nan, "Cr2O3": np.nan
    },
    "Quartz": {
        "Sample Name": "OM08-206A_2",
        "SiO2": 99.7, "TiO2": 0.0, "Al2O3": 0.0, "FeOt": 0.3, "MnO": 0.03,
        "MgO": 0.0, "CaO": 0.0, "Na2O": 0.01, "K2O": 0.02, "P2O5": np.nan, "Cr2O3": 0.0
    },
    "Rutile": {
        "Sample Name": "E2718C-1",
        "SiO2": 0.0091, "TiO2": 98.4337, "Al2O3": 0.0236, "FeOt": 0.2144, "MnO": 0.0089,
        "MgO": np.nan, "CaO": np.nan, "Na2O": np.nan, "K2O": np.nan, "P2O5": np.nan, "Cr2O3": 0.1976,
        "Nb2O5": 0.0, "Ta2O5": 0.0
    },
    "Serpentine": {
        "Sample Name": "OM15-6",
        "SiO2": 41.2574, "TiO2": np.nan, "Al2O3": 1.34979, "FeOt": 4.38996, "MnO": np.nan,
        "MgO": 39.5454, "CaO": np.nan, "Na2O": np.nan, "K2O": np.nan, "P2O5": np.nan, "Cr2O3": 0.26425
    },
    "Spinel": {
        "Sample Name": "HOR-11-01BY_SP503",
        "SiO2": 0.0554, "TiO2": 0.7736, "Al2O3": 22.1179, "FeOt": 23.5226, "MnO": np.nan,
        "MgO": 12.9254, "CaO": 0.0466, "Na2O": np.nan, "K2O": np.nan, "P2O5": np.nan, "Cr2O3": 39.0881
    },
    "Titanite": {
        "Sample Name": "REG-19-titanite-1",
        "SiO2": 29.3211, "TiO2": 33.0262, "Al2O3": 1.3941, "FeOt": 2.90473, "MnO": 0.0339,
        "MgO": 0.076, "CaO": 26.5311, "Na2O": 0.1435, "K2O": 0.0327, "P2O5": np.nan, "Cr2O3": 0.0
    },
    "Tourmaline": {
        "Sample Name": "Tourmaline1",
        "SiO2": 36.47, "TiO2": 0.82, "Al2O3": 30.79, "FeOt": 4.13, "MnO": np.nan,
        "MgO": 9.52, "CaO": 0.74, "Na2O": 2.36, "K2O": np.nan, "P2O5": np.nan, "Cr2O3": np.nan, "B2O3": 2.5
    },
    "Zircon": {
        "Sample Name": "Zrn-I",
        "SiO2": 32.816, "TiO2": 0.005, "Al2O3": 0.0, "FeOt": 0.007, "MnO": np.nan,
        "MgO": np.nan, "CaO": 0.0, "Na2O": 0.008, "K2O": np.nan, "P2O5": 0.027, "Cr2O3": np.nan,
        # Note: ZrO2 not reported; calculator should still run (Zr columns will be 0)
    },
}


# Helpers

def _finite_nonneg(s):
    return np.isfinite(s).all() and (s >= 0).all()

def _assert_cols(testcase, df, cols):
    for c in cols:
        testcase.assertIn(c, df.columns)

def _df(name, extra=None):
    d = dict(R[name])
    if extra:
        d.update(extra)
    return pd.DataFrame([d])


# Base + utilities

class TestBaseMineralCalculator(unittest.TestCase):
    def setUp(self):
        self.df = _df("Olivine")

    def test_invalid_base_instantiation(self):
        with self.assertRaises(NotImplementedError):
            mm.BaseMineralCalculator(self.df)

    def test_invalid_fe_combinations(self):
        bad = _df("Olivine")
        bad["Fe2O3"] = 1.0
        with self.assertRaises(ValueError):
            mm.OlivineCalculator(bad)

        bad = _df("Olivine")
        bad["FeO"] = 1.0
        with self.assertRaises(ValueError):
            mm.OlivineCalculator(bad)

        # Using FeO and Fe2O3 but missing one -> raises
        broken = pd.DataFrame({'SiO2':[40], 'MgO':[40], 'FeO':[10]})
        with self.assertRaises(ValueError):
            mm.OlivineCalculator(broken)

    def test_calculate_moles_ox_cats_exist(self):
        calc = mm.OlivineCalculator(self.df)
        moles = calc.calculate_moles()
        oxy = calc.calculate_oxygens()
        cats = calc.calculate_cations()
        _assert_cols(self, moles, ["SiO2_mols", "MgO_mols"])
        self.assertGreater(oxy.sum(axis=1).iloc[0], 0)
        self.assertTrue(_finite_nonneg(cats.sum(axis=1)))

    def test_calculate_all_column_presence(self):
        calc = mm.OlivineCalculator(self.df)
        full = calc.calculate_all()
        _assert_cols(self, full, ["SiO2_mols", "Mg_cat_4ox"])


class TestOxideElementConversions(unittest.TestCase):
    def test_roundtrip_oxide_element(self):
        # Use a mixed real sample (Biotite) to make sure round-trip is stable.
        df = _df("Biotite")
        el, _ = mm.oxide_to_element(df)
        ox, _ = mm.element_to_oxide(el)
        common = [c for c in df.columns if c in ox.columns]
        diff = (df[common].fillna(0) - ox[common].fillna(0)).abs().to_numpy()
        self.assertLess(np.nanmax(diff), 1e-6)

    def test_element_to_oxide_identity(self):
        # Build a small DataFrame with element-labeled columns
        df_el = pd.DataFrame({
            "Si": [25.0, 30.0],
            "Fe": [5.0, 8.0],
            "Mg": [3.0, 4.0],
            "Ca": [7.0, 6.0],
        })

        ox, factors = mm.element_to_oxide_identity(df_el)

        # Values should pass through unchanged (factor = 1)
        np.testing.assert_array_almost_equal(ox["SiO2"].values, df_el["Si"].values)
        np.testing.assert_array_almost_equal(ox["FeOt"].values, df_el["Fe"].values)
        np.testing.assert_array_almost_equal(ox["MgO"].values, df_el["Mg"].values)
        np.testing.assert_array_almost_equal(ox["CaO"].values, df_el["Ca"].values)

        # All conversion factors should be exactly 1
        for f in factors.values:
            self.assertEqual(f, 1)

        # Element columns should not appear in output
        for el in ["Si", "Fe", "Mg", "Ca"]:
            self.assertNotIn(el, ox.columns)

    def test_element_to_oxide_identity_missing_elements(self):
        # Elements not in the lookup should be silently skipped
        df_el = pd.DataFrame({
            "Si": [25.0],
            "Xx": [99.0],  # unknown element
        })

        ox, factors = mm.element_to_oxide_identity(df_el)

        self.assertIn("SiO2", ox.columns)
        self.assertNotIn("Xx", ox.columns)
        self.assertEqual(len(factors), 1)

    def test_element_to_oxide_identity_vs_converted(self):
        # Identity should differ from stoichiometric conversion
        df_el = pd.DataFrame({"Si": [25.0], "Mg": [3.0]})

        ox_identity, _ = mm.element_to_oxide_identity(df_el)
        ox_converted, _ = mm.element_to_oxide(df_el)

        # Stoichiometric conversion scales up by oxide_mass/element_mass,
        # so converted values must be larger than identity values
        self.assertGreater(ox_converted["SiO2"].iloc[0], ox_identity["SiO2"].iloc[0])
        self.assertGreater(ox_converted["MgO"].iloc[0], ox_identity["MgO"].iloc[0])


# Specific calculators (real data)

class TestOlivine(unittest.TestCase):
    def test_components_and_xfo_range(self):
        res = mm.OlivineCalculator(_df("Olivine")).calculate_components()
        _assert_cols(self, res, ["Fo", "M_site", "T_site", "Mg_cat_4ox"])
        xfo = float(res["Fo"].iloc[0])
        self.assertTrue(0.0 <= xfo <= 1.0)


class TestAmphibole(unittest.TestCase):
    def test_amphibole_real(self):
        res = mm.AmphiboleCalculator(_df("Amphibole")).calculate_components()
        # Presence-based due to suffixed columns
        self.assertTrue(any(k in res.columns for k in ["Mgno_leake"]))
        self.assertTrue(any("ridolfi" in c for c in res.columns))


class TestApatite(unittest.TestCase):
    def test_apatite_sites(self):
        res = mm.ApatiteCalculator(_df("Apatite")).calculate_components()
        _assert_cols(self, res, ["M_site", "T_site", "Ca_P"])
        self.assertTrue(_finite_nonneg(res[["M_site","T_site"]].iloc[0]))


class TestBiotite(unittest.TestCase):
    def test_biotite_sites(self):
        res = mm.BiotiteCalculator(_df("Biotite")).calculate_components()
        _assert_cols(self, res, ["X_site", "M_site", "T_site"])


class TestCarbonate(unittest.TestCase):
    def test_calcite_co2_injected(self):
        res = mm.CarbonateCalculator(_df("Calcite")).calculate_components()
        _assert_cols(self, res, ["CO2", "C_cat_3ox", "C_site", "M_site"])
        self.assertGreater(float(res["CO2"].iloc[0]), 0)


class TestChlorite(unittest.TestCase):
    def test_chlorite_sites(self):
        res = mm.ChloriteCalculator(_df("Chlorite")).calculate_components()
        _assert_cols(self, res, ["T_site", "M_site", "Mgno", "Al_IV", "Al_VI"])
        self.assertTrue(0 <= float(res["Mgno"].iloc[0]) <= 1)


class TestCpx(unittest.TestCase):
    def test_cpx_en_fs_wo(self):
        res = mm.ClinopyroxeneCalculator(_df("Clinopyroxene")).calculate_components()
        _assert_cols(self, res, ["En", "Fs", "Wo"])
        s = float((res["En"] + res["Fs"] + res["Wo"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)


class TestEpidote(unittest.TestCase):
    def test_epidote_fe_conversion_and_sites(self):
        res = mm.EpidoteCalculator(_df("Epidote")).calculate_components()
        _assert_cols(self, res, ["A_site", "M_site", "Z_site"])
        self.assertIn("Fe3t_cat_12.5ox", res.columns)


class TestFeldspar(unittest.TestCase):
    def test_feldspar_an_ab_or_plagioclase(self):
        res = mm.FeldsparCalculator(_df("Plagioclase")).calculate_components()
        _assert_cols(self, res, ["An", "Ab", "Or"])
        s = float((res["An"] + res["Ab"] + res["Or"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)

    def test_feldspar_classifier_kfeldspar(self):
        out = mm.FeldsparClassifier(_df("Alkali_Feldspar")).classify(subclass=True)
        _assert_cols(self, out, ["Mineral", "Submineral"])
        self.assertIn(out["Mineral"].iloc[0], ["Plagioclase","Alkali_Feldspar","Unclassified"])


class TestGarnet(unittest.TestCase):
    def test_garnet_droop_and_allfe2(self):
        df = _df("Garnet")
        droop = mm.GarnetCalculator(df).calculate_components(Fe_correction="Droop")
        allfe2 = mm.GarnetCalculator(df).calculate_components(Fe_correction="All_Fe2")
        _assert_cols(self, droop, ["Mgno","Alm","Prp","Grs","Sps","And"])
        _assert_cols(self, allfe2, ["Mgno","Alm","Prp","Grs","Sps","And"])
        self.assertTrue(0.0 <= float(droop["Mgno"].iloc[0]) <= 1.0)


class TestGlassCalculator(unittest.TestCase):
 
    def setUp(self):
        self.df = _df("Glass")
 
    def test_calculate_components_has_mgno(self):
        res = mm.GlassCalculator(self.df).calculate_components()
        _assert_cols(self, res, ["MgNo"])
 
    def test_mgno_range(self):
        res = mm.GlassCalculator(self.df).calculate_components()
        mgno = float(res["MgNo"].iloc[0])
        self.assertTrue(0.0 <= mgno <= 1.0)
 
    def test_mgno_value(self):
        # MgNo = MgO_mols / (MgO_mols + FeOt_mols)
        res = mm.GlassCalculator(self.df).calculate_components()
        mgo_mols = self.df["MgO"].iloc[0] / 40.3044
        feot_mols = self.df["FeOt"].iloc[0] / 71.844
        expected_mgno = mgo_mols / (mgo_mols + feot_mols)
        self.assertAlmostEqual(float(res["MgNo"].iloc[0]), expected_mgno, places=4)

    def test_moles_columns_present(self):
        res = mm.GlassCalculator(self.df).calculate_components()
        self.assertIn("MgO_mols", res.columns)
        self.assertIn("FeOt_mols", res.columns)
        self.assertIn("SiO2_mols", res.columns)

    def test_preserves_original_compositions(self):
        res = mm.GlassCalculator(self.df).calculate_components()
        self.assertAlmostEqual(float(res["SiO2"].iloc[0]), 49.96, places=4)
        self.assertAlmostEqual(float(res["MgO"].iloc[0]), 7.27, places=4)
 
    def test_zero_mg_and_fe_gives_zero_mgno(self):
        df_zero = pd.DataFrame([{
            "SiO2": 75.0, "TiO2": 0.0, "Al2O3": 13.0, "FeOt": 0.0, "MnO": 0.0,
            "MgO": 0.0, "CaO": 1.0, "Na2O": 4.0, "K2O": 5.0, "P2O5": 0.0, "Cr2O3": 0.0
        }])
        res = mm.GlassCalculator(df_zero).calculate_components()
        self.assertEqual(float(res["MgNo"].iloc[0]), 0.0)
 
    def test_multiple_rows(self):
        df = pd.DataFrame([
            {"SiO2": 50.0, "MgO": 7.0, "FeOt": 10.0, "CaO": 11.0, "Al2O3": 15.0},
            {"SiO2": 72.0, "MgO": 0.5, "FeOt": 2.0, "CaO": 1.0, "Al2O3": 14.0},
        ])
        res = mm.GlassCalculator(df).calculate_components()
        self.assertEqual(len(res), 2)
        # Basaltic glass should have higher MgNo than rhyolitic
        self.assertGreater(float(res["MgNo"].iloc[0]), float(res["MgNo"].iloc[1]))


class TestKalsilite(unittest.TestCase):
    def test_kalsilite(self):
        res = mm.KalsiliteCalculator(_df("Kalsilite")).calculate_components()
        _assert_cols(self, res, ["A_site","B_site","T_site"])


class TestLeucite(unittest.TestCase):
    def test_leucite(self):
        res = mm.LeuciteCalculator(_df("Leucite")).calculate_components()
        _assert_cols(self, res, ["Channel_site","T_site"])


class TestMelilite(unittest.TestCase):
    def test_melilite(self):
        res = mm.MeliliteCalculator(_df("Melilite")).calculate_components()
        _assert_cols(self, res, ["A_site","B_site","T_site"])
        self.assertTrue(_finite_nonneg(res[["A_site","B_site","T_site"]].iloc[0]))


class TestMuscovite(unittest.TestCase):
    def test_muscovite(self):
        res = mm.MuscoviteCalculator(_df("Muscovite")).calculate_components()
        _assert_cols(self, res, ["X_site","M_site","T_site","Al_IV","Al_VI"])
        self.assertTrue(_finite_nonneg(res[["Al_IV","Al_VI"]].iloc[0]))


class TestNepheline(unittest.TestCase):
    def test_nepheline(self):
        res = mm.NephelineCalculator(_df("Nepheline")).calculate_components()
        _assert_cols(self, res, ["A_B_site","A_site","B_site","T_site"])


class TestOpx(unittest.TestCase):
    def test_opx_components(self):
        res = mm.OrthopyroxeneCalculator(_df("Orthopyroxene")).calculate_components()
        _assert_cols(self, res, ["En","Fs","Wo","Al_IV","Al_VI"])
        s = float((res["En"] + res["Fs"] + res["Wo"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)


class TestQuartz(unittest.TestCase):
    def test_quartz(self):
        res = mm.QuartzCalculator(_df("Quartz")).calculate_components()
        _assert_cols(self, res, ["T_site"])


class TestRutile(unittest.TestCase):
    def test_rutile(self):
        res = mm.RutileCalculator(_df("Rutile")).calculate_components()
        _assert_cols(self, res, ["M_site","Ti_cat_2ox"])


class TestSerpentine(unittest.TestCase):
    def test_serpentine(self):
        res = mm.SerpentineCalculator(_df("Serpentine")).calculate_components()
        _assert_cols(self, res, ["M_site","T_site","Mgno","Feno"])
        self.assertTrue(0.0 <= float(res["Mgno"].iloc[0]) <= 1.0)


class TestNaPx(unittest.TestCase):
    def test_napx_en_fs_wo(self):
        res = mm.SodicPyroxeneCalculator(_df("Na-Pyroxene")).calculate_components()
        _assert_cols(self, res, ["En", "Fs", "Wo"])
        s = float((res["En"] + res["Fs"] + res["Wo"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)


class TestSpinel(unittest.TestCase):
    def test_spinel_droop(self):
        res = mm.SpinelCalculator(_df("Spinel")).calculate_components(Fe_correction="Droop")
        _assert_cols(self, res, ["XR2","XR3","XTi","Fe3_prop"])
        s = float((res["XR2"] + res["XR3"] + res["XTi"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)


class TestRhombohedralOxide(unittest.TestCase):
    def test_hematite_allfe3_and_ilmenite_droop(self):
        # Hematite-like: use All_Fe3 path
        hem = mm.RhombohedralOxideCalculator(_df("Hematite")).calculate_components(Fe_correction="All_Fe3")
        _assert_cols(self, hem, ["XR2","XR3","XTi","XHem","XIlm"])
        self.assertTrue(0.99 <= float((hem["XR2"]+hem["XR3"]+hem["XTi"]).iloc[0]) <= 1.01)

        # Ilmenite-like: Droop
        ilm = mm.RhombohedralOxideCalculator(_df("Ilmenite")).calculate_components(Fe_correction="Droop")
        _assert_cols(self, ilm, ["XR2","XR3","XTi","XHem","XIlm"])
        self.assertTrue(0.99 <= float((ilm["XR2"]+ilm["XR3"]+ilm["XTi"]).iloc[0]) <= 1.01)


class TestTitanite(unittest.TestCase):
    def test_titanite_feot_to_fe2o3t(self):
        res = mm.TitaniteCalculator(_df("Titanite")).calculate_components()
        _assert_cols(self, res, ["VII_site","M_site","T_site","Fe3t_cat_5ox"])


class TestTourmaline(unittest.TestCase):
    def test_tourmaline(self):
        res = mm.TourmalineCalculator(_df("Tourmaline")).calculate_components()
        _assert_cols(self, res, ["X_site","Y_site","Z_site","T_site"])


class TestZircon(unittest.TestCase):
    def test_zircon(self):
        res = mm.ZirconCalculator(_df("Zircon")).calculate_components()
        _assert_cols(self, res, ["M_site","T_site","Si_cat_4ox"])  # Zr may be 0; still check structure


# Classifiers (real data)

class TestPyroxeneClassifier(unittest.TestCase):
    def test_classify_fields_real(self):
        # Use a clinopyroxene-like row
        out = mm.PyroxeneClassifier(_df("Clinopyroxene")).classify(subclass=True)
        _assert_cols(self, out, ["Mineral","Submineral","En","Fs","Wo"])
        s = float((out["En"] + out["Fs"] + out["Wo"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)


class TestOxideClassifier(unittest.TestCase):
    def test_routing_real(self):
        rows = [
            dict(R["Hematite"], Predict_Mineral="Hematite"),
            dict(R["Spinel"],   Predict_Mineral="Spinels"),
        ]
        df = pd.DataFrame(rows)
        
        # Fill missing text/object values with an empty string
        string_cols = df.select_dtypes(include=['object', 'string']).columns
        df[string_cols] = df[string_cols].fillna("")
        
        # Modify to objects for safe insertion of np.nan later
        df[string_cols] = df[string_cols].astype(object)

        # Fill missing numeric oxides with 0.0
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

        out = mm.OxideClassifier(df).calculate_components()
        _assert_cols(self, out, ["XR2","XR3","XTi"])
        trip = out[["XR2","XR3","XTi"]].apply(pd.to_numeric, errors="coerce")
        mask = np.isfinite(trip.to_numpy()).all(axis=1)

        s = trip.loc[mask].sum(axis=1).to_numpy()
        self.assertTrue(np.all((0.99 <= s) & (s <= 1.01)))


try:
    from pyrolite.util.classification import TAS
    _has_pyrolite = True
except ImportError:
    _has_pyrolite = False
 
@unittest.skipUnless(_has_pyrolite, "pyrolite not installed")
class TestGlassClassifier(unittest.TestCase):
 
    def setUp(self):
        self.df = _df("Glass")
 
    def test_subclass_false_assigns_glass(self):
        res = mm.GlassClassifier(self.df).calculate_components(subclass=False)
        _assert_cols(self, res, ["Mineral", "MgNo"])
        self.assertEqual(res["Mineral"].iloc[0], "Glass")
        self.assertNotIn("TAS", res.columns)
  
    def test_tas_basalt_classification(self):
        # A basaltic glass (~50 wt% SiO2, ~3 wt% total alkalis) should not be "Unclassified"
        basalt = pd.DataFrame([{
            "SiO2": 50.0, "TiO2": 1.5, "Al2O3": 15.0, "FeOt": 10.0,
            "MgO": 7.0, "CaO": 11.0, "Na2O": 2.5, "K2O": 0.5,
        }])
        res = mm.GlassClassifier(basalt).calculate_components(subclass=True)
        self.assertNotEqual(res["TAS"].iloc[0], "Unclassified")
 
    def test_plot_returns_figure(self):
        clf = mm.GlassClassifier(self.df)
        df_class = clf.calculate_components(subclass=True)
        fig, ax = clf.plot(df_class=df_class)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(ax.get_xlabel(), "SiO$\\mathregular{_2}$ (wt%)")
        self.assertEqual(ax.get_ylabel(), "Na$\\mathregular{_2}$O + K$\\mathregular{_2}$O (wt%)")
        plt.close(fig)
 
    def test_plot_no_subclass(self):
        clf = mm.GlassClassifier(self.df)
        fig, ax = clf.plot(subclass=False)
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(len(ax.collections) > 0)
        plt.close(fig)
 
    def test_plot_existing_axes(self):
        clf = mm.GlassClassifier(self.df)
        fig_in, ax_in = plt.subplots()
        fig_out, ax_out = clf.plot(ax=ax_in)
        self.assertIs(fig_out, fig_in)
        self.assertIs(ax_out, ax_in)
        plt.close(fig_in)


# Test plotting 


class TestAmphibolePlot(unittest.TestCase):
    def test_amphibole_plot_returns_axes_and_draws(self):
        calc = mm.AmphiboleClassifier(_df("Amphibole"))
        df_class = calc.classify(subclass=True)

        fig, ax = calc.plot(df_class=df_class, subclass=True)
        try:
            # Basic type checks
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)

            # Axis labels & limits
            self.assertEqual(ax.get_xlabel(), "Si (apfu)")
            self.assertEqual(ax.get_ylabel(), "Mg# Amphibole")
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # x is inverted in function; first > second
            self.assertGreater(xlim[0], xlim[1])
            self.assertLessEqual(ylim[0], 0.0)
            self.assertGreaterEqual(ylim[1], 1.0)

            # Something was scattered
            # (matplotlib scatter adds a PathCollection to ax.collections)
            self.assertTrue(any(hasattr(col, "get_offsets") for col in ax.collections))

            # Legend should exist when subclass/hue provided
            self.assertIsNotNone(ax.get_legend())
        finally:
            plt.close(fig)


class TestFeldsparPlot(unittest.TestCase):
    def test_feldspar_plot_draws_on_ternary(self):
        clf = mm.FeldsparClassifier(_df("Plagioclase"))
        df_class = clf.classify(subclass=True)

        fig, tax = clf.plot(df_class=df_class, subclass=True, labels="short", figsize=(6, 6))
        try:
            # tax is a ternary axes wrapper; underlying Matplotlib axes exists
            ax = tax.get_axes()
            self.assertIsNotNone(ax)

            # Expect some artists: boundary lines or scatter points
            drew_lines = len(ax.lines) > 0
            drew_scatter = len(ax.collections) > 0
            self.assertTrue(drew_lines or drew_scatter)

            # Legend should exist if we drew labeled classes (may be empty for single sample, but created)
            # ternary puts legend on the underlying axes
            self.assertIsNotNone(ax.get_legend())
        finally:
            plt.close(fig)


class TestOxidePlot(unittest.TestCase):
    def test_oxide_plot_main_and_spinel_subplot(self):
        # Build a small dataframe with one rhombohedral oxide and one spinel
        rows = [
            dict(R["Hematite"], Predict_Mineral="Hematite"),
            dict(R["Spinel"],   Predict_Mineral="Spinels"),
        ]
        df = pd.DataFrame(rows)
        ox = mm.OxideClassifier(df)

        # Main ternary plot
        result = ox.plot(figsize=(6, 6), include_unclassified=True)
        fig, tax = result["ternary"]
        fig_spinel, ax_spinel = result["spinel"]

        try:
            ax = tax.get_axes()
            self.assertIsNotNone(ax)
            self.assertTrue(len(ax.lines) > 0 or len(ax.collections) > 0)
            self.assertIsNotNone(ax.get_legend())
        finally:
            plt.close(fig)
            if fig_spinel is not None:
                plt.close(fig_spinel)

        # Spinel sub-plot should return a valid (fig, ax) for spinel rows
        fig2, ax2 = ox.plot_spinel()
        try:
            self.assertIsNotNone(fig2)
            self.assertIsNotNone(ax2)
            self.assertTrue(len(ax2.lines) > 0 or len(ax2.collections) > 0)
            self.assertEqual(ax2.get_xlabel(), r"$\mathregular{Fe^{2+}/(Fe^{2+}+Mg^{2+})}$")
            self.assertEqual(ax2.get_ylabel(), r"$\mathregular{Fe^{3+}/(Fe^{3+}+Al^{3+})}$")
            # Legend may be absent when Subspinel column is missing or has one group
        finally:
            plt.close(fig2)

class TestPyroxenePlot(unittest.TestCase):
    def test_pyroxene_plot_draws_and_labels(self):
        clf = mm.PyroxeneClassifier(_df("Clinopyroxene"))
        df_class = clf.classify(subclass=True)

        fig, tax = clf.plot(df_class=df_class, subclass=True, labels="short", figsize=(6, 4))
        try:
            ax = tax.get_axes()
            self.assertIsNotNone(ax)

            # Expect boundary lines and scatter
            self.assertTrue(len(ax.lines) > 0 or len(ax.collections) > 0)

            # If subclass True, a legend should be present
            self.assertIsNotNone(ax.get_legend())
        finally:
            plt.close(fig)


class test_append_stoichiometry(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'Sample Name': ['ol1', 'pl1', 'x1', 'none1'],
            'SiO2': [40.2, 52.8, 45.0, 0.0], 'Al2O3': [0.0, 29.4, 10.0, 0.0],
            'FeOt': [12.1, 0.6, 10.0, 0.0], 'MgO': [47.5, 0.1, 10.0, 0.0],
            'CaO': [0.2, 12.3, 10.0, 0.0], 'Na2O': [0.0, 4.4, 1.0, 0.0], 'K2O': [0.0, 0.2, 0.0, 0.0],
            'Predict_Mineral': ['Olivine', 'Plagioclase', 'Unknown_Mineral', None],
            'Prediction_Score': [0.99, 0.98, 0.5, np.nan],
        }, index=[10, 11, 12, 13])

    def test_keeps_rows_index_and_columns(self):
        out = mm.append_stoichiometry(self.df)
        self.assertListEqual(list(out.index), list(self.df.index))
        pd.testing.assert_frame_equal(out[self.df.columns], self.df)

    def test_calculated_columns_per_mineral(self):
        out = mm.append_stoichiometry(self.df)
        self.assertTrue(np.isfinite(out.loc[10, "Fo"]))
        self.assertTrue(np.isnan(out.loc[11, "Fo"]))
        self.assertTrue(np.isfinite(out.loc[11, "An"]))
        # No calculator (unknown label) or unclassified: calculated columns stay NaN
        self.assertTrue(out.loc[[12, 13], ["Fo", "An"]].isna().all().all())

    def test_missing_mineral_column_raises(self):
        with self.assertRaises(ValueError):
            mm.append_stoichiometry(self.df.drop(columns="Predict_Mineral"))




# One real analysis (wt%) per field, the one closest to the median of all
# analyses the classifier puts in that field (training data natural rows,
# Cpx compilation, Jd_Omph). Oxides not reported by the source are omitted.
# "Wollastonite" is the classifier's label for Wo >= 0.5 (a Ca-rich cpx here).
PYROXENES = {
    # BT-19G_OPX 17-2 (JollesandLange2019)
    "Enstatite": dict(SiO2=52.01, TiO2=0.1, Al2O3=0.43, FeOt=25.1, MnO=0.89, MgO=20.21,
        CaO=1.11, Na2O=0.03, K2O=0, Cr2O3=0),
    # UC1248-2c-opx (Deeringetal2010)
    "Ferrosilite": dict(SiO2=50.418, TiO2=0.088, Al2O3=0.336, FeOt=31.928, MnO=1.676,
        MgO=14.879, CaO=0.877, Na2O=0.029),
    # LEPR entry 32895 (experimental run product; no clean natural pigeonite in the data)
    "Pigeonite": dict(SiO2=53.4, TiO2=0.2, Al2O3=2.47, FeOt=12.7, MnO=0.33, MgO=24.6,
        CaO=5.37, Na2O=0.11, Cr2O3=0.22),
    # Conboy cpx-8 (Hildreth and Fierstein, 1997)
    "Augite": dict(SiO2=51.399, TiO2=0.69, Al2O3=2.324, FeOt=8.674, MnO=0.242, MgO=15.668,
        CaO=20.025, Na2O=0.376, Cr2O3=0.039),
    # 27_C_2_798 (Ubideetal2019)
    "Diopside": dict(SiO2=49.18, TiO2=1.1237, Al2O3=4.16, FeOt=8.31, MnO=0.1696, MgO=13.62,
        CaO=21.81, Na2O=0.3806, Cr2O3=0.006),
    # samp. LFAF09 109A (Fieldetal2013)
    "Hedenbergite": dict(SiO2=47.5618, TiO2=0.4946, Al2O3=0.4124, FeOt=28.8126, MnO=1.2794,
        MgO=1.0008, CaO=19.9956, Na2O=0.4354, K2O=0, Cr2O3=0),
    # Ny21-005_NY21SM005_cpx1_rim-core_1911 (Molendijketal2024)
    "Wollastonite": dict(SiO2=46.1789, TiO2=3.0666, Al2O3=7.963, FeOt=6.8318, MnO=0.1414,
        MgO=12.2464, CaO=23.5189, Na2O=0.3882, K2O=0.022),
    # 2-1-Jd (Jd_Omph)
    "Jadeite": dict(SiO2=58.25, TiO2=0.04, Al2O3=20.81, FeOt=1.02, MnO=0, MgO=2.75, CaO=3.6,
        Na2O=13.54, K2O=0, Cr2O3=0.02),
    # samp. B 48 (Wiedendorderetal2016)
    "Aegirine": dict(SiO2=53.95, TiO2=0.3746, Al2O3=0.9965, FeOt=28.24, MnO=0.1636,
        MgO=1.1921, CaO=1.92, Na2O=12.22, K2O=0.0058, Cr2O3=0),
    # HO2_1b vein_159 (Schornetal2023)
    "Omphacite": dict(SiO2=54.522, TiO2=0.151, Al2O3=9.898, FeOt=5.265, MnO=0.044, MgO=8.955,
        CaO=14.943, Na2O=5.875, K2O=0.006),
    # samp. D885 A1 (Barkeretal2012)
    "Aegirine-Augite": dict(SiO2=50.8142, TiO2=0.6808, Al2O3=2.3487, FeOt=15.7983,
        MnO=0.7631, MgO=7.3147, CaO=17.8288, Na2O=3.5312, K2O=0, Cr2O3=0),
}
MAIN = {"Enstatite": "Orthopyroxene", "Ferrosilite": "Orthopyroxene",
        "Jadeite": "Na-Pyroxene", "Aegirine": "Na-Pyroxene", "Omphacite": "Na-Pyroxene",
        "Aegirine-Augite": "Na-Pyroxene"}


def _px_df(names):
    return pd.DataFrame([PYROXENES[n] for n in names], index=names).fillna(0.0)


class TestPyroxeneSubclasses(unittest.TestCase):

    def test_every_subclass(self):
        names = list(PYROXENES)
        out = mm.PyroxeneClassifier(_px_df(names)).classify(subclass=True)
        self.assertEqual(list(out["Submineral"]), names)
        self.assertEqual(list(out["Mineral"]),
                         [MAIN.get(n, "Clinopyroxene") for n in names])
        for col in ["En", "Fs", "Wo", "En_h", "Jd_h", "Aeg_h", "Di_h", "Hd_h"]:
            self.assertIn(col, out.columns)

    def test_without_subclass(self):
        out = mm.PyroxeneClassifier(_px_df(["Enstatite", "Jadeite"])).classify(subclass=False)
        self.assertNotIn("Submineral", out.columns)
        self.assertEqual(list(out["Mineral"]), ["Orthopyroxene", "Na-Pyroxene"])

    def test_empty_analysis_falls_back_without_subclass(self):
        df = _px_df(["Augite"])
        empty = pd.DataFrame([{c: 0.0 for c in df.columns}], index=["blank"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = mm.PyroxeneClassifier(pd.concat([df, empty])).classify(subclass=True)
        self.assertEqual(out["Submineral"].iat[0], "Augite")
        self.assertIn(out["Mineral"].iat[1], {"Orthopyroxene", "Clinopyroxene"})
        self.assertTrue(pd.isna(out["Submineral"].iat[1]))

    def test_without_na_k_ti_columns(self):
        # Regression: with no Na2O, K2O or TiO2 column the classifier used to
        # crash with "'int' object has no attribute 'clip'".
        df = (pd.DataFrame([PYROXENES["Pigeonite"], PYROXENES["Hedenbergite"]]).fillna(0.0)
              .drop(columns=["Na2O", "K2O", "TiO2"], errors="ignore"))
        out = mm.PyroxeneClassifier(df).classify(subclass=True)
        self.assertEqual(list(out["Submineral"]), ["Pigeonite", "Hedenbergite"])


class TestPyroxenePlotOptions(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_quadrilateral_only_returns_single_figure(self):
        clf = mm.PyroxeneClassifier(_px_df(["Enstatite", "Augite", "Diopside"]))
        fig, tax = clf.plot(labels="long")
        self.assertIsInstance(fig, plt.Figure)
        texts = [t.get_text() for t in tax.get_axes().texts]
        self.assertIn("Diopside", texts)

    def test_full_triangle_on_existing_axis_without_labels(self):
        clf = mm.PyroxeneClassifier(_px_df(["Pigeonite", "Augite"]))
        _, ax = plt.subplots()
        fig, tax = clf.plot(quad_only=False, labels=None, ax=ax)
        self.assertIs(tax.get_axes(), ax)
        self.assertNotIn("Au", [t.get_text() for t in ax.texts])   # no field labels

    def test_quadrilateral_on_existing_axis_uncoloured(self):
        df_class = mm.PyroxeneClassifier(_px_df(["Augite"])).classify(subclass=False)
        _, ax = plt.subplots()
        fig, tax = mm.PyroxeneClassifier(_px_df(["Augite"])).plot(df_class=df_class,
                                                               subclass=False, ax=ax)
        self.assertIsInstance(fig, plt.Figure)

    def test_full_triangle_new_figure(self):
        fig, _ = mm.PyroxeneClassifier(_px_df(["Augite"])).plot(quad_only=False)
        self.assertEqual(tuple(fig.get_size_inches()), (8.0, 8.0))

    def test_sodic_only_figure(self):
        clf = mm.PyroxeneClassifier(_px_df(["Jadeite", "Aegirine", "Omphacite"]))
        fig, tax = clf.plot(labels="short")
        texts = [t.get_text() for t in tax.get_axes().texts]
        self.assertIn("Omph", texts)
        fig2, tax2 = clf.plot(subclass=False, labels=None)
        self.assertNotIn("Omph", [t.get_text() for t in tax2.get_axes().texts])

    def test_mixed_returns_two_figures(self):
        figs = mm.PyroxeneClassifier(_px_df(["Augite", "Aegirine-Augite"])).plot(labels=True)
        self.assertEqual(len(figs), 2)

    def test_nothing_to_plot_returns_none(self):
        df_class = pd.DataFrame({"Mineral": ["Olivine"], "En": [0.5], "Fs": [0.1],
                                 "Wo": [0.4]})
        self.assertIsNone(mm.PyroxeneClassifier(_px_df(["Augite"])).plot(df_class=df_class))


# Real analyses (wt%): training data natural rows and the Cpx compilation
# Conboy cpx-8 (Hildreth and Fierstein, 1997)
AUGITE = dict(SiO2=51.399, TiO2=0.69, Al2O3=2.324, FeOt=8.674, MnO=0.242, MgO=15.668,
              CaO=20.025, Na2O=0.376, Cr2O3=0.039)
# 20210320-003_E5-1 (Kahletal2023)
OLIVINE = dict(SiO2=39.671, TiO2=0.008, Al2O3=0.042, FeOt=13.926, MnO=0.225, MgO=45.983,
               CaO=0.312, Cr2O3=0.03)
# HLY0102-D41-4 (Bennettetal2019)
PLAG = dict(SiO2=50.27, Al2O3=31.06, FeOt=0.38, MgO=0.18, CaO=14.11, Na2O=3.26, K2O=0.05)
# AC5-2-ox42_Tmt (Brahmetal2018)
MAGNETITE = dict(SiO2=0.081, TiO2=18.319, Al2O3=2.343, FeOt=73.543, MnO=0.739, MgO=2.575,
                 CaO=0, Cr2O3=0.023)


def _without(comp, *drop):
    return pd.DataFrame([{k: v for k, v in comp.items() if k not in drop}])


def _missing_warnings(fn):
    """Run fn and return (result, messages of any missing-oxide warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [str(w.message) for w in caught if "missing from the data" in str(w.message)]


class TestMissingOxideWarnings(unittest.TestCase):

    def test_complete_data_is_silent(self):
        for cls, comp in [(mm.PyroxeneClassifier, AUGITE), (mm.OlivineCalculator, OLIVINE),
                          (mm.FeldsparCalculator, PLAG)]:
            _, msgs = _missing_warnings(lambda: cls(pd.DataFrame([comp])).calculate_components())
            self.assertEqual(msgs, [], cls.__name__)

    def test_missing_minor_oxides_are_silent(self):
        df = _without(AUGITE, "TiO2", "MnO", "Na2O", "K2O", "Cr2O3")
        _, msgs = _missing_warnings(lambda: mm.PyroxeneClassifier(df).classify())
        self.assertEqual(msgs, [])

    def test_missing_sio2_warns_and_proceeds(self):
        out, msgs = _missing_warnings(
            lambda: mm.PyroxeneClassifier(_without(AUGITE, "SiO2")).classify())
        self.assertEqual(msgs, ["PyroxeneClassifier: SiO2 missing from the data and "
                                "treated as 0. Proceed with caution."])
        self.assertEqual(len(out), 1)                              # no KeyError any more

    def test_all_missing_oxides_named_in_one_warning(self):
        _, msgs = _missing_warnings(
            lambda: mm.OlivineCalculator(_without(OLIVINE, "SiO2", "MgO")).calculate_components())
        self.assertEqual(len(msgs), 1)
        self.assertIn("SiO2, MgO missing", msgs[0])

    def test_missing_major_no_longer_crashes_other_silicates(self):
        for cls, comp, drop in [(mm.FeldsparCalculator, PLAG, "Al2O3"),
                                (mm.OlivineCalculator, OLIVINE, "FeOt"),
                                (mm.ClinopyroxeneCalculator, AUGITE, "CaO")]:
            out, msgs = _missing_warnings(
                lambda: cls(_without(comp, drop)).calculate_components())
            self.assertIn(drop, msgs[0], cls.__name__)
            self.assertEqual(len(out), 1)

    def test_missing_and_empty_columns_are_zero_filled(self):
        df = _without(AUGITE, "MgO").assign(CaO=np.nan)
        _, msgs = _missing_warnings(lambda: mm.PyroxeneClassifier(df))
        self.assertIn("MgO, CaO missing", msgs[0])
        calc = mm.PyroxeneClassifier(df)
        self.assertTrue((calc.comps[["MgO", "CaO"]] == 0).all().all())

    def test_oxide_minerals_never_warn(self):
        _, msgs = _missing_warnings(
            lambda: mm.SpinelCalculator(pd.DataFrame([MAGNETITE])).calculate_components())
        self.assertEqual(msgs, [])
        self.assertEqual(mm.SpinelCalculator.EXPECTED_OXIDES, ())
        self.assertEqual(mm.RhombohedralOxideCalculator.EXPECTED_OXIDES, ())

    def test_every_silicate_expects_silica(self):
        silicates = [mm.AmphiboleCalculator, mm.BiotiteCalculator, mm.ChloriteCalculator,
                     mm.ClinopyroxeneCalculator, mm.EpidoteCalculator, mm.FeldsparCalculator,
                     mm.GarnetCalculator, mm.KalsiliteCalculator, mm.LeuciteCalculator,
                     mm.MeliliteCalculator, mm.MuscoviteCalculator, mm.NephelineCalculator,
                     mm.OlivineCalculator, mm.OrthopyroxeneCalculator, mm.PyroxeneClassifier,
                     mm.QuartzCalculator, mm.SerpentineCalculator, mm.SodicPyroxeneCalculator,
                     mm.TitaniteCalculator, mm.TourmalineCalculator, mm.ZirconCalculator]
        for cls in silicates:
            self.assertIn("SiO2", cls.EXPECTED_OXIDES, cls.__name__)
        self.assertEqual(BaseMineralCalculator.EXPECTED_OXIDES, ())


if __name__ == "__main__":
    unittest.main()
