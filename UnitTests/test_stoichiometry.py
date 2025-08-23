import unittest
import numpy as np
import pandas as pd

import mineralML as mm


# Helpers 
def _finite_nonneg(s):
    return np.isfinite(s).all() and (s >= 0).all()

def _assert_cols(testcase, df, cols):
    for c in cols:
        testcase.assertIn(c, df.columns)

def _make_basic_df(extra=None):
    """
    Minimal whole-rock style wt% row that many calculators can tolerate.
    ONLY FeOt (no FeO/Fe2O3) to satisfy your Fe guardrails.
    """
    base = dict(
        SiO2=50.0, TiO2=1.0, Al2O3=15.0, FeOt=9.0, MnO=0.1, MgO=12.0,
        CaO=10.0, Na2O=2.5, K2O=1.5, Cr2O3=0.05, P2O5=0.2
    )
    if extra:
        base.update(extra)
    return pd.DataFrame([base])


# Base + utilities
class TestBaseMineralCalculator(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'SiO2': [39.84], 'MgO': [43.12], 'FeOt': [17.39]})

    def test_invalid_base_instantiation(self):
        with self.assertRaises(NotImplementedError):
            mm.BaseMineralCalculator(self.df)

    def test_invalid_fe_combinations(self):
        bad = self.df.copy()
        bad["Fe2O3"] = 1.0
        with self.assertRaises(ValueError):
            mm.OlivineCalculator(bad)
        bad = self.df.copy()
        bad["FeO"] = 1.0
        with self.assertRaises(ValueError):
            mm.OlivineCalculator(bad)
        # Using FeO and Fe2O3 but missing one -> raises
        bad = pd.DataFrame({'SiO2':[40], 'MgO':[40], 'FeO':[10]})
        with self.assertRaises(ValueError):
            mm.OlivineCalculator(bad)

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
        df = _make_basic_df()
        el, f1 = mm.oxide_to_element(df)
        ox, f2 = mm.element_to_oxide(el)
        # Only compare keys present on both sides
        common = [c for c in df.columns if c in ox.columns]
        diff = (df[common] - ox[common]).abs().to_numpy()
        self.assertLess(np.nanmax(diff), 1e-6)


# Specific calculators (smoke + invariants)
class TestOlivine(unittest.TestCase):
    def test_components_and_xfo_range(self):
        df = pd.DataFrame({'SiO2':[39.84], 'MgO':[43.12], 'FeOt':[17.39]})
        res = mm.OlivineCalculator(df).calculate_components()
        _assert_cols(self, res, ["XFo", "M_site", "T_site", "Mg_cat_4ox"])
        xfo = float(res["XFo"].iloc[0])
        self.assertTrue(0.0 <= xfo <= 1.0)


class TestAmphibole(unittest.TestCase):
    def test_amphibole_minimal(self):
        df = _make_basic_df(extra={"H2O": 1.0, "F": 0.0, "Cl": 0.0})
        res = mm.AmphiboleCalculator(df).calculate_components()
        # Keys from ridolfi/leake pipelines should exist
        keys = ["Cation_Sum", "Mgno_leake", "Si_T_leake", "Charge_ridolfi", "Fe3_calc_ridolfi"]
        # Some may be suffixed; just ensure presence of leake/ridolfi outputs broadly:
        self.assertTrue(any(k in res.columns for k in ["Mgno_leake"]))
        self.assertTrue(any("ridolfi" in c for c in res.columns))


class TestApatite(unittest.TestCase):
    def test_apatite_sites(self):
        df = _make_basic_df(extra={"P2O5": 5.0, "CaO": 55.0, "Na2O": 0.5, "MnO": 0.1})
        res = mm.ApatiteCalculator(df).calculate_components()
        _assert_cols(self, res, ["M_site", "T_site", "Ca_P"])
        self.assertTrue(_finite_nonneg(res[["M_site","T_site"]].iloc[0]))


class TestBiotite(unittest.TestCase):
    def test_biotite_sites(self):
        df = _make_basic_df()
        res = mm.BiotiteCalculator(df).calculate_components()
        _assert_cols(self, res, ["X_site", "M_site", "T_site"])


class TestCalcite(unittest.TestCase):
    def test_calcite_co2_injected(self):
        df = pd.DataFrame({"CaO":[55.0], "MgO":[0.0], "MnO":[0.0], "FeOt":[0.0]})
        res = mm.CalciteCalculator(df).calculate_components()
        _assert_cols(self, res, ["CO2", "C_cat_3ox", "C_site", "M_site"])
        self.assertGreater(float(res["CO2"].iloc[0]), 0)


class TestChlorite(unittest.TestCase):
    def test_chlorite_sites(self):
        df = _make_basic_df()
        res = mm.ChloriteCalculator(df).calculate_components()
        _assert_cols(self, res, ["T_site", "M_site", "XMg", "Al_IV", "Al_VI"])
        self.assertTrue(0 <= float(res["XMg"].iloc[0]) <= 1)


class TestCpx(unittest.TestCase):
    def test_cpx_en_fs_wo(self):
        df = _make_basic_df()
        res = mm.ClinopyroxeneCalculator(df).calculate_components()
        _assert_cols(self, res, ["En", "Fs", "Wo"])
        s = float((res["En"] + res["Fs"] + res["Wo"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)


class TestEpidote(unittest.TestCase):
    def test_epidote_fe_conversion_and_sites(self):
        # Provide FeOt only; class will convert to Fe2O3t
        df = _make_basic_df()
        res = mm.EpidoteCalculator(df).calculate_components()
        _assert_cols(self, res, ["A_site", "M_site", "Z_site"])
        # Ensure FeOt was removed in internal conversion path
        # (Not strictly required to drop in final, but conversion must have happened)
        self.assertIn("Fe3t_cat_12.5ox", res.columns)


class TestFeldspar(unittest.TestCase):
    def test_feldspar_an_ab_or(self):
        df = _make_basic_df()
        res = mm.FeldsparCalculator(df).calculate_components()
        _assert_cols(self, res, ["An", "Ab", "Or"])
        s = float((res["An"] + res["Ab"] + res["Or"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)

    def test_feldspar_classifier(self):
        df = pd.DataFrame([{"SiO2":60.0,"Al2O3":23.0,"CaO":5.0,"Na2O":9.5,"K2O":1.0,"FeOt":0.5}])
        out = mm.FeldsparClassifier(df).classify(subclass=True)
        _assert_cols(self, out, ["Mineral", "Submineral"])
        self.assertIn(out["Mineral"].iloc[0], ["Plagioclase","KFeldspar","Unclassified"])


class TestGarnet(unittest.TestCase):
    def test_garnet_droop_and_allfe2(self):
        df = _make_basic_df(extra={"CaO": 5.0})
        droop = mm.GarnetCalculator(df).calculate_components(Fe_correction="Droop")
        allfe2 = mm.GarnetCalculator(df).calculate_components(Fe_correction="All_Fe2")
        _assert_cols(self, droop, ["XMg","Alm","Prp","Grs","Sps","And"])
        _assert_cols(self, allfe2, ["XMg","Alm","Prp","Grs","Sps","And"])
        self.assertTrue(0.0 <= float(droop["XMg"].iloc[0]) <= 1.0)


class TestKalsilite(unittest.TestCase):
    def test_kalsilite(self):
        df = pd.DataFrame([{"SiO2":45.0,"Al2O3":36.0,"K2O":17.0,"FeOt":2.0}])
        res = mm.KalsiliteCalculator(df).calculate_components()
        _assert_cols(self, res, ["A_site","B_site","T_site"])


class TestLeucite(unittest.TestCase):
    def test_leucite(self):
        df = pd.DataFrame([{"SiO2":53.0,"Al2O3":23.0,"K2O":20.0,"Na2O":2.5,"FeOt":1.0}])
        res = mm.LeuciteCalculator(df).calculate_components()
        _assert_cols(self, res, ["Channel_site","T_site"])


class TestMuscovite(unittest.TestCase):
    def test_muscovite(self):
        df = _make_basic_df()
        res = mm.MuscoviteCalculator(df).calculate_components()
        _assert_cols(self, res, ["X_site","M_site","T_site","Al_IV","Al_VI"])
        self.assertTrue(_finite_nonneg(res[["Al_IV","Al_VI"]].iloc[0]))


class TestNepheline(unittest.TestCase):
    def test_nepheline(self):
        df = pd.DataFrame([{"SiO2":44.0,"Al2O3":33.0,"Na2O":17.0,"K2O":4.5,"FeOt":1.0}])
        res = mm.NephelineCalculator(df).calculate_components()
        _assert_cols(self, res, ["A_B_site","A_site","B_site","T_site"])


class TestOpx(unittest.TestCase):
    def test_opx_components(self):
        df = _make_basic_df()
        res = mm.OrthopyroxeneCalculator(df).calculate_components()
        _assert_cols(self, res, ["En","Fs","Wo","Al_IV","Al_VI"])
        s = float((res["En"] + res["Fs"] + res["Wo"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)


class TestQuartz(unittest.TestCase):
    def test_quartz(self):
        df = pd.DataFrame([{"SiO2":99.5,"TiO2":0.3,"Al2O3":0.2}])
        res = mm.QuartzCalculator(df).calculate_components()
        _assert_cols(self, res, ["T_site"])


class TestRutile(unittest.TestCase):
    def test_rutile(self):
        df = pd.DataFrame([{"TiO2":99.0,"Nb2O5":0.5,"Ta2O5":0.2}])
        res = mm.RutileCalculator(df).calculate_components()
        _assert_cols(self, res, ["M_site","Ti_cat_2ox"])


class TestSerpentine(unittest.TestCase):
    def test_serpentine(self):
        df = pd.DataFrame([{"SiO2":42.0,"MgO":43.0,"FeOt":8.0,"Al2O3":2.0}])
        res = mm.SerpentineCalculator(df).calculate_components()
        _assert_cols(self, res, ["M_site","T_site","XMg","XFe"])
        self.assertTrue(0.0 <= float(res["XMg"].iloc[0]) <= 1.0)


class TestSpinel(unittest.TestCase):
    def test_spinel_droop(self):
        # Spinel-like: MgAl2O4-ish with Fe present
        df = pd.DataFrame([{"MgO":20.0,"Al2O3":40.0,"FeOt":20.0,"TiO2":2.0,"Cr2O3":5.0,"SiO2":3.0}])
        res = mm.SpinelCalculator(df).calculate_components(Fe_correction="Droop")
        _assert_cols(self, res, ["XR2","XR3","XTi","Fe3_prop"])
        s = float((res["XR2"] + res["XR3"] + res["XTi"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)


class TestRhombohedralOxide(unittest.TestCase):
    def test_oxide_droop_and_allfe3(self):
        df = pd.DataFrame([{"FeOt":65.0,"TiO2":10.0,"Al2O3":1.0}])
        r1 = mm.RhombohedralOxideCalculator(df).calculate_components(Fe_correction="Droop")
        r2 = mm.RhombohedralOxideCalculator(df).calculate_components(Fe_correction="All_Fe3")
        _assert_cols(self, r1, ["XR2","XR3","XTi","XHem","XIlm"])
        _assert_cols(self, r2, ["XR2","XR3","XTi","XHem","XIlm"])
        self.assertTrue(0.99 <= float((r1["XR2"]+r1["XR3"]+r1["XTi"]).iloc[0]) <= 1.01)


class TestTitanite(unittest.TestCase):
    def test_titanite_feot_to_fe2o3t(self):
        df = pd.DataFrame([{"CaO":28.0,"TiO2":40.0,"SiO2":31.0,"FeOt":1.0}])
        res = mm.TitaniteCalculator(df).calculate_components()
        _assert_cols(self, res, ["VII_site","M_site","T_site","Fe3t_cat_5ox"])


class TestTourmaline(unittest.TestCase):
    def test_tourmaline(self):
        df = pd.DataFrame([{"SiO2":35.0,"Al2O3":32.0,"FeOt":10.0,"MgO":8.0,"Na2O":2.0,"K2O":0.5,"B2O3":2.5}])
        res = mm.TourmalineCalculator(df).calculate_components()
        _assert_cols(self, res, ["X_site","Y_site","Z_site","T_site"])


class TestZircon(unittest.TestCase):
    def test_zircon(self):
        df = pd.DataFrame([{"SiO2":33.0,"ZrO2":65.0,"HfO2":1.0}])
        res = mm.ZirconCalculator(df).calculate_components()
        _assert_cols(self, res, ["M_site","T_site","Zr_cat_4ox"])


# Classifiers
class TestPyroxeneClassifier(unittest.TestCase):
    def test_classify_fields(self):
        df = _make_basic_df()
        out = mm.PyroxeneClassifier(df).classify(subclass=True)
        _assert_cols(self, out, ["Mineral","Submineral","En","Fs","Wo"])
        s = float((out["En"] + out["Fs"] + out["Wo"]).iloc[0])
        self.assertTrue(0.99 <= s <= 1.01)

class TestOxideClassifier(unittest.TestCase):
    def test_routing_rhomb_vs_spinel(self):
        rows = [
            {"Predict_Mineral":"Rhombohedral_Oxides","FeOt":65.0,"TiO2":8.0,"Al2O3":1.0},
            {"Predict_Mineral":"Spinels","MgO":20.0,"Al2O3":40.0,"FeOt":20.0,"TiO2":2.0,"Cr2O3":5.0},
        ]
        df = pd.DataFrame(rows)
        out = mm.OxideClassifier(df).calculate_components()
        _assert_cols(self, out, ["XR2","XR3","XTi"])
        # Both rows should have ternary fractions
        s = (out.loc[:, ["XR2","XR3","XTi"]].sum(axis=1)).to_numpy()
        self.assertTrue(np.all((0.99 <= s) & (s <= 1.01)))


if __name__ == "__main__":
    unittest.main()
