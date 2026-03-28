import unittest
from tempfile import TemporaryDirectory
import os
import numpy as np
import pandas as pd

import mineralML as mm


# ---------------------------------------------------------------------------
#  Helpers – build fixture DataFrames and write to temp files
# ---------------------------------------------------------------------------

def _save_csv(df, tmp_dir, name, **kwargs):
    """Save a DataFrame to CSV in tmp_dir and return the full path."""
    path = os.path.join(tmp_dir, name)
    df.to_csv(path, **kwargs)
    return path


def _make_probe4epma_df():
    """Flat-header DataFrame matching Probe4EPMA export format."""
    return pd.DataFrame({
        "SAMPLE":   ["Point1", "Point2"],
        "SiO2":     [50.0, 48.0],
        "FeOt":     [10.0, 11.0],
        "MgO":      [8.0, 9.0],
        "CaO":      [12.0, 11.0],
        "TOTAL":    [80.0, 79.0],
        "Si %ERR":  [2.0, 1.8],
        "Fe %ERR":  [3.0, 2.8],
        "Mg %ERR":  [2.5, 2.2],
        "Ca %ERR":  [1.5, 1.4],
    })


def _make_aztec_rows():
    """
    Headerless row list matching AZtec block export format.
    Returns a DataFrame to be written with header=False, index=False.
    """
    rows = [
        ["Point1", None,  None,  None],
        ["Element", "Oxide", "Oxide %", "Oxide % Sigma"],
        ["Si",  "SiO2", 50.0, 0.5],
        ["Fe",  "FeO",  10.0, 0.3],
        ["Mg",  "MgO",   8.0, 0.2],
        ["Total", None,  68.0, None],
        ["Point2", None,  None,  None],
        ["Element", "Oxide", "Oxide %", "Oxide % Sigma"],
        ["Si",  "SiO2", 48.0, 0.4],
        ["Ca",  "CaO",  12.0, 0.1],
        ["Total", None,  60.0, None],
    ]
    return pd.DataFrame(rows)


def _make_cameca_df():
    """
    MultiIndex-column DataFrame matching Cameca EPMA export format.
    Level 0 = category (Info / Oxide / StdDev wt%), level 1 = column name.
    """
    columns = pd.MultiIndex.from_tuples([
        ("Info",         "Comment"),
        ("Oxide",        "SiO2"),
        ("Oxide",        "FeO"),
        ("Oxide",        "MgO"),
        ("Oxide",        "Total"),
        ("StdDev wt%",   "Si"),
        ("StdDev wt%",   "Fe"),
        ("StdDev wt%",   "Mg"),
    ])
    data = [
        ["Point1", 50.0, 10.0, 8.0, 68.0, 0.9, 0.6, 0.45],
        ["Point2", 48.0, 11.0, 9.0, 68.0, 0.75, 0.54, 0.39],
    ]
    return pd.DataFrame(data, columns=columns)


# ---------------------------------------------------------------------------
#  get_oxide_from_elem
# ---------------------------------------------------------------------------

class TestGetOxideFromElem(unittest.TestCase):

    def test_fe_prefers_feot(self):
        self.assertEqual(mm.get_oxide_from_elem("Fe", ["SiO2", "FeOt", "FeO"]), "FeOt")

    def test_fe_falls_back_to_feo(self):
        self.assertEqual(mm.get_oxide_from_elem("Fe", ["SiO2", "FeO"]), "FeO")

    def test_fe_no_match_returns_element(self):
        self.assertEqual(mm.get_oxide_from_elem("Fe", ["SiO2", "MgO"]), "Fe")

    def test_simple_oxide_match(self):
        self.assertEqual(mm.get_oxide_from_elem("Ni", ["NiO", "SiO2"]), "NiO")
        self.assertEqual(mm.get_oxide_from_elem("Mg", ["MgO", "SiO2"]), "MgO")
        self.assertEqual(mm.get_oxide_from_elem("Ca", ["CaO", "SiO2"]), "CaO")

    def test_complex_oxide_suffixes(self):
        self.assertEqual(mm.get_oxide_from_elem("Si", ["SiO2", "MgO"]), "SiO2")
        self.assertEqual(mm.get_oxide_from_elem("Al", ["Al2O3", "MgO"]), "Al2O3")
        self.assertEqual(mm.get_oxide_from_elem("Na", ["Na2O", "SiO2"]), "Na2O")
        self.assertEqual(mm.get_oxide_from_elem("P", ["P2O5", "SiO2"]), "P2O5")
        self.assertEqual(mm.get_oxide_from_elem("Cr", ["Cr2O3", "SiO2"]), "Cr2O3")
        self.assertEqual(mm.get_oxide_from_elem("Ti", ["TiO2", "MgO"]), "TiO2")
        self.assertEqual(mm.get_oxide_from_elem("K", ["K2O", "MgO"]), "K2O")

    def test_unknown_element_fallback(self):
        self.assertEqual(mm.get_oxide_from_elem("Zz", ["SiO2", "MgO"]), "Zz")


# ---------------------------------------------------------------------------
#  format_for_thermobar
# ---------------------------------------------------------------------------

class TestFormatForThermobar(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            "Sample": ["Pt1", "Pt2"],
            "SiO2": [50.0, 48.0],
            "FeOt": [10.0, 11.0],
            "Total": [98.0, 97.0],
            "SiO2_1sigma": [0.3, 0.4],
            "Predict_Mineral": ["Olivine", "Olivine"],
            "Predict_Score": [0.95, 0.92],
            "Mineral": ["Olivine", "Olivine"],
            "Submineral": ["Fo80", "Fo75"],
            "SampleID": ["A", "B"],
        })

    def test_default_liq_suffix(self):
        out = mm.format_for_thermobar(self.df)

        # Oxide columns get suffix
        self.assertIn("SiO2_Liq", out.columns)
        self.assertIn("FeOt_Liq", out.columns)

        # Skipped columns remain unchanged
        for col in ["Sample", "SampleID", "Total", "Mineral", "Submineral",
                     "Predict_Mineral", "Predict_Score", "SiO2_1sigma"]:
            self.assertIn(col, out.columns)

    def test_custom_suffix(self):
        out = mm.format_for_thermobar(self.df, suffix="_Ol")
        self.assertIn("SiO2_Ol", out.columns)
        self.assertIn("FeOt_Ol", out.columns)

    def test_does_not_mutate_input(self):
        original_cols = list(self.df.columns)
        mm.format_for_thermobar(self.df)
        self.assertEqual(list(self.df.columns), original_cols)

    def test_values_preserved(self):
        out = mm.format_for_thermobar(self.df)
        pd.testing.assert_series_equal(
            out["SiO2_Liq"].reset_index(drop=True),
            self.df["SiO2"].reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
#  extract_probe4epma
# ---------------------------------------------------------------------------

class TestExtractProbe4EPMA(unittest.TestCase):

    def _extract(self, df=None):
        if df is None:
            df = _make_probe4epma_df()
        with TemporaryDirectory() as tmp:
            path = _save_csv(df, tmp, "probe.csv", index=False)
            return mm.extract_probe4epma(path)

    def test_basic_shape_and_columns(self):
        out = self._extract()
        self.assertEqual(len(out), 2)

        # Identifier renamed
        self.assertIn("Sample", out.columns)
        self.assertNotIn("SAMPLE", out.columns)

        # Oxides present
        for ox in ["SiO2", "FeOt", "MgO", "CaO"]:
            self.assertIn(ox, out.columns)

        # Total renamed from TOTAL
        self.assertIn("Total", out.columns)

    def test_sigma_columns_created(self):
        out = self._extract()
        for ox in ["SiO2", "FeOt", "MgO", "CaO"]:
            self.assertIn(f"{ox}_1sigma", out.columns)

    def test_relative_error_converted_to_absolute(self):
        out = self._extract()
        row = out.iloc[0]
        # Point1: SiO2=50.0, Si %ERR=2.0 -> absolute = 50.0 * 2.0/100 = 1.0
        self.assertAlmostEqual(row["SiO2_1sigma"], 50.0 * 2.0 / 100.0, places=6)
        # Point1: FeOt=10.0, Fe %ERR=3.0 -> 10.0 * 3.0/100 = 0.3
        self.assertAlmostEqual(row["FeOt_1sigma"], 10.0 * 3.0 / 100.0, places=6)

    def test_column_order(self):
        out = self._extract()
        cols = list(out.columns)

        # Sample first, then oxides, then Total, then sigmas
        self.assertEqual(cols[0], "Sample")

        total_idx = cols.index("Total")
        oxide_cols = cols[1:total_idx]
        sigma_cols = cols[total_idx + 1:]

        for ox in oxide_cols:
            self.assertFalse(ox.endswith("_1sigma"))
        for sig in sigma_cols:
            self.assertTrue(sig.endswith("_1sigma"))

    def test_o_column_skipped(self):
        df = pd.DataFrame({
            "SAMPLE":  ["Point1"],
            "SiO2":    [50.0],
            "O":       [45.0],
            "FeOt":    [10.0],
            "TOTAL":   [60.0],
            "Si %ERR": [2.0],
            "Fe %ERR": [3.0],
        })
        out = self._extract(df)
        self.assertNotIn("O", out.columns)

    def test_trailing_whitespace_stripped(self):
        df = pd.DataFrame({
            "SAMPLE ":   ["Point1"],
            "SiO2 ":     [50.0],
            "FeOt ":     [10.0],
            "TOTAL ":    [60.0],
            "Si %ERR ":  [2.0],
            "Fe %ERR ":  [3.0],
        })
        out = self._extract(df)
        self.assertIn("SiO2", out.columns)
        self.assertIn("Sample", out.columns)


# ---------------------------------------------------------------------------
#  extract_aztec
# ---------------------------------------------------------------------------

class TestExtractAZtec(unittest.TestCase):

    def _extract(self, block_df=None):
        if block_df is None:
            block_df = _make_aztec_rows()
        with TemporaryDirectory() as tmp:
            path = _save_csv(block_df, tmp, "aztec.csv", header=False, index=False)
            return mm.extract_aztec(path)

    def test_basic_shape_and_columns(self):
        out = self._extract()
        self.assertEqual(len(out), 2)
        self.assertIn("Sample", out.columns)
        self.assertNotIn("SampleID", out.columns)

    def test_feo_converted_to_feot(self):
        out = self._extract()
        self.assertIn("FeOt", out.columns)
        self.assertNotIn("FeO", out.columns)
        self.assertAlmostEqual(out.iloc[0]["FeOt"], 10.0, places=4)

    def test_sigma_columns_created(self):
        out = self._extract()
        self.assertIn("SiO2_1sigma", out.columns)
        self.assertIn("FeOt_1sigma", out.columns)
        self.assertAlmostEqual(out.iloc[0]["SiO2_1sigma"], 0.5, places=4)
        self.assertAlmostEqual(out.iloc[0]["FeOt_1sigma"], 0.3, places=4)

    def test_total_captured(self):
        out = self._extract()
        self.assertIn("Total", out.columns)
        self.assertAlmostEqual(out.iloc[0]["Total"], 68.0, places=4)
        self.assertAlmostEqual(out.iloc[1]["Total"], 60.0, places=4)

    def test_column_order(self):
        out = self._extract()
        cols = list(out.columns)
        self.assertEqual(cols[0], "Sample")
        sigma_start = next(i for i, c in enumerate(cols) if c.endswith("_1sigma"))
        for c in cols[sigma_start:]:
            self.assertTrue(c.endswith("_1sigma"))

    def test_wt_percent_fallback(self):
        # When Oxide % is missing, should fall back to Wt%
        rows = [
            ["Point1", None, None, None],
            ["Element", "Oxide", "Wt%", "Wt% Sigma"],
            ["Si", "SiO2", 50.0, 0.5],
            ["Total", None, 50.0, None],
        ]
        out = self._extract(pd.DataFrame(rows))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.iloc[0]["SiO2"], 50.0, places=4)
        self.assertAlmostEqual(out.iloc[0]["SiO2_1sigma"], 0.5, places=4)

    def test_no_blocks_returns_empty(self):
        out = self._extract(pd.DataFrame({"col1": ["val1"], "col2": ["val2"]}))
        self.assertTrue(out.empty)


# ---------------------------------------------------------------------------
#  extract_cameca
# ---------------------------------------------------------------------------

class TestExtractCameca(unittest.TestCase):

    def _extract(self, df=None):
        if df is None:
            df = _make_cameca_df()
        with TemporaryDirectory() as tmp:
            path = _save_csv(df, tmp, "cameca.csv", index=False)
            return mm.extract_cameca(path)

    def test_basic_shape_and_columns(self):
        out = self._extract()
        self.assertEqual(len(out), 2)

        # Comment renamed to Sample
        self.assertIn("Sample", out.columns)

        # Oxides present (FeO converted to FeOt)
        for ox in ["SiO2", "FeOt", "MgO"]:
            self.assertIn(ox, out.columns)

        self.assertIn("Total", out.columns)

    def test_feo_converted_to_feot(self):
        out = self._extract()
        self.assertIn("FeOt", out.columns)
        self.assertNotIn("FeO", out.columns)

    def test_sigma_columns_created(self):
        out = self._extract()
        for ox in ["SiO2", "FeOt", "MgO"]:
            self.assertIn(f"{ox}_1sigma", out.columns)

    def test_three_sigma_converted_to_one_sigma(self):
        out = self._extract()
        # Point1: raw StdDev for Si = 0.9 (3-sigma) -> 1-sigma = 0.3
        self.assertAlmostEqual(out.iloc[0]["SiO2_1sigma"], 0.9 / 3.0, places=6)
        self.assertAlmostEqual(out.iloc[0]["FeOt_1sigma"], 0.6 / 3.0, places=6)
        self.assertAlmostEqual(out.iloc[0]["MgO_1sigma"], 0.45 / 3.0, places=6)

    def test_column_order(self):
        out = self._extract()
        cols = list(out.columns)
        self.assertEqual(cols[0], "Sample")
        sigma_start = next(i for i, c in enumerate(cols) if c.endswith("_1sigma"))
        for c in cols[sigma_start:]:
            self.assertTrue(c.endswith("_1sigma"))

    def test_header_detection_with_preamble(self):
        # Simulate metadata rows before the actual header
        preamble = "Instrument,Cameca SX5\nDate,2025-01-15\n"
        cameca_df = _make_cameca_df().iloc[:1]  # single data row

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cameca_preamble.csv")
            with open(path, "w") as f:
                f.write(preamble)
                cameca_df.to_csv(f, index=False)

            out = mm.extract_cameca(path)
            self.assertIn("Sample", out.columns)
            self.assertIn("SiO2", out.columns)
            self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
