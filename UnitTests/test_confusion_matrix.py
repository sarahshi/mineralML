import warnings
import unittest
import numpy as np
import pandas as pd

# Use a non-interactive backend so tests don’t open windows
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import mineralML as mm


class TestConfusionMatrixDf(unittest.TestCase):
 
    def test_basic_canonical_minerals(self):
        # Two canonical minerals, perfect predictions
        given = ["Olivine", "Olivine", "Garnet", "Garnet"]
        pred  = ["Olivine", "Olivine", "Garnet", "Garnet"]
        cm = mm.confusion_matrix_df(given, pred)
 
        # Matrix should be a DataFrame with labeled axes
        self.assertIsInstance(cm, pd.DataFrame)
        self.assertIn("Olivine", cm.index)
        self.assertIn("Garnet", cm.index)
 
        # Diagonal should hold all counts
        self.assertEqual(cm.loc["Olivine", "Olivine"], 2)
        self.assertEqual(cm.loc["Garnet", "Garnet"], 2)
 
        # Off-diagonal between these two should be zero
        self.assertEqual(cm.loc["Olivine", "Garnet"], 0)
        self.assertEqual(cm.loc["Garnet", "Olivine"], 0)
 
    def test_misclassifications(self):
        # One Olivine misclassified as Garnet
        given = ["Olivine", "Olivine", "Garnet"]
        pred  = ["Olivine", "Garnet",  "Garnet"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertEqual(cm.loc["Olivine", "Olivine"], 1)
        self.assertEqual(cm.loc["Olivine", "Garnet"], 1)
        self.assertEqual(cm.loc["Garnet", "Garnet"], 1)
 
    def test_label_only_in_given_or_pred(self):
        # Amphibole only in truth, Biotite only in predictions
        given = ["Amphibole", "Amphibole"]
        pred  = ["Amphibole", "Biotite"]
        cm = mm.confusion_matrix_df(given, pred)
 
        # Both labels should appear as rows and columns
        self.assertIn("Amphibole", cm.columns)
        self.assertIn("Biotite", cm.columns)
        self.assertEqual(cm.loc["Amphibole", "Amphibole"], 1)
        self.assertEqual(cm.loc["Amphibole", "Biotite"], 1)
 
    # --- NaN handling ---
 
    def test_nan_warning_and_exclusion(self):
        given = ["Olivine", None, "Garnet"]
        pred  = ["Olivine", "Garnet", None]
        with self.assertWarns(UserWarning) as ctx:
            cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("NaN", str(ctx.warning))
 
        # Only the first row survives (both non-NaN)
        self.assertEqual(cm.loc["Olivine", "Olivine"], 1)
        self.assertEqual(cm.values.sum(), 1)
 
    def test_no_nan_no_warning(self):
        given = ["Olivine", "Olivine"]
        pred  = ["Olivine", "Olivine"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mm.confusion_matrix_df(given, pred)
            nan_warnings = [x for x in w if "NaN" in str(x.message)]
            self.assertEqual(len(nan_warnings), 0)
 
    # --- Spinel group merging ---
 
    def test_magnetite_merged_to_oxide(self):
        # magnetite -> Spinel_Group -> Oxide
        given = ["Magnetite", "Olivine"]
        pred  = ["Olivine",   "Olivine"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Oxide", cm.index)
        self.assertNotIn("Magnetite", cm.index)
        self.assertNotIn("Spinel_Group", cm.index)
        self.assertEqual(cm.loc["Oxide", "Olivine"], 1)
 
    def test_chromite_merged_to_oxide(self):
        given = ["Chromite", "Olivine"]
        pred  = ["Chromite", "Olivine"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Oxide", cm.index)
        self.assertNotIn("Chromite", cm.index)
        self.assertEqual(cm.loc["Oxide", "Oxide"], 1)
 
    def test_hercynite_merged_to_oxide(self):
        given = ["Hercynite"]
        pred  = ["Hercynite"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Oxide", cm.index)
        self.assertNotIn("Hercynite", cm.index)
 
    def test_ulvospinel_merged_to_oxide(self):
        given = ["Ulvospinel"]
        pred  = ["Ulvospinel"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Oxide", cm.index)
        self.assertNotIn("Ulvospinel", cm.index)
 
    def test_spinel_substring_merged_to_oxide(self):
        # Anything containing "spinel" should match
        given = ["Mg-Spinel"]
        pred  = ["Mg-Spinel"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Oxide", cm.index)
        self.assertNotIn("Mg-Spinel", cm.index)
 
    # --- Rhombohedral oxide merging ---
 
    def test_hematite_merged_to_oxide(self):
        given = ["Hematite", "Olivine"]
        pred  = ["Olivine",  "Olivine"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Oxide", cm.index)
        self.assertNotIn("Hematite", cm.index)
        self.assertNotIn("Rhombohedral_Oxides", cm.index)
 
    def test_ilmenite_merged_to_oxide(self):
        given = ["Ilmenite"]
        pred  = ["Ilmenite"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Oxide", cm.index)
        self.assertNotIn("Ilmenite", cm.index)
 
    # --- Na-pyroxene merging ---
 
    def test_na_pyroxene_merged_to_clinopyroxene(self):
        given = ["Na-Pyroxene", "Olivine"]
        pred  = ["Na-Pyroxene", "Olivine"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Clinopyroxene", cm.index)
        self.assertNotIn("Na-Pyroxene", cm.index)
        self.assertEqual(cm.loc["Clinopyroxene", "Clinopyroxene"], 1)
 
    # --- Dynamic parent-group merges (Feldspar) ---
 
    def test_feldspar_parent_merges_children(self):
        # When "Feldspar" appears, Alkali_Feldspar and Plagioclase merge into it
        given = ["Feldspar",        "Alkali_Feldspar", "Plagioclase"]
        pred  = ["Alkali_Feldspar", "Feldspar",        "Plagioclase"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Feldspar", cm.index)
        self.assertNotIn("Alkali_Feldspar", cm.index)
        self.assertNotIn("Plagioclase", cm.index)
 
        # All three should count as Feldspar-Feldspar
        self.assertEqual(cm.loc["Feldspar", "Feldspar"], 3)
 
    def test_no_feldspar_parent_keeps_children(self):
        # Without "Feldspar" label, children stay separate
        given = ["Alkali_Feldspar", "Plagioclase"]
        pred  = ["Alkali_Feldspar", "Plagioclase"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Alkali_Feldspar", cm.index)
        self.assertIn("Plagioclase", cm.index)
        self.assertNotIn("Feldspar", cm.index)
 
    # --- Dynamic parent-group merges (Pyroxene) ---
 
    def test_pyroxene_parent_merges_children(self):
        given = ["Pyroxene",       "Clinopyroxene", "Orthopyroxene"]
        pred  = ["Clinopyroxene",  "Pyroxene",      "Orthopyroxene"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Pyroxene", cm.index)
        self.assertNotIn("Clinopyroxene", cm.index)
        self.assertNotIn("Orthopyroxene", cm.index)
        self.assertEqual(cm.loc["Pyroxene", "Pyroxene"], 3)
 
    def test_no_pyroxene_parent_keeps_children(self):
        given = ["Clinopyroxene", "Orthopyroxene"]
        pred  = ["Clinopyroxene", "Orthopyroxene"]
        cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Clinopyroxene", cm.index)
        self.assertIn("Orthopyroxene", cm.index)
        self.assertNotIn("Pyroxene", cm.index)
 
    # --- Parent label ordering ---
 
    def test_feldspar_inserted_at_child_position(self):
        # Feldspar should appear where Alkali_Feldspar would have been
        given = ["Feldspar", "Olivine"]
        pred  = ["Feldspar", "Olivine"]
        cm = mm.confusion_matrix_df(given, pred)
 
        labels = list(cm.index)
        feldspar_idx = labels.index("Feldspar")
        olivine_idx = labels.index("Olivine")
 
        # In the canonical list Alkali_Feldspar (position 0) comes before Olivine (position 15)
        self.assertLess(feldspar_idx, olivine_idx)
 
    # --- Unrecognized label handling ---
 
    def test_unrecognized_label_warns_and_drops(self):
        given = ["Olivine", "Augite",  "Garnet"]
        pred  = ["Olivine", "Olivine", "Garnet"]
        with self.assertWarns(UserWarning) as ctx:
            cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Augite", str(ctx.warning))
        self.assertNotIn("Augite", cm.index)
        self.assertNotIn("Augite", cm.columns)
 
        # Only 2 valid rows should contribute
        self.assertEqual(cm.values.sum(), 2)
 
    def test_unrecognized_in_pred_warns_and_drops(self):
        given = ["Olivine", "Olivine"]
        pred  = ["Olivine", "Diopside"]
        with self.assertWarns(UserWarning) as ctx:
            cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Diopside", str(ctx.warning))
        self.assertEqual(cm.values.sum(), 1)
 
    def test_all_recognized_no_unrecognized_warning(self):
        given = ["Olivine", "Garnet"]
        pred  = ["Olivine", "Garnet"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mm.confusion_matrix_df(given, pred)
            unrec_warnings = [x for x in w if "Unrecognized" in str(x.message)]
            self.assertEqual(len(unrec_warnings), 0)
 
    # --- Combined merge + unrecognized ---
 
    def test_mixed_merges_and_unrecognized(self):
        # Magnetite -> Oxide (valid), "FakeMineral" -> dropped
        given = ["Magnetite",    "Olivine", "FakeMineral"]
        pred  = ["Olivine",      "Olivine", "Olivine"]
        with self.assertWarns(UserWarning):
            cm = mm.confusion_matrix_df(given, pred)
 
        self.assertIn("Oxide", cm.index)
        self.assertNotIn("FakeMineral", cm.index)
        self.assertNotIn("Magnetite", cm.index)
        # 2 valid rows: Oxide->Olivine and Olivine->Olivine
        self.assertEqual(cm.values.sum(), 2)


class TestInsertTotals(unittest.TestCase):
 
    def test_totals_values(self):
        df = pd.DataFrame([[5, 2],
                           [1, 7]],
                          index=["A", "B"],
                          columns=["A", "B"])
        mm.insert_totals(df)
 
        # Row sums
        self.assertEqual(df.loc["A", "sum_row"], 7)
        self.assertEqual(df.loc["B", "sum_row"], 8)
 
        # Column sums
        self.assertEqual(df.loc["sum_col", "A"], 6)
        self.assertEqual(df.loc["sum_col", "B"], 9)
 
        # Grand total
        self.assertEqual(df.loc["sum_col", "sum_row"], 15)
 
    def test_reinvocation_is_idempotent(self):
        df = pd.DataFrame([[5, 2],
                           [1, 7]],
                          index=["A", "B"],
                          columns=["A", "B"])
        mm.insert_totals(df)
        mm.insert_totals(df)
 
        # Should still be 3 rows and 3 columns, not 4
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(df.loc["sum_col", "sum_row"], 15)


class TestPPMatrixIntegration(unittest.TestCase):
    def setUp(self):
        # small confusion matrix with non-trivial values
        self.df = pd.DataFrame([[5, 2],
                                [1, 7]],
                               index=["A", "B"],
                               columns=["A", "B"])

    def _get_ax(self):
        # The function names the figure "Conf matrix default"
        fig = plt.figure("Conf matrix default")
        ax = fig.gca()
        return ax

    def test_ticks_are_disabled_and_texts_rewritten(self):
        df = self.df.copy()
        mm.pp_matrix(df)  # annot=True by default, so texts exist

        ax = self._get_ax()

        # --- ticks disabled (covers the for-loops over major ticks) ---
        for t in ax.xaxis.get_major_ticks():
            self.assertFalse(t.tick1On)
            self.assertFalse(t.tick2On)
        for t in ax.yaxis.get_major_ticks():
            self.assertFalse(t.tick1On)
            self.assertFalse(t.tick2On)

        # --- texts were rewritten & “sum” gid texts were added (covers add/remove loop) ---
        # There should be at least one text with gid="sum" (added for totals)
        sum_gids = [txt for txt in ax.texts if getattr(txt, "get_gid", lambda: None)() == "sum"]
        self.assertGreater(len(sum_gids), 0, "Expected new sum texts to be added with gid='sum'")

        # --- facecolors updated for totals (covers setting carr & bottom-right special carr) ---
        from matplotlib.collections import QuadMesh
        quad = ax.findobj(QuadMesh)[0]
        facecolors = quad.get_facecolors()

        # The mesh is (n_rows x n_cols) flattened. After insert_totals, matrix is 3x3.
        # Bottom-right grand-total cell should have the darker carr [0.17, 0.20, 0.17, 1.0]
        # We can infer last cell index as -1.
        self.assertTrue(np.allclose(facecolors[-1], [0.17, 0.20, 0.17, 1.0], atol=1e-3))

        # The last column-but-not-last-row (a column total) should have the lighter carr.
        # That cell’s flat index is the last column of row 0 => position 2 (row-major):
        # But QuadMesh flattens in draw order; safest check: at least one non-bottom-right total has the lighter carr.
        lighter = np.array([0.27, 0.30, 0.27, 1.0])
        self.assertTrue(any(np.allclose(c, lighter, atol=1e-3) for c in facecolors[:-1]))

        # --- input df was mutated to include totals (covers insert_totals path used by pp_matrix) ---
        self.assertNotIn("sum_row", df.columns, "pp_matrix should not mutate the input DataFrame")
        self.assertNotIn("sum_col", df.index, "pp_matrix should not mutate the input DataFrame")

    def test_pred_val_axis_y_transposes_and_labels(self):
        df = self.df.copy()
        mm.pp_matrix(df, pred_val_axis="y")  # exercise transpose branch

        ax = self._get_ax()
        # When pred_val_axis="y", x is Published[True], y is Predicted (per function)
        self.assertEqual(ax.get_xlabel(), "Published [True]")
        self.assertEqual(ax.get_ylabel(), "Predicted")


class TestConfigCellTextAndColorsUnit(unittest.TestCase):
 
    def _make_arr_and_ax(self):
        """Helper: 2x2 base -> 3x3 after insert_totals."""
        base = np.array([[5, 2],
                         [1, 7]])
        df = pd.DataFrame(base, index=["A", "B"], columns=["A", "B"])
        mm.insert_totals(df)
        arr = np.array(df.to_records(index=False).tolist())
        fig, ax = plt.subplots()
        return arr, ax
 
    def test_totals_branch_sets_texts_and_colors(self):
        arr, ax = self._make_arr_and_ax()
 
        # Place a text at (2.5, 2.5) because pp_matrix positions texts centered on cells at +0.5
        oText = ax.text(2.5, 2.5, "placeholder")
 
        # facecolors array with one slot; we will write into index 0 by passing posi=0
        facecolors = np.zeros((1, 4))
        text_add, text_del = mm.config_cell_text_and_colors(
            arr, lin=2, col=2, oText=oText, facecolors=facecolors, posi=0,
            fz=12, fmt=".2f", show_null_values=0
        )
 
        # original text should be scheduled for deletion
        self.assertEqual(len(text_del), 1)
        self.assertIs(text_del[0], oText)
 
        # three new stacked texts (value, %ok, %err)
        self.assertEqual(len(text_add), 3)
        self.assertTrue(any(d["text"].endswith("%") for d in text_add))
 
        # bottom-right should get darker carr
        self.assertTrue(np.allclose(facecolors[0], [0.17, 0.20, 0.17, 1.0], atol=1e-3))
 
    def test_row_total_cell(self):
        # Last column, non-last row -> row total
        arr, ax = self._make_arr_and_ax()
        oText = ax.text(2.5, 0.5, "placeholder")
        facecolors = np.zeros((1, 4))
 
        text_add, text_del = mm.config_cell_text_and_colors(
            arr, lin=0, col=2, oText=oText, facecolors=facecolors, posi=0,
            fz=12, fmt=".2f", show_null_values=0
        )
 
        # Should delete old text and add three new ones
        self.assertEqual(len(text_del), 1)
        self.assertEqual(len(text_add), 3)
 
        # Should get the lighter total-cell color
        self.assertTrue(np.allclose(facecolors[0], [0.27, 0.30, 0.27, 1.0], atol=1e-3))
 
    def test_column_total_cell(self):
        # Last row, non-last column -> column total
        arr, ax = self._make_arr_and_ax()
        oText = ax.text(0.5, 2.5, "placeholder")
        facecolors = np.zeros((1, 4))
 
        text_add, text_del = mm.config_cell_text_and_colors(
            arr, lin=2, col=0, oText=oText, facecolors=facecolors, posi=0,
            fz=12, fmt=".2f", show_null_values=0
        )
 
        self.assertEqual(len(text_del), 1)
        self.assertEqual(len(text_add), 3)
        self.assertTrue(np.allclose(facecolors[0], [0.27, 0.30, 0.27, 1.0], atol=1e-3))
 
    def test_regular_cell_branch_sets_text_and_color(self):
        arr, ax = self._make_arr_and_ax()
        oText = ax.text(0.5, 0.5, "")
        facecolors = np.zeros((1, 4))
 
        # Pick lin=0, col=1 (off-diagonal, not totals)
        text_add, text_del = mm.config_cell_text_and_colors(
            arr, lin=0, col=1, oText=oText, facecolors=facecolors, posi=0,
            fz=12, fmt=".2f", show_null_values=0
        )
        # No add/del lists used in regular cells; text is edited in place
        self.assertEqual(text_add, [])
        self.assertEqual(text_del, [])
 
        # Off-diagonal should be red text and not diagonal carr color
        self.assertEqual(oText.get_color(), "r")
        self.assertFalse(np.allclose(facecolors[0], [0.35, 0.8, 0.55, 1.0], atol=1e-3))
 
        # Now test main diagonal cell coloring branch
        oText2 = ax.text(1.5, 1.5, "")
        facecolors2 = np.zeros((1, 4))
        mm.config_cell_text_and_colors(
            arr, lin=1, col=1, oText=oText2, facecolors=facecolors2, posi=0,
            fz=12, fmt=".2f", show_null_values=0
        )
        self.assertEqual(oText2.get_color(), "k")  # diagonal text is black in your code
        self.assertTrue(np.allclose(facecolors2[0], [0.35, 0.8, 0.55, 1.0], atol=1e-3))
 
    def test_show_null_values_zero_hides_text(self):
        # Build a matrix where lin=0, col=1 has value 0
        base = np.array([[5, 0],
                         [1, 7]])
        df = pd.DataFrame(base, index=["A", "B"], columns=["A", "B"])
        mm.insert_totals(df)
        arr = np.array(df.to_records(index=False).tolist())
 
        fig, ax = plt.subplots()
        oText = ax.text(1.5, 0.5, "")
        facecolors = np.zeros((1, 4))
 
        mm.config_cell_text_and_colors(
            arr, lin=0, col=1, oText=oText, facecolors=facecolors, posi=0,
            fz=12, fmt=".2f", show_null_values=0
        )
        self.assertEqual(oText.get_text(), "")
 
    def test_show_null_values_one_shows_zero(self):
        base = np.array([[5, 0],
                         [1, 7]])
        df = pd.DataFrame(base, index=["A", "B"], columns=["A", "B"])
        mm.insert_totals(df)
        arr = np.array(df.to_records(index=False).tolist())
 
        fig, ax = plt.subplots()
        oText = ax.text(1.5, 0.5, "")
        facecolors = np.zeros((1, 4))
 
        mm.config_cell_text_and_colors(
            arr, lin=0, col=1, oText=oText, facecolors=facecolors, posi=0,
            fz=12, fmt=".2f", show_null_values=1
        )
        self.assertEqual(oText.get_text(), "0")
 
    def test_show_null_values_two_shows_zero_and_percent(self):
        base = np.array([[5, 0],
                         [1, 7]])
        df = pd.DataFrame(base, index=["A", "B"], columns=["A", "B"])
        mm.insert_totals(df)
        arr = np.array(df.to_records(index=False).tolist())
 
        fig, ax = plt.subplots()
        oText = ax.text(1.5, 0.5, "")
        facecolors = np.zeros((1, 4))
 
        mm.config_cell_text_and_colors(
            arr, lin=0, col=1, oText=oText, facecolors=facecolors, posi=0,
            fz=12, fmt=".2f", show_null_values=2
        )
        self.assertEqual(oText.get_text(), "0\n0.0%")
 

if __name__ == "__main__":
    unittest.main()
