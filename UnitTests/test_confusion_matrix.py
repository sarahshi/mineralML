import unittest
import numpy as np
import pandas as pd

# Use a non-interactive backend so tests don’t open windows
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

import mineralML as mm


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
        self.assertIn("sum_col", df.index)
        self.assertEqual(int(df.loc["sum_col", "sum_row"]), int(df.values[:-1, :-1].sum()))

    def test_pred_val_axis_y_transposes_and_labels(self):
        df = self.df.copy()
        mm.pp_matrix(df, pred_val_axis="y")  # exercise transpose branch

        ax = self._get_ax()
        # When pred_val_axis="y", x is Published[True], y is Predicted (per function)
        self.assertEqual(ax.get_xlabel(), "Published [True]")
        self.assertEqual(ax.get_ylabel(), "Predicted")


class TestConfigCellTextAndColorsUnit(unittest.TestCase):
    def test_totals_branch_sets_texts_and_colors(self):
        # 2x2 base -> after totals becomes 3x3
        base = np.array([[5, 2],
                         [1, 7]])
        df = pd.DataFrame(base, index=["A","B"], columns=["A","B"])
        mm.insert_totals(df)  # in-place
        arr = np.array(df.to_records(index=False).tolist())  # same representation used in pp_matrix

        # Make a dummy Axes/Text at the totals cell (bottom-right)
        fig, ax = plt.subplots()
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

    def test_regular_cell_branch_sets_text_and_color(self):
        # Build array with totals added so last row/col exist, but pick a non-total, off-diagonal cell
        base = np.array([[5, 2],
                         [1, 7]])
        df = pd.DataFrame(base, index=["A","B"], columns=["A","B"])
        mm.insert_totals(df)
        arr = np.array(df.to_records(index=False).tolist())

        fig, ax = plt.subplots()
        oText = ax.text(0.5, 0.5, "")  # position doesn’t matter for direct call here
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


if __name__ == "__main__":
    unittest.main()
