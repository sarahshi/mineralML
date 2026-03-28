import unittest
from tempfile import TemporaryDirectory
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mineralML as mm
from mineralML.constants import OXIDES


# ---------------------------------------------------------------------------
#  maps_to_df / df_to_maps
# ---------------------------------------------------------------------------

class TestMapsToDF(unittest.TestCase):

    def test_basic_round_trip(self):
        a = np.arange(12.0).reshape(3, 4)
        b = np.ones((3, 4)) * 5.0
        E = {"A": a, "B": b}

        df, shape = mm.maps_to_df(E)
        self.assertEqual(shape, (3, 4))
        self.assertEqual(len(df), 12)
        self.assertIn("A", df.columns)
        self.assertIn("B", df.columns)

        # Round-trip back
        maps = mm.df_to_maps(df, shape)
        np.testing.assert_array_equal(maps["A"], a)
        np.testing.assert_array_equal(maps["B"], b)

    def test_empty_dict_raises(self):
        with self.assertRaises(ValueError):
            mm.maps_to_df({})

    def test_inconsistent_shapes_raises(self):
        with self.assertRaises(ValueError):
            mm.maps_to_df({"A": np.zeros((3, 4)), "B": np.zeros((2, 4))})


# ---------------------------------------------------------------------------
#  renormalize_maps
# ---------------------------------------------------------------------------

class TestRenormalizeMaps(unittest.TestCase):

    def test_sums_to_100(self):
        ox = {
            "SiO2": np.array([[40.0, 20.0], [30.0, 10.0]]),
            "MgO":  np.array([[10.0, 30.0], [20.0, 40.0]]),
        }
        out = mm.renormalize_maps(ox)
        totals = out["SiO2"] + out["MgO"]
        np.testing.assert_allclose(totals, 100.0, atol=1e-6)

    def test_preserves_relative_proportions(self):
        ox = {
            "SiO2": np.array([[60.0]]),
            "MgO":  np.array([[30.0]]),
        }
        out = mm.renormalize_maps(ox)
        # 60/(60+30) = 2/3, 30/(60+30) = 1/3
        self.assertAlmostEqual(out["SiO2"][0, 0], 200 / 3.0, places=4)
        self.assertAlmostEqual(out["MgO"][0, 0], 100 / 3.0, places=4)

    def test_zero_total_pixel_becomes_nan(self):
        ox = {
            "SiO2": np.array([[0.0, 50.0]]),
            "MgO":  np.array([[0.0, 50.0]]),
        }
        out = mm.renormalize_maps(ox)
        self.assertTrue(np.isnan(out["SiO2"][0, 0]))
        self.assertAlmostEqual(out["SiO2"][0, 1], 50.0)


# ---------------------------------------------------------------------------
#  _ensure_columns
# ---------------------------------------------------------------------------

class TestEnsureColumns(unittest.TestCase):

    def test_reindex_to_oxides(self):
        df = pd.DataFrame({"SiO2": [50], "MgO": [8], "Extra": [99]})
        out = mm._ensure_columns(df)
        self.assertEqual(list(out.columns), OXIDES)
        self.assertNotIn("Extra", out.columns)
        self.assertEqual(out["SiO2"].iloc[0], 50)
        self.assertTrue(pd.isna(out["TiO2"].iloc[0]))

    def test_feo_renamed_to_feot(self):
        df = pd.DataFrame({"SiO2": [50], "FeO": [10]})
        out = mm._ensure_columns(df)
        self.assertIn("FeOt", out.columns)
        self.assertEqual(out["FeOt"].iloc[0], 10)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"SiO2": [50], "FeO": [10]})
        original_cols = list(df.columns)
        mm._ensure_columns(df)
        self.assertEqual(list(df.columns), original_cols)


# ---------------------------------------------------------------------------
#  _clean_labels_1d
# ---------------------------------------------------------------------------

class TestCleanLabels1D(unittest.TestCase):

    def test_basic_cleaning(self):
        arr = np.array(["Olivine", "  Garnet  ", "Olivine", "nan", None, "", "None"])
        out = mm._clean_labels_1d(arr)
        self.assertEqual(list(out), ["Olivine", "Garnet", "Olivine"])

    def test_2d_input_flattened(self):
        arr = np.array([["Olivine", "Garnet"], ["nan", "Olivine"]])
        out = mm._clean_labels_1d(arr)
        self.assertEqual(len(out), 2)

    def test_all_invalid_returns_empty(self):
        arr = np.array(["nan", "None", "", "null"])
        out = mm._clean_labels_1d(arr)
        self.assertTrue(out.empty)


# ---------------------------------------------------------------------------
#  pick_common_phases
# ---------------------------------------------------------------------------

class TestPickCommonPhases(unittest.TestCase):

    def test_sorted_by_frequency(self):
        arr = np.array(["A", "A", "A", "B", "B", "C"])
        phases = mm.pick_common_phases(arr)
        self.assertEqual(phases[0], "A")
        self.assertIn("B", phases)
        self.assertIn("C", phases)

    def test_top_k(self):
        arr = np.array(["A", "A", "A", "B", "B", "C"])
        phases = mm.pick_common_phases(arr, top_k=2)
        self.assertEqual(len(phases), 2)
        self.assertEqual(phases[0], "A")

    def test_empty_input(self):
        arr = np.array(["nan", "None", ""])
        self.assertEqual(mm.pick_common_phases(arr), [])


# ---------------------------------------------------------------------------
#  _make_palette
# ---------------------------------------------------------------------------

class TestMakePalette(unittest.TestCase):

    def test_returns_dict_with_rgb_tuples(self):
        labels = ["Olivine", "Garnet", "Glass"]
        palette = mm._make_palette(labels)
        self.assertEqual(set(palette.keys()), set(labels))
        for rgb in palette.values():
            self.assertEqual(len(rgb), 3)
            self.assertTrue(all(0 <= c <= 1 for c in rgb))

    def test_channel_capped_below_one(self):
        # Each channel is capped at 0.95 to avoid pure white
        labels = ["A"]
        palette = mm._make_palette(labels)
        for c in palette["A"]:
            self.assertLessEqual(c, 0.95)


# ---------------------------------------------------------------------------
#  _auto_bar_width / _auto_limits / _auto_figsize_from_array
# ---------------------------------------------------------------------------

class TestAutoHelpers(unittest.TestCase):

    def test_auto_bar_width_bounds(self):
        self.assertGreaterEqual(mm._auto_bar_width(1), 6.0)
        self.assertLessEqual(mm._auto_bar_width(100), 22.0)

    def test_auto_limits_std_mode(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        vmin, vmax = mm._auto_limits(data, mode="std")
        self.assertLess(vmin, vmax)
        self.assertAlmostEqual((vmin + vmax) / 2, np.mean(data), places=4)

    def test_auto_limits_percentile_mode(self):
        data = np.random.normal(50, 5, size=(100, 100))
        vmin, vmax = mm._auto_limits(data, mode="percentile", percentile=(5, 95))
        self.assertLess(vmin, vmax)
        self.assertGreater(vmin, data.min())
        self.assertLess(vmax, data.max())

    def test_auto_limits_all_nan(self):
        data = np.full((3, 3), np.nan)
        vmin, vmax = mm._auto_limits(data)
        self.assertEqual(vmin, 0.0)
        self.assertEqual(vmax, 1.0)

    def test_auto_figsize_returns_positive(self):
        for side in ("right", "left", "top", "bottom", "other"):
            w, h = mm._auto_figsize_from_array((100, 200), n_legend=5, legend_side=side)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)


# ---------------------------------------------------------------------------
#  _add_scalebar
# ---------------------------------------------------------------------------

class TestAddScalebar(unittest.TestCase):

    def test_returns_none_when_no_scalebar_um(self):
        fig, ax = plt.subplots()
        result = mm._add_scalebar(ax, scalebar_um=None, pixel_size_um=1.0)
        self.assertIsNone(result)
        plt.close(fig)

    def test_warns_when_no_pixel_size(self):
        fig, ax = plt.subplots()
        with self.assertWarns(UserWarning):
            result = mm._add_scalebar(ax, scalebar_um=100, pixel_size_um=None, warn=True)
        self.assertIsNone(result)
        plt.close(fig)

    def test_adds_artist_when_both_provided(self):
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((10, 10)))
        bar = mm._add_scalebar(ax, scalebar_um=50, pixel_size_um=5.0)
        self.assertIsNotNone(bar)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  remove_islands
# ---------------------------------------------------------------------------

class TestRemoveIslands(unittest.TestCase):

    def test_single_pixel_removed(self):
        # 5x5 map with one isolated pixel
        m = np.full((5, 5), "Olivine", dtype=object)
        m[2, 2] = "Garnet"
        cleaned = mm.remove_islands(m, min_size=2, fill_val="nan")
        self.assertEqual(cleaned[2, 2], "nan")

    def test_large_cluster_preserved(self):
        m = np.full((5, 5), "Olivine", dtype=object)
        m[0:3, 0:3] = "Garnet"
        cleaned = mm.remove_islands(m, min_size=2, fill_val="nan")
        # 3x3 = 9 pixels, well above min_size=2
        self.assertEqual(cleaned[1, 1], "Garnet")

    def test_phase_min_sizes(self):
        # Garnet has a 3-pixel cluster; set its min to 4 so it gets removed
        m = np.full((5, 5), "Olivine", dtype=object)
        m[0, 0:3] = "Garnet"
        cleaned = mm.remove_islands(m, min_size=2, phase_min_sizes={"Garnet": 4}, fill_val="nan")
        self.assertEqual(cleaned[0, 0], "nan")

    def test_grouped_phases(self):
        # Two adjacent pyroxenes treated as one group
        m = np.full((5, 5), "Olivine", dtype=object)
        m[0, 0] = "Clinopyroxene"
        m[0, 1] = "Orthopyroxene"
        # Individually each is 1 pixel (< min_size=2), but grouped they are 2
        cleaned = mm.remove_islands(
            m, min_size=2, grouped_phases=[("Clinopyroxene", "Orthopyroxene")], fill_val="nan"
        )
        self.assertEqual(cleaned[0, 0], "Clinopyroxene")
        self.assertEqual(cleaned[0, 1], "Orthopyroxene")

    def test_integer_map(self):
        m = np.ones((5, 5), dtype=int)
        m[2, 2] = 2
        cleaned = mm.remove_islands(m, min_size=2, fill_val=0)
        self.assertEqual(cleaned[2, 2], 0)


# ---------------------------------------------------------------------------
#  fill_phase_holes
# ---------------------------------------------------------------------------

class TestFillPhaseHoles(unittest.TestCase):

    def test_small_hole_filled(self):
        m = np.full((5, 5), "Olivine", dtype=object)
        m[2, 2] = np.nan
        filled = mm.fill_phase_holes(m, max_hole_size=10)
        self.assertEqual(str(filled[2, 2]), "Olivine")

    def test_excluded_phase_not_expanded(self):
        m = np.full((5, 5), "Glass", dtype=object)
        m[2, 2] = np.nan
        filled = mm.fill_phase_holes(m, max_hole_size=10, exclude_phases=["Glass"])
        # Glass is excluded from expansion; the hole should remain
        self.assertTrue(pd.isna(filled[2, 2]) or str(filled[2, 2]) in {"nan", "None"})

    def test_large_hole_not_filled(self):
        m = np.full((10, 10), "Olivine", dtype=object)
        m[2:8, 2:8] = np.nan  # 36-pixel hole
        filled = mm.fill_phase_holes(m, max_hole_size=5)
        # Center should still be empty
        self.assertTrue(pd.isna(filled[4, 4]) or str(filled[4, 4]) in {"nan", "None"})


# ---------------------------------------------------------------------------
#  load_element_maps (file I/O)
# ---------------------------------------------------------------------------

class TestLoadElementMaps(unittest.TestCase):

    def test_loads_matching_csvs(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        with TemporaryDirectory() as tmp:
            # Write two element CSVs
            pd.DataFrame(arr).to_csv(os.path.join(tmp, "Si_Ka.csv"), header=False, index=False)
            pd.DataFrame(arr * 2).to_csv(os.path.join(tmp, "Fe_Ka.csv"), header=False, index=False)

            out = mm.load_element_maps(tmp, verbose=False)
            self.assertIn("Si", out)
            self.assertIn("Fe", out)
            np.testing.assert_array_equal(out["Si"], arr)
            np.testing.assert_array_equal(out["Fe"], arr * 2)

    def test_skips_non_element_files(self):
        with TemporaryDirectory() as tmp:
            pd.DataFrame([[1]]).to_csv(os.path.join(tmp, "metadata.csv"), header=False, index=False)
            out = mm.load_element_maps(tmp, verbose=False)
            self.assertEqual(len(out), 0)

    def test_not_a_directory_raises(self):
        with self.assertRaises(NotADirectoryError):
            mm.load_element_maps("/nonexistent/path", verbose=False)

    def test_drop_trailing_blank(self):
        arr = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])
        with TemporaryDirectory() as tmp:
            pd.DataFrame(arr).to_csv(os.path.join(tmp, "Si_Ka.csv"), header=False, index=False)
            out = mm.load_element_maps(tmp, drop_trailing_blank=True, verbose=False)
            self.assertEqual(out["Si"].shape, (2, 2))

    def test_drop_trailing_blank_nan(self):
        arr = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, np.nan]])
        with TemporaryDirectory() as tmp:
            pd.DataFrame(arr).to_csv(os.path.join(tmp, "Si_Ka.csv"), header=False, index=False)
            out = mm.load_element_maps(tmp, drop_trailing_blank=True, verbose=False)
            self.assertEqual(out["Si"].shape, (2, 2))


# ---------------------------------------------------------------------------
#  parse_ctf_header
# ---------------------------------------------------------------------------

class TestParseCTFHeader(unittest.TestCase):

    def _write_ctf(self, tmp_dir, content):
        path = os.path.join(tmp_dir, "test.ctf")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_parses_dimensions_and_phases(self):
        ctf = (
            "Channel Text File\n"
            "XCells\t100\n"
            "YCells\t50\n"
            "Phases\t2\n"
            "3.24\t5.41\t7.50\t90\t90\t90\tAnorthite\tSodic plagioclase\t12345\n"
            "5.43\t5.43\t5.43\t90\t90\t90\tForsterite\tOlivine\t67890\n"
            "Phase\tX\tY\tBands\tError\n"
        )
        with TemporaryDirectory() as tmp:
            path = self._write_ctf(tmp, ctf)
            x, y, data_start, mapping = mm.parse_ctf_header(path)
            self.assertEqual(x, 100)
            self.assertEqual(y, 50)
            self.assertEqual(mapping[0], "Unindexed")
            self.assertIn(1, mapping)
            self.assertIn(2, mapping)

    def test_missing_header_raises(self):
        ctf = "Just some text\nNo valid header\n"
        with TemporaryDirectory() as tmp:
            path = self._write_ctf(tmp, ctf)
            with self.assertRaises(ValueError):
                mm.parse_ctf_header(path)


# ---------------------------------------------------------------------------
#  Plot smoke tests
# ---------------------------------------------------------------------------

class TestPlotPhaseMapSmoke(unittest.TestCase):

    def test_returns_figure_and_cleaned_map(self):
        m = np.array([["Olivine", "Olivine", "Garnet"],
                       ["Garnet",  "Olivine", "Garnet"]], dtype=object)
        fig, ax, cleaned = mm.plot_phase_map(m)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(cleaned.shape, m.shape)
        plt.close(fig)


class TestPlotPhaseCountsSmoke(unittest.TestCase):

    def test_returns_figure(self):
        m = np.array(["Olivine", "Olivine", "Garnet", "Garnet", "Glass"])
        fig, ax = mm.plot_phase_counts(m)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_labels(self):
        m = np.array(["nan", "None", ""])
        fig, ax = mm.plot_phase_counts(m)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPhaseProportionsSmoke(unittest.TestCase):

    def test_returns_figure(self):
        m = np.array([["Olivine", "Olivine"], ["Garnet", "Garnet"]], dtype=object)
        fig, ax = mm.plot_phase_proportions(m)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()