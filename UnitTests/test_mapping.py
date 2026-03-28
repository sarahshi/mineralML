import unittest
import unittest.mock
import warnings
from tempfile import TemporaryDirectory
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mineralML as mm
from mineralML.constants import OXIDES
from mineralML.mapping import (
    _ensure_columns,
    _clean_labels_1d,
    _make_palette,
    _auto_bar_width,
    _auto_limits,
    _auto_figsize_from_array,
    _add_scalebar,
    _plot_continuous_map,
)


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
        out = _ensure_columns(df)
        self.assertEqual(list(out.columns), OXIDES)
        self.assertNotIn("Extra", out.columns)
        self.assertEqual(out["SiO2"].iloc[0], 50)
        self.assertTrue(pd.isna(out["TiO2"].iloc[0]))

    def test_feo_renamed_to_feot(self):
        df = pd.DataFrame({"SiO2": [50], "FeO": [10]})
        out = _ensure_columns(df)
        self.assertIn("FeOt", out.columns)
        self.assertEqual(out["FeOt"].iloc[0], 10)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"SiO2": [50], "FeO": [10]})
        original_cols = list(df.columns)
        _ensure_columns(df)
        self.assertEqual(list(df.columns), original_cols)


# ---------------------------------------------------------------------------
#  _clean_labels_1d
# ---------------------------------------------------------------------------

class TestCleanLabels1D(unittest.TestCase):

    def test_basic_cleaning(self):
        arr = np.array(["Olivine", "  Garnet  ", "Olivine", "nan", None, "", "None"])
        out = _clean_labels_1d(arr)
        self.assertEqual(list(out), ["Olivine", "Garnet", "Olivine"])

    def test_2d_input_flattened(self):
        arr = np.array([["Olivine", "Garnet"], ["nan", "Olivine"]])
        out = _clean_labels_1d(arr)
        self.assertEqual(len(out), 3)

    def test_all_invalid_returns_empty(self):
        arr = np.array(["nan", "None", "", "null"])
        out = _clean_labels_1d(arr)
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
        palette = _make_palette(labels)
        self.assertEqual(set(palette.keys()), set(labels))
        for rgb in palette.values():
            self.assertEqual(len(rgb), 3)
            self.assertTrue(all(0 <= c <= 1 for c in rgb))

    def test_channel_capped_below_one(self):
        # Each channel is capped at 0.95 to avoid pure white
        labels = ["A"]
        palette = _make_palette(labels)
        for c in palette["A"]:
            self.assertLessEqual(c, 0.95)


# ---------------------------------------------------------------------------
#  _auto_bar_width / _auto_limits / _auto_figsize_from_array
# ---------------------------------------------------------------------------

class TestAutoHelpers(unittest.TestCase):

    def test_auto_bar_width_bounds(self):
        self.assertGreaterEqual(_auto_bar_width(1), 6.0)
        self.assertLessEqual(_auto_bar_width(100), 22.0)

    def test_auto_limits_std_mode(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        vmin, vmax = _auto_limits(data, mode="std")
        self.assertLess(vmin, vmax)
        self.assertAlmostEqual((vmin + vmax) / 2, np.mean(data), places=4)

    def test_auto_limits_percentile_mode(self):
        data = np.random.normal(50, 5, size=(100, 100))
        vmin, vmax = _auto_limits(data, mode="percentile", percentile=(5, 95))
        self.assertLess(vmin, vmax)
        self.assertGreater(vmin, data.min())
        self.assertLess(vmax, data.max())

    def test_auto_limits_all_nan(self):
        data = np.full((3, 3), np.nan)
        vmin, vmax = _auto_limits(data)
        self.assertEqual(vmin, 0.0)
        self.assertEqual(vmax, 1.0)

    def test_auto_figsize_returns_positive(self):
        for side in ("right", "left", "top", "bottom", "other"):
            w, h = _auto_figsize_from_array((100, 200), n_legend=5, legend_side=side)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)


# ---------------------------------------------------------------------------
#  _add_scalebar
# ---------------------------------------------------------------------------

class TestAddScalebar(unittest.TestCase):

    def test_returns_none_when_no_scalebar_um(self):
        fig, ax = plt.subplots()
        result = _add_scalebar(ax, scalebar_um=None, pixel_size_um=1.0)
        self.assertIsNone(result)
        plt.close(fig)

    def test_warns_when_no_pixel_size(self):
        fig, ax = plt.subplots()
        with self.assertWarns(UserWarning):
            result = _add_scalebar(ax, scalebar_um=100, pixel_size_um=None, warn=True)
        self.assertIsNone(result)
        plt.close(fig)

    def test_adds_artist_when_both_provided(self):
        fig, ax = plt.subplots()
        ax.imshow(np.zeros((10, 10)))
        bar = _add_scalebar(ax, scalebar_um=50, pixel_size_um=5.0)
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
        # Three adjacent pyroxenes treated as one group
        m = np.full((5, 5), "Olivine", dtype=object)
        m[0, 0] = "Clinopyroxene"
        m[0, 1] = "Orthopyroxene"
        m[0, 2] = "Clinopyroxene"
        # Individually each type is 1-2 pixels, but grouped they are 3 (> min_size=2)
        cleaned = mm.remove_islands(
            m, min_size=2, grouped_phases=[("Clinopyroxene", "Orthopyroxene")], fill_val="nan"
        )
        self.assertEqual(cleaned[0, 0], "Clinopyroxene")
        self.assertEqual(cleaned[0, 1], "Orthopyroxene")
        self.assertEqual(cleaned[0, 2], "Clinopyroxene")

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
            "3.24\t5.41\tAnorthite\t7.50\t90\t90\t90\n"
            "5.43\t5.43\tForsterite\t5.43\t90\t90\t90\n"
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
#  _plot_continuous_map
# ---------------------------------------------------------------------------

class TestPlotContinuousMap(unittest.TestCase):

    def test_returns_figure_and_axes(self):
        data = np.random.normal(50, 10, (10, 10))
        fig, ax = _plot_continuous_map(data, title="Test", cmap="viridis",
                                       vmin=30, vmax=70, cbar_label="wt%")
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(ax.get_title(), "Test")
        plt.close(fig)

    def test_existing_axes(self):
        data = np.random.normal(50, 10, (10, 10))
        fig, ax_in = plt.subplots()
        fig_out, ax_out = _plot_continuous_map(data, title="Custom", cmap="magma",
                                               vmin=0, vmax=100, cbar_label="val", ax=ax_in)
        self.assertIs(fig_out, fig)
        self.assertIs(ax_out, ax_in)
        plt.close(fig)

    def test_nan_background_masked(self):
        data = np.full((5, 5), np.nan)
        data[1:4, 1:4] = 50.0
        fig, ax = _plot_continuous_map(data, title="NaN", cmap="viridis",
                                       vmin=0, vmax=100, cbar_label="val")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  plot_phase_map
# ---------------------------------------------------------------------------

class TestPlotPhaseMap(unittest.TestCase):

    def test_returns_figure_and_cleaned_map(self):
        m = np.array([["Olivine", "Olivine", "Garnet"],
                       ["Garnet",  "Olivine", "Garnet"]], dtype=object)
        fig, ax, cleaned = mm.plot_phase_map(m)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(cleaned.shape, m.shape)
        plt.close(fig)

    def test_custom_phases_and_colors(self):
        m = np.array([["Olivine", "Garnet"], ["Glass", "Olivine"]], dtype=object)
        fig, ax, cleaned = mm.plot_phase_map(
            m, phases=["Olivine", "Garnet"], phase_colors={"Olivine": (0.2, 0.6, 0.2)}
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_remove_islands_and_fill_holes(self):
        m = np.full((10, 10), "Olivine", dtype=object)
        m[5, 5] = "Garnet"  # single isolated pixel
        m[2, 2] = np.nan    # small hole
        fig, ax, cleaned = mm.plot_phase_map(
            m, remove_islands_flag=True, fill_holes_flag=True, cleanup_min_size=2
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_legend_placement_sides(self):
        m = np.array([["Olivine", "Garnet"], ["Glass", "Olivine"]], dtype=object)
        for side in ("right", "left", "top", "bottom"):
            fig, ax, _ = mm.plot_phase_map(m, legend_side=side)
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)

    def test_existing_axes(self):
        m = np.array([["Olivine", "Garnet"], ["Garnet", "Olivine"]], dtype=object)
        fig, ax = plt.subplots()
        fig_out, ax_out, cleaned = mm.plot_phase_map(m, ax=ax)
        self.assertIs(ax_out, ax)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  plot_phase_counts
# ---------------------------------------------------------------------------

class TestPlotPhaseCounts(unittest.TestCase):

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

    def test_normalize_false(self):
        m = np.array(["Olivine", "Olivine", "Garnet"])
        fig, ax = mm.plot_phase_counts(m, normalize=False)
        self.assertEqual(ax.get_ylabel(), "Pixels")
        plt.close(fig)

    def test_explicit_phases(self):
        m = np.array(["Olivine", "Olivine", "Garnet", "Glass"])
        fig, ax = mm.plot_phase_counts(m, phases=["Olivine", "Garnet"])
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  plot_phase_proportions
# ---------------------------------------------------------------------------

class TestPlotPhaseProportions(unittest.TestCase):

    def test_returns_figure(self):
        m = np.array([["Olivine", "Olivine"], ["Garnet", "Garnet"]], dtype=object)
        fig, ax = mm.plot_phase_proportions(m)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_labels(self):
        m = np.array(["nan", "None"])
        fig, ax = mm.plot_phase_proportions(m)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_phases_and_colors(self):
        m = np.array(["Olivine", "Olivine", "Garnet", "Glass", "Glass"])
        fig, ax = mm.plot_phase_proportions(
            m, phases=["Olivine", "Garnet"], phase_colors={"Olivine": "#00FF00"}
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  plot_pred_score_histograms
# ---------------------------------------------------------------------------

class TestPlotPredScoreHistograms(unittest.TestCase):

    def test_returns_figure_and_axes(self):
        mineral_map = np.array([["Olivine", "Olivine"], ["Garnet", "Garnet"]], dtype=object)
        scores = np.array([[0.9, 0.85], [0.75, 0.95]])
        fig, axes = mm.plot_pred_score_histograms(scores, mineral_map, pred_score_threshold=0.5)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_empirical_phase_shows_text(self):
        mineral_map = np.array([["Zircon", "Zircon"], ["Olivine", "Olivine"]], dtype=object)
        scores = np.array([[0.9, 0.9], [0.85, 0.95]])
        fig, axes = mm.plot_pred_score_histograms(
            scores, mineral_map, pred_score_threshold=0.5,
            empirical_phases=("Zircon",)
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_phases_above_min_frac(self):
        mineral_map = np.full((2, 2), "nan", dtype=object)
        scores = np.full((2, 2), np.nan)
        fig, axes = mm.plot_pred_score_histograms(scores, mineral_map, pred_score_threshold=0.5)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
#  plot_score_map
# ---------------------------------------------------------------------------

class TestPlotScoreMap(unittest.TestCase):

    def _make_res(self):
        return {
            "mineral_map": np.array([["Olivine", "Garnet"], ["Olivine", "Garnet"]], dtype=object),
            "pred_score_map": np.array([[0.9, 0.8], [0.85, 0.95]]),
        }

    def test_returns_figure(self):
        fig, ax = mm.plot_score_map(self._make_res())
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_phase_filter(self):
        fig, ax = mm.plot_score_map(self._make_res(), phases=["Olivine"])
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_res_raises(self):
        with self.assertRaises(TypeError):
            mm.plot_score_map("not a dict")

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            mm.plot_score_map({"mineral_map": np.zeros((2, 2))})


# ---------------------------------------------------------------------------
#  plot_oxide_map
# ---------------------------------------------------------------------------

class TestPlotOxideMap(unittest.TestCase):

    def _make_res(self):
        return {
            "oxide_maps": {
                "SiO2": np.random.normal(50, 5, (10, 10)),
                "MgO":  np.random.normal(8, 2, (10, 10)),
            }
        }

    def test_returns_figure(self):
        fig, ax = mm.plot_oxide_map(self._make_res(), "SiO2")
        self.assertIsInstance(fig, plt.Figure)
        self.assertIn("SiO2", ax.get_title())
        plt.close(fig)

    def test_custom_title_and_label(self):
        fig, ax = mm.plot_oxide_map(self._make_res(), "MgO", title="Custom", cbar_label="wt%")
        self.assertEqual(ax.get_title(), "Custom")
        plt.close(fig)

    def test_missing_oxide_raises(self):
        with self.assertRaises(KeyError):
            mm.plot_oxide_map(self._make_res(), "FeOt")

    def test_invalid_res_raises(self):
        with self.assertRaises(TypeError):
            mm.plot_oxide_map("not a dict", "SiO2")

    def test_missing_oxide_maps_key_raises(self):
        with self.assertRaises(KeyError):
            mm.plot_oxide_map({}, "SiO2")


# ---------------------------------------------------------------------------
#  plot_component_composite
# ---------------------------------------------------------------------------

class TestPlotComponentComposite(unittest.TestCase):

    def _make_res(self):
        """Build a minimal res dict mimicking run_map output."""
        H, W = 10, 10
        mineral_map = np.full((H, W), "Olivine", dtype=object)
        mineral_map[0:4, :] = "Plagioclase"
        mineral_map[7:, :] = "Glass"

        # Synthetic component data
        ol_fo = np.full((H, W), np.nan)
        ol_fo[4:7, :] = np.random.uniform(0.6, 0.9, (3, W))

        feld_an = np.full((H, W), np.nan)
        feld_an[0:4, :] = np.random.uniform(0.3, 0.8, (4, W))

        return {
            "mineral_map": mineral_map,
            "component_maps": {
                "Olivine.XFo": ol_fo,
                "Feldspar.An": feld_an,
            },
        }

    def test_returns_figure_and_maps(self):
        res = self._make_res()
        fig, mineral_map, comp_maps = mm.plot_component_composite(res)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(mineral_map.shape, (10, 10))
        self.assertIsInstance(comp_maps, dict)
        plt.close(fig)

    def test_missing_mineral_map_raises(self):
        with self.assertRaises(ValueError):
            mm.plot_component_composite({"component_maps": {}})

    def test_existing_axes(self):
        res = self._make_res()
        fig, ax = plt.subplots()
        fig_out, _, _ = mm.plot_component_composite(res, ax=ax)
        self.assertIs(fig_out, fig)
        plt.close(fig)

    def test_save_path(self):
        res = self._make_res()
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "composite.png")
            fig, _, _ = mm.plot_component_composite(res, save_path=path)
            self.assertTrue(os.path.exists(path))
            plt.close(fig)

    def test_empty_component_maps(self):
        res = {
            "mineral_map": np.full((5, 5), "Glass", dtype=object),
            "component_maps": {},
        }
        fig, _, comp_maps = mm.plot_component_composite(res)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(comp_maps, {})
        plt.close(fig)


# ---------------------------------------------------------------------------
#  plot_ctf_phases (EBSD)
# ---------------------------------------------------------------------------

class TestPlotCTFPhases(unittest.TestCase):

    def _write_ctf(self, tmp_dir):
        """Write a minimal .ctf file fixture and return its path."""
        path = os.path.join(tmp_dir, "test.ctf")
        with open(path, "w") as f:
            f.write("Channel Text File\n")
            f.write("XCells\t3\n")
            f.write("YCells\t2\n")
            f.write("XStep\t1.0\n")
            f.write("Phases\t2\n")
            f.write("3.24\t5.41\tAnorthite\t7.50\t90\t90\t90\n")
            f.write("5.43\t5.43\tForsterite\t5.43\t90\t90\t90\n")
            f.write("Phase\tX\tY\tBands\tError\n")
            f.write("1\t0\t0\t5\t0\n")
            f.write("2\t1\t0\t5\t0\n")
            f.write("1\t2\t0\t5\t0\n")
            f.write("1\t0\t1\t5\t0\n")
            f.write("2\t1\t1\t5\t0\n")
            f.write("1\t2\t1\t5\t0\n")
        return path

    def test_returns_expected_outputs(self):
        with TemporaryDirectory() as tmp:
            path = self._write_ctf(tmp)
            fig, phase_map, raw_ids, mapping, unique_names = mm.plot_ctf_phases(path)

            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(phase_map.shape, (2, 3))
            self.assertEqual(raw_ids.shape, (2, 3))
            self.assertIn(0, mapping)  # Unindexed always present
            self.assertIn(1, mapping)
            self.assertIn(2, mapping)
            plt.close(fig)

    def test_rename_dict(self):
        with TemporaryDirectory() as tmp:
            path = self._write_ctf(tmp)
            fig, phase_map, _, mapping, _ = mm.plot_ctf_phases(
                path, rename_dict={"Anorthite": "Plagioclase"}
            )
            self.assertIn("Plagioclase", mapping.values())
            plt.close(fig)

    def test_custom_title(self):
        with TemporaryDirectory() as tmp:
            path = self._write_ctf(tmp)
            fig, _, _, _, _ = mm.plot_ctf_phases(path, title="My EBSD Map")
            plt.close(fig)

    def test_legend_off(self):
        with TemporaryDirectory() as tmp:
            path = self._write_ctf(tmp)
            fig, _, _, _, _ = mm.plot_ctf_phases(path, legend_on=False)
            plt.close(fig)

    def test_existing_axes(self):
        with TemporaryDirectory() as tmp:
            path = self._write_ctf(tmp)
            fig_in, ax_in = plt.subplots()
            fig_out, _, _, _, _ = mm.plot_ctf_phases(path, ax=ax_in)
            plt.close(fig_in)


# ---------------------------------------------------------------------------
#  run_map
# ---------------------------------------------------------------------------

class TestRunMap(unittest.TestCase):

    def _make_oxide_maps(self, shape=(4, 4)):
        """Build a minimal dict of synthetic oxide maps."""
        H, W = shape
        return {
            "SiO2": np.random.uniform(40, 60, (H, W)),
            "MgO":  np.random.uniform(5, 15, (H, W)),
            "FeOt": np.random.uniform(5, 12, (H, W)),
            "CaO":  np.random.uniform(8, 14, (H, W)),
            "Al2O3": np.random.uniform(10, 20, (H, W)),
        }

    def _make_mock_pred(self, n_pixels):
        """Build a synthetic df_pred DataFrame matching predict_class_prob output."""
        minerals = np.random.choice(["Olivine", "Plagioclase", "Clinopyroxene"], n_pixels)
        scores = np.random.uniform(0.6, 1.0, n_pixels)
        return pd.DataFrame({
            "Predict_Mineral": minerals,
            "Prediction_Score": scores,
        })

    @unittest.mock.patch("mineralML.mapping.predict_class_prob")
    @unittest.mock.patch.object(plt, "show")
    def test_dict_input_returns_result(self, _show, mock_pred):
        shape = (4, 4)
        ox_maps = self._make_oxide_maps(shape)
        mock_pred.return_value = self._make_mock_pred(shape[0] * shape[1])

        result = mm.run_map(ox_maps, show=False, n_iterations=1)

        self.assertIsInstance(result, dict)
        for key in ("figs", "shape", "oxide_maps", "df_pred",
                     "mineral_map", "pred_score_map", "kept_phases",
                     "component_maps", "component_frames"):
            self.assertIn(key, result)

        self.assertEqual(result["shape"], shape)
        self.assertEqual(result["mineral_map"].shape, shape)
        self.assertEqual(result["pred_score_map"].shape, shape)
        plt.close("all")

    @unittest.mock.patch("mineralML.mapping.predict_class_prob")
    @unittest.mock.patch.object(plt, "show")
    def test_stacked_bar_style(self, _show, mock_pred):
        shape = (4, 4)
        ox_maps = self._make_oxide_maps(shape)
        mock_pred.return_value = self._make_mock_pred(shape[0] * shape[1])

        result = mm.run_map(ox_maps, bar_style="stacked", show=False, n_iterations=1)
        self.assertIsInstance(result, dict)
        plt.close("all")

    @unittest.mock.patch("mineralML.mapping.predict_class_prob")
    @unittest.mock.patch.object(plt, "show")
    def test_epoxy_threshold_masks_pixels(self, _show, mock_pred):
        shape = (4, 4)
        ox_maps = self._make_oxide_maps(shape)
        # Set some pixels below threshold
        ox_maps["SiO2"][0, :] = 1.0
        mock_pred.return_value = self._make_mock_pred(shape[0] * shape[1])

        result = mm.run_map(ox_maps, epoxy_threshold=10.0, show=False, n_iterations=1)
        # Row 0 should have been masked to NaN across all oxides
        self.assertTrue(np.all(np.isnan(result["oxide_maps"]["MgO"][0, :])))
        plt.close("all")

    @unittest.mock.patch("mineralML.mapping.predict_class_prob")
    @unittest.mock.patch.object(plt, "show")
    def test_phases_and_exclude_warns(self, _show, mock_pred):
        shape = (4, 4)
        ox_maps = self._make_oxide_maps(shape)
        mock_pred.return_value = self._make_mock_pred(shape[0] * shape[1])

        with self.assertWarns(UserWarning):
            mm.run_map(
                ox_maps, phases=["Olivine"], exclude_phases=["Garnet"],
                show=False, n_iterations=1,
            )
        plt.close("all")

    def test_invalid_input_type_raises(self):
        with self.assertRaises(TypeError):
            mm.run_map(12345)

    def test_empty_dict_raises(self):
        with self.assertRaises(ValueError):
            mm.run_map({})


if __name__ == "__main__":
    unittest.main()