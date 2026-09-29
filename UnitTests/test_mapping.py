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
from matplotlib.backend_bases import MouseEvent, KeyEvent

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
    _coerce_profile_color,
    _profile_table_for_key,
    _resolve_profile_value_columns,
    _line_strip_geometry,
    _resolve_phases,
)


# ---------------------------------------------------------------------------
#  map loading
# ---------------------------------------------------------------------------


class TestLoadMapsFromDir(unittest.TestCase):

    def _write_element_csvs(self, tmp_dir, si_arr, mg_arr):
        """Write Si and Mg element CSVs to a temp directory."""
        pd.DataFrame(si_arr).to_csv(
            os.path.join(tmp_dir, "Si_Ka.csv"), header=False, index=False
        )
        pd.DataFrame(mg_arr).to_csv(
            os.path.join(tmp_dir, "Mg_Ka.csv"), header=False, index=False
        )

    def test_element_wt_percent_converts(self):
        si = np.array([[25.0, 30.0], [28.0, 26.0]])
        mg = np.array([[3.0, 4.0], [5.0, 2.0]])
        with TemporaryDirectory() as tmp:
            self._write_element_csvs(tmp, si, mg)
            ox = mm.load_maps_from_dir(tmp, units="element_wt%")

        # Stoichiometric conversion: oxide values should be larger than element values
        self.assertIn("SiO2", ox)
        self.assertIn("MgO", ox)
        self.assertEqual(ox["SiO2"].shape, (2, 2))
        self.assertTrue(np.all(ox["SiO2"] > si))
        self.assertTrue(np.all(ox["MgO"] > mg))

    def test_oxide_wt_percent_identity(self):
        si = np.array([[50.0, 55.0], [52.0, 48.0]])
        mg = np.array([[8.0, 9.0], [7.0, 10.0]])
        with TemporaryDirectory() as tmp:
            self._write_element_csvs(tmp, si, mg)
            ox = mm.load_maps_from_dir(tmp, units="oxide_wt%")

        # Identity: values should pass through unchanged
        self.assertIn("SiO2", ox)
        self.assertIn("MgO", ox)
        np.testing.assert_array_almost_equal(ox["SiO2"], si)
        np.testing.assert_array_almost_equal(ox["MgO"], mg)

    def test_renormalize_sums_to_100(self):
        si = np.array([[40.0, 50.0], [45.0, 35.0]])
        mg = np.array([[10.0, 20.0], [15.0, 25.0]])
        with TemporaryDirectory() as tmp:
            self._write_element_csvs(tmp, si, mg)
            ox = mm.load_maps_from_dir(tmp, units="oxide_wt%", renormalize=True)

        totals = ox["SiO2"] + ox["MgO"]
        np.testing.assert_allclose(totals, 100.0, atol=1e-6)

    def test_invalid_units_raises(self):
        si = np.array([[25.0]])
        mg = np.array([[3.0]])
        with TemporaryDirectory() as tmp:
            self._write_element_csvs(tmp, si, mg)
            with self.assertRaises(ValueError):
                mm.load_maps_from_dir(tmp, units="counts")


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
    def test_total_threshold_masks_pixels(self, _show, mock_pred):
        shape = (4, 4)
        ox_maps = self._make_oxide_maps(shape)
        # Set row 0 to very low values so the total falls below threshold
        for k in ox_maps:
            ox_maps[k][0, :] = 0.1
        mock_pred.return_value = self._make_mock_pred(shape[0] * shape[1])

        result = mm.run_map(ox_maps, total_threshold=50.0, show=False, n_iterations=1)
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


# ---------------------------------------------------------------------------
#  shared fixtures
# ---------------------------------------------------------------------------


def _make_res(H=12, W=16):
    """
    Minimal run_map-style result. SiO2 rises by 1 wt% per column and MgO
    falls by 0.5 wt% per row, so profile and region values can be checked
    by hand. Columns 0-7 are olivine, 8-15 plagioclase.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    sio2 = 40.0 + xx
    mgo = 20.0 - 0.5 * yy
    mineral_map = np.where(xx < 8, "Olivine", "Plagioclase").astype(object)
    return {
        "mineral_map": mineral_map,
        "oxide_maps": {"SiO2": sio2, "MgO": mgo, "Total": sio2 + mgo},
        "component_maps": {"Olivine.XFo": np.where(xx < 8, 0.9, np.nan)},
        "kept_phases": ["Olivine", "Plagioclase"],
        "shape": (H, W),
    }


def _draw(ax):
    ax.figure.canvas.draw()


def _click(ax, x, y, button=1):
    """Send a button press at data coordinates (x, y) on ax."""
    _draw(ax)
    px, py = ax.transData.transform((x, y))
    ev = MouseEvent("button_press_event", ax.figure.canvas, px, py, button=button)
    ax.figure.canvas.callbacks.process("button_press_event", ev)


def _key(fig, key):
    ev = KeyEvent("key_press_event", fig.canvas, key)
    fig.canvas.callbacks.process("key_press_event", ev)


def _drag(ax, x0, y0, x1, y1):
    """Press at (x0, y0), move and release at (x1, y1), in data coordinates."""
    _draw(ax)
    canvas = ax.figure.canvas
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    for name, (px, py) in [("button_press_event", p0),
                           ("motion_notify_event", p1),
                           ("button_release_event", p1)]:
        canvas.callbacks.process(name, MouseEvent(name, canvas, px, py, button=1))


# ---------------------------------------------------------------------------
#  profile helpers
# ---------------------------------------------------------------------------


class TestCoerceProfileColor(unittest.TestCase):

    def test_none_and_nan_return_fallback(self):
        self.assertEqual(_coerce_profile_color(None, fallback="red"), "red")
        self.assertEqual(_coerce_profile_color(float("nan"), fallback="red"), "red")

    def test_default_fallback_is_tab10_first(self):
        self.assertEqual(_coerce_profile_color(None), plt.get_cmap("tab10")(0))

    def test_valid_colors_pass_through(self):
        self.assertEqual(_coerce_profile_color("#1f77b4"), "#1f77b4")
        self.assertEqual(_coerce_profile_color("black"), "black")
        self.assertEqual(_coerce_profile_color((1.0, 0.0, 0.0, 1.0)), (1.0, 0.0, 0.0, 1.0))

    def test_serialised_tuple_string_is_parsed(self):
        out = _coerce_profile_color("(0.0, 0.5, 1.0, 1.0)", fallback="red")
        self.assertEqual(out, (0.0, 0.5, 1.0, 1.0))

    def test_unparseable_string_returns_fallback(self):
        self.assertEqual(_coerce_profile_color("not-a-colour", fallback="red"), "red")
        self.assertEqual(_coerce_profile_color("[1, 2", fallback="red"), "red")


class TestProfileTableForKey(unittest.TestCase):

    def test_renames_drops_and_orders(self):
        df = pd.DataFrame({
            "x1": [1.0], "value": [5.0], "bin": [0], "value_smoothed": [5.5],
            "distance_px": [0.5], "profile_id": [1], "key": ["SiO2"],
            "n_pixels": [3], "extra": ["e"], "x0": [0.0],
        })
        out = _profile_table_for_key(df, "SiO2")
        self.assertEqual(list(out.columns),
                         ["profile_id", "distance_px", "SiO2", "SiO2_smoothed",
                          "extra", "x0", "x1"])
        self.assertEqual(out["SiO2_smoothed"].iat[0], 5.5)


class TestResolveProfileValueColumns(unittest.TestCase):

    def test_generic_columns(self):
        df = pd.DataFrame(columns=["distance_px", "value", "value_smoothed"])
        self.assertEqual(_resolve_profile_value_columns(df), ("value", "value_smoothed"))

    def test_key_named_columns(self):
        df = pd.DataFrame(columns=["profile_id", "distance_px", "MgO", "MgO_smoothed", "x0"])
        self.assertEqual(_resolve_profile_value_columns(df), ("MgO", "MgO_smoothed"))

    def test_raw_only_uses_raw_for_smoothed(self):
        df = pd.DataFrame(columns=["distance_px", "CaO"])
        self.assertEqual(_resolve_profile_value_columns(df), ("CaO", "CaO"))

    def test_value_without_smoothed_falls_back_to_other_smoothed(self):
        df = pd.DataFrame(columns=["value", "other_smoothed"])
        self.assertEqual(_resolve_profile_value_columns(df), ("value", "other_smoothed"))

    def test_only_metadata_raises(self):
        df = pd.DataFrame(columns=["profile_id", "distance_px", "x0", "y0"])
        with self.assertRaises(KeyError):
            _resolve_profile_value_columns(df)


class TestLineStripGeometry(unittest.TestCase):

    def test_horizontal_line(self):
        g = _line_strip_geometry((0, 0), (4, 0), 2.0)
        self.assertEqual(g["length_px"], 4.0)
        np.testing.assert_allclose(g["direction"], [1, 0])
        np.testing.assert_allclose(g["normal"], [0, 1])
        self.assertEqual(g["half_width"], 1.0)
        self.assertEqual(g["outline"].shape, (5, 2))
        np.testing.assert_allclose(g["outline"][0], g["outline"][-1])

    def test_nan_width_raises(self):
        with self.assertRaises(ValueError):
            _line_strip_geometry((0, 0), (4, 0), float("nan"))
        with self.assertRaises(ValueError):
            mm.extract_line_profile(np.ones((5, 5)), (0, 2), (4, 2), width_px=np.nan)

    def test_negative_width_is_clamped(self):
        self.assertEqual(_line_strip_geometry((0, 0), (1, 1), -3)["half_width"], 0.0)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            _line_strip_geometry((0, 0, 0), (1, 1), 1)
        with self.assertRaises(ValueError):
            _line_strip_geometry((2, 2), (2, 2), 1)


class TestResolvePhases(unittest.TestCase):

    def setUp(self):
        self.map = np.array([["Olivine", "Plagioclase"],
                             [None, "nan"],
                             ["Glass", "Olivine"]], dtype=object)

    def test_none_means_no_filter(self):
        self.assertIsNone(_resolve_phases(self.map, None))

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(_resolve_phases(self.map, "  oLiViNe "), ["Olivine"])

    def test_list_keeps_candidate_order_and_dedupes(self):
        out = _resolve_phases(self.map, ["glass", "OLIVINE", "Olivine"])
        self.assertEqual(out, ["Olivine", "Glass"])            # map order

    def test_unknown_names_warn_and_are_dropped(self):
        with self.assertWarns(UserWarning) as cm:
            out = _resolve_phases(self.map, ["Olivine", "Quartz"])
        self.assertEqual(out, ["Olivine"])
        self.assertIn("'Quartz' not found", str(cm.warning))
        self.assertNotIn("nan", str(cm.warning))                # empty labels excluded

    def test_explicit_candidates(self):
        out = _resolve_phases(self.map, "glass", candidates=["Glass", "Olivine"])
        self.assertEqual(out, ["Glass"])
        with self.assertWarns(UserWarning):
            self.assertEqual(_resolve_phases(self.map, "Plagioclase",
                                             candidates=["Olivine"]), [])


# ---------------------------------------------------------------------------
#  get_profile_map
# ---------------------------------------------------------------------------


class TestGetProfileMap(unittest.TestCase):

    def setUp(self):
        self.res = _make_res()

    def test_auto_finds_oxide_then_component(self):
        np.testing.assert_array_equal(mm.get_profile_map(self.res, "SiO2"),
                                      self.res["oxide_maps"]["SiO2"])
        out = mm.get_profile_map(self.res, "Olivine.XFo")
        self.assertEqual(out.dtype, float)

    def test_explicit_sources(self):
        mm.get_profile_map(self.res, "MgO", source="oxide")
        mm.get_profile_map(self.res, "Olivine.XFo", source="component")
        with self.assertRaises(KeyError):
            mm.get_profile_map(self.res, "Olivine.XFo", source="oxide")
        with self.assertRaises(KeyError):
            mm.get_profile_map(self.res, "SiO2", source="component")

    def test_plain_oxide_dict(self):
        plain = {"SiO2": np.ones((2, 2))}
        np.testing.assert_array_equal(mm.get_profile_map(plain, "SiO2"), np.ones((2, 2)))

    def test_errors(self):
        with self.assertRaises(TypeError):
            mm.get_profile_map([1, 2], "SiO2")
        with self.assertRaises(ValueError):
            mm.get_profile_map(self.res, "SiO2", source="bogus")
        with self.assertRaises(KeyError):
            mm.get_profile_map(self.res, "FeOt")


# ---------------------------------------------------------------------------
#  extract_line_profile / plot_line_profile
# ---------------------------------------------------------------------------


class TestExtractLineProfile(unittest.TestCase):

    def setUp(self):
        self.sio2 = _make_res()["oxide_maps"]["SiO2"]    # 40 + column index

    def test_method_none_returns_every_pixel(self):
        prof, samples = mm.extract_line_profile(self.sio2, (2, 5), (10, 5), width_px=1.0)
        self.assertIs(prof, samples)
        np.testing.assert_allclose(prof["value"], np.arange(42, 51))
        np.testing.assert_allclose(prof["distance_px"], np.arange(0, 9))
        self.assertTrue(prof["distance_um"].isna().all())
        self.assertEqual(prof.attrs["length_px"], 8.0)
        self.assertEqual(prof.attrs["method"], "none")

    def test_method_none_smoothing_and_pixel_size(self):
        prof, _ = mm.extract_line_profile(self.sio2, (2, 5), (10, 5), pixel_size_um=2.0,
                                          smooth_window=3)
        np.testing.assert_allclose(prof["distance_um"], prof["distance_px"] * 2.0)
        self.assertAlmostEqual(prof["value_smoothed"].iat[0], 42.5)   # edge window of 2
        self.assertAlmostEqual(prof["value_smoothed"].iat[4], prof["value"].iat[4])

    def test_mean_binning(self):
        prof, samples = mm.extract_line_profile(self.sio2, (2, 5), (10, 5), n_bins=4,
                                                method="mean")
        np.testing.assert_allclose(prof["distance_px"], [1, 3, 5, 7])
        np.testing.assert_allclose(prof["value"], [42.5, 44.5, 46.5, 49.0])
        np.testing.assert_array_equal(prof["n_pixels"], [2, 2, 2, 3])
        self.assertEqual(len(samples), 9)
        self.assertEqual(samples.attrs["start"], (2.0, 5.0))

    def test_median_differs_from_mean_with_outlier(self):
        data = self.sio2.copy()
        data[5, 10] = 100.0
        mean, _ = mm.extract_line_profile(data, (2, 5), (10, 5), n_bins=4, method="mean")
        med, _ = mm.extract_line_profile(data, (2, 5), (10, 5), n_bins=4, method="median")
        self.assertAlmostEqual(mean["value"].iat[-1], (48 + 49 + 100) / 3)
        self.assertAlmostEqual(med["value"].iat[-1], 49.0)

    def test_default_bins_and_smoothing(self):
        prof, _ = mm.extract_line_profile(self.sio2, (2, 5), (10, 5), method="mean",
                                          smooth_window=3, pixel_size_um=0.5)
        self.assertEqual(len(prof), 8)                     # ceil(length) bins
        np.testing.assert_allclose(prof["distance_um"], prof["distance_px"] * 0.5)
        self.assertFalse(prof["value_smoothed"].equals(prof["value"]))

    def test_wide_diagonal_strip(self):
        _, samples = mm.extract_line_profile(self.sio2, (1, 1), (10, 9), width_px=3.0,
                                             method="mean")
        self.assertGreater(len(samples), 20)
        self.assertTrue((samples["perp_distance_px"].abs() <= 1.5).all())

    def test_nan_strip_gives_empty_bins(self):
        data = np.full((6, 6), np.nan)
        prof, samples = mm.extract_line_profile(data, (0, 2), (5, 2), n_bins=5,
                                                method="mean", pixel_size_um=2.0)
        self.assertTrue(samples.empty)
        self.assertTrue(prof["value"].isna().all())
        self.assertTrue((prof["n_pixels"] == 0).all())
        np.testing.assert_allclose(prof["distance_um"], prof["distance_px"] * 2.0)

        prof2, _ = mm.extract_line_profile(data, (0, 2), (5, 2), n_bins=5, method="mean")
        self.assertTrue(prof2["distance_um"].isna().all())

    def test_errors(self):
        with self.assertRaises(ValueError):
            mm.extract_line_profile(np.arange(5.0), (0, 0), (1, 0))
        with self.assertRaises(ValueError):
            mm.extract_line_profile(self.sio2, (0, 0), (5, 0), method="max")
        with self.assertRaises(ValueError):
            mm.extract_line_profile(self.sio2, (0, 0), (5, 0), method="mean", n_bins=0)


class TestPlotLineProfile(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_pixel_distances_and_counts(self):
        sio2 = _make_res()["oxide_maps"]["SiO2"]
        prof, _ = mm.extract_line_profile(sio2, (2, 5), (10, 5), n_bins=4, method="mean")
        ax = mm.plot_line_profile(prof, label="SiO2", show_counts=True)
        self.assertEqual(ax.get_xlabel(), "Distance (px)")
        self.assertIsNotNone(ax.get_legend())
        self.assertEqual(len(ax.figure.axes), 2)                      # twin count axis

    def test_micron_distances_and_existing_axis(self):
        sio2 = _make_res()["oxide_maps"]["SiO2"]
        prof, _ = mm.extract_line_profile(sio2, (2, 5), (10, 5), pixel_size_um=1.5)
        fig, ax = plt.subplots()
        out = mm.plot_line_profile(_profile_table_for_key(prof, "SiO2"), ax=ax)
        self.assertIs(out, ax)
        self.assertEqual(ax.get_xlabel(), "Distance (µm)")
        self.assertIsNone(ax.get_legend())


# ---------------------------------------------------------------------------
#  extract_region_stats
# ---------------------------------------------------------------------------


class TestExtractRegionStats(unittest.TestCase):

    def setUp(self):
        self.res = _make_res()
        self.sio2 = self.res["oxide_maps"]["SiO2"]

    def test_box_stats(self):
        df, stats = mm.extract_region_stats(self.sio2, 1.2, 1.2, 4.8, 3.8)
        self.assertEqual(stats["n_pixels"], 20)                       # x 1-5, y 1-4
        self.assertAlmostEqual(stats["mean"], 43.0)
        self.assertEqual(stats["min"], 41.0)
        self.assertEqual(stats["max"], 45.0)
        self.assertEqual(set(df.columns), {"x", "y", "value"})

    def test_reversed_corners_and_phase_column(self):
        df, stats = mm.extract_region_stats(self.sio2, 9, 3, 6, 1,
                                            mineral_map=self.res["mineral_map"])
        self.assertEqual(stats["n_pixels"], 12)
        self.assertEqual(set(df["phase"]), {"Olivine", "Plagioclase"})

    def test_clipped_to_map_and_nan_excluded(self):
        data = self.sio2.copy()
        data[0, 0] = np.nan
        _, stats = mm.extract_region_stats(data, -5, -5, 1, 1)
        self.assertEqual(stats["n_pixels"], 3)

    def test_all_nan_region(self):
        _, stats = mm.extract_region_stats(np.full((4, 4), np.nan), 0, 0, 2, 2)
        self.assertEqual(stats["n_pixels"], 0)
        self.assertTrue(np.isnan(stats["mean"]))

    def test_non_2d_raises(self):
        with self.assertRaises(ValueError):
            mm.extract_region_stats(np.arange(4.0), 0, 0, 1, 1)


# ---------------------------------------------------------------------------
#  batch_extract_line_profiles
# ---------------------------------------------------------------------------


class TestBatchExtractLineProfiles(unittest.TestCase):

    def setUp(self):
        self.res = _make_res()
        self.transects = [
            {"x0": 2, "y0": 5, "x1": 10, "y1": 5, "width_px": 1.0},
            {"x0": 3, "y0": 1, "x1": 3, "y1": 9, "width_px": 1.0, "n_bins": 4,
             "pixel_size_um": 0.5},
        ]

    def test_binned_wide_table(self):
        out = mm.batch_extract_line_profiles(self.res, self.transects)
        for col in ["SiO2", "SiO2_smoothed", "MgO", "MgO_smoothed"]:
            self.assertIn(col, out.columns)
        self.assertEqual(sorted(out["profile_id"].unique()), [1, 2])
        p1 = out[out["profile_id"] == 1]
        self.assertEqual(len(p1), 8)
        self.assertAlmostEqual(p1["SiO2"].iat[0], 42.0)
        self.assertTrue(np.allclose(p1["MgO"], 17.5))
        p2 = out[out["profile_id"] == 2]
        self.assertEqual(len(p2), 4)
        np.testing.assert_allclose(p2["distance_um"], p2["distance_px"] * 0.5)

    def test_single_key_string_and_return_long(self):
        wide, long = mm.batch_extract_line_profiles(self.res, self.transects, keys="SiO2",
                                                    method="median", return_long=True)
        self.assertNotIn("MgO", wide.columns)
        self.assertEqual(set(long["key"]), {"SiO2"})

    def test_method_none_keeps_pixels(self):
        out = mm.batch_extract_line_profiles(self.res, self.transects[:1],
                                             keys=["SiO2", "MgO"], method="none",
                                             pixel_size_um=2.0)
        self.assertEqual(len(out), 9)
        np.testing.assert_allclose(out["SiO2"], np.arange(42, 51))
        np.testing.assert_allclose(out["distance_um"], out["distance_px"] * 2.0)

    def test_plain_oxide_dict_and_component_source(self):
        out = mm.batch_extract_line_profiles(self.res["oxide_maps"], self.transects[:1])
        self.assertIn("SiO2", out.columns)
        comp = mm.batch_extract_line_profiles(self.res, self.transects[:1],
                                              keys="Olivine.XFo", source="component")
        self.assertIn("Olivine.XFo", comp.columns)

    def test_width_column_absent_defaults_to_one_pixel(self):
        tr = [{"x0": 2, "y0": 5, "x1": 10, "y1": 5}]
        out = mm.batch_extract_line_profiles(self.res, tr, keys="SiO2", method="none")
        self.assertEqual(len(out), 9)
        self.assertTrue((out["width_px"] == 1.0).all())

    def test_width_missing_on_some_rows_defaults_to_one_pixel(self):
        # Regression: a partly filled width_px column used to leave NaN widths,
        # which selected no pixels and returned all-NaN profiles.
        tr = [{"x0": 2, "y0": 5, "x1": 10, "y1": 5},
              {"x0": 3, "y0": 1, "x1": 3, "y1": 9, "width_px": 3.0}]
        with self.assertWarns(UserWarning) as cm:
            out = mm.batch_extract_line_profiles(self.res, tr, keys="SiO2", method="none")
        self.assertIn("different strip widths (1, 3 px)", str(cm.warning))
        p1 = out[out["profile_id"] == 1]
        self.assertEqual(len(p1), 9)                                # 1 px strip
        self.assertTrue((p1["width_px"] == 1.0).all())
        self.assertFalse(p1["SiO2"].isna().any())

    def test_width_override_applies_to_every_transect(self):
        tr = [{"x0": 2, "y0": 5, "x1": 10, "y1": 5, "width_px": 1.0},
              {"x0": 3, "y0": 1, "x1": 3, "y1": 9, "width_px": 5.0}]
        with warnings.catch_warnings():
            warnings.simplefilter("error")                          # no mixed-width warning
            out = mm.batch_extract_line_profiles(self.res, tr, keys="SiO2",
                                                 method="none", width_px=3.0)
        self.assertTrue((out["width_px"] == 3.0).all())
        self.assertEqual(len(out[out["profile_id"] == 1]), 27)      # 9 columns x 3 rows

    def test_matching_widths_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mm.batch_extract_line_profiles(self.res, self.transects, keys="SiO2")

    def test_errors(self):
        with self.assertRaises(ValueError):
            mm.batch_extract_line_profiles({"oxide_maps": {"Foo": np.ones((3, 3))}},
                                           self.transects)
        with self.assertRaises(ValueError):
            mm.batch_extract_line_profiles(self.res, self.transects, keys=[])
        with self.assertRaises(KeyError):
            mm.batch_extract_line_profiles(self.res, [{"x0": 0, "y0": 0}])


# ---------------------------------------------------------------------------
#  plot_locations
# ---------------------------------------------------------------------------


class TestPlotLocations(unittest.TestCase):

    def setUp(self):
        self.res = _make_res()

    def tearDown(self):
        plt.close("all")

    def test_transects_on_map(self):
        tr = pd.DataFrame({"x0": [2, 3], "y0": [5, 1], "x1": [10, 3], "y1": [5, 9],
                           "width_px": [3.0, np.nan],
                           "color": ["(1.0, 0.0, 0.0, 1.0)", "not-a-colour"]})
        fig, ax = mm.plot_locations(self.res, tr, map_key="SiO2")
        self.assertEqual(ax.get_title(), "Profile Locations: SiO2")
        self.assertEqual(len(ax.patches), 1)                       # one width strip
        self.assertEqual(len(fig.axes), 2)                         # map + colorbar

    def test_transects_blank_canvas_no_annotation(self):
        tr = [{"x0": 1, "y0": 1, "x1": 8, "y1": 8}]
        fig, ax = mm.plot_locations(self.res, tr, annotate=False, show_width=False,
                                    title="Mine")
        self.assertEqual(ax.get_title(), "Mine")
        self.assertEqual(len(ax.texts), 0)

    def test_pixel_picks(self):
        picks = pd.DataFrame({"x": [1, 5], "y": [2, 6]})
        _, ax = mm.plot_locations(self.res, picks)
        self.assertEqual(ax.get_title(), "Pixel Pick Locations")
        self.assertEqual(len(ax.texts), 2)
        _, ax2 = mm.plot_locations(self.res, picks, map_key="MgO", vmin=10, vmax=20)
        self.assertEqual(ax2.get_title(), "Pixel Pick Locations: MgO")

    def test_regions(self):
        regions = pd.DataFrame({"x0": [1], "y0": [1], "x1": [5], "y1": [4],
                                "height_px": [3.0]})
        _, ax = mm.plot_locations(self.res, regions)
        self.assertEqual(ax.get_title(), "Region Locations")
        self.assertEqual(len(ax.patches), 1)
        _, ax2 = mm.plot_locations(self.res, regions, map_key="SiO2")
        self.assertEqual(ax2.get_title(), "Region Locations: SiO2")

    def test_existing_axis(self):
        fig, ax = plt.subplots()
        fig_out, ax_out = mm.plot_locations(self.res, [{"x0": 0, "y0": 0, "x1": 3, "y1": 3}],
                                            ax=ax)
        self.assertIs(fig_out, fig)
        self.assertIs(ax_out, ax)

    def test_errors(self):
        with self.assertRaises(KeyError):
            mm.plot_locations(self.res, [{"x0": 0, "y0": 0}])
        with self.assertRaises(KeyError):
            mm.plot_locations({"oxide_maps": {}}, [{"x0": 0, "y0": 0, "x1": 1, "y1": 1}])


# ---------------------------------------------------------------------------
#  interactive tools, driven with synthetic mouse and key events
# ---------------------------------------------------------------------------


def _quiet(fn, *args, **kwargs):
    """Call an interactive tool, asserting the non-interactive-backend warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    assert any("interactive Matplotlib backend" in str(w.message) for w in caught)
    return out


class TestInteractiveLineProfile(unittest.TestCase):

    def setUp(self):
        self.res = _make_res()

    def tearDown(self):
        plt.close("all")

    def test_click_pair_builds_profile_and_keys(self):
        ctl = _quiet(mm.interactive_line_profile, self.res, "SiO2", pixel_size_um=1.0)
        ax_map = ctl["fig"].axes[0]
        self.assertIsNone(ctl["get_profile"]())
        self.assertIsNone(ctl["get_samples"]())

        _click(ax_map, 1.6, 5.0)
        _click(ax_map, 10.4, 5.0)
        self.assertEqual(len(ctl["profiles"]), 1)
        self.assertEqual(len(ctl["get_samples"]()), 27)             # 9 columns x 3 rows
        self.assertIn("SiO2", ctl["get_profile"]().columns)
        coords = ctl["get_coordinates"]()
        self.assertEqual(coords[["x0", "y0", "x1", "y1"]].iloc[0].tolist(), [2, 5, 10, 5])

        _click(ax_map, 3.2, 1.6)                                    # second profile
        _click(ax_map, 3.2, 9.4)
        self.assertEqual(len(ctl["profiles"]), 2)
        self.assertEqual(len(ctl["profiles_df"]["profile_id"].unique()), 2)

        _key(ctl["fig"], "u")                                       # undo last
        self.assertEqual(len(ctl["profiles"]), 1)
        _click(ax_map, 4.0, 4.0)                                    # half a pair ...
        _key(ctl["fig"], "r")                                       # ... then reset
        _key(ctl["fig"], "c")                                       # clear all
        self.assertEqual(ctl["profiles"], [])
        self.assertIsNone(ctl["profiles_df"])

        _key(ctl["fig"], "q")                                       # disconnect
        _click(ax_map, 1.6, 5.0)
        _click(ax_map, 10.4, 5.0)
        self.assertEqual(ctl["profiles"], [])

    def test_single_mode_replaces_and_third_click_restarts(self):
        ctl = _quiet(mm.interactive_line_profile, self.res, "MgO", multi=False,
                     width_px=0.0, method="mean", layout="horizontal")
        ax_map = ctl["fig"].axes[0]
        _click(ax_map, 1.6, 2.0)
        _click(ax_map, 10.4, 2.0)
        _click(ax_map, 2.0, 1.6)
        _click(ax_map, 2.0, 9.4)
        self.assertEqual(len(ctl["profiles"]), 1)
        self.assertEqual(ctl["get_coordinates"]()["x0"].iat[0], 2)
        _key(ctl["fig"], "u")                                       # width 0: two artists
        self.assertEqual(ctl["profiles"], [])

    def test_clicks_outside_map_and_phase_mask(self):
        ctl = _quiet(mm.interactive_line_profile, self.res, "SiO2", phase="Olivine",
                     width_px=1.0)
        ax_map, ax_profile = ctl["fig"].axes[0], ctl["fig"].axes[1]
        _click(ax_profile, 0.5, 0.5)                                # ignored
        _click(ax_map, 1.6, 5.0)
        _click(ax_map, 12.4, 5.0)
        vals = ctl["get_samples"]()["value"]
        self.assertTrue((vals < 48).all())                          # plagioclase masked

    def test_phase_is_case_insensitive_and_unknown_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctl = mm.interactive_line_profile(self.res, "SiO2", phase=["OLIVINE", "Quartz"],
                                              width_px=1.0)
        self.assertTrue(any("'Quartz' not found" in str(w.message) for w in caught))
        ax_map = ctl["fig"].axes[0]
        _click(ax_map, 1.6, 5.0)
        _click(ax_map, 12.4, 5.0)
        vals = ctl["get_samples"]()["value"]
        self.assertEqual(len(vals), 6)                              # olivine columns 2-7
        self.assertTrue((vals < 48).all())

    def test_bad_layout_raises(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(ValueError):
                mm.interactive_line_profile(self.res, "SiO2", layout="diagonal")


class TestInteractiveRegion(unittest.TestCase):

    def setUp(self):
        self.res = _make_res()

    def tearDown(self):
        plt.close("all")

    def test_drag_records_region_and_oxides(self):
        ctl = _quiet(mm.interactive_region, self.res, "SiO2", pixel_size_um=2.0)
        ax_map = ctl["fig"].axes[0]
        self.assertIsNone(ctl["get_region"]())
        self.assertIsNone(ctl["get_samples"]())

        _drag(ax_map, 1.2, 1.2, 4.8, 3.8)
        self.assertEqual(len(ctl["regions"]), 1)
        rec = ctl["get_region"]()
        self.assertEqual(rec["n_pixels"], 20)
        self.assertAlmostEqual(rec["mean"], 43.0)
        self.assertIn("mean_MgO", rec)
        self.assertFalse(np.isnan(rec["area_um2"]))
        self.assertIn("MgO", ctl["get_samples"]().columns)

        _drag(ax_map, 9.2, 2.2, 12.8, 6.8)
        self.assertEqual(len(ctl["regions_df"]), 2)
        _key(ctl["fig"], "u")
        self.assertEqual(len(ctl["regions"]), 1)
        _key(ctl["fig"], "r")
        _key(ctl["fig"], "c")
        self.assertIsNone(ctl["regions_df"])
        _key(ctl["fig"], "q")

    def test_single_mode_phase_mask_without_oxides(self):
        ctl = _quiet(mm.interactive_region, self.res, "MgO", phase=["Plagioclase"],
                     multi=False, include_oxides=False)
        ax_map = ctl["fig"].axes[0]
        _drag(ax_map, 1.2, 1.2, 4.8, 3.8)                           # olivine: all masked
        self.assertEqual(ctl["get_region"]()["n_pixels"], 0)
        _drag(ax_map, 9.2, 1.2, 12.8, 3.8)
        self.assertEqual(len(ctl["regions"]), 1)
        self.assertEqual(ctl["get_region"]()["n_pixels"], 20)
        self.assertNotIn("mean_SiO2", ctl["get_region"]())


class TestInteractiveRegionPhaseCase(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_lower_case_phase_matches_exact_name(self):
        counts = []
        for phase in ("Plagioclase", "plagioclase"):
            ctl = _quiet(mm.interactive_region, _make_res(), "SiO2", phase=phase)
            _drag(ctl["fig"].axes[0], 6.2, 1.2, 10.8, 3.8)            # straddles the contact
            counts.append(ctl["get_region"]()["n_pixels"])
        self.assertEqual(counts, [16, 16])                          # plag x 8-11, y 1-4


class TestInteractivePixels(unittest.TestCase):

    def setUp(self):
        self.res = _make_res()

    def tearDown(self):
        plt.close("all")

    def test_clicks_average_same_phase_box(self):
        ctl = _quiet(mm.interactive_pixels, self.res, region=3,
                     phase_colors={"Olivine": "green", "Quartz": "grey"})
        ax_map, fig = ctl["fig"].axes[0], ctl["fig"]
        _click(ax_map, 3.2, 4.1)                                    # olivine interior
        picks = ctl["picks"]
        self.assertEqual(picks["n_pixels"].iat[0], 9)
        self.assertAlmostEqual(picks["SiO2"].iat[0], 43.0)
        self.assertEqual(picks["phase"].iat[0], "Olivine")

        _click(ax_map, 7.9, 6.0)                                    # plag edge: 2 of 3 columns
        self.assertEqual(ctl["picks"]["n_pixels"].iat[1], 6)

        _key(fig, "u")
        self.assertEqual(len(ctl["picks"]), 1)
        _key(fig, "c")
        self.assertTrue(ctl["picks"].empty)
        _key(fig, "q")
        _click(ax_map, 3.2, 4.1)
        self.assertTrue(ctl["picks"].empty)

    def test_single_pixel_phase_filter_and_heatmap(self):
        res = _make_res()
        res["oxide_maps"]["Total_raw"] = res["oxide_maps"]["Total"] * 0.99
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctl = mm.interactive_pixels(res, region=1, phase=["Olivine", "Quartz"],
                                        oxide_key="SiO2")
        self.assertTrue(any("'Quartz' not found" in str(w.message) for w in caught))
        ax_map = ctl["fig"].axes[0]
        _click(ax_map, 12.0, 4.0)                                   # plagioclase: ignored
        self.assertTrue(ctl["picks"].empty)
        _click(ax_map, 2.0, 3.0)
        self.assertEqual(ctl["picks"]["n_pixels"].iat[0], 1)
        self.assertIn("Total_raw", ctl["picks"].columns)
        _click(ctl["fig"].axes[1], 0.5, 0.5)                        # legend axis: ignored
        self.assertEqual(len(ctl["picks"]), 1)

    def test_phase_filter_is_case_insensitive_for_clicks(self):
        # Regression: clicks used to be filtered on the raw `phase` argument,
        # so phase="olivine" silently ignored every olivine click.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctl = mm.interactive_pixels(self.res, region=1, phase="olivine")
        _click(ctl["fig"].axes[0], 2.0, 3.0)
        self.assertEqual(len(ctl["picks"]), 1)

    def test_even_region_raises(self):
        with self.assertRaises(ValueError):
            mm.interactive_pixels(self.res, region=4)


if __name__ == "__main__":
    unittest.main()