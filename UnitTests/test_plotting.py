import unittest
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import mineralML as mm
from mineralML import plotting as mmp


VOLCANOES = ["Mt. Hood", "Mt. Rainier", "Mt. St. Helens"]


def _rows(base, n=6, jitter=None, **extra):
    """n analyses around `base`, with a Volcano column and any extra columns."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame([base] * n)
    for col, sd in (jitter or {}).items():
        df[col] = df[col] + rng.normal(0, sd, n)
    df["Volcano"] = [VOLCANOES[i % len(VOLCANOES)] for i in range(n)]
    for k, v in extra.items():
        df[k] = v
    return df


GLASS = _rows({"SiO2": 50.0, "TiO2": 1.5, "Al2O3": 15.0, "FeOt": 10.0, "MnO": 0.2, "MgO": 7.0,
               "CaO": 11.0, "Na2O": 2.5, "K2O": 0.5}, jitter={"SiO2": 4, "TiO2": 0.4, "Na2O": 0.5})
PLAG = _rows({"SiO2": 52.8, "TiO2": 0.05, "Al2O3": 29.4, "FeOt": 0.6, "MgO": 0.1, "CaO": 12.3,
              "Na2O": 4.4, "K2O": 0.2}, jitter={"CaO": 1.0, "Na2O": 0.5})
AMPH = _rows({"SiO2": 43.0, "TiO2": 2.0, "Al2O3": 12.0, "FeOt": 11.0, "MnO": 0.2, "MgO": 15.0,
              "CaO": 11.5, "Na2O": 2.5, "K2O": 0.5}, jitter={"SiO2": 0.5, "MgO": 0.5})
CPX = _rows({"SiO2": 50.9, "TiO2": 0.9, "Al2O3": 3.8, "FeOt": 7.6, "MnO": 0.2, "MgO": 15.1,
             "CaO": 20.6, "Na2O": 0.4, "Cr2O3": 0.3}, jitter={"MgO": 0.5, "CaO": 0.5})
OPX = _rows({"SiO2": 54.0, "TiO2": 0.2, "Al2O3": 1.5, "FeOt": 15.0, "MnO": 0.3, "MgO": 27.5,
             "CaO": 1.5, "Na2O": 0.02}, n=3, jitter={"MgO": 0.5})
OXIDES = pd.concat([
    _rows({"SiO2": 0.1, "TiO2": 12.5, "Al2O3": 2.9, "FeOt": 78.4, "MnO": 0.5, "MgO": 2.1},
          n=4, jitter={"TiO2": 1.0}, Mineral="Spinel"),  # plot_spinel picks spinels by name
    _rows({"SiO2": 0.05, "TiO2": 48.0, "Al2O3": 0.3, "FeOt": 46.0, "MnO": 0.6, "MgO": 3.0},
          n=4, jitter={"TiO2": 1.0}, Mineral="Ilmenite"),
], ignore_index=True)


def _legend_labels(ax):
    """Legend text on ax, or on any colorbar axis of its figure."""
    for a in [ax] + list(ax.get_figure().axes):
        leg = a.get_legend()
        if leg is not None:
            return [t.get_text() for t in leg.get_texts()]
    return []


class TestHelpers(unittest.TestCase):
    def test_category_slots_fold_beyond_limit(self):
        s = pd.Series(list("aaabbc") + [f"x{i}" for i in range(10)])
        slots = mmp.category_slots(s, limit=8)
        self.assertEqual(len(slots), 7)  # 7 keep slots, the rest fold into Other
        self.assertEqual(slots[:2], ("a", "b"))
        self.assertTrue(mmp.folds(s, 8))

    def test_category_slots_skip_unclassified(self):
        s = pd.Series(["Augite", "Unclassified", "Augite", "Diopside"])
        self.assertEqual(mmp.category_slots(s), ("Augite", "Diopside"))

    def test_is_continuous(self):
        self.assertTrue(mmp.is_continuous(pd.Series([0.02, 0.05, 0.9, 12.5])))  # few, but decimals
        self.assertFalse(mmp.is_continuous(pd.Series([1, 2, 3, 1])))  # group numbers
        self.assertTrue(mmp.is_continuous(pd.Series(range(20))))
        self.assertFalse(mmp.is_continuous(pd.Series(["a", "b"])))

    def test_symbol_names_and_markers(self):
        self.assertEqual(mmp.symbol_spec("○ Open circle"), ("o", False))
        self.assertEqual(mmp.symbol_spec("open circle"), ("o", False))
        self.assertEqual(mmp.symbol_spec("Star"), ("*", True))
        self.assertEqual(mmp.symbol_spec("^"), ("^", True))
        with self.assertRaises(ValueError):
            mmp.symbol_spec("not a symbol")

    def test_default_symbols_are_distinct(self):
        self.assertEqual(len(set(mmp.SYMBOLS.values())), mmp.MAX_CATEGORIES)


class TestScatterPoints(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.x = np.arange(24.0)

    def tearDown(self):
        plt.close(self.fig)

    def test_same_mode_gives_unique_color_symbol_pairs(self):
        data = pd.DataFrame({"Volcano": [f"V{i:02d}" for i in range(24)]})
        out = mmp.scatter_points(self.ax, self.x, self.x, data, color="Volcano")
        pairs = {(to_hex(h.get_color()), h.get_marker(), h.get_markerfacecolor() == "none")
                 for h in out["colors"]}
        self.assertEqual(len(out["colors"]), 24)
        self.assertEqual(len(pairs), 24)

    def test_separate_symbol_column(self):
        data = pd.DataFrame({"Volcano": ["A", "B"] * 12, "Mineral": ["Cpx", "Opx", "Ol"] * 8})
        out = mmp.scatter_points(self.ax, self.x, self.x, data, color="Volcano", symbol="Mineral")
        self.assertEqual([h.get_label() for h in out["colors"]], ["A", "B"])
        self.assertEqual(sorted(h.get_label() for h in out["symbols"]), ["Cpx", "Ol", "Opx"])
        self.assertEqual(len({h.get_marker() for h in out["symbols"]}), 3)

    def test_separate_symbols_cap_colors_at_palette(self):
        data = pd.DataFrame({"Volcano": [f"V{i:02d}" for i in range(12)] * 2, "M": ["a"] * 24})
        out = mmp.scatter_points(self.ax, self.x, self.x, data, color="Volcano", symbol="M")
        labels = [h.get_label() for h in out["colors"]]
        self.assertEqual(len(labels), len(mmp.PALETTE))  # 7 colors + Other
        self.assertEqual(labels[-1], "Other")

    def test_numeric_color_gives_colorbar(self):
        data = pd.DataFrame({"TiO2": np.linspace(0.5, 3.0, 24)})
        out = mmp.scatter_points(self.ax, self.x, self.x, data, color="TiO2")
        self.assertIsNotNone(out["mappable"])
        self.assertEqual(out["colors"], [])

    def test_user_colors_and_symbols(self):
        data = pd.DataFrame({"Volcano": ["Hood", "Rainier"] * 12})
        out = mmp.scatter_points(self.ax, self.x, self.x, data, color="Volcano",
                                 colors={"Hood": "#000000"}, symbols={"Hood": "star"})
        hood = next(h for h in out["colors"] if h.get_label() == "Hood")
        self.assertEqual(to_hex(hood.get_color()), "#000000")
        self.assertEqual(hood.get_marker(), "*")

    def test_numeric_category_keys(self):
        data = pd.DataFrame({"Group": [1, 2, 3] * 8})
        out = mmp.scatter_points(self.ax, self.x, self.x, data, color="Group", colors={1: "#000000"})
        one = next(h for h in out["colors"] if h.get_label() == "1")
        self.assertEqual(to_hex(one.get_color()), "#000000")

    def test_missing_column_raises(self):
        with self.assertRaises(KeyError):
            mmp.scatter_points(self.ax, self.x, self.x, pd.DataFrame({"a": range(24)}), color="Volcano")

    def test_color_in_scatter_kw_raises(self):
        with self.assertRaises(TypeError):
            mmp.scatter_points(self.ax, self.x, self.x, pd.DataFrame(index=range(24)), scatter_kw={"c": "k"})


class TestClassifierPlots(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def tearDown(self):
        plt.close("all")

    def test_tas_color_by_oxide_and_symbol_by_user_column(self):
        fig, ax = mm.GlassClassifier(GLASS).plot(color="TiO2", symbol="Volcano")
        self.assertEqual(len(fig.axes), 2)  # plot + colorbar
        self.assertEqual(sorted(l for l in _legend_labels(ax)), sorted(VOLCANOES))

    def test_tas_default_still_colors_by_rock_type(self):
        fig, ax = mm.GlassClassifier(GLASS).plot()
        self.assertEqual(ax.get_legend().get_title().get_text(), "Rock Type")

    def test_feldspar_kwargs_reach_points(self):
        # **kwargs used to be ignored; now they style the points
        fig, tax = mm.FeldsparClassifier(PLAG).plot(s=80, marker="^")
        ax = tax.get_axes()
        sizes = [c.get_sizes()[0] for c in ax.collections if len(c.get_offsets())]
        self.assertTrue(sizes and all(s > 80 for s in sizes))  # triangles are scaled up to match circles

    def test_tas_edgecolors_still_pass_through(self):
        fig, ax = mm.GlassClassifier(GLASS).plot(edgecolors="k")
        edges = [c.get_edgecolor() for c in ax.collections if len(c.get_offsets())]
        self.assertTrue(edges and all(to_hex(e[0]) == "#000000" for e in edges))

    def test_size_wins_over_s(self):
        fig, ax = mm.GlassClassifier(GLASS).plot(size=90, s=10)
        sizes = [c.get_sizes()[0] for c in ax.collections if len(c.get_offsets())]
        self.assertTrue(sizes and all(s >= 90 for s in sizes))

    def test_feldspar_color_by_user_column(self):
        fig, tax = mm.FeldsparClassifier(PLAG).plot(color="Volcano")
        self.assertEqual(sorted(_legend_labels(tax.get_axes())), sorted(VOLCANOES))

    def test_amphibole_hue_alias(self):
        fig, ax = mm.AmphiboleClassifier(AMPH).plot(hue="Volcano")
        self.assertEqual(sorted(_legend_labels(ax)), sorted(VOLCANOES))

    def test_pyroxene_subclass_false_colors_by_mineral(self):
        fig, tax = mm.PyroxeneClassifier(pd.concat([CPX, OPX], ignore_index=True)).plot(subclass=False)
        self.assertEqual(sorted(_legend_labels(tax.get_axes())), ["Clinopyroxene", "Orthopyroxene"])

    def test_oxide_ternary_colors_by_suboxide(self):
        # Used to read a "Submineral" column that classify() never makes, drawing every point as Unclassified
        figs = mm.OxideClassifier(OXIDES).plot()
        labels = _legend_labels(figs["ternary"][1].get_axes())
        self.assertIn("Ilmenite", labels)
        self.assertNotEqual(labels, ["Unclassified"])

    def test_spinel_one_color_without_legend(self):
        fig, ax = mm.OxideClassifier(OXIDES).plot_spinel(color=False)
        self.assertIsNone(ax.get_legend())

    def test_legend_false(self):
        fig, ax = mm.GlassClassifier(GLASS).plot(legend=False)
        self.assertIsNone(ax.get_legend())


if __name__ == "__main__":
    unittest.main()
