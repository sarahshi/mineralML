import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
import os
import numpy as np
import pandas as pd
import matplotlib

# Headless backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mineralML as mm
from mineralML.constants import (
    CATION_TO_OXIDE_MAP,
    OXYGEN_NUMBERS,
    CATION_NUMBERS,
    OXIDE_MASSES,
    VALENCES,
)

def _mk_ssg(endmembers, oxygen_basis=4, **kwargs):
    # Let the class use mineralML.constants
    return mm.SolidSolutionGenerator(
        endmembers=endmembers,
        oxygen_basis=oxygen_basis,
        validate_fn=lambda _: True,
        **kwargs,
    )

AVAILABLE = set(CATION_TO_OXIDE_MAP.keys())

def pick_alt(*candidates):
    """Return the first candidate cation that exists in the constants map."""
    for el in candidates:
        if el in AVAILABLE:
            return el
    raise unittest.SkipTest("No suitable cation present in CATION_TO_OXIDE_MAP")

def pick_two_distinct(*candidates):
    """Return two distinct available cations from a preference list."""
    chosen = []
    for el in candidates:
        if el in AVAILABLE and el not in chosen:
            chosen.append(el)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        raise unittest.SkipTest("Not enough distinct cations available in constants")
    return tuple(chosen)


class TestSolidSolutionGenerator(unittest.TestCase):
    def setUp(self):
        np.random.seed(12345)

    def test_constants_are_from_mm(self):
        ssg = _mk_ssg({"Em": {"Mg": 2, "Si": 1, "O": 4}})
        self.assertIs(ssg.CATION_TO_OXIDE_MAP, CATION_TO_OXIDE_MAP)
        self.assertIs(ssg.OXYGEN_NUMBERS, OXYGEN_NUMBERS)
        self.assertIs(ssg.CATION_NUMBERS, CATION_NUMBERS)
        self.assertIs(ssg.OXIDE_MASSES, OXIDE_MASSES)
        self.assertIs(ssg.VALENCES, VALENCES)

    # --- __init__ defaults ---

    def test_init_defaults(self):
        ssg = mm.SolidSolutionGenerator(
            endmembers={"Em": {"Mg": 2, "Si": 1, "O": 4}},
            oxygen_basis=4,
        )
        # minor_elements defaults to empty dict
        self.assertEqual(ssg.minor_elements, {})
        # validate_fn defaults to always-true callable
        self.assertTrue(ssg.validate_fn({}))
        self.assertTrue(ssg.validate_fn({"MgO": 50}))

    def test_suffix_reflects_oxygen_basis(self):
        ssg6 = _mk_ssg({"Em": {"Mg": 2, "Si": 1, "O": 6}}, oxygen_basis=6)
        self.assertEqual(ssg6.suffix, "_cat_6ox")

    # --- _validate_endmember_oxygens ---

    def test_validate_endmember_oxygens_pass_and_fail(self):
        ok = {"Em1": {"Mg": 2, "Si": 1, "O": 4}}
        _ = mm.SolidSolutionGenerator(ok, oxygen_basis=4)  # should not raise

        bad = {"Em1": {"Mg": 2, "Si": 1, "O": 5}}
        with self.assertRaises(ValueError):
            mm.SolidSolutionGenerator(bad, oxygen_basis=4)

    def test_validate_endmember_oxygens_missing_o_key(self):
        # When O key is absent, defaults to oxygen_basis — should not raise
        no_o = {"Em1": {"Mg": 2, "Si": 1}}
        ssg = mm.SolidSolutionGenerator(no_o, oxygen_basis=4)
        self.assertIsNotNone(ssg)

    # --- _generate_mixing_fraction ---

    def test_generate_mixing_fraction_variants(self):
        B = pick_alt("Fe", "Ca", "Na", "K", "Mg")
        # beta
        ssg = _mk_ssg({"A": {"Mg": 2, "Si": 1, "O": 4}, "B": {B: 2, "Si": 1, "O": 4}},
                      mixing_dist="beta", mixing_params={"a": 2, "b": 2})
        x = ssg._generate_mixing_fraction()
        self.assertTrue(0.0 <= x <= 1.0)

        # uniform
        ssg.mixing_dist = "uniform"
        x = ssg._generate_mixing_fraction()
        self.assertTrue(0.0 <= x <= 1.0)

        # dirichlet (vector whose entries sum to 1)
        B, C = pick_two_distinct("Fe", "Ca", "Na", "K", "Mg")
        ssg3 = _mk_ssg(
            {"A": {"Mg": 2, "Si": 1, "O": 4},
             "B": {B: 2, "Si": 1, "O": 4},
             "C": {C: 2, "Si": 1, "O": 4}},
            mixing_dist="dirichlet", mixing_params={"alpha": [1, 1, 1]}
        )
        v = ssg3._generate_mixing_fraction()
        if isinstance(v, np.ndarray):
            self.assertEqual(v.shape, (3,))
            self.assertTrue(np.all((v >= 0) & (v <= 1)))
            self.assertAlmostEqual(float(v.sum()), 1.0, places=6)
        else:
            self.assertTrue(0.0 <= v <= 1.0)

        # bad name
        ssg_bad = _mk_ssg({"A": {"Mg": 2, "O": 4}, "B": {B: 2, "O": 4}})
        ssg_bad.mixing_dist = "nope"
        with self.assertRaises(ValueError):
            ssg_bad._generate_mixing_fraction()

    def test_generate_mixing_fraction_beta_default_params(self):
        # When mixing_params lacks a/b, defaults to a=2, b=2
        B = pick_alt("Fe", "Ca", "Na", "K", "Mg")
        ssg = _mk_ssg({"A": {"Mg": 2, "Si": 1, "O": 4}, "B": {B: 2, "Si": 1, "O": 4}},
                      mixing_dist="beta", mixing_params={})
        x = ssg._generate_mixing_fraction()
        self.assertTrue(0.0 <= x <= 1.0)

    # --- _add_minor_elements ---

    def test_add_minor_elements_exponential(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        ssg.minor_elements = {"Na": {"distribution": "exponential", "scale": 0.01, "max_fraction": 0.1}}
        base = {"Mg": 2.0, "Si": 1.0}
        total_before = sum(base.values())
        out = ssg._add_minor_elements(base.copy())

        self.assertIn("Na", out)
        self.assertGreaterEqual(out["Na"], 0.0)
        self.assertLess(out["Mg"] + out["Si"], total_before)
        self.assertAlmostEqual(out["Mg"] + out["Si"] + out["Na"], total_before, places=6)

    def test_add_minor_elements_empty_is_noop(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        ssg.minor_elements = {}
        base = {"Mg": 2.0, "Si": 1.0}
        out = ssg._add_minor_elements(base.copy())
        self.assertEqual(out, base)

    def test_add_minor_elements_max_fraction_caps(self):
        # Force a large exponential draw; max_fraction should cap the amount
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        ssg.minor_elements = {"Na": {"distribution": "exponential", "scale": 100.0, "max_fraction": 0.01}}
        base = {"Mg": 2.0, "Si": 1.0}
        total_before = sum(base.values())

        # Run many times; Na should never exceed max_fraction * total
        for _ in range(50):
            out = ssg._add_minor_elements(base.copy())
            self.assertLessEqual(out["Na"], 0.01 * total_before + 1e-12)

    # --- _apply_site_variation ---

    def test_apply_site_variation_scalar_and_dict(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}},
                      site_variation=0.2, element_noise_scale=0.1, min_site_fraction=0.2)
        varied = ssg._apply_site_variation({"M": 3.0})
        self.assertIn("M", varied)
        self.assertGreaterEqual(varied["M"], 0.2 * 3.0)

        ssg.site_variation = {"M": 0.05}
        varied2 = ssg._apply_site_variation({"M": 3.0})
        self.assertGreaterEqual(varied2["M"], 0.2 * 3.0)

    def test_apply_site_variation_min_site_fraction_clamp(self):
        # With very large variation, min_site_fraction should prevent near-zero values
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}},
                      site_variation=5.0, min_site_fraction=0.5)

        # Over many draws, the floor should hold
        for _ in range(100):
            varied = ssg._apply_site_variation({"bulk": 3.0})
            self.assertGreaterEqual(varied["bulk"], 0.5 * 3.0)

    # --- _add_element_noise ---

    def test_add_element_noise_preserves_oxygen_basis_and_nonneg(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}}, element_noise_scale=0.05)
        c = {"Mg": 2.0, "Si": 1.0}
        out = ssg._add_element_noise(c)
        self.assertTrue(all(v >= 0 for v in out.values()))
        o_sum = sum(out[e] * ssg.OXYGEN_NUMBERS[ssg.CATION_TO_OXIDE_MAP[e]] for e in out)
        self.assertAlmostEqual(o_sum, ssg.oxygen_basis, delta=ssg.oxygen_basis * 0.05)

    def test_add_element_noise_zero_scale_preserves_values(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}}, element_noise_scale=0.0)
        c = {"Mg": 2.0, "Si": 1.0}
        out = ssg._add_element_noise(c)
        # With zero noise scale, values should remain very close to originals
        self.assertAlmostEqual(out["Mg"], 2.0, places=4)
        self.assertAlmostEqual(out["Si"], 1.0, places=4)

    # --- _get_element_valence ---

    def test_get_element_valence_known_and_unknown(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        # Known elements should match VALENCES
        self.assertEqual(ssg._get_element_valence("Si"), VALENCES["Si"])
        self.assertEqual(ssg._get_element_valence("Mg"), VALENCES["Mg"])
        # Unknown element defaults to 2
        self.assertEqual(ssg._get_element_valence("Xx"), 2)

    # --- _total_charge ---

    def test_total_charge_strips_suffix(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        cations = {f"Mg{ssg.suffix}": 2.0, "Si": 1.0}
        self.assertAlmostEqual(ssg._total_charge(cations), 8.0, places=6)

    def test_total_charge_bare_keys(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        cations = {"Mg": 2.0, "Si": 1.0}
        # Mg(2+)*2 + Si(4+)*1 = 8
        self.assertAlmostEqual(ssg._total_charge(cations), 8.0, places=6)

    # --- _check_charge_balance_add_noise ---

    def test_check_charge_balance_calls_noise(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        cations = {"Mg": 2.0, "Si": 1.0}
        with patch.object(ssg, "_add_element_noise", wraps=ssg._add_element_noise) as w:
            _ = ssg._check_charge_balance_add_noise(cations)
            w.assert_called_once()

    # --- _calculate_oxide_wt_percent ---

    def test_calculate_oxide_wt_percent_normalizes_to_100_and_raises_on_unknown(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        wt = ssg._calculate_oxide_wt_percent({"Mg": 2.0, "Si": 1.0})
        self.assertTrue(set(wt.keys()) >= {"MgO", "SiO2"})
        self.assertAlmostEqual(sum(wt.values()), 100.0, places=6)
        with self.assertRaises(ValueError):
            ssg._calculate_oxide_wt_percent({"Xx": 1.0})

    def test_calculate_oxide_wt_percent_skips_oxygen_key(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        wt = ssg._calculate_oxide_wt_percent({"Mg": 2.0, "Si": 1.0, "O": 4.0})
        # O key should be skipped; result same as without it
        self.assertNotIn("O", wt)
        self.assertAlmostEqual(sum(wt.values()), 100.0, places=6)

    # --- generate ---

    @patch.object(plt, "show")
    def test_generate_binary_single_dirichlet(self, _show):
        # Binary path (use available cation for B)
        B = pick_alt("Fe", "Ca", "Na", "K", "Mg")
        ssg_bin = _mk_ssg({"A": {"Mg": 2.0, "Si": 1.0, "O": 4},
                           "B": {B: 2.0, "Si": 1.0, "O": 4}},
                          mixing_dist="beta", mixing_params={"a": 2, "b": 2})
        df_bin = ssg_bin.generate(n_samples=20)
        self.assertFalse(df_bin.empty)
        self.assertTrue(any(c.endswith(ssg_bin.suffix) for c in df_bin.columns))
        self.assertIn("MgO", df_bin.columns)

        # Single endmember path
        ssg_one = _mk_ssg({"Only": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        df_one = ssg_one.generate(n_samples=10)
        self.assertFalse(df_one.empty)

        # >2 endmembers path (dirichlet) — ensure vector mixing fractions
        B, C = pick_two_distinct("Fe", "Ca", "Na", "K", "Mg")
        ssg_tri = _mk_ssg({"A": {"Mg": 2.0, "Si": 1.0, "O": 4},
                           "B": {B: 2.0, "Si": 1.0, "O": 4},
                           "C": {C: 2.0, "Si": 1.0, "O": 4}},
                          mixing_dist="dirichlet", mixing_params={"alpha": [1, 1, 1]})
        with patch.object(ssg_tri, "_generate_mixing_fraction",
                          return_value=np.array([0.2, 0.5, 0.3])):
            df_tri = ssg_tri.generate(n_samples=15)
        self.assertFalse(df_tri.empty)

    def test_generate_validation_rejects_all_returns_empty(self):
        # validate_fn always returns False -> no rows kept
        ssg = mm.SolidSolutionGenerator(
            endmembers={"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}},
            oxygen_basis=4,
            validate_fn=lambda _: False,
        )
        df = ssg.generate(n_samples=10)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_generate_with_minor_elements(self):
        B = pick_alt("Na", "Ca", "K")
        ssg = _mk_ssg(
            {"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}},
            minor_elements={B: {"distribution": "exponential", "scale": 0.01, "max_fraction": 0.05}},
        )
        df = ssg.generate(n_samples=20)
        self.assertFalse(df.empty)
        # The minor element's oxide should appear in some rows
        minor_oxide = CATION_TO_OXIDE_MAP[B]
        self.assertIn(minor_oxide, df.columns)

    # --- compare_distributions ---

    @patch.object(plt, "show")
    def test_compare_distributions_returns_stats_and_handles_twin_axis(self, _show):
        ssg = _mk_ssg({"Only": {"Mg": 2.0, "Si": 1.0, "O": 4}}, oxygen_basis=4)
        suffix = ssg.suffix
        base = pd.DataFrame({
            f"Mg{suffix}": np.random.lognormal(mean=0, sigma=0.1, size=50) + 1.5,
            f"Si{suffix}": np.random.lognormal(mean=0, sigma=0.1, size=50) + 0.5,
            "MgO": np.random.normal(loc=40, scale=2, size=50),
            "SiO2": np.random.normal(loc=60, scale=2, size=50),
        })
        synth = pd.DataFrame({
            f"Mg{suffix}": np.random.lognormal(mean=0, sigma=0.1, size=60) + 1.6,
            f"Si{suffix}": np.random.lognormal(mean=0, sigma=0.1, size=60) + 0.6,
            "MgO": np.random.normal(loc=41, scale=2, size=60),
            "SiO2": np.random.normal(loc=59, scale=2, size=60),
        })
        fig, stats = ssg.compare_distributions(base_df=base, synth_df=synth, ncols=2, figsize_per=(3, 2))
        self.assertIsInstance(fig, plt.Figure) 
        self.assertIsInstance(stats, pd.DataFrame)
        plt.close(fig)
        self.assertTrue({"ks_stat", "p_value", "mean_base", "mean_synth", "std_base", "std_synth"}.issubset(stats.columns))
        self.assertTrue(all(idx.endswith(suffix) for idx in stats.index))

    @patch.object(plt, "show")
    def test_compare_distributions_no_matching_columns(self, _show):
        ssg = _mk_ssg({"Only": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        base = pd.DataFrame({"foo": [1, 2, 3]})
        out = ssg.compare_distributions(base_df=base, synth_df=pd.DataFrame({"bar": [1, 2]}))
        self.assertTrue(isinstance(out, pd.DataFrame))
        self.assertTrue(out.empty)

    @patch.object(plt, "show")
    def test_compare_distributions_auto_generates_synth(self, _show):
        # When synth_df=None, compare_distributions should call generate internally
        ssg = _mk_ssg({"Only": {"Mg": 2.0, "Si": 1.0, "O": 4}}, oxygen_basis=4)
        suffix = ssg.suffix
        base = pd.DataFrame({
            f"Mg{suffix}": np.random.normal(loc=2.0, scale=0.1, size=30),
            f"Si{suffix}": np.random.normal(loc=1.0, scale=0.1, size=30),
            "MgO": np.random.normal(loc=40, scale=2, size=30),
            "SiO2": np.random.normal(loc=60, scale=2, size=30),
        })
        fig, stats = ssg.compare_distributions(base_df=base, synth_df=None, n_samples=20)
        self.assertIsInstance(fig, plt.Figure)
        self.assertFalse(stats.empty)
        plt.close(fig)

    @patch.object(plt, "show")
    def test_compare_distributions_savefig_and_suptitle(self, _show):
        ssg = _mk_ssg({"Only": {"Mg": 2.0, "Si": 1.0, "O": 4}}, oxygen_basis=4)
        suffix = ssg.suffix
        base = pd.DataFrame({
            f"Mg{suffix}": np.random.normal(loc=2.0, scale=0.1, size=20),
            f"Si{suffix}": np.random.normal(loc=1.0, scale=0.1, size=20),
        })
        synth = pd.DataFrame({
            f"Mg{suffix}": np.random.normal(loc=2.0, scale=0.1, size=20),
            f"Si{suffix}": np.random.normal(loc=1.0, scale=0.1, size=20),
        })
        with TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_fig")
            fig, stats = ssg.compare_distributions(
                base_df=base, synth_df=synth, suptitle="Test Title", savefig_path=save_path,
            )
            self.assertIsInstance(fig, plt.Figure)

            # suptitle should be set on the figure
            self.assertEqual(fig._suptitle.get_text(), "Test Title")

            # File should have been saved
            self.assertTrue(os.path.exists(save_path))
            plt.close(fig)

    @patch.object(plt, "show")
    def test_compare_distributions_cation_only_no_oxide_twin(self, _show):
        # When oxide columns are absent, twin axis should not be created
        ssg = _mk_ssg({"Only": {"Mg": 2.0, "Si": 1.0, "O": 4}}, oxygen_basis=4)
        suffix = ssg.suffix
        # Only cation columns, no MgO/SiO2
        base = pd.DataFrame({
            f"Mg{suffix}": np.random.normal(loc=2.0, scale=0.1, size=20),
        })
        synth = pd.DataFrame({
            f"Mg{suffix}": np.random.normal(loc=2.0, scale=0.1, size=20),
        })
        fig, stats = ssg.compare_distributions(base_df=base, synth_df=synth)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(stats), 1)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()