import unittest
from unittest.mock import patch
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
    # No overrides here—let the class use mineralML.constants
    return mm.SolidSolutionGenerator(
        endmembers=endmembers,
        oxygen_basis=oxygen_basis,
        validate_fn=lambda _: True,
        **kwargs,
    )

class TestSolidSolutionGenerator(unittest.TestCase):
    def setUp(self):
        np.random.seed(12345)

    def test_constants_are_from_mm(self):
        ssg = _mk_ssg({"Em": {"Mg": 2, "Si": 1, "O": 4}})
        # sanity: the instance got the module constants
        self.assertIs(ssg.CATION_TO_OXIDE_MAP, CATION_TO_OXIDE_MAP)
        self.assertIs(ssg.OXYGEN_NUMBERS, OXYGEN_NUMBERS)
        self.assertIs(ssg.CATION_NUMBERS, CATION_NUMBERS)
        self.assertIs(ssg.OXIDE_MASSES, OXIDE_MASSES)
        self.assertIs(ssg.VALENCES, VALENCES)

    def test_validate_endmember_oxygens_pass_and_fail(self):
        ok = {"Em1": {"Mg": 2, "Si": 1, "O": 4}}
        _ = mm.SolidSolutionGenerator(ok, oxygen_basis=4)  # should not raise

        bad = {"Em1": {"Mg": 2, "Si": 1, "O": 5}}
        with self.assertRaises(ValueError):
            mm.SolidSolutionGenerator(bad, oxygen_basis=4)

    def test_generate_mixing_fraction_variants(self):
        # beta
        ssg = _mk_ssg({"A": {"Mg": 2, "Si": 1, "O": 4}, "B": {"Fe": 2, "Si": 1, "O": 4}},
                    mixing_dist="beta", mixing_params={"a": 2, "b": 2})
        x = ssg._generate_mixing_fraction()
        self.assertTrue(0.0 <= x <= 1.0)

        # uniform
        ssg.mixing_dist = "uniform"
        x = ssg._generate_mixing_fraction()
        self.assertTrue(0.0 <= x <= 1.0)

        # dirichlet (expects a vector whose entries sum to 1)
        ssg3 = _mk_ssg(
            {"A": {"Mg": 2, "Si": 1, "O": 4},
            "B": {"Fe": 2, "Si": 1, "O": 4},
            "C": {"Ca": 2, "Si": 1, "O": 4}},
            mixing_dist="dirichlet", mixing_params={"alpha": [1, 1, 1]}
        )
        v = ssg3._generate_mixing_fraction()
        if isinstance(v, np.ndarray):
            self.assertEqual(v.shape, (3,))
            self.assertTrue(np.all((v >= 0) & (v <= 1)))
            self.assertAlmostEqual(float(v.sum()), 1.0, places=6)
        else:
            # If the implementation returns a scalar, at least assert it's in [0,1]
            self.assertTrue(0.0 <= v <= 1.0)

        # bad name
        ssg_bad = _mk_ssg({"A": {"Mg": 2, "O": 4}, "B": {"Fe": 2, "O": 4}})
        ssg_bad.mixing_dist = "nope"
        with self.assertRaises(ValueError):
            ssg_bad._generate_mixing_fraction()

    def test_add_minor_elements_exponential(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        ssg.minor_elements = {"Na": {"distribution": "exponential", "scale": 0.01, "max_fraction": 0.1}}
        base = {"Mg": 2.0, "Si": 1.0}
        total_before = sum(base.values())
        out = ssg._add_minor_elements(base.copy())

        self.assertIn("Na", out)
        self.assertGreaterEqual(out["Na"], 0.0)
        # majors were reduced to make room for Na
        self.assertLess(out["Mg"] + out["Si"], total_before)
        # total should be roughly conserved (allowing very small fp noise)
        self.assertAlmostEqual(out["Mg"] + out["Si"] + out["Na"], total_before, places=6)

    def test_apply_site_variation_scalar_and_dict(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}},
                      site_variation=0.2, element_noise_scale=0.1, min_site_fraction=0.2)
        varied = ssg._apply_site_variation({"M": 3.0})
        self.assertIn("M", varied)
        self.assertGreaterEqual(varied["M"], 0.2 * 3.0)

        ssg.site_variation = {"M": 0.05}
        varied2 = ssg._apply_site_variation({"M": 3.0})
        self.assertGreaterEqual(varied2["M"], 0.2 * 3.0)

    def test_add_element_noise_preserves_oxygen_basis_and_nonneg(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}},
                      element_noise_scale=0.05)
        c = {"Mg": 2.0, "Si": 1.0}
        out = ssg._add_element_noise(c)
        self.assertTrue(all(v >= 0 for v in out.values()))

        # sum(c_i * oxygen_number(oxide_i)) ≈ oxygen_basis
        o_sum = sum(out[e] * ssg.OXYGEN_NUMBERS[ssg.CATION_TO_OXIDE_MAP[e]] for e in out)
        self.assertAlmostEqual(o_sum, ssg.oxygen_basis, delta=ssg.oxygen_basis * 0.05)

    def test_calculate_oxide_wt_percent_normalizes_to_100_and_raises_on_unknown(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        wt = ssg._calculate_oxide_wt_percent({"Mg": 2.0, "Si": 1.0})
        self.assertTrue(set(wt.keys()) >= {"MgO", "SiO2"})
        self.assertAlmostEqual(sum(wt.values()), 100.0, places=6)

        with self.assertRaises(ValueError):
            ssg._calculate_oxide_wt_percent({"Xx": 1.0})

    @patch.object(plt, "show")
    def test_generate_binary_single_dirichlet(self, _show):
        # Binary path
        ssg_bin = _mk_ssg({"A": {"Mg": 2.0, "Si": 1.0, "O": 4},
                           "B": {"Fe": 2.0, "Si": 1.0, "O": 4}},
                          mixing_dist="beta", mixing_params={"a": 2, "b": 2})
        df_bin = ssg_bin.generate(n_samples=20)
        self.assertFalse(df_bin.empty)
        self.assertTrue(any(c.endswith(ssg_bin.suffix) for c in df_bin.columns))
        self.assertIn("MgO", df_bin.columns)  # oxide weights present

        # Single endmember path
        ssg_one = _mk_ssg({"Only": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        df_one = ssg_one.generate(n_samples=10)
        self.assertFalse(df_one.empty)

        # >2 endmembers path (dirichlet)
        ssg_tri = _mk_ssg({"A": {"Mg": 2.0, "Si": 1.0, "O": 4},
                           "B": {"Fe": 2.0, "Si": 1.0, "O": 4},
                           "C": {"Ca": 2.0, "Si": 1.0, "O": 4}},
                          mixing_dist="dirichlet", mixing_params={"alpha": [1, 1, 1]})
        df_tri = ssg_tri.generate(n_samples=15)
        self.assertFalse(df_tri.empty)

    @patch.object(plt, "show")
    def test_compare_distributions_returns_stats_and_handles_twin_axis(self, _show):
        ssg = _mk_ssg({"Only": {"Mg": 2.0, "Si": 1.0, "O": 4}}, oxygen_basis=4)

        # Build base/synth with both cation and oxide columns so twin-axis branch is exercised
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

        stats = ssg.compare_distributions(base_df=base, synth_df=synth, ncols=2, figsize_per=(3, 2))
        self.assertIsInstance(stats, pd.DataFrame)
        self.assertTrue({"ks_stat", "p_value", "mean_base", "mean_synth", "std_base", "std_synth"}.issubset(stats.columns))
        # Index is the cation column names
        self.assertTrue(all(idx.endswith(suffix) for idx in stats.index))

    @patch.object(plt, "show")
    def test_compare_distributions_no_matching_columns(self, _show):
        ssg = _mk_ssg({"Only": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        base = pd.DataFrame({"foo": [1, 2, 3]})
        out = ssg.compare_distributions(base_df=base, synth_df=pd.DataFrame({"bar": [1, 2]}))
        self.assertTrue(isinstance(out, pd.DataFrame))
        self.assertTrue(out.empty)

    def test_check_charge_balance_calls_noise(self):
        ssg = _mk_ssg({"Em": {"Mg": 2.0, "Si": 1.0, "O": 4}})
        # Add a suffix-ed key to ensure _total_charge strips suffix correctly
        cations = {f"Mg{ssg.suffix}": 2.0, "Si": 1.0}
        with patch.object(ssg, "_add_element_noise", wraps=ssg._add_element_noise) as w:
            _ = ssg._check_charge_balance_add_noise(cations)
            w.assert_called_once()


if __name__ == "__main__":
    unittest.main()
