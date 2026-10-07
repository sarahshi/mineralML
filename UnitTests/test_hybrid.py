import types
import os
import io
import contextlib
import warnings
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
from unittest.mock import patch
import numpy as np
import pandas as pd
from math import sqrt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mineralML as mm
from mineralML.constants import OXIDES

def _get_oxides():
    # Be tolerant to where OXIDES lives
    if hasattr(mm, "constants") and hasattr(mm.constants, "OXIDES"):
        return mm.constants.OXIDES
    if hasattr(mm, "OXIDES"):
        return mm.OXIDES
    raise AttributeError("Could not find OXIDES in mineralML.")

def tiny_loader(n=32, in_features=11, n_classes=23, batch=16):
    x = torch.randn(n, in_features)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)

def _fake_scaler_series(oxides):
    # Series with index=oxides as your norm_data expects
    mean = pd.Series(np.zeros(len(oxides)), index=oxides)
    std  = pd.Series(np.ones(len(oxides)), index=oxides)
    return mean, std

class mineralML_supervised(unittest.TestCase):
    def setUp(self):
        self.data = {
            "SampleID": [72065, 72066, 31890, 31891, 59237, 59238, 37643, 37644],
            "Mineral": [
                "Amphibole", "Amphibole",
                "Pyroxene",  "Pyroxene",
                "Garnet",    "Garnet",
                "Olivine",   "Olivine",
            ],
            "SiO2":  [40, 39.7, 51.49, 51.15, 39.8, 40.2, 40.31, 38.99],
            "TiO2":  [3.1, 3.2, 0.6,   0.53,  0.6,  0.6,  0.01,  0.08],
            "Al2O3": [16.1, 16,  2.57,  2.57, 22.5, 23.4, 0.01,  np.nan],
            "Cr2O3": [0.08, 0.06, 0.24, 0.19, np.nan, np.nan, 0.06, 0.05],
            "FeOt":  [12,   13,   6.98, 5.55, 16.1, 15.1, 11.88, 19.2],
            "MnO":   [0.16, 0.17, 0.22, 0.16, 0.5,  0.5,  0.18,  0.25],
            "MgO":   [10.2, 9.5,  16.77,16.37,9.8, 12.3, 47.02, 40.73],
            "CaO":   [10,   10.7, 19.42,21.36,10.7, 7.9,  0.08,  0.26],
            "Na2O":  [3.1,  2.9,  0.25, 0.32, np.nan, np.nan, np.nan, np.nan],
            "K2O":   [1.9,  1.7,  np.nan,np.nan, np.nan, np.nan, np.nan, np.nan],
            "P2O5":  [np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan, np.nan],
        }
        self.df = pd.DataFrame(self.data)

    def test_load_mineral_classes(self):
        min_cat, mapping = mm.load_mineral_classes()

        # Robust checks (less brittle than fully hard-coding)
        self.assertIsInstance(min_cat, list)
        self.assertIsInstance(mapping, dict)
        # Keys are 0..N-1 and values match min_cat order
        self.assertEqual(sorted(mapping.keys()), list(range(len(min_cat))))
        self.assertEqual([mapping[i] for i in range(len(min_cat))], min_cat)

        # Sanity: core classes exist (names from your current mapping)
        required = {"Amphibole", "Pyroxene", "Garnet", "Olivine", "Leucite"}
        self.assertTrue(required.issubset(set(min_cat)))

    def test_prep_df(self):
        df_cleaned = mm.prep_df(self.df.copy())

        # No NaNs after cleaning (oxides filled with 0, Mineral preserved)
        self.assertEqual(int(df_cleaned.isnull().sum().sum()), 0)
        # self.assertEqual(df_cleaned.index.name, "SampleID")

        oxides = set(_get_oxides())
        # Required columns: all oxides + ZrO2 + Mineral
        expected_cols = oxides.union({"ZrO2", "Mineral"})
        self.assertTrue(expected_cols.issubset(set(df_cleaned.columns)))

    def test_norm_data(self):
        # Prepare
        df_cleaned = mm.prep_df(self.df.copy())
        oxides = _get_oxides()

        # Under test
        normalized_data = mm.norm_data(df_cleaned)
        self.assertEqual(normalized_data.shape, (len(df_cleaned), len(oxides)))

        # Compute expected normalization directly from the scaler
        mean, std = mm.load_scaler("scaler_nn_v0030.npz")
        # Ensure Series aligned to oxides
        mean = mean.reindex(oxides)
        std = std.reindex(oxides)

        expected = (df_cleaned[oxides] - mean.values) / std.values
        np.testing.assert_allclose(
            normalized_data, expected.to_numpy(), rtol=1e-6, atol=1e-6
        )

    def test_unique_mapping(self):
        # Use indices that exist in the CURRENT mapping (no more Clinopyroxene/Spinel)
        # Choose a small set: Amphibole(0), Pyroxene(15), Garnet(7), Olivine(14), Spinels(20)
        pred_class = np.array([0, 15, 7, 14, 20, 0, 7, 20])

        unique, valid_mapping = mm.unique_mapping(pred_class)

        # Expected unique set (order not guaranteed)
        expected_unique = np.array([0, 7, 14, 15, 20])
        np.testing.assert_array_equal(np.sort(unique), np.sort(expected_unique))

        # Names from the **loaded** mapping (robust to future reorderings)
        _, mapping = mm.load_mineral_classes()
        expected_valid_mapping = {i: mapping[i] for i in expected_unique}
        self.assertEqual(valid_mapping, expected_valid_mapping)

    def test_class2mineral(self):
        pred_class = np.array([0, 15, 7, 14, 20, 0, 7, 20])
        pred_mineral = mm.class2mineral(pred_class)

        # Build expected labels from the current mapping
        _, mapping = mm.load_mineral_classes()
        expected_pred_mineral = np.array([mapping[i] for i in pred_class])

        np.testing.assert_array_equal(pred_mineral, expected_pred_mineral)


class test_variational_layer(unittest.TestCase):
    def setUp(self):
        self.input_features = 11
        self.output_features = 3
        self.layer = mm.VariationalLayer(self.input_features, self.output_features)
        self.input = torch.randn(11, self.input_features)

    def test_initialization(self):
        self.assertEqual(
            self.layer.weight_mu.size(), (self.output_features, self.input_features)
        )
        self.assertEqual(
            self.layer.weight_rho.size(), (self.output_features, self.input_features)
        )
        self.assertEqual(self.layer.bias_mu.size(), (self.output_features,))
        self.assertEqual(self.layer.bias_rho.size(), (self.output_features,))

        std = 1.0 / sqrt(self.input_features)
        self.assertTrue(
            torch.all(self.layer.weight_mu.data <= std)
            and torch.all(self.layer.weight_mu.data >= -std)
        )
        self.assertTrue(
            torch.all(self.layer.weight_rho.data <= std)
            and torch.all(self.layer.weight_rho.data >= -std)
        )

    def test_forward_pass(self):
        output = self.layer(self.input)
        self.assertEqual(output.size(), (self.input_features, self.output_features))

    def test_kl_divergence(self):
        kl_div = self.layer.kl_divergence()
        self.assertIsInstance(kl_div, torch.Tensor)
        self.assertGreaterEqual(kl_div.item(), 0.0)


# class TestMultiClassClassifier(unittest.TestCase):
#     def test_forward_and_predict_shapes(self):
#         model = mm.MultiClassClassifier(input_dim=11, classes=7, hidden_layer_sizes=[16, 8, 4], dropout_rate=0.0)
#         x = torch.randn(5, 11)
#         logits = model(x)
#         self.assertEqual(logits.shape, (5, 7))
#         pred = model.predict(x)
#         self.assertEqual(pred.shape, (5,))
#         self.assertTrue((pred >= 0).all() and (pred < 7).all())


# class TestTrainNN(unittest.TestCase):
#     def test_train_nn_runs_and_early_stops(self):
#         in_features, n_classes = 10, 3
#         model = mm.MultiClassClassifier(input_dim=in_features, classes=n_classes, hidden_layer_sizes=[8, 4], dropout_rate=0.0)
#         opt = torch.optim.SGD(model.parameters(), lr=1e-2)
#         crit = nn.CrossEntropyLoss()
#         train_loader = tiny_loader(n=48, in_features=in_features, n_classes=n_classes, batch=16)
#         valid_loader = tiny_loader(n=48, in_features=in_features, n_classes=n_classes, batch=16)
#         out = mm.train_nn(
#             model=model,
#             optimizer=opt,
#             train_loader=train_loader,
#             valid_loader=valid_loader,
#             n_epoch=20,               # small
#             criterion=crit,
#             kl_weight_decay=0.1,      # small increments
#             kl_decay_epochs=5,        # ramp quickly
#             patience=3,               # force early stop quickly
#         )
#         train_out, valid_out, train_losses, valid_losses, best_valid, best_state = out
#         # minimal sanity checks
#         self.assertIsNotNone(best_state)
#         self.assertGreater(len(train_losses), 0)
#         self.assertGreater(len(valid_losses), 0)
#         self.assertIsInstance(best_valid, float)


# class TestPredictTrainLoop(unittest.TestCase):
#     def test_predict_class_prob_nn_train_stats_shape(self):
#         # Model that injects noise so std > 0
#         class NoisyModel(nn.Module):
#             def __init__(self, in_f=6, classes=5):
#                 super().__init__()
#                 self.fc = nn.Linear(in_f, classes)
#             def forward(self, x):
#                 return self.fc(x) + torch.randn_like(self.fc(x))*0.01

#         model = NoisyModel(in_f=6, classes=5)
#         x = torch.randn(4, 6)
#         mean, std = mm.predict_class_prob_nn_train(model, x, n_iterations=8)
#         self.assertEqual(mean.shape, (4, 5))
#         self.assertEqual(std.shape, (4, 5))
#         self.assertTrue(np.allclose(mean.sum(axis=1), 1.0, atol=1e-5))


class TestPredictClassProbNN(unittest.TestCase):
    @patch("mineralML.hybrid.load_model", side_effect=lambda model, opt, path: None)  # no file I/O
    @patch("mineralML.hybrid.norm_data")
    @patch("mineralML.hybrid.load_mineral_classes")
    @patch("mineralML.hybrid.class2mineral",
           side_effect=lambda idx: np.array([f"C{int(i)}" for i in idx]))
    def test_predict_class_prob_nn_contract(self, p_c2m, p_classes, p_norm, _p_load_model):
        K = 6
        fake_classes = [f"C{i}" for i in range(K)]
        fake_map = dict(enumerate(fake_classes))
        p_classes.return_value = (fake_classes, fake_map)

        ox = mm.constants.OXIDES
        N = 5
        df = pd.DataFrame(0.0, columns=list(ox) + ["ZrO2"], index=[f"S{i}" for i in range(N)])

        zircon_rows = [0, 2]
        non_zircon_rows = [i for i in range(N) if i not in zircon_rows]

        df.loc[df.index[zircon_rows], ["ZrO2", "SiO2", "TiO2"]] = [60.0, 30.0, 1.0]
        df.loc[df.index[non_zircon_rows], ["SiO2", "TiO2", "Al2O3"]] = [50.0, 1.0, 1.0]

        p_norm.side_effect = lambda d, *args, **kwargs: np.zeros((d.shape[0], len(ox)), dtype=np.float32)

        out_df = mm.predict_class_prob(df, n_iterations=1)

        self.assertEqual(len(out_df), N)
        self.assertTrue({"Predict_Mineral", "Prediction_Score"}.issubset(out_df.columns))

        for i in zircon_rows:
            self.assertEqual(out_df.iloc[i]["Predict_Mineral"], "Zircon")
            self.assertTrue(np.isnan(float(out_df.iloc[i]["Prediction_Score"])))


class TestBalance(unittest.TestCase):
    def test_balance_groups_with_mocks(self):
        # Build minimal df with “special” and “other” classes
        ox = mm.constants.OXIDES
        rows = []
        def row(mineral):
            r = {c: 0.0 for c in ox}
            r["Mineral"] = mineral
            return r
        for mineral in ["Clinopyroxene", "Orthopyroxene", "Plagioclase", "Alkali_Feldspar",
                        "Hematite", "Ilmenite", "Spinel", "Magnetite", "Glass",
                        "Garnet"]:
            rows.append(row(mineral))
        df = pd.DataFrame(rows)
        is_glass = df["Mineral"] == "Glass"
        df.loc[is_glass, "SiO2"] = 50.0      # passes SiO2 > 40 filter
        df.loc[is_glass, "Na2O"] = 0.5       # optional, used in TAS features
        df.loc[is_glass, "K2O"]  = 0.5
        # df.loc[is_glass, "TAS"] = "Bs"

        # --- mock imblearn + pyrolite so balance() doesn't require those deps ---
        fake_imblearn = types.ModuleType("imblearn")
        fake_os = types.ModuleType("over_sampling")
        class FakeROS:
            def __init__(self, sampling_strategy=None, random_state=None): pass
            def fit_resample(self, X, y):
                # simple passthrough: return X,y unchanged
                return X.values, y.values
        fake_os.RandomOverSampler = FakeROS
        fake_imblearn.over_sampling = fake_os

        fake_pyrolite = types.ModuleType("pyrolite")
        fake_util = types.ModuleType("util")
        fake_cls = types.ModuleType("classification")
        class FakeTAS:
            def __init__(self): pass
            def predict(self, df_):
                # bucket everything into two bins to exercise logic
                return pd.Series(np.where((df_.get("SiO2", 0) > 40), "A", "B"), index=df_.index)
        fake_cls.TAS = FakeTAS
        fake_util.classification = fake_cls
        fake_pyrolite.util = fake_util

        with mock.patch.dict("sys.modules", {
            "imblearn": fake_imblearn,
            "imblearn.over_sampling": fake_os,
            "pyrolite": fake_pyrolite,
            "pyrolite.util": fake_util,
            "pyrolite.util.classification": fake_cls,
        }):
            balanced = mm.balance(df, n=2)

        # Result should contain combined group names
        self.assertIn("Pyroxene", balanced["Mineral"].unique())
        self.assertIn("Feldspar", balanced["Mineral"].unique())
        self.assertIn("Rhombohedral_Oxides", balanced["Mineral"].unique())
        self.assertIn("Spinel_Group", balanced["Mineral"].unique())
        # Glass handled (either present or empty frame)
        self.assertTrue("Glass" in balanced["Mineral"].unique() or "Glass" not in df["Mineral"].unique())


class TestConfusionMatrixDF(unittest.TestCase):
    def test_confusion_matrix_df_merges_and_shape(self):
        given = ["Magnetite", "Plagioclase", "Hematite", "Zircon"]
        pred  = ["Spinel_Group",  "Alkali_Feldspar",  "Ilmenite", "Zircon"]
        cm = mm.confusion_matrix_df(given, pred)
        # Square with the fixed label set
        self.assertEqual(cm.shape[0], cm.shape[1])
        # Zircon row/col present
        self.assertIn("Zircon", cm.index)
        self.assertIn("Zircon", cm.columns)


def _toy_df(n=60):
    """Tiny synthetic dataset with required columns."""
    ox = mm.constants.OXIDES
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(50, 10, size=(n, len(ox))), columns=ox)
    # Add a few minerals (>= 2 classes so stratify works)
    minerals = np.array(["Garnet", "Olivine", "Pyroxene"])
    y = pd.Series(minerals[rng.integers(0, len(minerals), size=n)], name="Mineral")
    df = pd.concat([X, y], axis=1)
    # also required by prep/nn paths sometimes, but not strictly used here
    if "ZrO2" not in df.columns:
        df["ZrO2"] = 0.0
    # include an index like SampleID (not required by this function, but common)
    df.insert(0, "SampleID", [f"S{i}" for i in range(n)])
    return df


# ---------------------------------------------------------------------------
#  convert_fe_to_feot
# ---------------------------------------------------------------------------

class TestConvertFeToFeot(unittest.TestCase):

    def test_feo_only(self):
        df = pd.DataFrame({"FeO": [10.0], "SiO2": [50.0]})
        out = mm.convert_fe_to_feot(df)
        self.assertAlmostEqual(out["FeOt"].iloc[0], 10.0, places=4)
        self.assertNotIn("FeO", out.columns)

    def test_feot_only_passthrough(self):
        df = pd.DataFrame({"FeOt": [10.0], "SiO2": [50.0]})
        out = mm.convert_fe_to_feot(df)
        self.assertAlmostEqual(out["FeOt"].iloc[0], 10.0, places=4)

    def test_fe2o3_only_converted(self):
        fe2o3_val = 5.0
        fe_conv = 159.69 / (2 * 71.844)  # constants.OXIDE_MASSES
        expected = fe2o3_val / fe_conv
        df = pd.DataFrame({"Fe2O3": [fe2o3_val], "SiO2": [50.0]})
        out = mm.convert_fe_to_feot(df)
        self.assertAlmostEqual(out["FeOt"].iloc[0], expected, places=4)
        self.assertNotIn("Fe2O3", out.columns)

    def test_feo_plus_fe2o3_summed(self):
        fe_conv = 159.69 / (2 * 71.844)  # constants.OXIDE_MASSES
        df = pd.DataFrame({"FeO": [8.0], "Fe2O3": [2.0], "SiO2": [50.0]})
        out = mm.convert_fe_to_feot(df)
        expected = 8.0 + 2.0 / fe_conv
        self.assertAlmostEqual(out["FeOt"].iloc[0], expected, places=4)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"FeO": [10.0], "SiO2": [50.0]})
        original_cols = list(df.columns)
        mm.convert_fe_to_feot(df)
        self.assertEqual(list(df.columns), original_cols)


# ---------------------------------------------------------------------------
#  prep_df extended options
# ---------------------------------------------------------------------------


class TestConvertFeMixed(unittest.TestCase):

    COLS = ("FeO", "FeOt", "Fe2O3", "Fe2O3t")
    F = 2 * 71.844 / 159.69  # wt% FeO per wt% Fe2O3, from constants.OXIDE_MASSES

    def _convert(self, df):
        """Converts with warnings recorded, returning (out, warning messages)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = mm.convert_fe_to_feot(df)
        return out, [str(x.message) for x in w]

    def _combos_df(self):
        import itertools
        rows = []
        for r in range(1, 5):
            for combo in itertools.combinations(self.COLS, r):
                rows.append({c: (10.0 if c in combo else np.nan) for c in self.COLS})
        return pd.DataFrame(rows)

    # --- Chemistry ---

    def test_fe_moles_conserved(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"FeO": rng.uniform(0.1, 30, 200),
                           "Fe2O3": rng.uniform(0.1, 30, 200)})
        out, _ = self._convert(df)
        fe_in = df["FeO"] / 71.844 + 2 * df["Fe2O3"] / 159.69
        np.testing.assert_allclose(out["FeOt"] / 71.844, fe_in, rtol=1e-12)

    def test_every_fe_combination_gives_feot(self):
        out, _ = self._convert(self._combos_df())
        self.assertFalse(out["FeOt"].isna().any())

    def test_previously_handled_combinations_unchanged(self):
        # The nine combinations handled before the rewrite, with 10 wt% each
        f = self.F
        expected = {
            ("FeO",): 10.0,
            ("FeOt",): 10.0,
            ("Fe2O3",): 10.0 * f,
            ("Fe2O3t",): 10.0 * f,
            ("FeO", "Fe2O3"): 10.0 + 10.0 * f,
            ("FeO", "FeOt", "Fe2O3"): 10.0,
            ("FeO", "Fe2O3", "Fe2O3t"): 10.0 * f,
            ("FeOt", "Fe2O3"): 10.0,
            ("Fe2O3", "Fe2O3t"): 10.0 * f,
        }
        for combo, value in expected.items():
            df = pd.DataFrame([{c: (10.0 if c in combo else np.nan) for c in self.COLS}])
            out, _ = self._convert(df)
            self.assertAlmostEqual(out["FeOt"].iloc[0], value, places=10, msg=combo)

    def test_preference_order(self):
        f = self.F
        df = pd.DataFrame({
            "FeO":    [5.0,    np.nan, 5.0,    5.0],
            "FeOt":   [8.0,    8.0,    np.nan, np.nan],
            "Fe2O3":  [3.0,    np.nan, 3.0,    3.0],
            "Fe2O3t": [np.nan, 9.0,    9.0,    np.nan],
        })
        out, _ = self._convert(df)
        np.testing.assert_allclose(out["FeOt"], [8.0, 8.0, 9.0 * f, 5.0 + 3.0 * f])

    # --- Zeros and missing values ---

    def test_zero_placeholder_does_not_override_reported_fe(self):
        f = self.F
        df = pd.DataFrame({
            "FeOt":   [0.0, 0.0,    np.nan],
            "Fe2O3t": [9.0, np.nan, np.nan],
            "FeO":    [np.nan, 5.0, 5.0],
            "Fe2O3":  [np.nan, 2.0, 0.0],
        })
        out, _ = self._convert(df)
        np.testing.assert_allclose(out["FeOt"], [9.0 * f, 5.0 + 2.0 * f, 5.0])

    def test_fe_free_rows_stay_zero(self):
        df = pd.DataFrame({"FeOt": [0.0, 0.0], "Fe2O3t": [np.nan, 0.0]})
        out, _ = self._convert(df)
        self.assertEqual(list(out["FeOt"]), [0.0, 0.0])

    def test_no_fe_reported_gives_nan(self):
        df = pd.DataFrame({"SiO2": [99.0], "FeOt": [np.nan], "Fe2O3t": [np.nan]})
        out, _ = self._convert(df)
        self.assertTrue(np.isnan(out["FeOt"].iloc[0]))

    def test_negative_value_alone_is_kept(self):
        # Small negative values (e.g., EDS) pass through when nothing else is reported
        df = pd.DataFrame({"FeOt": [-0.02, -0.02], "Fe2O3t": [np.nan, 9.0]})
        out, _ = self._convert(df)
        np.testing.assert_allclose(out["FeOt"], [-0.02, 9.0 * self.F])

    # --- Non-numeric values ---

    def test_text_value_treated_as_not_reported(self):
        df = pd.DataFrame({"FeOt": ["bdl"], "Fe2O3t": [9.0]})
        out, msgs = self._convert(df)
        self.assertAlmostEqual(out["FeOt"].iloc[0], 9.0 * self.F)
        self.assertTrue(any("'bdl'" in m for m in msgs))

    def test_numbers_stored_as_text(self):
        df = pd.DataFrame({"FeO": ["5", "n.d."], "Fe2O3": ["2", 1.0]})
        out, msgs = self._convert(df)
        np.testing.assert_allclose(out["FeOt"], [5.0 + 2.0 * self.F, 1.0 * self.F])
        self.assertEqual(out["FeOt"].dtype, float)
        self.assertTrue(any("'n.d.'" in m for m in msgs))

    # --- Warnings ---

    def test_mixed_forms_across_rows_warns(self):
        df = pd.DataFrame({"FeOt": [8.0, np.nan], "Fe2O3t": [np.nan, 9.0]})
        out, msgs = self._convert(df)
        self.assertTrue(any("FeOt (1 row), Fe2O3t (1 row)" in m for m in msgs))
        self.assertFalse(out["FeOt"].isna().any())

    def test_single_form_no_warning(self):
        df = pd.DataFrame({"FeO": [8.0, 7.0], "Fe2O3": [1.0, 2.0]})
        _, msgs = self._convert(df)
        self.assertEqual(msgs, [])

    def test_agreeing_totals_no_warning(self):
        # FeOt and Fe2O3t from the same analysis, rounded to 2 decimals
        df = pd.DataFrame({"FeOt": [10.0, 4.5], "Fe2O3t": [11.11, 5.0]})
        out, msgs = self._convert(df)
        self.assertEqual(msgs, [])
        self.assertEqual(list(out["FeOt"]), [10.0, 4.5])

    def test_disagreeing_totals_warn_with_row_labels(self):
        df = pd.DataFrame({"FeOt": [10.0, 10.0], "Fe2O3t": [11.11, 20.0]},
                          index=["ok", "bad"])
        out, msgs = self._convert(df)
        self.assertEqual(list(out["FeOt"]), [10.0, 10.0])
        warning = [m for m in msgs if "differ" in m]
        self.assertEqual(len(warning), 1)
        self.assertIn("1 row(s)", warning[0])
        self.assertIn("'bad'", warning[0])

    # --- Structure ---

    def test_index_and_other_columns_preserved(self):
        df = pd.DataFrame({"SiO2": [50.0, 40.0], "Mineral": ["Olivine", "Spinel"],
                           "FeOt": [8.0, np.nan], "Fe2O3t": [np.nan, 9.0]},
                          index=[7, 7])
        out, _ = self._convert(df)
        self.assertEqual(list(out.index), [7, 7])
        self.assertEqual(list(out["Mineral"]), ["Olivine", "Spinel"])
        np.testing.assert_allclose(out["FeOt"], [8.0, 9.0 * self.F])
        for col in ("FeO", "Fe2O3", "Fe2O3t"):
            self.assertNotIn(col, out.columns)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"FeOt": pd.Series(dtype=float), "Fe2O3t": pd.Series(dtype=float)})
        out, msgs = self._convert(df)
        self.assertEqual(len(out), 0)
        self.assertIn("FeOt", out.columns)
        self.assertEqual(msgs, [])

    def test_prep_df_converts_mixed_feot_and_fe2o3t_columns(self):
        df = pd.DataFrame({"SiO2": [50.0, 50.0], "MgO": [10.0, 10.0],
                           "FeOt": [8.0, np.nan], "Fe2O3t": [np.nan, 8.9]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = mm.prep_df(df, convert_fe=True, verbose=False)
        self.assertGreater(out["FeOt"].iloc[1], 0)
        self.assertNotIn("Fe2O3t", out.columns)

    def test_prep_df_warns_when_feot_zero_placeholder(self):
        df = pd.DataFrame({"SiO2": [50.0], "MgO": [10.0],
                           "FeOt": [0.0], "Fe2O3t": [8.9]})
        with self.assertWarns(UserWarning) as ctx:
            mm.prep_df(df, convert_fe=False, verbose=False)
        self.assertTrue(any("convert_fe=True" in str(x.message) for x in ctx.warnings))

    def test_prep_df_warns_when_fe_only_in_other_columns(self):
        df = pd.DataFrame({"SiO2": [50.0, 50.0], "MgO": [10.0, 10.0],
                           "FeOt": [8.0, np.nan], "Fe2O3t": [np.nan, 8.9]})
        with self.assertWarns(UserWarning) as ctx:
            mm.prep_df(df, convert_fe=False, verbose=False)
        self.assertTrue(any("convert_fe=True" in str(x.message) for x in ctx.warnings))


class TestLabelMask(unittest.TestCase):

    def test_nullable_string_nulls_are_false(self):
        from mineralML.hybrid import _label_mask
        labels = pd.Series(["Pyroxene", None, "Olivine"], dtype="string")
        mask = _label_mask(labels, ["Pyroxene"])
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(list(mask), [True, False, False])
        # Usable for .loc indexing without raising on <NA>
        self.assertEqual(list(labels.loc[mask]), ["Pyroxene"])


class TestPrepDfOptions(unittest.TestCase):

    def test_convert_fe_true(self):
        df = pd.DataFrame({
            "SiO2": [50.0], "FeO": [10.0], "MgO": [8.0],
            "Mineral": ["Olivine"],
        })
        out = mm.prep_df(df, convert_fe=True, verbose=False)
        self.assertIn("FeOt", out.columns)
        self.assertNotIn("FeO", out.columns)

    def test_fe_variants_without_feot_raises(self):
        df = pd.DataFrame({
            "SiO2": [50.0], "FeO": [10.0], "MgO": [8.0],
            "Mineral": ["Olivine"],
        })
        with self.assertRaises(ValueError):
            mm.prep_df(df, convert_fe=False, verbose=False)

    def test_drop_empty_rows(self):
        df = pd.DataFrame({
            "SiO2": [50.0, 0.0], "FeOt": [10.0, 0.0], "MgO": [8.0, 0.0],
            "Mineral": ["Olivine", "Unknown"],
        })
        out = mm.prep_df(df, drop_empty_rows=True, min_oxide_count=2, verbose=False)
        # Second row has 0 non-zero oxides, should be dropped
        self.assertEqual(len(out), 1)


# ---------------------------------------------------------------------------
#  format_oxide_label
# ---------------------------------------------------------------------------


class TestFormatOxideLabel(unittest.TestCase):

    def test_total_passthrough(self):
        self.assertEqual(mm.format_oxide_label("Total"), "Total")

    def test_subscript_formatting(self):
        label = mm.format_oxide_label("SiO2")
        self.assertIn("_2", label)
        self.assertTrue(label.startswith("$"))

    def test_feot_formatting(self):
        label = mm.format_oxide_label("FeOt")
        self.assertIn("_t", label)


# ---------------------------------------------------------------------------
#  _mineral_colormap
# ---------------------------------------------------------------------------


class TestMineralColormap(unittest.TestCase):

    def test_returns_cmap_and_norm(self):
        from mineralML.hybrid import _mineral_colormap
        cmap, norm = _mineral_colormap(10)
        self.assertIsNotNone(cmap)
        self.assertIsNotNone(norm)


# ---------------------------------------------------------------------------
#  Model architecture classes
# ---------------------------------------------------------------------------


class TestFeatureExtractor(unittest.TestCase):

    def test_forward_shape(self):
        model = mm.FeatureExtractor(input_dim=11, classes=7, hidden_layer_sizes=[16, 8])
        x = torch.randn(5, 11)
        logits = model(x)
        self.assertEqual(logits.shape, (5, 7))

    def test_forward_with_features(self):
        model = mm.FeatureExtractor(input_dim=11, classes=7, hidden_layer_sizes=[16, 8])
        x = torch.randn(5, 11)
        logits, h = model(x, return_features=True)
        self.assertEqual(logits.shape, (5, 7))
        self.assertEqual(h.shape, (5, 8))  # last hidden layer size

    def test_bayesian_classifier_head(self):
        model = mm.FeatureExtractor(
            input_dim=11, classes=7, hidden_layer_sizes=[16, 8],
            use_bayesian_classifier=True
        )
        self.assertIsInstance(model.classifier, mm.VariationalLayer)
        x = torch.randn(3, 11)
        logits = model(x)
        self.assertEqual(logits.shape, (3, 7))


class TestLatentProjector(unittest.TestCase):

    def test_nonlinear_forward(self):
        proj = mm.LatentProjector(feat_dim=8, hidden=16, nonlinear=True)
        h = torch.randn(5, 8)
        z2 = proj(h)
        self.assertEqual(z2.shape, (5, 2))

    def test_linear_forward(self):
        proj = mm.LatentProjector(feat_dim=8, nonlinear=False)
        h = torch.randn(5, 8)
        z2 = proj(h)
        self.assertEqual(z2.shape, (5, 2))


class TestReconstructionDecoder(unittest.TestCase):

    def test_forward_shape(self):
        dec = mm.ReconstructionDecoder(z_dim=2, output_dim=11, decoder_hidden_sizes=[16, 8])
        z2 = torch.randn(5, 2)
        recon = dec(z2)
        self.assertEqual(recon.shape, (5, 11))


class TestReconstructionWrapper(unittest.TestCase):

    def _make_wrapper(self):
        clf = mm.FeatureExtractor(input_dim=11, classes=7, hidden_layer_sizes=[16, 8])
        mapper = mm.LatentProjector(feat_dim=8, hidden=16)
        decoder = mm.ReconstructionDecoder(z_dim=2, output_dim=11)
        return mm.ReconstructionWrapper(clf, mapper, decoder)

    def test_forward_returns_three_tensors(self):
        wrapper = self._make_wrapper()
        x = torch.randn(5, 11)
        logits, recon, z2 = wrapper(x)
        self.assertEqual(logits.shape, (5, 7))
        self.assertEqual(recon.shape, (5, 11))
        self.assertEqual(z2.shape, (5, 2))

    def test_latent_dim_attribute(self):
        wrapper = self._make_wrapper()
        self.assertEqual(wrapper.latent_dim, 2)


# ---------------------------------------------------------------------------
#  kl_divergence_sum
# ---------------------------------------------------------------------------

class TestKLDivergenceSum(unittest.TestCase):

    def test_model_with_variational_layers(self):
        model = mm.FeatureExtractor(
            input_dim=11, classes=7, hidden_layer_sizes=[16, 8],
            use_bayesian_feature_layer=True
        )
        # Need a forward pass to populate weights
        _ = model(torch.randn(2, 11))
        kl = mm.kl_divergence_sum(model)
        self.assertIsInstance(kl, (float, torch.Tensor))
        self.assertGreaterEqual(kl.detach().item(), 0.0)

    def test_model_without_variational_layers(self):
        # Plain linear model with no VariationalLayers
        model = nn.Sequential(nn.Linear(11, 8), nn.Linear(8, 4))
        kl = mm.kl_divergence_sum(model)
        self.assertEqual(float(kl), 0.0)


# ---------------------------------------------------------------------------
#  enable_mc_sampling
# ---------------------------------------------------------------------------

class TestEnableMCSampling(unittest.TestCase):

    def _make_model(self):
        return mm.FeatureExtractor(
            input_dim=11, classes=7, hidden_layer_sizes=[16, 8],
            use_bayesian_feature_layer=True, dropout_rate=0.2
        )

    def test_batchnorm_stays_eval(self):
        model = self._make_model()
        mm.enable_mc_sampling(model, enable_dropout=True)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                self.assertFalse(m.training)

    def test_variational_set_to_train(self):
        model = self._make_model()
        mm.enable_mc_sampling(model, enable_dropout=False)
        for m in model.modules():
            if isinstance(m, mm.VariationalLayer):
                self.assertTrue(m.training)

    def test_dropout_enabled_when_requested(self):
        model = self._make_model()
        mm.enable_mc_sampling(model, enable_dropout=True)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                self.assertTrue(m.training)

    def test_dropout_disabled_when_not_requested(self):
        model = self._make_model()
        mm.enable_mc_sampling(model, enable_dropout=False)
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                self.assertFalse(m.training)


# ---------------------------------------------------------------------------
#  _downsample
# ---------------------------------------------------------------------------

class TestDownsample(unittest.TestCase):

    def test_small_array_unchanged(self):
        from mineralML.hybrid import _downsample
        Z = np.random.randn(100, 2)
        labels = np.arange(100)
        Z_out, labels_out = _downsample(Z, labels, max_points=200)
        np.testing.assert_array_equal(Z_out, Z)
        np.testing.assert_array_equal(labels_out, labels)

    def test_large_array_downsampled(self):
        from mineralML.hybrid import _downsample
        Z = np.random.randn(1000, 2)
        labels = np.arange(1000)
        Z_out, labels_out = _downsample(Z, labels, max_points=100)
        self.assertEqual(Z_out.shape[0], 100)
        self.assertEqual(labels_out.shape[0], 100)

    def test_none_input(self):
        from mineralML.hybrid import _downsample
        Z_out, labels_out = _downsample(None, None, max_points=100)
        self.assertIsNone(Z_out)
        self.assertIsNone(labels_out)

    def test_no_labels(self):
        from mineralML.hybrid import _downsample
        Z = np.random.randn(500, 2)
        Z_out, labels_out = _downsample(Z, labels=None, max_points=50)
        self.assertEqual(Z_out.shape[0], 50)
        self.assertIsNone(labels_out)


# ---------------------------------------------------------------------------
#  build_model_from_config
# ---------------------------------------------------------------------------

class TestBuildModelFromConfig(unittest.TestCase):

    def test_builds_wrapper(self):
        config = {
            "input_dim": 11,
            "classes": 7,
            "hidden_layer_sizes": [16, 8],
            "feat_dim": 8,
            "dropout_rate": 0.1,
            "use_bayesian_feature_layer": True,
            "use_bayesian_classifier": False,
            "mapper_hidden": 16,
            "mapper_nonlinear": True,
            "decoder_hidden_sizes": [16, 8],
        }
        wrapper = mm.build_model_from_config(config, device="cpu")
        self.assertIsInstance(wrapper, mm.ReconstructionWrapper)

        # Verify forward pass works
        x = torch.randn(3, 11)
        logits, recon, z2 = wrapper(x)
        self.assertEqual(logits.shape, (3, 7))
        self.assertEqual(recon.shape, (3, 11))
        self.assertEqual(z2.shape, (3, 2))

    def test_mismatched_feat_dim_raises(self):
        config = {
            "input_dim": 11,
            "classes": 7,
            "hidden_layer_sizes": [16, 8],
            "feat_dim": 99,  # doesn't match last hidden layer (8)
        }
        with self.assertRaises(ValueError):
            mm.build_model_from_config(config, device="cpu")


# ---------------------------------------------------------------------------
#  Helpers for training-loop tests
# ---------------------------------------------------------------------------

def _tiny_model(in_dim=11, n_classes=23, hls=None):
    """Build a small FeatureExtractor for fast tests."""
    hls = hls or [8, 4]
    return mm.FeatureExtractor(
        input_dim=in_dim, classes=n_classes, hidden_layer_sizes=hls,
        dropout_rate=0.0, use_bayesian_feature_layer=True,
    )

def _tiny_wrapper(in_dim=11, n_classes=23, hls=None, feat_dim=4):
    """Build a small ReconstructionWrapper for fast tests."""
    hls = hls or [8, 4]
    clf = mm.FeatureExtractor(
        input_dim=in_dim, classes=n_classes, hidden_layer_sizes=hls,
        dropout_rate=0.0, use_bayesian_feature_layer=True,
    )
    mapper = mm.LatentProjector(feat_dim=feat_dim, hidden=8, nonlinear=True)
    decoder = mm.ReconstructionDecoder(z_dim=2, output_dim=in_dim, decoder_hidden_sizes=[8])
    return mm.ReconstructionWrapper(clf, mapper, decoder)


# ---------------------------------------------------------------------------
#  train_nn_hybrid_classifier
# ---------------------------------------------------------------------------

class TestTrainNNHybridClassifier(unittest.TestCase):

    def test_returns_loss_dicts_and_best_state(self):
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader = tiny_loader(n=32, in_features=11, n_classes=23, batch=16)

        train_losses, valid_losses, best_valid, best_state = mm.train_nn_hybrid_classifier(
            model, optimizer, loader, loader,
            n_epoch=3, kl_weight_decay=0.01, kl_decay_epochs=2, patience=10,
        )

        # Loss dicts have the right keys
        for key in ("total", "classification", "kl"):
            self.assertIn(key, train_losses)
            self.assertIn(key, valid_losses)
            self.assertEqual(len(train_losses[key]), 3)

        # Best state is a valid state_dict
        self.assertIsInstance(best_state, dict)
        self.assertIsInstance(best_valid, float)
        self.assertGreater(best_valid, 0.0)

    def test_early_stopping(self):
        # With patience=1 and a tiny model, early stopping should kick in
        model = _tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)  # very low LR -> little improvement
        loader = tiny_loader(n=32, in_features=11, n_classes=23, batch=16)

        train_losses, valid_losses, _, _ = mm.train_nn_hybrid_classifier(
            model, optimizer, loader, loader,
            n_epoch=100, kl_weight_decay=0.0, kl_decay_epochs=1, patience=1,
        )

        # Should have stopped well before 100 epochs
        self.assertLess(len(train_losses["total"]), 100)


# ---------------------------------------------------------------------------
#  train_nn_hybrid_bottleneck
# ---------------------------------------------------------------------------

class TestTrainNNHybridBottleneck(unittest.TestCase):

    def test_returns_loss_dicts_and_best_states(self):
        clf = _tiny_model()
        mapper = mm.LatentProjector(feat_dim=4, hidden=8)
        decoder = mm.ReconstructionDecoder(z_dim=2, output_dim=11, decoder_hidden_sizes=[8])
        optimizer = torch.optim.Adam(
            list(mapper.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        loader = tiny_loader(n=32, in_features=11, n_classes=23, batch=16)

        train_losses, valid_losses, best_valid, best_mapper, best_decoder = (
            mm.train_nn_hybrid_bottleneck(
                clf, mapper, decoder, optimizer, loader, loader,
                n_epoch=3, patience=10, plot_latent=False,
            )
        )

        self.assertIn("reconstruction", train_losses)
        self.assertIn("reconstruction", valid_losses)
        self.assertEqual(len(train_losses["reconstruction"]), 3)

        self.assertIsInstance(best_valid, float)
        self.assertIsInstance(best_mapper, dict)
        self.assertIsInstance(best_decoder, dict)

    def test_classifier_stays_frozen(self):
        clf = _tiny_model()
        mapper = mm.LatentProjector(feat_dim=4, hidden=8)
        decoder = mm.ReconstructionDecoder(z_dim=2, output_dim=11, decoder_hidden_sizes=[8])
        optimizer = torch.optim.Adam(
            list(mapper.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        loader = tiny_loader(n=32, in_features=11, n_classes=23, batch=16)

        # Snapshot classifier weights before training
        clf_before = {k: v.clone() for k, v in clf.state_dict().items()}

        mm.train_nn_hybrid_bottleneck(
            clf, mapper, decoder, optimizer, loader, loader,
            n_epoch=2, patience=10, plot_latent=False,
        )

        # Classifier weights should be unchanged
        for key, before in clf_before.items():
            after = clf.state_dict()[key]
            self.assertTrue(torch.equal(before, after), f"Classifier param {key} was modified")

    @patch.object(plt, "show")
    def test_plot_latent_runs(self, _show):
        clf = _tiny_model()
        mapper = mm.LatentProjector(feat_dim=4, hidden=8)
        decoder = mm.ReconstructionDecoder(z_dim=2, output_dim=11, decoder_hidden_sizes=[8])
        optimizer = torch.optim.Adam(
            list(mapper.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        loader = tiny_loader(n=32, in_features=11, n_classes=23, batch=16)

        train_losses, valid_losses, _, _, _ = mm.train_nn_hybrid_bottleneck(
            clf, mapper, decoder, optimizer, loader, loader,
            n_epoch=2, patience=10,
            plot_latent=True, plot_every=1, plot_on="valid",
        )

        # plt.show should have been called at least once (epoch 0 and epoch 1)
        self.assertGreaterEqual(_show.call_count, 1)
        self.assertEqual(len(train_losses["reconstruction"]), 2)
        plt.close("all")

    @patch.object(plt, "show")
    def test_plot_latent_on_train(self, _show):
        clf = _tiny_model()
        mapper = mm.LatentProjector(feat_dim=4, hidden=16)
        decoder = mm.ReconstructionDecoder(z_dim=2, output_dim=11, decoder_hidden_sizes=[64, 32])

        optimizer = torch.optim.Adam(
            list(mapper.parameters()) + list(decoder.parameters()), lr=1e-3
        )
        loader = tiny_loader(n=32, in_features=11, n_classes=23, batch=16)

        # Exercise the plot_on="train" branch
        mm.train_nn_hybrid_bottleneck(
            clf, mapper, decoder, optimizer, loader, loader,
            n_epoch=2, patience=10,
            plot_latent=True, plot_every=1, plot_on="train",
        )

        self.assertGreaterEqual(_show.call_count, 1)
        plt.close("all")


# ---------------------------------------------------------------------------
#  compute_z2_from_df
# ---------------------------------------------------------------------------

class TestComputeZ2FromDf(unittest.TestCase):

    @patch("mineralML.hybrid.norm_data")
    def test_returns_z2_and_preds(self, mock_norm):
        n_classes = 4
        wrapper = _tiny_wrapper(n_classes=n_classes)
        wrapper.eval()

        oxides = _get_oxides()
        N = 10
        df = pd.DataFrame(np.random.rand(N, len(oxides)), columns=oxides)

        mock_norm.return_value = np.zeros((N, len(oxides)), dtype=np.float32)

        Z2, preds = mm.compute_z2_from_df(df, wrapper, device="cpu")

        self.assertEqual(Z2.shape, (N, 2))
        self.assertEqual(preds.shape, (N,))
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds < n_classes))


# ---------------------------------------------------------------------------
#  train_hybrid_model
# ---------------------------------------------------------------------------


class TestTrainHybridModel(unittest.TestCase):

    def _make_df(self, n=80):
        """Build a small synthetic DataFrame with required columns."""
        oxides = _get_oxides()
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            rng.normal(50, 10, size=(n, len(oxides))), columns=oxides
        )
        minerals = rng.choice(
            ["Olivine", "Garnet", "Pyroxene", "Amphibole"], size=n
        )
        df["Mineral"] = minerals
        df["ZrO2"] = 0.0
        return df

    @patch("mineralML.hybrid.plot_loss_curves")
    @patch("mineralML.hybrid.plot_latent_space_training")
    @patch("mineralML.hybrid.train_nn_hybrid_bottleneck")
    @patch("mineralML.hybrid.train_nn_hybrid_classifier")
    @patch("mineralML.hybrid.balance", side_effect=lambda d, n=1000: d)
    def test_returns_best_state_balanced(
        self, _balance, mock_cls, mock_bottle, mock_plot_latent, mock_plot_loss
    ):
        from tempfile import TemporaryDirectory

        df = self._make_df(n=80)

        # Fake Stage A: return loss dicts and a real small state dict
        tiny = _tiny_model(n_classes=23, hls=[8, 4])
        mock_cls.return_value = (
            {"total": [1.0], "classification": [0.8], "kl": [0.01]},
            {"total": [1.1], "classification": [0.9], "kl": [0.01]},
            0.9,
            tiny.state_dict(),
        )

        # Fake Stage B: return loss dicts and real small state dicts
        mapper = mm.LatentProjector(feat_dim=4, hidden=8)
        decoder = mm.ReconstructionDecoder(z_dim=2, output_dim=11, decoder_hidden_sizes=[8])
        mock_bottle.return_value = (
            {"reconstruction": [2.0]},
            {"reconstruction": [2.5]},
            2.5,
            mapper.state_dict(),
            decoder.state_dict(),
        )

        # Fake latent space plots
        mock_plot_latent.return_value = (
            np.random.randn(10, 2), np.random.randint(0, 4, 10)
        )

        with TemporaryDirectory() as tmp:
            original_dir = os.getcwd()
            try:
                os.chdir(tmp)
                best_state = mm.train_hybrid_model(
                    df=df,
                    hls_list=[[8, 4]],
                    kl_weight_decay_list=[0.01],
                    lr=1e-3, wd=1e-4, dr=0.0, ep=2, n=0.2,
                    balanced=True,
                    ep_bottle=2,
                    name="test_run",
                )
            finally:
                os.chdir(original_dir)

        self.assertIsInstance(best_state, dict)
        _balance.assert_called_once()
        self.assertEqual(mock_cls.call_count, 1)  # 1 hls x 1 kl = 1 call
        mock_bottle.assert_called_once()
        self.assertEqual(mock_plot_latent.call_count, 2)  # train + valid
        mock_plot_loss.assert_called_once()

    @patch("mineralML.hybrid.plot_loss_curves")
    @patch("mineralML.hybrid.plot_latent_space_training")
    @patch("mineralML.hybrid.train_nn_hybrid_bottleneck")
    @patch("mineralML.hybrid.train_nn_hybrid_classifier")
    @patch("mineralML.hybrid.balance", side_effect=lambda d, n=1000: d)
    def test_unbalanced_skips_balance(
        self, mock_balance, mock_cls, mock_bottle, mock_plot_latent, mock_plot_loss
    ):
        from tempfile import TemporaryDirectory

        df = self._make_df(n=80)

        tiny = _tiny_model(n_classes=23, hls=[8, 4])
        mock_cls.return_value = (
            {"total": [1.0], "classification": [0.8], "kl": [0.01]},
            {"total": [1.1], "classification": [0.9], "kl": [0.01]},
            0.9,
            tiny.state_dict(),
        )

        mapper = mm.LatentProjector(feat_dim=4, hidden=8)
        decoder = mm.ReconstructionDecoder(z_dim=2, output_dim=11, decoder_hidden_sizes=[8])
        mock_bottle.return_value = (
            {"reconstruction": [2.0]},
            {"reconstruction": [2.5]},
            2.5,
            mapper.state_dict(),
            decoder.state_dict(),
        )
        mock_plot_latent.return_value = (
            np.random.randn(10, 2), np.random.randint(0, 4, 10)
        )

        with TemporaryDirectory() as tmp:
            original_dir = os.getcwd()
            try:
                os.chdir(tmp)
                mm.train_hybrid_model(
                    df=df,
                    hls_list=[[8, 4]],
                    kl_weight_decay_list=[0.01],
                    lr=1e-3, wd=1e-4, dr=0.0, ep=2, n=0.2,
                    balanced=False,
                    ep_bottle=2,
                    name="test_unbal",
                )
            finally:
                os.chdir(original_dir)

        mock_balance.assert_not_called()

    @patch("mineralML.hybrid.plot_loss_curves")
    @patch("mineralML.hybrid.plot_latent_space_training")
    @patch("mineralML.hybrid.train_nn_hybrid_bottleneck")
    @patch("mineralML.hybrid.train_nn_hybrid_classifier")
    @patch("mineralML.hybrid.balance", side_effect=lambda d, n=1000: d)
    def test_grid_sweep_calls_classifier_per_combo(
        self, _balance, mock_cls, mock_bottle, mock_plot_latent, mock_plot_loss
    ):
        from tempfile import TemporaryDirectory

        df = self._make_df(n=80)

        tiny = _tiny_model(n_classes=23, hls=[8, 4])
        mock_cls.return_value = (
            {"total": [1.0], "classification": [0.8], "kl": [0.01]},
            {"total": [1.1], "classification": [0.9], "kl": [0.01]},
            0.9,
            tiny.state_dict(),
        )

        mapper = mm.LatentProjector(feat_dim=4, hidden=8)
        decoder = mm.ReconstructionDecoder(z_dim=2, output_dim=11, decoder_hidden_sizes=[8])
        mock_bottle.return_value = (
            {"reconstruction": [2.0]},
            {"reconstruction": [2.5]},
            2.5,
            mapper.state_dict(),
            decoder.state_dict(),
        )
        mock_plot_latent.return_value = (
            np.random.randn(10, 2), np.random.randint(0, 4, 10)
        )

        with TemporaryDirectory() as tmp:
            original_dir = os.getcwd()
            try:
                os.chdir(tmp)
                mm.train_hybrid_model(
                    df=df,
                    hls_list=[[8, 4], [16, 8]],
                    kl_weight_decay_list=[0.01, 0.1],
                    lr=1e-3, wd=1e-4, dr=0.0, ep=2, n=0.2,
                    balanced=True,
                    ep_bottle=2,
                    name="test_sweep",
                )
            finally:
                os.chdir(original_dir)

        # 2 hls x 2 kl = 4 classifier training calls
        self.assertEqual(mock_cls.call_count, 4)
        # Bottleneck only runs once with the best classifier
        mock_bottle.assert_called_once()


# ---------------------------------------------------------------------------
#  plot_loss_curves
# ---------------------------------------------------------------------------

class TestPlotLossCurves(unittest.TestCase):

    def test_saves_figure(self):
        from tempfile import TemporaryDirectory
        train = {
            "cls_classification": [1.0, 0.8, 0.6],
            "cls_kl": [0.01, 0.02, 0.03],
            "cls_total": [1.01, 0.82, 0.63],
            "dec_reconstruction": [5.0, 3.0, 2.0],
        }
        valid = {
            "cls_classification": [1.1, 0.9, 0.7],
            "cls_kl": [0.01, 0.02, 0.03],
            "cls_total": [1.11, 0.92, 0.73],
            "dec_reconstruction": [5.5, 3.5, 2.5],
        }
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "losses.pdf")
            mm.plot_loss_curves(train, valid, path)
            self.assertTrue(os.path.exists(path))

    def test_handles_empty_histories(self):
        from tempfile import TemporaryDirectory
        train = {"cls_classification": [], "cls_kl": [], "cls_total": [], "dec_reconstruction": []}
        valid = {"cls_classification": [], "cls_kl": [], "cls_total": [], "dec_reconstruction": []}
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "losses_empty.pdf")
            mm.plot_loss_curves(train, valid, path)
            self.assertTrue(os.path.exists(path))


# ---------------------------------------------------------------------------
#  plot_latent_space_training
# ---------------------------------------------------------------------------

class TestPlotLatentSpaceTraining(unittest.TestCase):

    def test_returns_latents_and_labels(self):
        from tempfile import TemporaryDirectory
        import matplotlib
        matplotlib.use("Agg")

        wrapper = _tiny_wrapper()
        wrapper.eval()

        N, in_dim, n_classes = 20, 11, 4
        x = torch.randn(N, in_dim)
        y = torch.randint(0, n_classes, (N,))
        dataset = TensorDataset(x, y)

        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "latent.pdf")
            latents, labels = mm.plot_latent_space_training(
                model=wrapper, dataset=dataset, title="Test", filename=path,
            )

            self.assertEqual(latents.shape, (N, 2))
            self.assertEqual(labels.shape, (N,))
            self.assertTrue(os.path.exists(path))


# ---------------------------------------------------------------------------
#  plot_latent_space (mock-heavy, depends on checkpoint + reference data)
# ---------------------------------------------------------------------------

class TestPlotLatentSpace(unittest.TestCase):

    @patch("mineralML.hybrid.load_mineral_classes")
    @patch("mineralML.hybrid.load_hybrid_checkpoint")
    @patch("mineralML.hybrid.compute_z2_from_df")
    @patch("mineralML.hybrid.np.load")
    @patch("mineralML.hybrid.os.path.exists", return_value=True)
    @patch.object(plt, "show")
    def test_runs_without_error(self, _show, _exists, mock_npload,
                                 mock_z2, mock_ckpt, mock_classes):
        oxides = _get_oxides()
        N = 10
        n_classes = 4

        # Mock load_mineral_classes so unique_mapping doesn't hit np.load
        fake_cats = [f"C{i}" for i in range(n_classes)]
        mock_classes.return_value = (fake_cats, dict(enumerate(fake_cats)))

        # Mock the reference latent data file
        mock_npload.return_value.__enter__ = lambda s: {
            "valid_latents": np.random.randn(50, 2).astype(np.float32),
            "valid_labels": np.random.randint(0, n_classes, 50),
        }
        mock_npload.return_value.__exit__ = lambda s, *a: None

        # Mock the model checkpoint
        wrapper = _tiny_wrapper(n_classes=n_classes)
        mock_ckpt.return_value = (wrapper, None, None)

        # Mock compute_z2_from_df
        mock_z2.return_value = (
            np.random.randn(N, 2).astype(np.float32),
            np.random.randint(0, n_classes, N),
        )

        # Build the input DataFrame
        df = pd.DataFrame(np.random.rand(N, len(oxides)), columns=oxides)
        df["Predict_Mineral"] = "C0"

        mm.plot_latent_space(df, label_column="Predict_Mineral")
        plt.close("all")


# ---------------------------------------------------------------------------
#  plot_harker (mock-heavy, depends on checkpoint + reference data)
# ---------------------------------------------------------------------------

class TestPlotHarker(unittest.TestCase):

    def setUp(self):
        oxides = _get_oxides()
        self.df_train = pd.DataFrame(
            np.random.rand(30, len(oxides)) * 60,
            columns=oxides,
        )
        self.df_train["Mineral"] = np.random.choice(
            ["Olivine", "Garnet", "Pyroxene"], 30
        )

    @patch.object(plt, "show")
    def test_basic_no_data(self, _show):
        mm.plot_harker()
        plt.close("all")

    @patch.object(plt, "show")
    def test_train_background(self, _show):
        mm.plot_harker(
            df_train=self.df_train,
            train_minerals=["Olivine", "Garnet"],
        )
        plt.close("all")

    @patch.object(plt, "show")
    def test_overlay_plain_dataframe(self, _show):
        oxides = _get_oxides()
        overlay = pd.DataFrame(
            np.random.rand(10, len(oxides)) * 60, columns=oxides
        )
        mm.plot_harker(
            df_train=self.df_train,
            train_minerals=["Olivine"],
            overlay_datasets={"Study A": overlay},
        )
        plt.close("all")

    @patch.object(plt, "show")
    def test_overlay_with_custom_kws(self, _show):
        oxides = _get_oxides()
        overlay = pd.DataFrame(
            np.random.rand(10, len(oxides)) * 60, columns=oxides
        )
        mm.plot_harker(
            overlay_datasets={
                "Study B": (overlay, {"s": 100, "marker": "D"}),
            },
        )
        plt.close("all")

    @patch.object(plt, "show")
    def test_extra_pairs(self, _show):
        mm.plot_harker(
            df_train=self.df_train,
            train_minerals=["Olivine"],
            extra_pairs=[("CaO", "Na2O")],
        )
        plt.close("all")

    @patch.object(plt, "show")
    def test_plot_totals(self, _show):
        oxides = _get_oxides()
        overlay = pd.DataFrame(
            np.random.rand(5, len(oxides)) * 60, columns=oxides
        )
        mm.plot_harker(
            df_train=self.df_train,
            train_minerals=["Olivine"],
            overlay_datasets={"Study": overlay},
            plot_totals=True,
        )
        plt.close("all")

    @patch.object(plt, "show")
    def test_plot_totals_with_overlay_kws(self, _show):
        oxides = _get_oxides()
        overlay = pd.DataFrame(
            np.random.rand(5, len(oxides)) * 60, columns=oxides
        )
        mm.plot_harker(
            overlay_datasets={"Study": (overlay, {"marker": "^"})},
            plot_totals=True,
        )
        plt.close("all")

    @patch.object(plt, "show")
    def test_custom_title(self, _show):
        mm.plot_harker(
            df_train=self.df_train,
            train_minerals=["Olivine"],
            title="My Harker Diagram",
        )
        plt.close("all")


def _silent(fn, *args, **kwargs):
    """Call fn with stdout captured; return (result, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()




# Real analyses (wt%) spanning the network classes and the empirical rules
# (training data natural rows and the Cpx compilation), plus an empty row
ANALYSES = pd.DataFrame([
    # 20210320-003_E5-1 (Kahletal2023)
    dict(Sample="ol", SiO2=39.671, TiO2=0.008, Al2O3=0.042, FeOt=13.926, MnO=0.225,
         MgO=45.983, CaO=0.312, Cr2O3=0.03),
    # HLY0102-D41-4 (Bennettetal2019)
    dict(Sample="plag", SiO2=50.27, Al2O3=31.06, FeOt=0.38, MgO=0.18, CaO=14.11, Na2O=3.26,
         K2O=0.05),
    # Conboy cpx-8 (Hildreth and Fierstein, 1997)
    dict(Sample="cpx", SiO2=51.399, TiO2=0.69, Al2O3=2.324, FeOt=8.674, MnO=0.242,
         MgO=15.668, CaO=20.025, Na2O=0.376, Cr2O3=0.039),
    # Gon05262iph-g (Kleinsasseretal2008)
    dict(Sample="qz", SiO2=97.25, TiO2=0.01, Al2O3=0, FeOt=0.03, MnO=0, MgO=0, CaO=0.01,
         Na2O=0, K2O=0),
    # Z131 (Geisler1999)
    dict(Sample="zrc", SiO2=31.53, FeOt=0, CaO=0.014, P2O5=0.09, ZrO2=65.77),
    # CG-2b_67 (Myintetal2022)
    dict(Sample="cc", SiO2=0, Al2O3=0.0043, FeOt=0.301, MnO=0.497, MgO=0.261, CaO=54.165),
    dict(Sample="blank"),
]).reindex(columns=["Sample"] + OXIDES + ["ZrO2"])


# ---------------------------------------------------------------------------
#  prep_df / norm_data branches
# ---------------------------------------------------------------------------


class TestPrepDfBranches(unittest.TestCase):

    def test_fe_variants_raise_or_convert(self):
        df = pd.DataFrame({"SiO2": [50.0], "FeO": [10.0], "MgO": [30.0]})
        with self.assertRaises(ValueError):
            mm.prep_df(df.copy(), verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, printed = _silent(mm.prep_df, df.copy(), convert_fe=True)
        self.assertIn("Converted iron columns", printed)
        self.assertAlmostEqual(out["FeOt"].iat[0], 10.0)

    def test_sample_index_is_recovered_as_column(self):
        df = pd.DataFrame({"SiO2": [50.0, 45.0], "MgO": [30.0, 40.0]},
                          index=pd.Index(["a", "b"], name="Sample"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = mm.prep_df(df, verbose=False)
        self.assertEqual(list(out["Sample"]), ["a", "b"])
        self.assertEqual(out.columns[0], "Sample")

    def test_non_numeric_values_warn_and_zero(self):
        df = pd.DataFrame({"SiO2": ["bdl", 45.0], "MgO": [30.0, 40.0]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = mm.prep_df(df, verbose=False)
        self.assertTrue(any("Non-numeric" in str(w.message) and "'bdl'" in str(w.message)
                            for w in caught))
        self.assertEqual(out["SiO2"].iat[0], 0.0)

    def test_renormalise_and_drop_empty_rows(self):
        df = pd.DataFrame({"SiO2": [40.0, 0.0, 30.0], "MgO": [40.0, 0.0, 0.0]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, printed = _silent(mm.prep_df, df, renormalize=True, drop_empty_rows=True)
        self.assertIn("Renormalized 2 row(s)", printed)
        self.assertIn("2 dropped", printed)
        self.assertTrue(any("were dropped" in str(w.message) for w in caught))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[OXIDES + ["ZrO2"]].sum(axis=1).iat[0], 100.0)


class TestNormDataBranches(unittest.TestCase):

    def test_scaler_must_be_series(self):
        with mock.patch("mineralML.hybrid.load_scaler",
                        return_value=(np.zeros(11), np.ones(11))):
            with self.assertRaises(ValueError):
                mm.norm_data(pd.DataFrame({c: [1.0] for c in OXIDES}))

    def test_scaler_missing_column(self):
        mean = pd.Series(0.0, index=OXIDES[:-1])
        with mock.patch("mineralML.hybrid.load_scaler", return_value=(mean, mean + 1)):
            with self.assertRaises(ValueError):
                mm.norm_data(pd.DataFrame({c: [1.0] for c in OXIDES}))

    def test_nan_inputs_are_prepped(self):
        mean, std = pd.Series(1.0, index=OXIDES), pd.Series(2.0, index=OXIDES)
        df = pd.DataFrame({c: [3.0] for c in OXIDES})
        df.loc[0, "MnO"] = np.nan
        with mock.patch("mineralML.hybrid.load_scaler", return_value=(mean, std)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, _ = _silent(mm.norm_data, df)
        self.assertEqual(x.shape, (1, len(OXIDES)))
        self.assertAlmostEqual(x[0, OXIDES.index("SiO2")], 1.0)
        self.assertAlmostEqual(x[0, OXIDES.index("MnO")], -0.5)   # NaN filled with 0

    def test_absent_columns_are_added_as_zero(self):
        # Regression: an absent oxide column used to raise KeyError before the
        # prep_df fallback could run.
        mean, std = pd.Series(1.0, index=OXIDES), pd.Series(2.0, index=OXIDES)
        df = pd.DataFrame({c: [3.0] for c in OXIDES if c not in ("MnO", "P2O5")})
        with mock.patch("mineralML.hybrid.load_scaler", return_value=(mean, std)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                x, _ = _silent(mm.norm_data, df)
        self.assertTrue(any("were missing" in str(w.message) for w in caught))
        self.assertAlmostEqual(x[0, OXIDES.index("MnO")], -0.5)
        self.assertAlmostEqual(x[0, OXIDES.index("SiO2")], 1.0)


# ---------------------------------------------------------------------------
#  balance, with pyrolite mocked (it is not a runtime dependency)
# ---------------------------------------------------------------------------


def _fake_pyrolite():
    fake_pyrolite = types.ModuleType("pyrolite")
    fake_util = types.ModuleType("util")
    fake_cls = types.ModuleType("classification")

    class FakeTAS:
        def predict(self, df_):
            return pd.Series(np.where(df_["SiO2"] > 60, "Dacite", "Andesite"),
                             index=df_.index)

    fake_cls.TAS = FakeTAS
    fake_util.classification = fake_cls
    fake_pyrolite.util = fake_util
    return {"pyrolite": fake_pyrolite, "pyrolite.util": fake_util,
            "pyrolite.util.classification": fake_cls}


class TestBalanceBranches(unittest.TestCase):

    def _df(self):
        rng = np.random.default_rng(0)
        counts = {"Olivine": 12, "Amphibole": 20, "Garnet": 3, "Clinopyroxene": 12,
                  "Orthopyroxene": 12, "Plagioclase": 12, "Alkali_Feldspar": 12,
                  "Ilmenite": 6, "Hematite": 2, "Magnetite": 6, "Spinel": 2,
                  "Glass": 10, "Apatite": 3, "Titanite": 1000}
        rows = []
        for mineral, n in counts.items():
            block = pd.DataFrame(rng.uniform(0, 50, size=(n, len(OXIDES))), columns=OXIDES)
            block["Mineral"] = mineral
            rows.append(block)
        df = pd.concat(rows, ignore_index=True)
        glass = df["Mineral"] == "Glass"
        df.loc[glass, "SiO2"] = np.linspace(50, 75, glass.sum())
        return df

    def test_group_sizes(self):
        with mock.patch.dict("sys.modules", _fake_pyrolite()):
            out = mm.balance(self._df(), n=4)
        counts = out["Mineral"].value_counts()
        self.assertEqual(counts["Olivine"], 4)            # kmeans, remainder allocated
        self.assertEqual(counts["Amphibole"], 8)          # 2n
        self.assertEqual(counts["Garnet"], 3)             # fewer rows than n: all kept
        self.assertEqual(counts["Pyroxene"], 8)           # 4 cpx + 4 opx
        self.assertEqual(counts["Feldspar"], 8)
        self.assertEqual(counts["Rhombohedral_Oxides"], 6)  # 4 ilmenite (capped) + 2 hematite
        self.assertEqual(counts["Spinel_Group"], 6)
        self.assertEqual(counts["Glass"], 8)              # TAS-stratified to 2n
        self.assertEqual(counts["Apatite"], 4)            # oversampled up to n
        self.assertEqual(counts["Titanite"], 4)           # >= 1000 rows: capped
        self.assertNotIn("TAS", out.columns)

    def test_missing_pyrolite_raises(self):
        blocked = {k: None for k in _fake_pyrolite()}
        with mock.patch.dict("sys.modules", blocked):
            with self.assertRaises(ImportError):
                mm.balance(self._df(), n=4)


# ---------------------------------------------------------------------------
#  predict_class_prob: empirical rules, subclassing, reconstruction
# ---------------------------------------------------------------------------


class TestPredictClassProbBranches(unittest.TestCase):

    def test_rules_subclasses_and_reconstruction(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, printed = _silent(mm.predict_class_prob, ANALYSES.copy(), n_iterations=5,
                                   return_recon_oxides=True, seed=42)
        pred = dict(zip(out["Sample"], out["Predict_Mineral"]))
        self.assertEqual(pred["ol"], "Olivine")
        self.assertEqual(pred["plag"], "Plagioclase")
        self.assertEqual(pred["cpx"], "Clinopyroxene")
        self.assertEqual(pred["qz"], "SiO2_Polymorph")
        self.assertEqual(pred["zrc"], "Zircon")
        self.assertEqual(pred["cc"], "Carbonate")
        self.assertTrue(pd.isna(pred["blank"]))
        self.assertIn("classified by neural network", printed)
        for ox in OXIDES:
            self.assertIn(f"{ox}_recon", out.columns)
        nn_rows = out["Sample"].isin(["ol", "plag", "cpx"])
        self.assertTrue(out.loc[nn_rows, "SiO2_recon"].notna().all())
        self.assertTrue(out.loc[~nn_rows, "SiO2_recon"].isna().all())

    def test_raw_data_with_absent_columns_skips_prep_df(self):
        raw = pd.DataFrame([
            dict(SiO2=51.5, TiO2=0.8, Al2O3=3.0, FeOt=9.0, MgO=15.5, CaO=19.0, Na2O=0.4),
            dict(SiO2=52.0, Al2O3=30.0, FeOt=0.5, CaO=13.0, Na2O=4.0, K2O=0.1),
        ])                                          # no MnO, Cr2O3 or P2O5 columns
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, _ = _silent(mm.predict_class_prob, raw, n_iterations=5, seed=42,
                             verbose=False)
        self.assertEqual(list(out["Predict_Mineral"]), ["Clinopyroxene", "Plagioclase"])
        self.assertTrue(any("were missing" in str(w.message) for w in caught))

    def test_deprecated_wrapper(self):
        with self.assertWarns(DeprecationWarning):
            out, _ = _silent(mm.predict_class_prob_nnwr, ANALYSES.iloc[:1].copy(),
                             n_iterations=2, verbose=False)
        self.assertEqual(out["Predict_Mineral"].iat[0], "Olivine")


# ---------------------------------------------------------------------------
#  load_hybrid_checkpoint
# ---------------------------------------------------------------------------


class TestLoadHybridCheckpoint(unittest.TestCase):

    def setUp(self):
        self.model, self.ckpt, self.config = mm.load_hybrid_checkpoint(device="cpu")
        self.tmp = TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self, ckpt):
        path = os.path.join(self.tmp.name, "ckpt.pt")
        torch.save(ckpt, path)
        return path

    def test_default_load_is_eval(self):
        self.assertFalse(self.model.training)
        self.assertIn("model_state_dict", self.ckpt)
        self.assertTrue(self.config)

    def test_eval_mode_off(self):
        model, _, _ = mm.load_hybrid_checkpoint(device="cpu", eval_mode=False)
        self.assertTrue(model.training)

    def test_missing_state_or_config_raises(self):
        no_state = {k: v for k, v in self.ckpt.items() if k != "model_state_dict"}
        with self.assertRaises(KeyError):
            mm.load_hybrid_checkpoint(model_path=self._save(no_state), device="cpu")
        empty_cfg = dict(self.ckpt, model_config={})
        with self.assertRaises(KeyError):
            mm.load_hybrid_checkpoint(model_path=self._save(empty_cfg), device="cpu")

    def test_non_strict_warns_about_missing_keys(self):
        state = dict(self.ckpt["model_state_dict"])
        state.pop(next(iter(state)))
        path = self._save(dict(self.ckpt, model_state_dict=state))
        with self.assertWarns(UserWarning):
            mm.load_hybrid_checkpoint(model_path=path, device="cpu", strict=False)

    def test_optimizer_state_restored_from_either_key(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=0.123)
        for key in ("optimizer_state_dict", "optimizer"):
            path = self._save(dict(self.ckpt, **{key: opt.state_dict()}))
            fresh = mm.build_model_from_config(self.config, device="cpu")
            fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1.0)
            mm.load_hybrid_checkpoint(model_path=path, device="cpu", optimizer=fresh_opt)
            self.assertAlmostEqual(fresh_opt.param_groups[0]["lr"], 0.123)


# ---------------------------------------------------------------------------
#  plot_latent_space (real bundled model and reference latents)
# ---------------------------------------------------------------------------


class TestPlotLatentSpaceBranches(unittest.TestCase):

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pred, _ = _silent(mm.predict_class_prob, ANALYSES.iloc[:3].copy(),
                                   n_iterations=2, seed=42)

    def tearDown(self):
        plt.close("all")

    def test_labels_rollup_oxide_and_unmapped_warning(self):
        df = pd.concat([self.pred, self.pred.iloc[:2]], ignore_index=True)
        df.loc[3, "Predict_Mineral"] = "Oxide"
        df.loc[3, "Submineral"] = "Spinel_Group"
        df.loc[4, "Predict_Mineral"] = "Zircon"
        df = pd.concat([df, df.iloc[[0]].assign(Predict_Mineral="Unobtainium")],
                       ignore_index=True)
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "z2.png")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mm.plot_latent_space(df, filename=path,
                                     ref_kws={"color": {"Olivine": "red"}},
                                     new_kws={"color": "black"})
            self.assertTrue(os.path.exists(path))
        msgs = " ".join(str(w.message) for w in caught)
        self.assertIn("Empirical labels", msgs)
        self.assertIn("Unobtainium", msgs)

    def test_integer_labels_and_colour_overrides(self):
        df = self.pred.assign(label_id=[0, 1, 2])
        mm.plot_latent_space(df, label_column="label_id",
                             ref_kws={"color": "grey"},
                             new_kws={"color": {"Olivine": "blue"}})
        self.assertEqual(len(plt.get_fignums()), 1)        # drawn, left open by plt.show()

    def test_missing_label_column_raises(self):
        with self.assertRaises(KeyError):
            mm.plot_latent_space(self.pred.drop(columns="Predict_Mineral"))

    def test_deprecated_overlay_wrapper(self):
        with self.assertWarns(DeprecationWarning):
            mm.plot_z2_overlay(self.pred)


class TestDeprecatedLoaders(unittest.TestCase):

    def test_load_minclass_nn(self):
        with self.assertWarns(DeprecationWarning):
            classes = mm.load_minclass_nn()
        self.assertEqual(list(classes), list(mm.load_mineral_classes()))


if __name__ == "__main__":
    unittest.main()