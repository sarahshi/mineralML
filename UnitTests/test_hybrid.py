import types
import os
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
        fe_conv = 159.688 / (2 * 71.8464)
        expected = fe2o3_val / fe_conv
        df = pd.DataFrame({"Fe2O3": [fe2o3_val], "SiO2": [50.0]})
        out = mm.convert_fe_to_feot(df)
        self.assertAlmostEqual(out["FeOt"].iloc[0], expected, places=4)
        self.assertNotIn("Fe2O3", out.columns)

    def test_feo_plus_fe2o3_summed(self):
        fe_conv = 159.688 / (2 * 71.8464)
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


if __name__ == "__main__":
    unittest.main()