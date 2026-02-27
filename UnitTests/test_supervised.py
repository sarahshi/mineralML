import types
import unittest
from unittest import mock
from unittest.mock import patch
import numpy as np
import pandas as pd
from math import sqrt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import mineralML as mm

def _get_oxides():
    # Be tolerant to where OXIDES lives
    if hasattr(mm, "constants") and hasattr(mm.constants, "OXIDES"):
        return mm.constants.OXIDES
    if hasattr(mm, "OXIDES"):
        return mm.OXIDES
    raise AttributeError("Could not find OXIDES in mineralML.")

def tiny_loader(n=32, in_features=11, n_classes=4, batch=16):
    x = torch.randn(n, in_features)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)

def _fake_scaler_series(oxides):
    # Series with index=oxides as your norm_data_nn expects
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

    def test_load_minclass_nn(self):
        min_cat, mapping = mm.load_minclass_nn()

        # Robust checks (less brittle than fully hard-coding)
        self.assertIsInstance(min_cat, list)
        self.assertIsInstance(mapping, dict)
        # Keys are 0..N-1 and values match min_cat order
        self.assertEqual(sorted(mapping.keys()), list(range(len(min_cat))))
        self.assertEqual([mapping[i] for i in range(len(min_cat))], min_cat)

        # Sanity: core classes exist (names from your current mapping)
        required = {"Amphibole", "Pyroxene", "Garnet", "Olivine", "Spinels"}
        self.assertTrue(required.issubset(set(min_cat)))

    def test_prep_df_nn(self):
        df_cleaned = mm.prep_df_nn(self.df.copy())

        # No NaNs after cleaning (oxides filled with 0, Mineral preserved)
        self.assertEqual(int(df_cleaned.isnull().sum().sum()), 0)
        # self.assertEqual(df_cleaned.index.name, "SampleID")

        oxides = set(_get_oxides())
        # Required columns: all oxides + ZrO2 + Mineral
        expected_cols = oxides.union({"ZrO2", "Mineral"})
        self.assertTrue(expected_cols.issubset(set(df_cleaned.columns)))

    def test_norm_data_nn(self):
        # Prepare
        df_cleaned = mm.prep_df_nn(self.df.copy())
        oxides = _get_oxides()

        # Under test
        normalized_data = mm.norm_data_nn(df_cleaned)
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

    def test_unique_mapping_nn(self):
        # Use indices that exist in the CURRENT mapping (no more Clinopyroxene/Spinel)
        # Choose a small set: Amphibole(0), Pyroxene(15), Garnet(7), Olivine(14), Spinels(20)
        pred_class = np.array([0, 15, 7, 14, 20, 0, 7, 20])

        unique, valid_mapping = mm.unique_mapping_nn(pred_class)

        # Expected unique set (order not guaranteed)
        expected_unique = np.array([0, 7, 14, 15, 20])
        np.testing.assert_array_equal(np.sort(unique), np.sort(expected_unique))

        # Names from the **loaded** mapping (robust to future reorderings)
        _, mapping = mm.load_minclass_nn()
        expected_valid_mapping = {i: mapping[i] for i in expected_unique}
        self.assertEqual(valid_mapping, expected_valid_mapping)

    def test_class2mineral_nn(self):
        pred_class = np.array([0, 15, 7, 14, 20, 0, 7, 20])
        pred_mineral = mm.class2mineral_nn(pred_class)

        # Build expected labels from the current mapping
        _, mapping = mm.load_minclass_nn()
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


class TestMultiClassClassifier(unittest.TestCase):
    def test_forward_and_predict_shapes(self):
        model = mm.MultiClassClassifier(input_dim=11, classes=7, hidden_layer_sizes=[16, 8, 4], dropout_rate=0.0)
        x = torch.randn(5, 11)
        logits = model(x)
        self.assertEqual(logits.shape, (5, 7))
        pred = model.predict(x)
        self.assertEqual(pred.shape, (5,))
        self.assertTrue((pred >= 0).all() and (pred < 7).all())


class TestTrainNN(unittest.TestCase):
    def test_train_nn_runs_and_early_stops(self):
        in_features, n_classes = 10, 3
        model = mm.MultiClassClassifier(input_dim=in_features, classes=n_classes, hidden_layer_sizes=[8, 4], dropout_rate=0.0)
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        crit = nn.CrossEntropyLoss()
        train_loader = tiny_loader(n=48, in_features=in_features, n_classes=n_classes, batch=16)
        valid_loader = tiny_loader(n=48, in_features=in_features, n_classes=n_classes, batch=16)
        out = mm.train_nn(
            model=model,
            optimizer=opt,
            train_loader=train_loader,
            valid_loader=valid_loader,
            n_epoch=20,               # small
            criterion=crit,
            kl_weight_decay=0.1,      # small increments
            kl_decay_epochs=5,        # ramp quickly
            patience=3,               # force early stop quickly
        )
        train_out, valid_out, train_losses, valid_losses, best_valid, best_state = out
        # minimal sanity checks
        self.assertIsNotNone(best_state)
        self.assertGreater(len(train_losses), 0)
        self.assertGreater(len(valid_losses), 0)
        self.assertIsInstance(best_valid, float)


class TestPredictTrainLoop(unittest.TestCase):
    def test_predict_class_prob_nn_train_stats_shape(self):
        # Model that injects noise so std > 0
        class NoisyModel(nn.Module):
            def __init__(self, in_f=6, classes=5):
                super().__init__()
                self.fc = nn.Linear(in_f, classes)
            def forward(self, x):
                return self.fc(x) + torch.randn_like(self.fc(x))*0.01

        model = NoisyModel(in_f=6, classes=5)
        x = torch.randn(4, 6)
        mean, std = mm.predict_class_prob_nn_train(model, x, n_iterations=8)
        self.assertEqual(mean.shape, (4, 5))
        self.assertEqual(std.shape, (4, 5))
        self.assertTrue(np.allclose(mean.sum(axis=1), 1.0, atol=1e-5))


class TestPredictClassProbNN(unittest.TestCase):
    @patch("mineralML.hybrid.load_model", side_effect=lambda model, opt, path: None)  # no file I/O
    @patch("mineralML.hybrid.norm_data_nn")
    @patch("mineralML.hybrid.load_minclass_nn")
    @patch("mineralML.hybrid.class2mineral_nn",
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

        df.loc[df.index[zircon_rows], ["ZrO2", "SiO2"]] = [60.0, 30.0]
        df.loc[df.index[non_zircon_rows], ["SiO2", "TiO2"]] = [50.0, 1.0]

        p_norm.side_effect = lambda d, *args, **kwargs: np.zeros((d.shape[0], len(ox)), dtype=np.float32)

        out_df, prob = mm.predict_class_prob_nnwr(df, n_iterations=1)

        self.assertEqual(len(out_df), N)
        self.assertTrue({"Predict_Mineral", "Predict_Probability"}.issubset(out_df.columns))

        for i in zircon_rows:
            self.assertEqual(out_df.iloc[i]["Predict_Mineral"], "Zircon")
            self.assertEqual(float(out_df.iloc[i]["Predict_Probability"]), 1.0)

        self.assertIsInstance(prob, np.ndarray)
        self.assertEqual(prob.ndim, 2)
        self.assertEqual(prob.shape[0], len(non_zircon_rows))
        # self.assertGreater(prob.shape[1], 0)


class TestBalance(unittest.TestCase):
    def test_balance_groups_with_mocks(self):
        # Build minimal df with “special” and “other” classes
        ox = mm.constants.OXIDES
        rows = []
        def row(mineral):
            r = {c: 0.0 for c in ox}
            r["Mineral"] = mineral
            return r
        for mineral in ["Clinopyroxene", "Orthopyroxene", "Plagioclase", "KFeldspar",
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
        self.assertIn("Spinels", balanced["Mineral"].unique())
        # Glass handled (either present or empty frame)
        self.assertTrue("Glass" in balanced["Mineral"].unique() or "Glass" not in df["Mineral"].unique())


class TestConfusionMatrixDF(unittest.TestCase):
    def test_confusion_matrix_df_merges_and_shape(self):
        given = ["Magnetite", "Plagioclase", "Hematite", "Zircon"]
        pred  = ["Spinels",   "KFeldspar",  "Ilmenite", "Zircon"]
        cm = mm.confusion_matrix_df(given, pred)
        # Square with the fixed label set
        self.assertEqual(cm.shape[0], cm.shape[1])
        # Merge: Magnetite -> Spinels should contribute to Spinels column
        self.assertGreaterEqual(cm.loc["Spinels", "Spinels"], 1)
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


class TestNeuralNetworkDriver(unittest.TestCase):
    @patch("mineralML.supervised.balance")
    @patch("mineralML.supervised.train_nn")
    @patch("mineralML.supervised.save_model_nn")
    @patch("mineralML.supervised.classification_report")
    @patch("mineralML.supervised.np.savez")
    @patch("mineralML.supervised.MultiClassClassifier.load_state_dict", return_value=None)
    @patch("mineralML.supervised.torch.cuda.is_available", return_value=False)
    def test_grid_search_wiring_and_best_selection(
        self,
        _cuda_off,
        _p_ld,
        p_savez,
        p_class_report,
        p_save_model,
        p_train_nn,
        p_balance,
    ):
        # --- Arrange ---
        df = _toy_df(n=60)

        p_balance.side_effect = lambda d, n=1000: d

        # Fake train_nn returning different "best" losses per (call index)
        fake_states = [{"w": 1}, {"w": 2}, {"w": 3}, {"w": 4}]
        best_losses = [0.50, 0.40, 0.60, 0.35]  # lowest on 4th call
        def _train_return(*args, **kwargs):
            idx = p_train_nn.call_count - 1  # 0-based
            return (None, None, [1.0], [1.0], best_losses[idx], fake_states[idx])
        p_train_nn.side_effect = _train_return

        p_class_report.return_value = {"macro avg": {"f1-score": 0.5}}
        p_savez.return_value = None

        hls_list = [[8, 4], [16, 8]]
        kl_list = [0.01, 0.1]

        # --- Act ---
        best_state = mm.neuralnetwork(
            df=df,
            hls_list=hls_list,
            kl_weight_decay_list=kl_list,
            lr=1e-3,
            wd=1e-4,
            dr=0.1,
            ep=2,
            n=0.2,
            balanced=True,
        )

        # --- Assert ---
        p_balance.assert_called_once()
        self.assertEqual(p_train_nn.call_count, len(hls_list) * len(kl_list))
        self.assertEqual(best_state, fake_states[-1])
        self.assertTrue(p_save_model.called)
        args, kwargs = p_save_model.call_args
        self.assertIs(args[1], fake_states[-1])
        self.assertIsInstance(args[0], torch.optim.Optimizer)
        self.assertIsInstance(args[2], str)
        self.assertGreaterEqual(p_class_report.call_count, 2)
        self.assertGreaterEqual(p_savez.call_count, 3)

    @patch("mineralML.supervised.balance")
    @patch("mineralML.supervised.train_nn")
    @patch("mineralML.supervised.save_model_nn")
    @patch("mineralML.supervised.classification_report")
    @patch("mineralML.supervised.np.savez")
    @patch("mineralML.supervised.MultiClassClassifier.load_state_dict", return_value=None)
    @patch("mineralML.supervised.torch.cuda.is_available", return_value=False)
    def test_unbalanced_path_and_small_grid(
        self, _cuda_off, _p_ld, p_savez, p_class_report, p_save_model, p_train_nn, p_balance
    ):
        # Arrange: tiny grid, balanced=False (balance() should NOT be called)
        df = _toy_df(n=45)
        p_balance.side_effect = lambda d, n=1000: d
        p_train_nn.return_value = (None, None, [1.0], [1.0], 0.99, {"w": 99})
        p_class_report.return_value = {"macro avg": {"f1-score": 0.1}}

        best_state = mm.neuralnetwork(
            df=df,
            hls_list=[[8]],
            kl_weight_decay_list=[0.2],
            lr=1e-3,
            wd=1e-4,
            dr=0.1,
            ep=1,
            n=0.25,
            balanced=False,
        )

        # Assert
        p_balance.assert_not_called()
        self.assertEqual(best_state, {"w": 99})
        p_train_nn.assert_called_once()
        self.assertGreaterEqual(p_savez.call_count, 2)


if __name__ == "__main__":
    unittest.main()
