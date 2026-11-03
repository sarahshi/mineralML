"""
Random Forest and Gaussian Naive Bayes baselines for the comparison with
mineralML. Like mineralML, they are trained on grouped labels (Pyroxene,
Feldspar), and those predictions are split by mineralML's PyroxeneClassifier /
FeldsparClassifier. Imported by the Baseline_*.py scripts.

Run directly, it repeats the grouped five-fold cross-validation grid (folds
grouped by training data source) that selected TUNED, and prints the scores.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/GitHub/mineralML")
sys.path.insert(0, f"{REPO}/src")
import mineralML as mm
from mineralML.constants import OXIDES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

SEED = 42
EXCLUDE = {"Zircon", "Carbonate", "SiO2_Polymorph"}
# selected by the cross-validation grid at the bottom of this file
TUNED = dict(n_estimators=200, max_depth=None, min_samples_leaf=1,
             class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)


def merge_oxides(df):
    df = df.copy()
    df["Mineral"] = (df["Mineral"].astype(str)
        .replace(["Hematite", "Ilmenite"], "Rhombohedral_Oxides")
        .replace(["Magnetite", "Spinel"], "Spinel_Group"))
    return df


def merge_feldspar(s):
    return pd.Series(s).astype(str).replace(
        {"Plagioclase": "Feldspar", "Alkali_Feldspar": "Feldspar",
         "Feldspar_Miscibility_Gap": "Feldspar"})


def merge_pyroxene(s):
    return pd.Series(s).astype(str).replace({"Na-Pyroxene": "Clinopyroxene"})


def oxide_frame(d):
    cols = OXIDES + ["ZrO2"]
    o = d[[c for c in cols if c in d.columns]].copy()
    for c in cols:
        if c not in o.columns:
            o[c] = 0.0
    return o.fillna(0)


def empirical_rules(ox_all):
    si = ox_all.get("SiO2", pd.Series(0.0, index=ox_all.index))
    zr = ox_all.get("ZrO2", pd.Series(0.0, index=ox_all.index))
    total = ox_all.sum(axis=1)
    invalid = ((ox_all.fillna(0) != 0).sum(axis=1) < 1) | (total < 50)
    lab = pd.Series(np.nan, index=ox_all.index, dtype=object)
    lab[(si < 5) & (total < 70) & ~invalid] = "Carbonate"
    lab[(si > 90) & ~invalid] = "SiO2_Polymorph"
    lab[(zr > 50) & ~invalid] = "Zircon"
    lab[invalid] = np.nan
    return lab, invalid


CARBONATE_FAMILY = ["Calcite", "Dolomite", "Ankerite", "Magnesite", "Siderite",
    "Strontianite", "Witherite", "Rhodochrosite", "Kutnahorite", "Alstonite",
    "Barytocalcite", "Norsethite", "Breunnerite", "Magnesiosiderite",
    "Pirssonite", "Shortite", "Nyerereite", "Gregoryite", "Olekminskite"]
SIO2_FAMILY = ["Quartz", "Quartz-new", "Cristobalite", "Tridymite", "Opal"]


def map_empirical_labels(df):
    df = df.copy()
    ren = {c: "Carbonate" for c in CARBONATE_FAMILY}
    ren.update({c: "SiO2_Polymorph" for c in SIO2_FAMILY})
    df["Mineral"] = df["Mineral"].astype(str).replace(ren)
    return df


def resolve_groups(raw, lab):
    """What predict_class_prob does after its network: Pyroxene / Feldspar
    rows go through PyroxeneClassifier / FeldsparClassifier."""
    lab = pd.Series(lab, index=raw.index, dtype=object).copy()
    ox_cols = [c for c in OXIDES if c in raw.columns]
    for grp, Clf in (("Pyroxene", mm.PyroxeneClassifier),
                     ("Feldspar", mm.FeldsparClassifier)):
        m = (lab == grp).to_numpy()
        if m.any():
            sub = raw.loc[m, ox_cols].copy()
            sub["Predict_Mineral"] = grp
            lab.loc[m] = Clf(sub).classify(subclass=True)["Mineral"].to_numpy()
    return lab


def training_set():
    """mm.balance() on the 80% training split, and the classes scored."""
    df = pd.read_csv(f"{REPO}/Training_Data/min_df_v4_synth.csv", low_memory=False)
    tr, _ = train_test_split(df, test_size=0.20, random_state=SEED)
    classes = sorted(set(merge_oxides(tr)["Mineral"]) - EXCLUDE)
    bal = merge_oxides(mm.balance(tr.copy(), n=1000))
    return bal.loc[~bal["Mineral"].isin(EXCLUDE)].reset_index(drop=True), classes


class Baseline:
    """sklearn classifier, then the stoichiometric split of Pyroxene / Feldspar."""

    def __init__(self, clf, ss):
        self.clf, self.ss = clf, ss

    def predict_raw(self, X_df):
        """No empirical front end, for scripts that never applied one."""
        p = pd.Series(self.clf.predict(self.ss.transform(X_df[OXIDES].fillna(0).to_numpy(float))),
                      index=X_df.index, dtype=object)
        return resolve_groups(X_df, p)

    def labels(self, d):
        """Empirical front end, then the classifier."""
        ox_all = oxide_frame(d)
        ox = ox_all[OXIDES]
        emp, invalid = empirical_rules(ox_all)
        valid = (ox.sum(axis=1) > 0) & ~invalid
        to_model = valid & emp.isna()
        p = self.predict_raw(d.loc[to_model])
        return pd.concat([p, emp[valid & emp.notna()]]).reindex(d.index)


def fit_baselines():
    bal, classes = training_set()
    ss = StandardScaler().fit(bal[OXIDES].fillna(0))
    X = ss.transform(bal[OXIDES].fillna(0))
    y = bal["Mineral"].to_numpy()
    return {"Random Forest (tuned)": Baseline(RandomForestClassifier(**TUNED).fit(X, y), ss),
            "Gaussian Naive Bayes": Baseline(GaussianNB().fit(X, y), ss)}, classes


if __name__ == "__main__":
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score

    bal, _ = training_set()
    X = StandardScaler().fit_transform(bal[OXIDES].fillna(0))
    classes = sorted(set(bal["Mineral"]))
    y = bal["Mineral"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    src = bal["Source"]
    groups = src.where(src.notna() & (src.astype(str) != "nan"),
                       pd.Series([f"synthetic_{i}" for i in range(len(src))])).astype(str).to_numpy()
    cv = list(GroupKFold(n_splits=5).split(X, y, groups))

    best = (-1, None)
    for n_est in (200, 500):
        for max_depth in (None, 20):
            for min_leaf in (1, 2):
                for cw in (None, "balanced_subsample"):
                    kw = dict(n_estimators=n_est, max_depth=max_depth, min_samples_leaf=min_leaf,
                              class_weight=cw, n_jobs=-1, random_state=SEED)
                    score = np.mean([balanced_accuracy_score(y[te], RandomForestClassifier(**kw)
                                     .fit(X[tr], y[tr]).predict(X[te])) for tr, te in cv])
                    print(f"n={n_est} depth={max_depth} leaf={min_leaf} cw={cw}: {100*score:.2f}")
                    if score > best[0]:
                        best = (score, kw)
    print(f"best (CV balanced accuracy {100*best[0]:.2f}):", best[1])
