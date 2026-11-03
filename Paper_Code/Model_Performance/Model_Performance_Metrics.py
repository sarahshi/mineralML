# %%
"""
Per-class precision / recall / F1 for the shipped hybrid NN (v0030) on the
held-out validation split, secondary standards, GEOROC, and EDS maps scored
against EBSD. Classes resolved by the empirical rules are flagged `empirical`.
Writes Model_Performance_{perclass,overall}.csv for the supplement figure (Supp_Model_Performance.py)
and GEOROC_truth_pred.csv.gz for the main-text confusion matrix.
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '../../src')
import torch
torch.set_num_threads(8)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mineralML as mm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

mm.same_seeds(42)
SEED = 42
EXCLUDE = {"Zircon", "Carbonate", "SiO2_Polymorph"}

OXIDE_MAP = {"Hematite": "Rhombohedral_Oxides", "Ilmenite": "Rhombohedral_Oxides",
             "Magnetite": "Spinel_Group", "Spinel": "Spinel_Group"}
FELDSPAR_MAP = {f: "Feldspar" for f in ["Plagioclase", "Alkali_Feldspar", "Feldspar_Miscibility_Gap"]}
# reference names for the phases the empirical rules emit as one class
EMPIRICAL_MAP = {**{c: "Carbonate" for c in [
    "Calcite", "Dolomite", "Ankerite", "Magnesite", "Siderite", "Strontianite", "Witherite",
    "Rhodochrosite", "Kutnahorite", "Alstonite", "Barytocalcite", "Norsethite", "Breunnerite",
    "Magnesiosiderite", "Pirssonite", "Shortite", "Nyerereite", "Gregoryite", "Olekminskite"]},
    **{c: "SiO2_Polymorph" for c in ["Quartz", "Quartz-new", "Cristobalite", "Tridymite", "Opal"]}}
TRUTH_MAP = {**OXIDE_MAP, **EMPIRICAL_MAP}

per_rows, overall_rows = [], []


def nn_labels(d):
    out = mm.predict_class_prob(d, n_iterations=250, seed=SEED, verbose=False)
    lab = out["Predict_Mineral"].copy()
    oxide = lab == "Oxide"
    lab[oxide] = out.loc[oxide, "Submineral"]
    return lab.astype(str).replace(OXIDE_MAP)


def metrics(t, q, labels, dataset):
    pr, rc, f1, sup = precision_recall_fscore_support(t, q, labels=labels, zero_division=0)
    per_rows.append(pd.DataFrame({"dataset": dataset, "mineral": labels, "precision": 100*pr,
                                  "recall": 100*rc, "f1": 100*f1, "support": sup,
                                  "empirical": [c in EXCLUDE for c in labels]}))
    overall_rows.append({"dataset": dataset, "n": len(t),
                         "accuracy": 100*accuracy_score(t, q),
                         "balanced_accuracy": 100*balanced_accuracy_score(t, q),
                         "macro_precision": 100*pr.mean(), "macro_recall": 100*rc.mean(),
                         "macro_f1": 100*f1.mean(), "n_classes": len(labels)})


def score(df, dataset, collapse_feldspar=False):
    d = mm.prep_df(df, verbose=False)
    d["Mineral"] = d["Mineral"].astype(str).replace(TRUTH_MAP)
    keep, pred_map = classes | EXCLUDE, {"Na-Pyroxene": "Clinopyroxene"}
    if collapse_feldspar:
        d["Mineral"] = d["Mineral"].replace(FELDSPAR_MAP)
        keep = {FELDSPAR_MAP.get(c, c) for c in keep}
        pred_map.update(FELDSPAR_MAP)
    d = d[d["Mineral"].isin(keep)].reset_index(drop=True)
    q = nn_labels(d).replace(pred_map)
    m = q.notna().to_numpy()  # drop analyses the network could not classify
    t, q = d["Mineral"].to_numpy()[m], q.to_numpy()[m]
    metrics(t, q, sorted(set(t)), dataset)
    return t, q


# %% Held-out validation, secondary standards, GEOROC

train_df, valid_df = train_test_split(pd.read_csv("../../Training_Data/min_df_v4_synth.csv", low_memory=False),
                                      test_size=0.20, random_state=SEED)
classes = set(train_df["Mineral"].astype(str).replace(OXIDE_MAP)) - EXCLUDE

score(valid_df.copy(), "Held-out validation")
score(pd.read_csv("../../Validation_Data/Standard_Compilation_July24.csv"), "Secondary standards")
# GEOROC labels cannot resolve plagioclase vs alkali feldspar
t, q = score(pd.read_csv("../../Validation_Data/GEOROC_validationdata_Fe_202602.csv"), "GEOROC", collapse_feldspar=True)
# every GEOROC truth/prediction pair, for the main-text confusion matrix (Main_GEOROC_Confusion.py)
pd.DataFrame({"truth": t, "pred": q}).to_csv("GEOROC_truth_pred.csv.gz", index=False)

# %% EDS maps scored per pixel against EBSD

SAMPLES = {
    "MH0811b": dict(
        map_dir="../../EDS_EBSD_Maps/MountHood_MH0811b",
        ctf="../../EDS_EBSD_Maps/EBSD_CTF/MH0811b_EBSD_EDS.ctf",
        merge={"Andesine": "Plagioclase", "Orthoclase": "Alkali_Feldspar",
               "Augite": "Clinopyroxene", "Enstatite": "Orthopyroxene",
               "Magnetite": "Spinel_Group", "Ilmenite": "Rhombohedral_Oxides",
               "Quartz-new": "SiO2_Polymorph", "Cristobalite": "SiO2_Polymorph"}),
    "Bii": dict(
        map_dir="../../EDS_EBSD_Maps/Tuolumne_Bii",
        ctf="../../EDS_EBSD_Maps/EBSD_CTF/Bii_EBSD_EDS.ctf",
        merge={"Anorthite": "Plagioclase", "Andesine": "Plagioclase",
               "Orthoclase": "Alkali_Feldspar", "Hornblende": "Amphibole",
               "Augite": "Clinopyroxene", "Enstatite": "Orthopyroxene",
               "Magnetite": "Spinel_Group", "Ilmenite": "Rhombohedral_Oxides",
               "Quartz-new": "SiO2_Polymorph"}),
}
NON_CRYST = ["Unindexed", "Glass", "Vesicles", "nan", "None", "NaN"]
MIN_SUPPORT = 30  # truth pixels needed to score a class

for sample, cfg in SAMPLES.items():
    res = mm.run_map(cfg["map_dir"], units="element_wt%", pred_score_threshold=0,
                     renormalize=True, total_threshold=None,
                     remove_islands_flag=True, fill_holes_flag=True, show=False)
    shape = res["mineral_map"].shape
    pred = nn_labels(res["df_pred"])

    _, eb, *_ = mm.plot_ctf_phases(cfg["ctf"], rename_dict=cfg["merge"], title=None, legend_on=False)
    plt.close("all")
    eb = np.asarray(eb, dtype=object)
    h, w = min(eb.shape[0], shape[0]), min(eb.shape[1], shape[1])
    ebf = pd.Series(eb[:h, :w].ravel()).replace(cfg["merge"]).astype(str).replace(TRUTH_MAP)
    pc = pd.Series(np.asarray(pred.reindex(range(shape[0]*shape[1]))).reshape(shape)[:h, :w].ravel())

    m = (ebf.notna() & ~ebf.isin(NON_CRYST) & pc.notna()).to_numpy()
    t, q = ebf.to_numpy()[m], pc.to_numpy()[m]
    name = f"EDS map {sample}"

    # pooled feldspar: EBSD has no counterpart for Feldspar_Miscibility_Gap
    tp, qp = pd.Series(t).replace(FELDSPAR_MAP), pd.Series(q).replace(FELDSPAR_MAP)
    vcp = tp.value_counts()
    metrics(tp.to_numpy(), qp.to_numpy(), sorted(vcp[vcp >= MIN_SUPPORT].index), f"{name}_pool")
    vc = pd.Series(t).value_counts()
    metrics(t, q, sorted(vc[vc >= MIN_SUPPORT].index), name)

# %%
pd.concat(per_rows, ignore_index=True).to_csv("Model_Performance_perclass.csv", index=False)
overall = pd.DataFrame(overall_rows)
overall.to_csv("Model_Performance_overall.csv", index=False)
print(overall.round(2).to_string(index=False))
