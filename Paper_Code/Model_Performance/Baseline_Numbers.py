# %%
"""
Every Text S5 number comparing mineralML with the baselines, on the held-out
split, the secondary standards, GEOROC and the two EDS-EBSD maps.

mineralML: held-out, standards and maps run as in Model_Performance_Metrics.py
(n_iterations=250, seed=42); GEOROC read from Model_Performance_*.csv. ECE: 15
equal-width bins on the top score, rows with a score only.

Writes Baseline_Numbers.json, Baseline_perclass.csv (GEOROC) and
Baseline_labelmaps_<sample>.npz.
"""
import os, json, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__))
from Baseline_Models import *
from mineralML.mapping import maps_to_df, _ensure_columns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NN_ITERS = 250
OUT = {}


def nn_run(d):
    o = mm.predict_class_prob(d, n_iterations=NN_ITERS, seed=SEED, verbose=False)
    lab = o["Predict_Mineral"].astype(object)
    m = lab == "Oxide"
    lab.loc[m] = o.loc[m, "Submineral"]
    return lab, o["Prediction_Score"]


def bl_run(bl, d):
    """Baseline labels plus the classifier's top probability (NaN where the
    empirical filter decided)."""
    lab = bl.labels(d)
    ox_all = oxide_frame(d)
    emp, invalid = empirical_rules(ox_all)
    to_model = (ox_all[OXIDES].sum(axis=1) > 0) & ~invalid & emp.isna()
    conf = pd.Series(np.nan, index=d.index)
    conf[to_model] = bl.clf.predict_proba(
        bl.ss.transform(d.loc[to_model, OXIDES].fillna(0).to_numpy(float))).max(axis=1)
    return lab, conf


def clean(lab):
    p = merge_oxides(pd.DataFrame({"Mineral": lab}))["Mineral"]
    return merge_pyroxene(p).where(lab.notna())


def ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    return float(sum((b == k).mean() * abs(correct[b == k].mean() - conf[b == k].mean())
                     for k in range(n_bins) if (b == k).any()))


fitted, classes = fit_baselines()
MODELS = {"RF grouped": fitted["Random Forest (tuned)"], "GNB grouped": fitted["Gaussian Naive Bayes"]}
NAMES = ["mineralML"] + list(MODELS)


def run_all(d):
    L, C = {}, {}
    L["mineralML"], C["mineralML"] = nn_run(d)
    for k, bl in MODELS.items():
        L[k], C[k] = bl_run(bl, d)
    return {k: clean(v) for k, v in L.items()}, C


# %% ---- held-out split: accuracy and ECE --------------------------------------
_df = pd.read_csv(f"{REPO}/Training_Data/min_df_v4_synth.csv", low_memory=False)
_, valid_df = train_test_split(_df, test_size=0.20, random_state=SEED)
vd = map_empirical_labels(merge_oxides(mm.prep_df(valid_df.copy(), verbose=False)))
vd = vd.loc[vd["Mineral"].isin(set(classes) | EXCLUDE)].reset_index(drop=True)
L, C = run_all(vd)
truth = vd["Mineral"].astype(str)
OUT["heldout"] = {}
for k in NAMES:
    m = L[k].notna()
    acc = 100 * accuracy_score(truth[m], L[k][m].astype(str))
    s = m & C[k].notna()
    e = ece(C[k][s].to_numpy(float), (L[k][s].astype(str) == truth[s]).to_numpy())
    OUT["heldout"][k] = {"accuracy": acc, "ece": e}
    print(f"held-out {k:<14} acc {acc:6.2f}  ECE {e:.4f}")

# %% ---- secondary standards: accuracy, balanced accuracy, per-class recall ---
raw = pd.read_csv(f"{REPO}/Validation_Data/Standard_Compilation_July24.csv")
sd = map_empirical_labels(merge_oxides(mm.prep_df(raw, verbose=False)))
sd = sd.loc[sd["Mineral"].isin(set(classes) | EXCLUDE)].reset_index(drop=True)
L, C = run_all(sd)
truth = sd["Mineral"].astype(str)
OUT["standards"] = {}
for k in NAMES:
    m = L[k].notna()
    t, q = truth[m], L[k][m].astype(str)
    rec = {c: 100 * (q[t == c] == c).mean() for c in sorted(t.unique())}
    wrong = {c: q[(t == c) & (q != c)].value_counts().head(3).to_dict() for c in rec if rec[c] < 100}
    OUT["standards"][k] = {"n": int(m.sum()), "accuracy": 100 * accuracy_score(t, q),
                           "balanced": 100 * balanced_accuracy_score(t, q),
                           "recall": rec, "wrong": wrong}
    print(f"standards {k:<14} acc {OUT['standards'][k]['accuracy']:6.2f} "
          f"bal {OUT['standards'][k]['balanced']:6.2f}  wrong {wrong}")
ol = _df[_df.Mineral == "Olivine"]
OUT["olivine_train_FeOt_MgO"] = [float(ol.FeOt.mean()), float(ol.FeOt.std()),
                                 float(ol.MgO.mean()), float(ol.MgO.std())]

# %% ---- GEOROC: baselines scored per class, feldspar pooled ---------------------
g = map_empirical_labels(merge_oxides(mm.prep_df(pd.read_csv(
    f"{REPO}/Validation_Data/GEOROC_validationdata_Fe_202602.csv", low_memory=False), verbose=False)))
g["Mineral"] = merge_feldspar(g["Mineral"])
g = g.loc[g["Mineral"].isin(set(merge_feldspar(pd.Series(classes))) | EXCLUDE)].reset_index(drop=True)
nn_ov = pd.read_csv(os.path.join(HERE, "Model_Performance_overall.csv")).set_index("dataset")
OUT["georoc"] = {"mineralML": float(nn_ov.loc["GEOROC", "accuracy"])}
per_rows = []
for name, bl in fitted.items():
    p = merge_oxides(pd.DataFrame({"Mineral": bl.labels(g)}))["Mineral"]
    p = merge_feldspar(merge_pyroxene(p).where(p.notna()))
    m = p.notna().to_numpy()
    t, q = g["Mineral"].to_numpy()[m], p[m].to_numpy()
    labels = sorted(set(t))
    pr, rc, f1, sup = precision_recall_fscore_support(t, q, labels=labels, zero_division=0)
    per_rows.append(pd.DataFrame({"dataset": "GEOROC", "model": name, "mineral": labels,
                                  "precision": 100*pr, "recall": 100*rc, "f1": 100*f1, "support": sup,
                                  "empirical": [c in EXCLUDE for c in labels]}))
    OUT["georoc"][f"{name} grouped"] = 100 * accuracy_score(t, q)
per = pd.concat(per_rows, ignore_index=True)
per.to_csv(os.path.join(HERE, "Baseline_perclass.csv"), index=False)

nn = pd.read_csv(os.path.join(HERE, "Model_Performance_perclass.csv"))
nn = nn[nn.dataset == "GEOROC"].set_index("mineral")
rf = per[per.model == "Random Forest (tuned)"].set_index("mineral")
learned = ~nn.empirical.astype(bool)
for met in ("precision", "recall"):
    dlt = (nn[met] - rf[met])[learned]
    OUT["georoc"][f"{met} counts grouped"] = {
        "n_phases": int(len(dlt)), "mML_higher": int((dlt >= 0.5).sum()),
        "RF_higher": int((dlt <= -0.5).sum()), "within_0.5": int((dlt.abs() < 0.5).sum())}
print(OUT["georoc"])

# %% ---- EDS-EBSD maps ---------------------------------------------------------
SAMPLES = {
    "MH0811b": ("EDS_EBSD_Maps/MountHood_MH0811b", "EDS_EBSD_Maps/EBSD_CTF/MH0811b_EBSD_EDS.ctf",
                {"Andesine": "Plagioclase", "Orthoclase": "Alkali_Feldspar",
                 "Augite": "Clinopyroxene", "Enstatite": "Orthopyroxene",
                 "Magnetite": "Spinel_Group", "Ilmenite": "Rhombohedral_Oxides",
                 "Quartz-new": "SiO2_Polymorph", "Cristobalite": "SiO2_Polymorph"}),
    "Bii": ("EDS_EBSD_Maps/Tuolumne_Bii", "EDS_EBSD_Maps/EBSD_CTF/Bii_EBSD_EDS.ctf",
            {"Anorthite": "Plagioclase", "Andesine": "Plagioclase",
             "Orthoclase": "Alkali_Feldspar", "Hornblende": "Amphibole",
             "Augite": "Clinopyroxene", "Enstatite": "Orthopyroxene",
             "Magnetite": "Spinel_Group", "Ilmenite": "Rhombohedral_Oxides",
             "Quartz-new": "SiO2_Polymorph"}),
}
NON_CRYST = ["Unindexed", "Glass", "Vesicles", "nan", "None", "NaN"]
FSP = {"Plagioclase": "Feldspar", "Alkali_Feldspar": "Feldspar",
       "Feldspar_Miscibility_Gap": "Feldspar", "Feldspar": "Feldspar"}
OUT["maps"] = {}
for sample, (mdir, ctf, merge) in SAMPLES.items():
    with contextlib.redirect_stdout(io.StringIO()):
        ox_maps = mm.load_maps_from_dir(f"{REPO}/{mdir}", units="element_wt%", renormalize=True)
        raw_maps = mm.load_maps_from_dir(f"{REPO}/{mdir}", units="element_wt%", renormalize=False)
        flat, shape = maps_to_df(ox_maps)
        flat_raw, _ = maps_to_df(raw_maps)
        X = _ensure_columns(flat, expected=list(OXIDES) + ["ZrO2"])
        Xraw = _ensure_columns(flat_raw, expected=list(OXIDES) + ["ZrO2"])
        _f, eb, *_ = mm.plot_ctf_phases(f"{REPO}/{ctf}", rename_dict=merge, title=None, legend_on=False)
    plt.close("all")
    L, _ = run_all(X)
    eb = np.asarray(eb, dtype=object)
    h, w = min(eb.shape[0], shape[0]), min(eb.shape[1], shape[1])
    ebs = map_empirical_labels(merge_oxides(pd.DataFrame(
        {"Mineral": pd.Series(eb[:h, :w].ravel()).replace(merge)})))["Mineral"].astype(str)
    eb2d = ebs.to_numpy().reshape(h, w)
    edge = np.zeros((h, w), bool)
    dx = eb2d[:, :-1] != eb2d[:, 1:]; edge[:, :-1] |= dx; edge[:, 1:] |= dx
    dy = eb2d[:-1, :] != eb2d[1:, :]; edge[:-1, :] |= dy; edge[1:, :] |= dy
    edge = edge.ravel()
    idx = np.arange(shape[0] * shape[1]).reshape(shape)[:h, :w].ravel()
    scoreable = (~ebs.isin(NON_CRYST)).to_numpy()
    P = {k: pd.Series(np.asarray(v.reindex(range(shape[0] * shape[1])), dtype=object)[idx]) for k, v in L.items()}
    np.savez_compressed(os.path.join(HERE, f"Baseline_labelmaps_{sample}.npz"), ebsd=eb2d, on_edge=edge.reshape(h, w),
                        **{f"pred::{k}": P[k].astype(str).to_numpy().reshape(h, w) for k in P})
    res = {"unindexed_pct": 100 * (~scoreable).mean(), "n_pixels": int(h * w)}
    for k, p in P.items():
        valid = p.notna().to_numpy()
        m = scoreable & valid
        t, q = ebs[m].to_numpy(), p[m].astype(str).to_numpy()
        tp, qp = pd.Series(t).replace(FSP).to_numpy(), pd.Series(q).replace(FSP).to_numpy()
        ok = tp == qp
        e = edge[m]
        vc = pd.Series(t).value_counts()
        rec = {c: 100 * (q[t == c] == c).mean() for c in vc[vc >= 30].index}
        r = {"n": int(m.sum()), "agreement": 100 * (t == q).mean(), "agreement_pool": 100 * ok.mean(),
             "balanced": 100 * balanced_accuracy_score(t, q), "balanced_pool": 100 * balanced_accuracy_score(tp, qp),
             "interior_error_pool": 100 * (~ok[~e]).mean(), "interior_error": 100 * (t != q)[~e].mean(),
             "boundary_share_of_errors_pool": 100 * e[~ok].mean(),
             "fsp_to_glass": int((np.isin(t, ["Plagioclase", "Alkali_Feldspar"]) & (q == "Glass")).sum()),
             "glass_pct_of_map": 100 * (p == "Glass").sum() / (h * w),
             "glass_pct_of_classified": 100 * (p[valid] == "Glass").mean(),
             "glass_on_crystalline": int((q == "Glass").sum()),
             "opx_to_amph": int(((t == "Orthopyroxene") & (q == "Amphibole")).sum()),
             "opx_to_cpx": int(((t == "Orthopyroxene") & (q == "Clinopyroxene")).sum()),
             "cpx_to_amph": int(((t == "Clinopyroxene") & (q == "Amphibole")).sum()),
             "amph_to_cpx": int(((t == "Amphibole") & (q == "Clinopyroxene")).sum()),
             "spinel_to_rhomb": int(((t == "Spinel_Group") & (q == "Rhombohedral_Oxides")).sum()),
             "recall": rec}
        res[k] = r
        print(f"{sample:<8}{k:<14} agree {r['agreement']:5.2f} pool {r['agreement_pool']:5.2f} "
              f"bal_pool {r['balanced_pool']:5.2f} int_err_pool {r['interior_error_pool']:5.2f}")
    OUT["maps"][sample] = res

    # ---- trace oxides: MH0811b interior orthopyroxene, raw exports --------------
    if sample == "MH0811b":
        io_ = (ebs == "Orthopyroxene").to_numpy() & ~edge
        R = Xraw.iloc[idx].reset_index(drop=True)[OXIDES]
        tr = {"interior_opx_mean_Na2O": float(R[io_].Na2O.mean()), "interior_opx_mean_K2O": float(R[io_].K2O.mean())}
        amph = io_ & (P["RF grouped"] == "Amphibole").to_numpy()
        opx = io_ & (P["RF grouped"] == "Orthopyroxene").to_numpy()
        med = pd.DataFrame({"amph": R[amph].median(), "opx": R[opx].median()})
        tr["RF grouped"] = {"n_amph": int(amph.sum()), "n_opx": int(opx.sum()),
                            "median_amph": R[amph].median().round(3).to_dict(), "median_opx": R[opx].median().round(3).to_dict(),
                            "total_amph": float(R[amph].sum(axis=1).median()), "total_opx": float(R[opx].sum(axis=1).median()),
                            "max_major_diff": float((med.amph - med.opx)[["SiO2", "Al2O3", "FeOt", "MgO", "CaO"]].abs().max()),
                            "mML_opx_on_these": 100 * float((P["mineralML"][amph] == "Orthopyroxene").mean())}
        OUT["trace"] = tr

# training-set trace-oxide thresholds
for col, thr in [("Na2O", 0.2), ("K2O", 0.1), ("K2O", 0.05)]:
    OUT.setdefault("train_thresholds", {})[f"{col}>{thr}"] = {
        m: 100 * float((_df.loc[_df.Mineral == m, col].fillna(0) > thr).mean())
        for m in ["Amphibole", "Orthopyroxene", "Clinopyroxene"]}

json.dump(OUT, open(os.path.join(HERE, "Baseline_Numbers.json"), "w"), indent=1, default=float)
