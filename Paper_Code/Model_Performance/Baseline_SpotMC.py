# %%
"""
Text S5 'Analytical Noise': Monte Carlo of the MH0811b spot analyses (each
oxide drawn with the spot's reported Oxide % Sigma at 1-4x, negatives clipped,
no renormalisation) and of the stored map sites (1x1/3x3/5x5 means plus each
phase's spot sigma at 0-4x, 50 draws), classified by mineralML and the
baselines. Prints the tables; writes nothing.
"""
from Baseline_Models import *
from scipy.ndimage import uniform_filter


def read_aztec_blocks(raw):
    rows, name, hdr = [], None, None
    for _, r in raw.iterrows():
        a = r.iloc[0]
        if isinstance(a, str) and a not in ("Element", "Total") and pd.isna(r.iloc[2]):
            name = a.strip(); continue
        if a == "Element":
            hdr = list(r.values); continue
        if name is None or hdr is None or not isinstance(a, str) or a in ("O", "Total"):
            continue
        rec = dict(zip(hdr, r.values))
        ox = "FeOt" if rec["Oxide"] == "FeO" else rec["Oxide"]
        rows.append(dict(Analysis=name, Oxide=ox, wt=float(rec["Oxide %"]), sigma=float(rec["Oxide % Sigma"])))
    return pd.DataFrame(rows)


box = read_aztec_blocks(pd.read_csv(f"{REPO}/Paper_Code/AZTEC/AZTEC_boxpx.csv", header=None, encoding="utf-8-sig"))
box = box[box.Oxide.isin(OXIDES)]
SPOT = {"MH0811_plag": "PLAG", "MH0811_opx": "OPX", "MH0811_cpx": "CPX",
        "MH0811_spinel": "SPL", "MH0811_rhombox": "RHOMB OX", "Bii_amphibole": "AMPH"}
box = box[box.Analysis.isin(SPOT)].assign(Sample=lambda d: d.Analysis.map(SPOT))
mu = box.pivot_table(index="Sample", columns="Oxide", values="wt").reindex(columns=OXIDES).fillna(0)
sig = box.pivot_table(index="Sample", columns="Oxide", values="sigma").reindex(columns=OXIDES).fillna(0)
print(mu.round(2).to_string())
TRUE = {"PLAG": "Plagioclase", "OPX": "Orthopyroxene", "CPX": "Clinopyroxene",
        "SPL": "Spinel_Group", "RHOMB OX": "Rhombohedral_Oxides"}

f, _ = fit_baselines()
MODELS = {"RF grouped": f["Random Forest (tuned)"], "GNB grouped": f["Gaussian Naive Bayes"]}


def classify(D):
    out = {k: bl.predict_raw(D).astype(str).to_numpy() for k, bl in MODELS.items()}
    o = mm.predict_class_prob(D[OXIDES], n_iterations=50, seed=SEED, verbose=False)
    lab = o["Predict_Mineral"].astype(object)
    lab[lab == "Oxide"] = o.loc[lab == "Oxide", "Submineral"]
    out["mineralML"] = lab.astype(str).to_numpy()
    return out


rng = np.random.default_rng(42)
spots = [s for s in TRUE]
rows = []
for k in [1, 2, 3, 4]:
    d = rng.normal(mu.loc[spots].to_numpy()[:, None, :], k * sig.loc[spots].to_numpy()[:, None, :],
                   size=(len(spots), 1000, len(OXIDES)))
    D = pd.DataFrame(np.clip(d, 0, None).reshape(-1, len(OXIDES)), columns=OXIDES)
    D["Sample"] = np.repeat(spots, 1000)
    for m, lab in classify(D).items():
        for s in spots:
            sel = (D.Sample == s).to_numpy()
            wrong = pd.Series(lab[sel][lab[sel] != TRUE[s]]).value_counts()
            rows.append({"Sigma": k, "Sample": s, "Model": m, "pct_correct": 100 * (lab[sel] == TRUE[s]).mean(),
                         "top_wrong": f"{wrong.index[0]} ({100*wrong.iloc[0]/sel.sum():.0f}%)" if len(wrong) else ""})
spot = pd.DataFrame(rows)
print(spot.pivot_table(index=["Sample", "Model"], columns="Sigma", values="pct_correct").round(1).to_string())
print(spot[spot.top_wrong != ""].to_string(index=False))

# ---- map regions with spot sigma ---------------------------------------------
mp = np.load(f"{REPO}/Paper_Code/AZTEC/AZTEC_MC_MapSites.npz")
sites = pd.DataFrame(mp["sites"])
reg = []
for smp in ["MH0811b", "Bii"]:
    s = sites[sites.Map == smp]
    for b in [1, 3, 5]:
        means = {ox: uniform_filter(mp[f"{ox}_{smp}"].astype(float), size=b, mode="nearest") for ox in OXIDES}
        reg.append(pd.DataFrame({ox: means[ox][s.Row, s.Col] for ox in OXIDES}).assign(
            Site=s.Site.values, Map=smp, Phase=s.Phase.values, Spot=s.Spot.values, Box=b))
reg = pd.concat(reg, ignore_index=True)
draws = [reg.assign(Sigma=0)]
for k in [1, 2, 3, 4]:
    d = rng.normal(reg[OXIDES].to_numpy()[:, None, :], k * sig.loc[reg.Spot].to_numpy()[:, None, :],
                   size=(len(reg), 50, len(OXIDES)))
    D = pd.DataFrame(np.clip(d, 0, None).reshape(-1, len(OXIDES)), columns=OXIDES)
    for c in ["Site", "Map", "Phase", "Spot", "Box"]:
        D[c] = np.repeat(reg[c].to_numpy(), 50)
    draws.append(D.assign(Sigma=k))
R = pd.concat(draws, ignore_index=True)
print(f"{len(R):,} region compositions")
lab = classify(R)
res = (R.assign(**{m: lab[m] == R.Phase.to_numpy() for m in lab})
       .groupby(["Map", "Phase", "Box", "Sigma"])[list(lab)].mean().mul(100).round(1))
mean = res.groupby(["Box", "Sigma"]).mean().round(1)
print(mean.to_string())
print("max |change from sigma 0| of the mean over minerals, per model:")
print((mean - mean.xs(0, level="Sigma").reindex(mean.index, level="Box")).abs().max().round(1).to_string())
