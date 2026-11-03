# %%
"""
Map-region test of AZTEC_SinglePx_Figure.py.

Sites are those stored in AZTEC_MC_MapSites.npz plus up to 200 each of Bii
titanite, epidote and alkali feldspar, every one centred in a 7x7 window of a
single EBSD phase. Each site is classified as its real 1x1, 3x3 and 5x5 mean,
and as its 5x5 mean plus N(0, sigma) per oxide (50 draws, negatives clipped
to 0), sigma from AZtec's single pixel of that phase, either as drawn
("independent") or rescaled to the 5x5 total ("closure").

RF and GNB are trained on mm.balance()'s grouped labels, and their Pyroxene /
Feldspar predictions are split as mineralML's are.

Writes AZTEC_SinglePx_Results.npz: sites, scatter, real, regions.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.expanduser("~/GitHub/mineralML")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, f"{REPO}/Paper_Code/Model_Performance")  # for Baseline_Models
from Baseline_Models import resolve_groups
import numpy as np, pandas as pd
import mineralML as mm
from mineralML.constants import OXIDES
from scipy.ndimage import binary_erosion, uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

SEED, N_DRAWS = 42, 50
rng = np.random.default_rng(SEED)
MODELS = ["mineralML", "RF", "GNB"]
pd.set_option("display.width", 200)

PX = {  # AZtec single-pixel analysis -> (map, EBSD phase)
    "MH0811b_opx": ("MH0811b", "Orthopyroxene"), "MH0811b_cpx": ("MH0811b", "Clinopyroxene"),
    "MH0811b_plag": ("MH0811b", "Plagioclase"), "MH0811b_spinel": ("MH0811b", "Spinel_Group"),
    "MH0811b_rhombox": ("MH0811b", "Rhombohedral_Oxides"),
    "Bii_amph": ("Bii", "Amphibole"), "Bii_plag": ("Bii", "Plagioclase"),
    "Bii_spinel": ("Bii", "Spinel_Group"), "Bii_titanite": ("Bii", "Titanite"),
    "Bii_epidote": ("Bii", "Epidote"), "Bii_kfeldspar": ("Bii", "Alkali_Feldspar")}
BOX = {"MH0811_opx": "MH0811b_opx", "MH0811_cpx": "MH0811b_cpx", "MH0811_plag": "MH0811b_plag",
       "MH0811_spinel": "MH0811b_spinel", "MH0811_rhombox": "MH0811b_rhombox",
       "Bii_amphibole": "Bii_amph", "Bii_plagioclase": "Bii_plag"}   # AZTEC_boxpx.csv name -> PX name


def read_aztec(path):
    """One row per analysis and oxide from an AZtec 'Element / Oxide % / Oxide % Sigma' export."""
    rows, name, hdr = [], None, None
    for r in pd.read_csv(path, header=None, encoding="utf-8-sig").itertuples(index=False):
        a = r[0]
        if isinstance(a, str) and a not in ("Element", "Total") and pd.isna(r[2]):
            name = a.strip()                       # analysis-name row
        elif a == "Element":
            hdr = list(r)
        elif hdr and isinstance(a, str) and a not in ("O", "Total"):
            rec = dict(zip(hdr, r))
            rows.append(dict(Analysis=name, Oxide="FeOt" if rec["Oxide"] == "FeO" else rec["Oxide"],
                             wt=float(rec["Oxide %"]), sigma=float(rec["Oxide % Sigma"])))
    return pd.DataFrame(rows)


px = read_aztec(f"{HERE}/AZTEC_singlepx.csv")
box = read_aztec(f"{HERE}/AZTEC_boxpx.csv").replace({"Analysis": BOX})
sig = px.pivot_table(index="Analysis", columns="Oxide", values="sigma").reindex(columns=OXIDES).fillna(0)
maps = np.load(f"{HERE}/AZTEC_MC_MapSites.npz")

# %% AZtec sigma: single pixel against box, and against the scatter between neighboring pixels
cmp = px.merge(box, on=["Analysis", "Oxide"], suffixes=("_px", "_box"))
cmp["rel_px"], cmp["rel_box"] = 100 * cmp.sigma_px / cmp.wt_px, 100 * cmp.sigma_box / cmp.wt_box
print(cmp.round(3).to_string(index=False))
print("median sigma ratio (pixel/box):",
      (cmp.sigma_px / cmp.sigma_box).groupby(cmp.Analysis).median().round(1).to_dict())


def interior(smp, ph):
    return binary_erosion(maps[f"ebsd_{smp}"] == ph, np.ones((7, 7), bool))


scatter = []
for a, (smp, ph) in PX.items():
    inside = interior(smp, ph)
    for r in px[px.Analysis == a].itertuples():
        x = maps[f"{r.Oxide}_{smp}"].astype(float)
        m1, m2 = uniform_filter(x, 3), uniform_filter(x * x, 3)
        sd3 = np.sqrt(np.clip(m2 - m1 * m1, 0, None) * 9 / 8)        # sample SD of each 3x3 block
        scatter.append(dict(Map=smp, Phase=ph, Oxide=r.Oxide, sigma_px=r.sigma,
                            map_local_sd=np.median(sd3[inside])))
scatter = pd.DataFrame(scatter)
print("\nneighbor-pixel SD / pixel sigma:")
print((scatter.map_local_sd / scatter.sigma_px).groupby(scatter.Map).agg(["min", "max"]).round(2))

# %% models
train, _ = train_test_split(pd.read_csv(f"{REPO}/Training_Data/min_df_v4_synth.csv"),
                            test_size=0.20, random_state=SEED)
train = mm.balance(train, n=1000)
train["Mineral"] = train["Mineral"].replace({"Hematite": "Rhombohedral_Oxides", "Ilmenite": "Rhombohedral_Oxides",
                                             "Magnetite": "Spinel_Group", "Spinel": "Spinel_Group"})
train = train[~train["Mineral"].isin(["Zircon", "Carbonate", "SiO2_Polymorph"])]
ss = StandardScaler().fit(train[OXIDES].fillna(0).to_numpy(float))
Xtr = ss.transform(train[OXIDES].fillna(0).to_numpy(float))
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
                            random_state=SEED).fit(Xtr, train["Mineral"])
gnb = GaussianNB().fit(Xtr, train["Mineral"])


def classify(D):
    """Add each model's label to D."""
    Xs = ss.transform(D[OXIDES].to_numpy(float))
    for name, clf in [("RF", rf), ("GNB", gnb)]:
        D[name] = resolve_groups(D[OXIDES], pd.Series(clf.predict(Xs), index=D.index)).to_numpy()
    o = mm.predict_class_prob(D[OXIDES], n_iterations=50, seed=SEED, verbose=False)
    D["mineralML"] = np.where(o.Predict_Mineral == "Oxide", o.Submineral, o.Predict_Mineral)
    return D


def score(D, by=("Map", "Phase")):
    """% classified as the EBSD phase."""
    return D[MODELS].eq(D.Phase, axis=0).groupby([D[c] for c in by]).mean().mul(100).round(1)


def report(title, D):
    s = score(D)
    print(f"\n=== {title} ===\n{s.to_string()}\nmean per map:\n{s.groupby('Map').mean().round(1).to_string()}")
    for (smp, ph), g in D.groupby(["Map", "Phase"]):
        print(f"  {smp:<8} {ph:<20} wrong:", {m: g.loc[g[m] != ph, m].value_counts().head(2).to_dict() for m in MODELS})

# %% sites
sites = pd.DataFrame(maps["sites"])[["Site", "Map", "Phase", "Row", "Col"]]
for ph, tag in [("Titanite", "TIT"), ("Epidote", "EPI"), ("Alkali_Feldspar", "KFS")]:
    ys, xs = np.nonzero(interior("Bii", ph))
    gen = np.random.default_rng(SEED + 1) if ph == "Alkali_Feldspar" else rng   # the original alkali feldspar sites
    pick = gen.choice(len(ys), min(200, len(ys)), replace=False)
    sites = pd.concat([sites, pd.DataFrame({"Site": [f"Bii_{tag}_{i:03d}" for i in range(len(pick))],
                                            "Map": "Bii", "Phase": ph, "Row": ys[pick], "Col": xs[pick]})],
                      ignore_index=True)
sites["Pixel"] = [{v: k for k, v in PX.items()}[(m, p)] for m, p in zip(sites.Map, sites.Phase)]


def box_means(b):
    """b x b mean composition at every site."""
    out = []
    for smp, s in sites.groupby("Map", sort=False):
        m = {ox: uniform_filter(maps[f"{ox}_{smp}"].astype(float), size=b, mode="nearest") for ox in OXIDES}
        out.append(pd.DataFrame({ox: m[ox][s.Row, s.Col] for ox in OXIDES}, index=s.index))
    return pd.concat(out).join(sites)

# %% real pixels, no added error
real = classify(pd.concat([box_means(b).assign(Box=b) for b in [1, 3, 5]], ignore_index=True))
for b, g in real.groupby("Box"):
    report(f"real {b}x{b}", g)

# %% 5x5 means + single-pixel sigma, as drawn and rescaled to the 5x5 total
reg = box_means(5)
mu, sd = reg[OXIDES].to_numpy(float), sig.loc[reg.Pixel].to_numpy(float)
d = np.clip(rng.normal(mu[:, None], sd[:, None], (len(reg), N_DRAWS, len(OXIDES))), 0, None)
tot = mu.sum(1)[:, None]                                      # each site's 5x5 total
ind = pd.DataFrame(d.reshape(-1, len(OXIDES)), columns=OXIDES).assign(
    Map=np.repeat(reg.Map.to_numpy(), N_DRAWS), Phase=np.repeat(reg.Phase.to_numpy(), N_DRAWS),
    dTot=(d.sum(2) - tot).ravel())
clo = ind.copy()
clo[OXIDES] = (d * (tot / d.sum(2))[..., None]).reshape(-1, len(OXIDES))
classify(ind); classify(clo)
report("5x5 + pixel sigma, independent draws", ind)
report("5x5 + pixel sigma, closure", clo)
ok = ind.mineralML == ind.Phase
print("\nindependent draws, mineralML % correct by total change (wt.%):")
print(ok.groupby([ind.Map, ind.Phase, pd.cut(ind.dTot, [-99, -4, -2, 2, 4, 99])], observed=True)
      .mean().mul(100).round(1).unstack().to_string())

# %% save


def rec(df):
    """Table as a record array, so np.load reads it without pickle."""
    return df.to_records(index=False, column_dtypes={c: "U32" for c in df.select_dtypes(exclude="number")})


regions = pd.concat({"independent": score(ind), "closure": score(clo)}, names=["Draws"])
np.savez_compressed(f"{HERE}/AZTEC_SinglePx_Results.npz", sites=rec(sites.drop(columns="Pixel")),
                    scatter=rec(scatter), real=rec(score(real, ["Map", "Phase", "Box"]).reset_index()),
                    regions=rec(regions.reset_index()))
