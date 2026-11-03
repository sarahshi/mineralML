# %%
"""
Figure S5: mineralML versus the Random Forest and Gaussian Naive Bayes
baselines. A, agreement on every evaluation dataset; B-F, per-class comparison
on GEOROC. Hatched / grey classes are resolved by the empirical composition
rules, so they score the same for every model.
Reads Model_Performance_*.csv, Baseline_perclass.csv and Baseline_Numbers.json.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as _mp
from matplotlib.colors import LinearSegmentedColormap

STYLE = "../../style.mplstyle"
plt.style.use(STYLE)
plt.rcParams["mathtext.default"] = "regular"

J = json.load(open("Baseline_Numbers.json"))
nn = pd.read_csv("Model_Performance_perclass.csv")
nn_ov = pd.read_csv("Model_Performance_overall.csv").set_index("dataset")["accuracy"]
bl = pd.read_csv("Baseline_perclass.csv")

# ---- A: agreement on every evaluation dataset ----
ORDER = ["Gaussian Naive Bayes", "Random Forest (tuned)", "mineralML"]
SHORT = {"Gaussian Naive Bayes": "Gaussian Naive Bayes (grouped)",
         "Random Forest (tuned)": "Random Forest (grouped)",
         "mineralML": "mineralML"}
C = {"Gaussian Naive Bayes": "#eda100",
     "Random Forest (tuned)": "#eb6834",
     "mineralML": "#2a78d6"}
A = {"mineralML": [nn_ov["Held-out validation"], nn_ov["Secondary standards"], nn_ov["GEOROC"],
                   nn_ov["EDS map MH0811b_pool"], nn_ov["EDS map Bii_pool"]]}
for m, k in [("Random Forest (tuned)", "RF grouped"), ("Gaussian Naive Bayes", "GNB grouped")]:
    A[m] = [J["heldout"][k]["accuracy"], J["standards"][k]["accuracy"], J["georoc"][f"{m} grouped"],
            J["maps"]["MH0811b"][k]["agreement_pool"], J["maps"]["Bii"][k]["agreement_pool"]]
DSETS = ["Held-out\nvalidation", "Secondary\nstandards", "GEOROC", "EDS map\nMH0811b", "EDS map\nBii"]

fig = plt.figure(figsize=(20, 16))
outer = fig.add_gridspec(2, 1, height_ratios=[0.34, 1.0], hspace=0.20, left=0.06, right=0.97)

axA = fig.add_subplot(outer[0])
x = np.arange(len(DSETS)); w = 0.26
for i, m in enumerate(ORDER):
    v = A[m]
    axA.bar(x + (i - 1) * w, v, w, color=C[m], edgecolor="k", lw=0.6, label=SHORT[m], zorder=3)
    for xi, vi in zip(x + (i - 1) * w, v):
        axA.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", fontsize=11, rotation=90, va="bottom")
axA.set_xticks(x)
axA.set_xticklabels(DSETS, fontsize=13)
axA.set_ylabel("Agreement with reference (%)", fontsize=14)
axA.set_ylim(0, 120)
axA.tick_params(labelsize=13)
axA.grid(axis="y", color="0.92", lw=0.8, zorder=0)
axA.spines[["top", "right"]].set_visible(False)
axA.legend(frameon=False, loc="upper right", ncol=3, fontsize=12, bbox_to_anchor=(1.0, 1.02))
axA.set_title("A.", fontsize=15, loc="left", pad=10)

# ---- B-F: per-class comparison on GEOROC ----
plt.style.use(STYLE)
plt.rcParams['mathtext.default'] = 'regular'
FS_TITLE = 14      # panel titles
FS_AXIS  = 14      # axis labels
FS_TICK  = 12      # tick labels, legends, in-panel values
FS_HEAT  = 12      # everything inside the F1 heatmap
FS_NOTE  = 12      # small italic annotations
plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 14, 'axes.labelsize': 14,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12,
})

EMPIRICAL = {"Zircon", "Carbonate", "SiO2_Polymorph"}
ORDER = ["mineralML", "Random Forest (tuned)", "Gaussian Naive Bayes"]
SHORT = {"mineralML": "mineralML",
         "Random Forest (tuned)": "Random Forest\n(grouped)",
         "Gaussian Naive Bayes": "Gaussian\nNaive Bayes\n(grouped)"}
COLORS = {"mineralML": "#2a78d6", "Random Forest (tuned)": "#eb6834",
          "Gaussian Naive Bayes": "#eda100"}
C_NN, C_RF = "#2a78d6", "#eb6834"

per = pd.concat([nn[nn.dataset == "GEOROC"].assign(model="mineralML"), bl[bl.dataset == "GEOROC"]],
                ignore_index=True)
# raw concordance on GEOROC
bal = {"mineralML": float(nn_ov["GEOROC"])}
for m in ORDER[1:]:
    bal[m] = float(J["georoc"][f"{m} grouped"])

f1 = per.pivot_table(index="model", columns="mineral", values="f1")
f1 = f1.loc[[m for m in ORDER if m in f1.index]]
f1 = f1[sorted(f1.columns)]

SHORT_PHASE = {"Rhombohedral_Oxides": "Rhomb. Oxides"}
# abbreviations from the MINERAL_SUFFIX of each calculator in mineralML.stoichiometry
ABBREV = {
    "Amphibole": "Amp", "Apatite": "Ap", "Biotite": "Bt", "Carbonate": "Cal",
    "Chlorite": "Chl", "Clinopyroxene": "Cpx", "Epidote": "Ep",
    "Feldspar": "Feld", "Garnet": "Grt", "Glass": "Gl", "Kalsilite": "Kls",
    "Leucite": "Lct", "Melilite": "Mll", "Muscovite": "Ms",
    "Nepheline": "Nph", "Olivine": "Ol", "Orthopyroxene": "Opx",
    "SiO2_Polymorph": "Qz", "Rhombohedral_Oxides": "Ox", "Rutile": "Rt",
    "Serpentine": "Srp", "Na-Pyroxene": "NaPx", "Spinel_Group": "Spl",
    "Titanite": "Ttn", "Tourmaline": "Tur", "Zircon": "Zrn",
}


def pretty(m):
    if m in SHORT_PHASE:
        return SHORT_PHASE[m]
    return m.replace("_", " ").replace("SiO2", "SiO$_2$")


def abbrev(m):
    return ABBREV.get(m, pretty(m))


gs = outer[1].subgridspec(2, 4, height_ratios=[0.40, 1], hspace=0.0, wspace=0.45)
axB = fig.add_subplot(gs[0, :])     # per-class F1 heatmap
axG = fig.add_subplot(gs[1, 0])     # aggregate concordance
axP = fig.add_subplot(gs[1, 1])     # precision, RF vs mineralML
axR = fig.add_subplot(gs[1, 2])     # recall, RF vs mineralML
axC = fig.add_subplot(gs[1, 3])     # deltas

# ---- C: aggregate ----
y = np.arange(len(ORDER))
vals = [bal[m] for m in ORDER]
axG.barh(y, vals, 0.62, color=[COLORS[m] for m in ORDER], edgecolor="k", lw=0.6, zorder=3)
for yi, v in zip(y, vals):
    axG.text(v + 0.25, yi, f"{v:.1f}", va="center", fontsize=FS_TICK)
axG.set_yticks(y)
axG.set_yticklabels([SHORT[m].replace("\n", " ") for m in ORDER], fontsize=FS_TICK)
axG.invert_yaxis()
axG.set_xlim(80, 100)
axG.set_ylim(len(ORDER) - 0.45, -0.55)
axG.set_xlabel("Concordance (%)", fontsize=FS_AXIS)
axG.tick_params(labelsize=FS_TICK)
axG.grid(axis="x", color="0.92", lw=0.8, zorder=0)
axG.spines[["top", "right"]].set_visible(False)
axG.set_title("C.", fontsize=FS_TITLE, loc="left", pad=10)

# ---- B: per-class F1 heatmap ----
cmap = LinearSegmentedColormap.from_list(
    "f1", ["#f7fbff", "#c6dcef", "#6aaed6", "#2a78d6", "#0b3d80"])
vmin = max(55.0, float(np.floor(f1.min().min() / 5) * 5))
im = axB.imshow(f1.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=100)
axB.set_xticks(range(f1.shape[1]))
axB.set_xticklabels([pretty(c) for c in f1.columns], rotation=45, ha="right", fontsize=FS_HEAT)
axB.set_yticks(range(f1.shape[0]))
axB.set_yticklabels([SHORT[m] for m in f1.index], fontsize=FS_HEAT)
for i in range(f1.shape[0]):
    for j in range(f1.shape[1]):
        v = f1.values[i, j]
        axB.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=FS_HEAT,
                 color="w" if v > vmin + 0.62 * (100 - vmin) else "0.15")
for j, c in enumerate(f1.columns):
    if c in EMPIRICAL:
        axB.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, f1.shape[0],
                                    fill=False, hatch="///", edgecolor="0.35", lw=0.0, zorder=5))
        axB.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, f1.shape[0],
                                    fill=False, edgecolor="0.2", lw=1.4, zorder=6))
axB.set_title("B.", fontsize=FS_TITLE, loc="left", pad=10)
_hatch = _mp.Patch(facecolor="white", edgecolor="0.2", hatch="///", lw=1.2,
                   label="resolved by empirical composition filter")
axB.legend(handles=[_hatch], loc="lower right", bbox_to_anchor=(1.0, 1.02),
           frameon=False, fontsize=FS_HEAT)

# ---- F: precision and recall deltas ----
p = per.pivot_table(index="mineral", columns="model", values="precision")
q = per.pivot_table(index="mineral", columns="model", values="recall")
sup = per.groupby("mineral")["support"].first()
d = pd.DataFrame({
    "d_prec": p["mineralML"] - p["Random Forest (tuned)"],
    "d_rec": q["mineralML"] - q["Random Forest (tuned)"],
    "support": sup,
}).sort_index(ascending=False)

yy = np.arange(len(d)); hh = 0.38
_is_emp = np.array([m in EMPIRICAL for m in d.index])
axC.barh(yy - hh/2, d.d_rec, hh, color=np.where(_is_emp, "0.80", "#7ba7d7"),
         edgecolor="k", lw=0.5, label="recall", zorder=3)
axC.barh(yy + hh/2, d.d_prec, hh, color=np.where(_is_emp, "0.60", "#2a4d78"),
         edgecolor="k", lw=0.5, label="precision", zorder=3)
for yi, m in zip(yy, d.index):
    if m in EMPIRICAL:
        axC.text(0.5, yi, "empirical", va="center", ha="left",
                 fontsize=8.5, style="italic", color="0.45", zorder=5)
axC.axvline(0, color="k", lw=1.0, zorder=4)
axC.set_yticks(yy)
axC.set_yticklabels([pretty(m) for m in d.index], fontsize=FS_NOTE)
axC.tick_params(axis="y", pad=2)
axC.set_xlabel("mineralML $-$ Random Forest\n(percentage points)", fontsize=FS_AXIS)
axC.tick_params(axis="x", labelsize=FS_TICK)
axC.grid(axis="x", color="0.92", lw=0.8, zorder=0)
axC.spines[["top", "right"]].set_visible(False)
axC.legend(handles=[_mp.Patch(fc="#7ba7d7", ec="k", lw=0.5, label="recall"),
                    _mp.Patch(fc="#2a4d78", ec="k", lw=0.5, label="precision")],
           loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
           frameon=False, fontsize=FS_NOTE)
axC.set_title("F.", fontsize=FS_TITLE, loc="left", pad=10)

# ---- D, E: Random Forest against mineralML, one panel per metric ----
# Points above the 1:1 line favour mineralML; differences under TIE pp are grey.
# Marker area scales with the log of class support.
TIE = 0.5
sz = 26 + 150 * (np.log10(d.support) - np.log10(d.support.min())) / (
     np.log10(d.support.max()) - np.log10(d.support.min()))
_lrn_idx = [m for m in d.index if m not in EMPIRICAL]


def _label(ax, m, xy, spec, fs):
    """spec is an (dx, dy) offset in points, or ("data", x, y)."""
    if spec[0] == "data":
        kw = dict(xytext=spec[1:], textcoords="data",
                  ha="left" if spec[1] >= xy[0] else "right")
    else:
        kw = dict(xytext=spec, textcoords="offset points",
                  ha="center" if spec[0] == 0 else
                     ("left" if spec[0] > 0 else "right"))
    ax.annotate(abbrev(m), xy=xy, fontsize=fs, color="0.2", va="center",
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.7,
                                shrinkA=1, shrinkB=4),
                annotation_clip=False, zorder=6, **kw)


def _points(ax, rf, nnv, col, lo, hi, scale=1.0):
    ax.fill_between([lo, hi], [lo, hi], hi, color=C_NN, alpha=0.06, lw=0, zorder=0)
    ax.fill_between([lo, hi], lo, [lo, hi], color=C_RF, alpha=0.06, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.3", lw=1.0, ls="--", zorder=1)
    ax.scatter(rf, nnv, s=scale * sz[_lrn_idx], c=col, ec="k", lw=0.5, zorder=3)


def parity(ax, tab, lim, ticks, metric, letter, labels,
           zoom=None, inset_at=None, inset_labels=None, inset_ticks=None):
    """Phases inside `zoom` are labelled in a magnified inset at `inset_at`."""
    rf, nnv = tab.loc[_lrn_idx, "Random Forest (tuned)"], tab.loc[_lrn_idx, "mineralML"]
    diff = nnv - rf
    col = np.where(diff >= TIE, C_NN, np.where(diff <= -TIE, C_RF, "0.75"))
    lo, hi = lim
    _points(ax, rf, nnv, col, lo, hi)
    for m, spec in labels.items():
        _label(ax, m, (rf[m], nnv[m]), spec, FS_NOTE)
    if zoom is not None:
        zl, zh = zoom
        axi = ax.inset_axes(inset_at)
        _points(axi, rf, nnv, col, zl, zh)
        for m, spec in inset_labels.items():
            _label(axi, m, (rf[m], nnv[m]), spec, FS_NOTE - 1)
        axi.set_xlim(zl, zh); axi.set_ylim(zl, zh)
        axi.set_xticks(inset_ticks); axi.set_yticks(inset_ticks)
        axi.tick_params(labelsize=FS_NOTE - 3, length=2, pad=1)
        axi.grid(color="0.92", lw=0.6, zorder=0)
        for sp in axi.spines.values():
            sp.set_edgecolor("0.4"); sp.set_linewidth(0.8)
        _rect, _conn = ax.indicate_inset_zoom(axi, edgecolor="0.4", lw=0.8, alpha=1)
        for c in _conn:
            c.set_visible(False)
    missing = set(_lrn_idx) - set(labels) - set(inset_labels or {})
    assert not missing, f"unlabelled phases in {metric}: {missing}"
    n_nn, n_rf = int((diff >= TIE).sum()), int((diff <= -TIE).sum())
    n_tie = len(diff) - n_nn - n_rf
    ax.text(0.04, 0.96, f"mineralML higher: {n_nn}", transform=ax.transAxes,
            color=C_NN, fontsize=FS_TICK, weight="bold", va="top")
    ax.text(0.96, 0.04, f"Random Forest higher: {n_rf}", transform=ax.transAxes,
            color=C_RF, fontsize=FS_TICK, weight="bold", ha="right")
    ax.text(0.96, 0.11, f"within {TIE} pp (grey): {n_tie}", transform=ax.transAxes,
            color="0.45", fontsize=FS_TICK - 1, ha="right")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xlabel(f"Random Forest (grouped) {metric} (%)", fontsize=FS_AXIS)
    ax.set_ylabel(f"mineralML {metric} (%)", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(color="0.92", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(letter, fontsize=FS_TITLE, loc="left", pad=10)


parity(axP, p, (62, 102), range(65, 101, 5), "precision", "D.", {
    "Rhombohedral_Oxides": (14, -10),
    "Kalsilite":           (12, -8),
    "Muscovite":           (12, 4),
    "Nepheline":           (14, -6),
    "Amphibole":           (-14, 10),
    "Glass":               (12, -12),
    "Leucite":             (14, -6),
}, zoom=(93.5, 102.0), inset_at=[0.09, 0.47, 0.39, 0.39],
   inset_ticks=[95, 100], inset_labels={
    "Biotite":             (12, -6),
    "Orthopyroxene":       (0, -16),
    "Garnet":              (12, -8),
    "Apatite":             (10, -14),
    "Titanite":            (-10, 12),
    "Spinel_Group":        (-4, 16),
    "Olivine":             (10, 10),
    "Feldspar":            (12, -2),
    "Clinopyroxene":       (6, -16),
})
parity(axR, q, (88, 101), range(88, 101, 2), "recall", "E.", {
    "Rhombohedral_Oxides": (8, 18),
    "Olivine":             (16, 6),
    "Spinel_Group":        (14, -14),
    "Kalsilite":           (-14, 10),
    "Clinopyroxene":       (-14, 12),
    "Leucite":             (-14, 10),
    "Orthopyroxene":       (14, -10),
    "Glass":               (14, -8),
    "Feldspar":            (-12, 14),
    "Garnet":              (10, -14),
    "Muscovite":           (6, 14),
    "Apatite":             ("data", 99.4, 96.6),
    "Titanite":            ("data", 99.4, 95.8),
    "Nepheline":           ("data", 99.4, 95.0),
    "Biotite":             ("data", 99.4, 94.2),
    "Amphibole":           ("data", 99.4, 93.4),
})

for _ax in (axG, axP, axR, axC):
    _ax.set_box_aspect(1)

fig.savefig("Combined_Comparison_Parity_grouped.pdf", bbox_inches="tight")
