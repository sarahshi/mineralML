# %%
"""
Figure: mineralML versus conventional classifiers on the GEOROC compilation.

Numbers read from Model_Performance_perclass.csv / _overall.csv (mineralML,
written by Model_Performance_Metrics.py) and Model_Performance_RF_perclass.csv
/ _RF_overall.csv (baselines, written by Model_Performance_RF_perclass.py).
Every model is scored on the same rows, behind the same empirical front end,
with the feldspar split collapsed because GEOROC labels cannot resolve it.

  A  per-class F1, every model against every phase -- 19 phases against 3
     models, so this panel spans the full width rather than being squared
  B  concordance on the full compilation -- the aggregate view
  C  precision against recall, one arrow per phase running from the Random
     Forest to mineralML, so the direction of the change is visible
  D  the same differences per phase, since the two metrics disagree and the
     largest single gap is a precision one (rhombohedral oxides)

Phases are ordered alphabetically throughout, so the same row or column
refers to the same mineral in every panel.

The Random Forest is tuned by grouped cross-validation on the training set
only, so it cannot be dismissed as a default-parameter strawman; an untuned
Random Forest scores within a point of it and is reported in the text rather
than plotted.

Zircon, Carbonate and the SiO2 polymorphs are included because they are part
of what the pipeline delivers, but they are assigned by the empirical
composition rules ahead of the network rather than by any classifier, so their
scores are identical for every model by construction. They are marked in every
panel so that they are not read as a model comparison: hatched in B, and shown
once as grey points in C and D.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.style.use('./style.mplstyle')
plt.rcParams['mathtext.default'] = 'regular'

# One type scale for the whole figure. The heatmap spans the full width and
# carries far more cells than the other panels, so it sits one step down
# (FS_HEAT) while B, C and D share FS_TICK / FS_AXIS.
FS_TITLE = 14      # panel titles
FS_AXIS  = 14      # axis labels
FS_TICK  = 12      # tick labels, legends, in-panel values
FS_HEAT  = 12       # everything inside the F1 heatmap
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

import os
# Panel letters, overridable so this block can be embedded in a larger figure
PANEL = globals().get("PANEL_LETTERS_OVERRIDE",
                      os.environ.get("PANEL_LETTERS", "A.,B.,C.,D.,E.").split(","))

nn = pd.read_csv("Model_Performance_perclass.csv")
nn_ov = pd.read_csv("Model_Performance_overall.csv")
bl = pd.read_csv("Model_Performance_RF_perclass.csv")
bl_ov = pd.read_csv("Model_Performance_RF_overall.csv")

nn_g = nn[nn.dataset == "GEOROC"].assign(model="mineralML")
bl_g = bl[bl.dataset == "GEOROC"]
per = pd.concat([nn_g, bl_g], ignore_index=True)
# empirical classes retained; flagged rather than dropped

# Raw concordance: the fraction of GEOROC analyses assigned the same name as
# the compilation. Balanced concordance is within a point of it for every
# model here and is reported in Model_Performance_RF_overall.csv.
bal = {"mineralML": float(nn_ov[nn_ov.dataset == "GEOROC"]["accuracy"].iloc[0])}
for m in ORDER[1:]:
    row = bl_ov[(bl_ov.dataset == "GEOROC") & (bl_ov.model == m)]
    bal[m] = float(row["accuracy"].iloc[0])

f1 = per.pivot_table(index="model", columns="mineral", values="f1")
f1 = f1.loc[[m for m in ORDER if m in f1.index]]
f1 = f1[sorted(f1.columns)]

SHORT_PHASE = {"Rhombohedral_Oxides": "Rhomb. Oxides"}

# Mineral abbreviations as defined by the MINERAL_SUFFIX of each calculator in
# mineralML.stoichiometry, so the figure uses the same shorthand as the code.
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
    """Short form for the crowded precision-recall panel."""
    return ABBREV.get(m, pretty(m))


# The heatmap carries 19 phases against 3 models, so squaring it would force
# cells about six times taller than wide. It spans the full width at its own
# aspect on the top row; the three panels below it are square.
# HOST_GS lets another script embed this block natively in a larger figure
# rather than pasting a rendered image, so both share one canvas and one scale.
HOST = globals().get("HOST_FIG")
if HOST is None:
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.40, 1], hspace=0.0, wspace=0.5)
else:
    fig = HOST
    gs = globals()["HOST_GS"].subgridspec(2, 4, height_ratios=[0.40, 1],
                                          hspace=0.0, wspace=0.45)
axB = fig.add_subplot(gs[0, :])     # per-class F1 heatmap
axA = fig.add_subplot(gs[1, 0])     # aggregate concordance
axP = fig.add_subplot(gs[1, 1])     # precision, RF vs mineralML
axR = fig.add_subplot(gs[1, 2])     # recall, RF vs mineralML
axC = fig.add_subplot(gs[1, 3])     # deltas

# ---- A: aggregate ----------------------------------------------------------
y = np.arange(len(ORDER))
vals = [bal[m] for m in ORDER]
axA.barh(y, vals, 0.62, color=[COLORS[m] for m in ORDER], edgecolor="k", lw=0.6,
         zorder=3)
for yi, v in zip(y, vals):
    axA.text(v + 0.25, yi, f"{v:.1f}", va="center", fontsize=FS_TICK)
axA.set_yticks(y)
axA.set_yticklabels([SHORT[m].replace("\n", " ") for m in ORDER], fontsize=FS_TICK)
axA.invert_yaxis()
axA.set_xlim(80, 100)
axA.set_ylim(len(ORDER) - 0.45, -0.55)
axA.set_xlabel("Concordance (%)", fontsize=FS_AXIS)
axA.tick_params(labelsize=FS_TICK)
axA.grid(axis="x", color="0.92", lw=0.8, zorder=0)
axA.spines[["top", "right"]].set_visible(False)
axA.set_title(PANEL[1], fontsize=FS_TITLE,
              loc="left", pad=10)

# ---- B: per-class F1 heatmap ----------------------------------------------
cmap = LinearSegmentedColormap.from_list(
    "f1", ["#f7fbff", "#c6dcef", "#6aaed6", "#2a78d6", "#0b3d80"])
vmin = max(55.0, float(np.floor(f1.min().min() / 5) * 5))
im = axB.imshow(f1.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=100)
axB.set_xticks(range(f1.shape[1]))
axB.set_xticklabels([pretty(c) for c in f1.columns], rotation=45, ha="right",
                    fontsize=FS_HEAT)
axB.set_yticks(range(f1.shape[0]))
axB.set_yticklabels([SHORT[m] for m in f1.index], fontsize=FS_HEAT)
for i in range(f1.shape[0]):
    for j in range(f1.shape[1]):
        v = f1.values[i, j]
        axB.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=FS_HEAT,
                 color="w" if v > vmin + 0.62 * (100 - vmin) else "0.15")
# hatch the empirically-assigned columns: identical for every model by design
for j, c in enumerate(f1.columns):
    if c in EMPIRICAL:
        axB.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, f1.shape[0],
                                    fill=False, hatch="///", edgecolor="0.35",
                                    lw=0.0, zorder=5))
        axB.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, f1.shape[0],
                                    fill=False, edgecolor="0.2", lw=1.4,
                                    zorder=6))
axB.set_title(PANEL[0], fontsize=FS_TITLE, loc="left", pad=10)
# No colourbar: the value is printed in every cell, so the colour is a
# reading aid rather than the data. A legend for the hatched columns is more
# useful, as those are resolved by the empirical filters, not the network.
import matplotlib.patches as _mp
_hatch = _mp.Patch(facecolor="white", edgecolor="0.2", hatch="///", lw=1.2,
                   label="resolved by empirical composition filter")
axB.legend(handles=[_hatch], loc="lower right", bbox_to_anchor=(1.0, 1.02),
           frameon=False, fontsize=FS_HEAT)

# ---- C: precision and recall deltas ---------------------------------------
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
axC.barh(yy - hh/2, d.d_rec, hh,
         color=np.where(_is_emp, "0.80", "#7ba7d7"),
         edgecolor="k", lw=0.5, label="recall", zorder=3)
axC.barh(yy + hh/2, d.d_prec, hh,
         color=np.where(_is_emp, "0.60", "#2a4d78"),
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
axC.set_title(PANEL[4], fontsize=FS_TITLE,
              loc="left", pad=10)

# ---- D, E: Random Forest against mineralML, one panel per metric -----------
# Each phase is one point at (Random Forest, mineralML). Above the 1:1 line
# mineralML scores higher; below it the Random Forest does. Points are
# coloured by which side they fall on, with differences under TIE pp shown
# grey, so the count of each colour is the answer to "which model does better
# on this metric". Marker area scales with the log of class support.
TIE = 0.5
sz = 26 + 150 * (np.log10(d.support) - np.log10(d.support.min())) / (
     np.log10(d.support.max()) - np.log10(d.support.min()))
_lrn_idx = [m for m in d.index if m not in EMPIRICAL]


def _label(ax, m, xy, spec, fs):
    """Label one point. spec is an (dx, dy) offset in points, or
    ("data", x, y) to place the label at a fixed data position."""
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
    """Every phase is labelled: in the main panel, or -- for phases inside
    `zoom` (lo, hi) -- in a magnified inset placed at `inset_at` (axes
    fraction), so the cluster near 100% stays legible on a linear scale."""
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
        # Outline the magnified region; connector lines would cross labels,
        # and the matching frame and tick values make the link clear.
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


# Every phase is labelled. Precision spans 66-100%, so the nine phases above
# 94% are shown in a magnified inset rather than labelled in place.
parity(axP, p, (62, 102), range(65, 101, 5), "precision", PANEL[2], {
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
# Recall spans 89-100%, so the cluster is labelled in place, with the five
# phases near 96-98% led out to a column in the empty lower-right corner.
parity(axR, q, (88, 101), range(88, 101, 2), "recall", PANEL[3], {
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

# One square box per panel, so the four read as a set regardless of how many
# categories each one carries.
for _ax in (axA, axP, axR, axC):
    _ax.set_box_aspect(1)

if HOST is None:
    fig.savefig("GEOROC_PerClass_Parity.pdf", bbox_inches="tight")
    fig.savefig("GEOROC_PerClass_Parity.png", bbox_inches="tight", dpi=170)
print("concordance:", {k: round(v, 2) for k, v in bal.items()})
if HOST is None:
    print("wrote GEOROC_PerClass_Figure.pdf / .png")
else:
    print("  (per-class block drawn into host figure)")

# %%
