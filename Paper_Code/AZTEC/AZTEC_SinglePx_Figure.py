# %%
"""
Figure: how the map regions are sampled, how large AZtec's errors are, and how
each model classifies the regions.

  A, B  EBSD maps of MH0811b and Tuolumne Bii with every sampled 5x5 region
  C     one example site per mineral: EBSD neighbourhood and a diagnostic
        oxide, with the 1x1, 3x3, 5x5 boxes inside the 7x7 same-phase window
  D     AZtec single-pixel 1 sigma against the measured scatter between
        neighboring map pixels
  E     % of regions classified as the EBSD phase: real pixels (1x1, 3x3,
        5x5, no added error), and 5x5 means with single-pixel AZtec error

Reads AZTEC_SinglePx_Results.npz (AZTEC_singlepx_test.py) and AZTEC_MC_MapSites.npz.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.expanduser("~/GitHub/mineralML/style.mplstyle"))
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["font.size"] = 14
FS_T, FS_L, FS_K, FS_S = 16, 14, 14, 12          # titles, labels, ticks, small

MODELS = ["mineralML", "RF", "GNB"]
NAME = {"mineralML": "mineralML", "RF": "Random Forest (grouped)", "GNB": "Gaussian Naive Bayes (grouped)"}
COLORS = {"mineralML": "#2a78d6", "RF": "#eb6834", "GNB": "#eda100"}
MARKERS = {"mineralML": "o", "RF": "s", "GNB": "^"}
# panel E: the Random Forest is an orange outline drawn on top, so a mineralML
# circle behind it stays visible where the two models agree
FILL = {"mineralML": dict(c=COLORS["mineralML"], edgecolors="k", linewidths=0.6, zorder=3),
        "RF": dict(facecolors="none", edgecolors=COLORS["RF"], linewidths=2.0, zorder=4),
        "GNB": dict(c=COLORS["GNB"], edgecolors="k", linewidths=0.6, zorder=3)}
MAPS = {"MH0811b": "MH0811b", "Bii": "Tuolumne Bii"}
MAP_MARKER = {"MH0811b": "o", "Bii": "D"}

PHASE_COLORS = {"Plagioclase": "#66C4C4", "Alkali_Feldspar": "#FEF7C2",
                "Clinopyroxene": "#E57A7A", "Orthopyroxene": "#931d1d",
                "Amphibole": "#5E2910", "Spinel_Group": "#2E2DCE",
                "Rhombohedral_Oxides": "#6FB2E4", "SiO2_Polymorph": "#A8A5A5",
                "Apatite": "#000000", "Titanite": "#E1D6DF", "Epidote": "magenta",
                "Zircon": "cyan", "Unindexed": "white"}
LABEL = {"Spinel_Group": "Spinel", "Rhombohedral_Oxides": "Ilmenite",
         "SiO2_Polymorph": "SiO$_2$", "Alkali_Feldspar": "Alkali Feldspar"}
# rows of panel E, top to bottom: map, EBSD phase, row label, panel D label and
# colour (darker than the map tone, to read on white)
ROWS = [("MH0811b", "Orthopyroxene", "Orthopyroxene", "Opx", "#931d1d"),
        ("MH0811b", "Clinopyroxene", "Clinopyroxene", "Cpx", "#E57A7A"),
        ("MH0811b", "Plagioclase", "Plagioclase", "Plag", "#3a9e9e"),
        ("MH0811b", "Spinel_Group", "Magnetite", "Mag", "#2E2DCE"),
        ("MH0811b", "Rhombohedral_Oxides", "Ilmenite", "Ilm", "#6FB2E4"),
        ("Bii", "Amphibole", "Amphibole", "Amp", "#5E2910"),
        ("Bii", "Plagioclase", "Plagioclase", "Plag", "#66C4C4"),
        ("Bii", "Alkali_Feldspar", "Alkali Feldspar", "Kfs", "#D9C35A"),
        ("Bii", "Spinel_Group", "Spinel", "Spl", "#0b0b6e"),
        ("Bii", "Titanite", "Titanite", "Ttn", "#9c8a99"),
        ("Bii", "Epidote", "Epidote", "Ep", "magenta")]
EXAMPLES = [("MH0811b", "Orthopyroxene", "Orthopyroxene", "MgO"),
            ("MH0811b", "Rhombohedral_Oxides", "Ilmenite", "TiO2"),
            ("Bii", "Amphibole", "Amphibole", "Al2O3"),
            ("Bii", "Spinel_Group", "Spinel", "FeOt"),
            ("Bii", "Titanite", "Titanite", "TiO2"),
            ("Bii", "Alkali_Feldspar", "Alkali Feldspar", "K2O")]
OX_LABEL = {"MgO": "MgO", "Al2O3": "Al$_2$O$_3$", "FeOt": "FeO$_t$", "TiO2": "TiO$_2$", "K2O": "K$_2$O"}
OX11 = ["SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO", "CaO", "Na2O", "K2O", "Cr2O3", "P2O5"]
CROP = 8

maps = np.load(os.path.join(HERE, "AZTEC_MC_MapSites.npz"))
res = np.load(os.path.join(HERE, "AZTEC_SinglePx_Results.npz"))
sites = pd.DataFrame(res["sites"])
scat = pd.DataFrame(res["scatter"])
real = pd.DataFrame(res["real"]).set_index(["Map", "Phase", "Box"])
regions = pd.DataFrame(res["regions"]).set_index(["Draws", "Map", "Phase"])
# panel E values: model -> (row, column), columns 1x1, 3x3, 5x5, 5x5 + pixel sigma
KEYS = [(m, p) for m, p, *_ in ROWS]
COLS = ["1×1", "3×3", "5×5", "+1× pixel $\\sigma$"]
VAL = {md: np.column_stack([real[md].unstack("Box").loc[KEYS], regions.loc["closure"][md].loc[KEYS]])
       for md in MODELS}


def phase_rgb(lab):
    rgb = np.ones(lab.shape + (3,))
    for ph in np.unique(lab):
        rgb[lab == ph] = to_rgb(PHASE_COLORS.get(ph, "#DDDDDD"))
    return rgb


fig = plt.figure(figsize=(20, 20.5))
# spacer rows (1, 3) hold the legends and titles that sit between the panel rows
outer = fig.add_gridspec(5, 1, height_ratios=[0.95, 0.17, 0.80, 0.28, 0.80], hspace=0.10)

# ---- A, B: maps with every sampled region --------------------------------------
top = outer[0].subgridspec(1, 2, width_ratios=[500 / 494, 570 / 285], wspace=0.04)
present, map_axes = [], []
for k, smp in enumerate(MAPS):
    ax = fig.add_subplot(top[k])
    lab = maps[f"ebsd_{smp}"]
    ax.imshow(phase_rgb(lab), interpolation="none")
    s = sites[sites.Map == smp]
    for r in s.itertuples():
        ax.add_patch(Rectangle((r.Col - 2.5, r.Row - 2.5), 5, 5, fill=False, ec="k", lw=0.35, zorder=3))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{'AB'[k]}. {MAPS[smp]}: {len(s)} sampled regions, {s.Phase.nunique()} minerals",
                 fontsize=FS_T, loc="left", pad=50)
    # data quality: plagioclase (in both maps) pixel sigma, and single-pixel totals at the sites
    pl = scat[(scat.Map == smp) & (scat.Phase == "Plagioclase")].set_index("Oxide").sigma_px
    tot = sum(maps[f"{ox}_{smp}"][s.Row, s.Col].astype(float) for ox in OX11)
    ax.text(0, 1.012, f"plagioclase pixel 1$\\sigma$: {pl['SiO2']:.1f} wt.% SiO$_2$, {pl['Na2O']:.1f} wt.% Na$_2$O\n"
                      f"pixel total {tot.mean():.1f} ± {np.std(tot, ddof=1):.1f} wt.%",
            transform=ax.transAxes, fontsize=FS_S + 1, ha="left", va="bottom", color="0.3")
    map_axes.append(ax)
    present += [p for p in PHASE_COLORS if (lab == p).any() and p not in present]
phase_handles = [Rectangle((0, 0), 1, 1, fc=PHASE_COLORS[p], ec="k", lw=0.4,
                           label=LABEL.get(p, p.replace("_", " "))) for p in present]

# ---- C: example sites -----------------------------------------------------------
mid = outer[2].subgridspec(1, 2, width_ratios=[2.0, 1], wspace=0.10)
row_cd = []                                      # axes of the C/D row, aligned below
ex = mid[0].subgridspec(2, len(EXAMPLES), hspace=0.06, wspace=0.04)
for j, (smp, ph, name, ox) in enumerate(EXAMPLES):
    r = sites[(sites.Map == smp) & (sites.Phase == ph)].iloc[0]
    rr, cc = int(r.Row), int(r.Col)
    lab = maps[f"ebsd_{smp}"]
    r0, r1 = max(rr - CROP, 0), min(rr + CROP + 1, lab.shape[0])
    c0, c1 = max(cc - CROP, 0), min(cc + CROP + 1, lab.shape[1])
    ext = (c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5)
    axE = fig.add_subplot(ex[0, j])
    axE.imshow(phase_rgb(lab[r0:r1, c0:c1]), extent=ext, interpolation="none")
    axO = fig.add_subplot(ex[1, j])
    im = axO.imshow(maps[f"{ox}_{smp}"][r0:r1, c0:c1], extent=ext, cmap="viridis", interpolation="none")
    for a, lw, ec in ((axE, 1.5, "k"), (axO, 1.2, "w")):
        for size, ls in [(7, "--"), (5, "-"), (3, "-"), (1, "-")]:
            a.add_patch(Rectangle((cc - size / 2, rr - size / 2), size, size, fill=False, lw=lw, ls=ls,
                                  ec=ec, zorder=4))
        a.set_xticks([]); a.set_yticks([])
    axE.set_title(f"{name}\n({smp})", fontsize=FS_K)
    cb = fig.colorbar(im, ax=axO, orientation="horizontal", pad=0.04, fraction=0.07)
    row_cd += [axE, axO, cb.ax]
    cb.set_label(f"{OX_LABEL[ox]} (wt.%)", fontsize=FS_S)
    cb.ax.tick_params(labelsize=10)
    # round ticks kept clear of the bar ends, so labels of neighbouring colorbars cannot touch
    lo, hi = im.get_clim()
    cb.set_ticks([t for t in MaxNLocator(nbins=4).tick_values(lo, hi)
                  if lo + 0.08 * (hi - lo) <= t <= hi - 0.08 * (hi - lo)])
    if j == 0:
        axE.set_ylabel("EBSD phase", fontsize=FS_L)
        axO.set_ylabel("EDS pixels", fontsize=FS_L)
        axE_first = axE

# ---- D: single-pixel sigma against neighbor-pixel scatter -------------------------
axS = fig.add_subplot(mid[1].subgridspec(1, 3, width_ratios=[0.20, 1, 0.02])[1])
row_cd.append(axS)
for m, p, _, _, col in ROWS:
    g = scat[(scat.Map == m) & (scat.Phase == p)]
    axS.scatter(g.sigma_px, g.map_local_sd, s=70, c=col, ec="k", lw=0.5, zorder=3, marker=MAP_MARKER[m])
lim = (0.1, 6)
# shade the span of each map's points, so the two noise levels read at a glance;
# names right of / below each box, clear of the legend and the other map's points
for smp, g in scat.groupby("Map"):
    x0, x1 = g.sigma_px.min() / 1.15, g.sigma_px.max() * 1.15
    y0, y1 = g.map_local_sd.min() / 1.15, g.map_local_sd.max() * 1.15
    axS.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="0.5", alpha=0.10, ec="0.4",
                            lw=0.8, ls="--", zorder=1))
    if smp == "MH0811b":
        axS.text(x1 * 1.08, y0, smp, fontsize=FS_S, color="0.25", ha="left", va="bottom")
    else:
        axS.text(x1, y0 / 1.08, smp, fontsize=FS_S, color="0.25", ha="right", va="top")
axS.plot(lim, lim, color="0.4", ls=":", lw=1, zorder=1)
axS.text(3.0, 3.9, "1:1", fontsize=FS_S, color="0.35", rotation=45, ha="center")
axS.set_xscale("log"); axS.set_yscale("log"); axS.set_xlim(lim); axS.set_ylim(lim)
axS.set_xlabel("Single pixel 1$\\sigma$ (wt.%)", fontsize=FS_L)
axS.set_ylabel("Neighbor pixel 1$\\sigma$ (wt.%)", fontsize=FS_L)
axS.set_title("D. Single pixel vs neighbor pixel 1$\\sigma$", fontsize=FS_T, loc="left", pad=8)
axS.set_box_aspect(1); axS.tick_params(labelsize=FS_K); axS.grid(color="0.92", lw=0.8, zorder=0)
# legend columns: MH0811b, then Bii; blanks pad the shorter MH0811b column
col_handles = [[Line2D([], [], ls="", marker=MAP_MARKER[m], ms=8, mfc=col, mec="k", mew=0.5, label=short)
                for m, _, _, short, col in ROWS if m == smp] for smp in MAPS]
n_bii = len(col_handles[1])
sig_handles = col_handles[0] + [Line2D([], [], ls="", marker="", label=" ")] * (n_bii - len(col_handles[0])) \
    + col_handles[1]

# ---- E: results -----------------------------------------------------------------
bot = outer[4].subgridspec(1, len(COLS), wspace=0.08)
pct = lambda v: np.clip(np.asarray(v) / 100, 0.001, 0.999)   # logit axis: 0% and 100% at its limits
n = len(ROWS)
ys = np.arange(n)
# one mean row per map, below the minerals: the two maps differ greatly in quality
MEANS = [(smp, n + 0.75 + i, [k for k, r in enumerate(ROWS) if r[0] == smp]) for i, smp in enumerate(MAPS)]
axes = []
for j, title in enumerate(COLS):
    ax = fig.add_subplot(bot[j], sharey=axes[0] if axes else None)
    axes.append(ax)
    ax.axhspan(4.5, n - 0.5, color="0.96", zorder=0)                  # Bii rows
    ax.axhspan(MEANS[1][1] - 0.5, MEANS[1][1] + 0.5, color="0.96", zorder=0)   # Bii mean
    for md in MODELS[::-1]:
        ax.scatter(pct(VAL[md][:, j]), ys, s=80, marker=MARKERS[md], **FILL[md])
        for smp, y, idx in MEANS:
            mv = VAL[md][idx, j].mean()
            ax.scatter([pct(mv)], [y], s=128, marker=MARKERS[md], **FILL[md])
            if md != "GNB":            # mineralML then Random Forest, in the empty left of the row
                ax.annotate(f"{mv:.1f}", (0.03 if md == "mineralML" else 0.15, y),
                            xycoords=("axes fraction", "data"), ha="left", va="center",
                            fontsize=FS_S, fontweight="bold", color=COLORS[md])
    ax.axhline(n - 0.1, color="0.6", lw=0.8)
    ax.set_xscale("logit")
    ax.set_xlim(0.0007, 0.9993)
    ax.set_xticks([0.001, 0.1, 0.5, 0.9, 0.99, 0.999], ["0.1", "10", "50", "90", "99", "99.9"])
    ax.minorticks_off()
    ax.tick_params(labelsize=FS_K, labelleft=j == 0)
    ax.grid(axis="x", color="0.9", lw=0.8, zorder=0)
    ax.set_title(title, fontsize=FS_L, pad=8)
axes[0].set_yticks(list(ys) + [y for _, y, _ in MEANS],
                   [r[2] for r in ROWS] + [f"Mean, {smp} ({len(idx)})" for smp, _, idx in MEANS], fontsize=FS_K)
axes[0].set_ylim(MEANS[-1][1] + 0.6, -0.6)


# rows C/D and E span the same width as the maps, whose extent is fixed by their aspect
def fit_row(axs, L, R):
    fig.canvas.draw()                             # apply each axes' aspect before measuring
    pos = [a.get_position() for a in axs]
    cl, cr = min(q.x0 for q in pos), max(q.x1 for q in pos)
    f = (R - L) / (cr - cl)
    for a, q in zip(axs, pos):
        a.set_position([L + (q.x0 - cl) * f, q.y0, q.width * f, q.height])


fig.canvas.draw()
_L, _R = map_axes[0].get_position().x0, map_axes[1].get_position().x1
fit_row(row_cd, _L, _R)
fit_row(axes, _L, _R)
fig.canvas.draw()

# column-group headers, placed from the axes positions so they stay aligned
pA, pC, pD = (axes[i].get_position() for i in (0, 2, 3))
yh = pA.y1 + 0.028
for x0, x1, txt in [(pA.x0, pC.x1, "Sampled Map Pixels, no added error"),
                    (pD.x0, pD.x1, "5×5 means, total held")]:
    fig.add_artist(Line2D([x0, x1], [yh - 0.004] * 2, color="0.3", lw=1.2, transform=fig.transFigure))
    fig.text((x0 + x1) / 2, yh, txt, ha="center", va="bottom", fontsize=FS_L)
fig.text(0.125, yh + 0.016, "E. Classification of the sampled regions", fontsize=FS_T, ha="left", va="bottom")
xc = (pA.x0 + pD.x1) / 2
fig.text(xc, pA.y0 - 0.026, "% of regions classified as the EBSD phase (logit scale)",
         ha="center", va="top", fontsize=FS_L)
fig.legend(handles=[Line2D([], [], ls="", marker=MARKERS[m], ms=10, label=NAME[m],
                           **({"mfc": "none", "mec": COLORS[m], "mew": 2.0} if m == "RF"
                              else {"mfc": COLORS[m], "mec": "k", "mew": 0.6})) for m in MODELS],
           loc="upper center", bbox_to_anchor=(xc, pA.y0 - 0.040), ncol=4, fontsize=FS_K, frameon=False)
# map brackets left of the row labels, in the style of the column headers
inv = fig.transFigure.inverted()
lab_x0 = min(t.get_window_extent(fig.canvas.get_renderer()).x0 for t in axes[0].get_yticklabels() if t.get_text())
x_line = inv.transform((lab_x0, 0))[0] - 0.008
# x at the left limit: x = 0 is undefined on the logit axis
to_fig = lambda y: inv.transform(axes[0].transData.transform((axes[0].get_xlim()[0], y)))[1]
for smp, _, idx in MEANS:
    y0, y1 = to_fig(min(idx) - 0.35), to_fig(max(idx) + 0.35)
    fig.add_artist(Line2D([x_line] * 2, [y0, y1], color="0.3", lw=1.2, transform=fig.transFigure))
    fig.text(x_line - 0.006, (y0 + y1) / 2, smp, rotation=90, ha="right", va="center", fontsize=FS_L)
# phase legend under the maps, C title above C, D legend in D's empty upper-left corner
fig.legend(handles=phase_handles, loc="upper center",
           bbox_to_anchor=((_L + _R) / 2, map_axes[0].get_position().y0 - 0.004),
           ncol=len(phase_handles), fontsize=FS_S, frameon=False, handlelength=1.2, columnspacing=1.2)
fig.text(0.125, axE_first.get_position().y1 + 0.026, "C. Example Regions: 1×1, 3×3 and 5×5 boxes",
         fontsize=FS_T, ha="left", va="bottom")
axS.legend(handles=sig_handles, loc="upper left", ncol=2, fontsize=FS_S - 1, frameon=False,
           handletextpad=0.1, columnspacing=0.5, borderaxespad=0.3, labelspacing=0.25,
           handlelength=1.2, markerscale=0.85,
           title="MH0811b  Bii", title_fontsize=FS_S - 1, alignment="left")

# Uncompressed PDF: with compression on, matplotlib stores images of <= 256 colours
# (every phase map) as low-bit-depth indexed images, which Illustrator misreads,
# showing only one phase. Uncompressed, they are written as plain RGB.
with plt.rc_context({"pdf.compression": 0}):
    fig.savefig(os.path.join(HERE, "AZTEC_SinglePx_Figure_logit.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "AZTEC_SinglePx_Figure_logit.png"), dpi=110, bbox_inches="tight")

# %%
