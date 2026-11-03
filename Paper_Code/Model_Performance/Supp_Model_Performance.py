# %%
"""
Figure S2: training losses and held-out performance of the shipped hybrid NN (v0030).
Metrics come from Model_Performance_Metrics.py. * marks classes resolved by
the empirical rules rather than the network.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

plt.style.use('../../style.mplstyle')

ck = torch.load("../../src/mineralML/nnwr_best_model_v0030.pt", map_location="cpu", weights_only=False)
tl, vl = ck["train_losses"], ck["valid_losses"]
perclass = pd.read_csv("Model_Performance_perclass.csv")
overall = pd.read_csv("Model_Performance_overall.csv").set_index("dataset")


class HandlerThin(HandlerPatch):
    """Legend swatch at a fraction of the row height, matching the precision outline."""

    def __init__(self, frac, **kw):
        self.frac = frac
        super().__init__(**kw)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        src = orig_handle.patches[0]
        h = height * self.frac
        p = mpatches.Rectangle((-xdescent, -ydescent + (height - h) / 2), width, h,
                               facecolor=src.get_facecolor(), edgecolor=src.get_edgecolor(),
                               linewidth=max(src.get_linewidth(), 0.8), linestyle=src.get_linestyle(),
                               hatch=src.get_hatch(), alpha=src.get_alpha())
        p.set_transform(trans)
        return [p]


TAIL_FRAC = 0.55     # precision outline height, as a fraction of the recall bar
HDR_XY = (0.75, 0.97)
C_TRAIN, C_VALID = "#0C7BDC", "k"

fig = plt.figure(figsize=(24, 15))
gs = fig.add_gridspec(2, 6, hspace=0.3, wspace=0.8)


def _panel(ax, letter, title):
    ax.set_title(title, fontsize=20)
    ax.text(-0.1, 1.0, letter, transform=ax.transAxes, fontsize=22, fontweight="bold", va="bottom")
    ax.grid(True, alpha=0.25, lw=0.7)


def short(m):
    return m.replace("_", " ").replace("Rhombohedral", "Rhomb.")


# ---- A: classification loss ----
axA = fig.add_subplot(gs[0, 0:3])
ce_t, ce_v = np.asarray(tl["cls_classification"]), np.asarray(vl["cls_classification"])
best = int(ce_v.argmin())
axA.plot(ce_t, color=C_TRAIN, lw=1.6, label="Train")
axA.plot(ce_v, color=C_VALID, lw=1.6, label="Valid")
axA.axvline(best, color="k", ls=":", lw=1.3)
axA.plot([best], [ce_v[best]], "o", color="k", ms=7, zorder=5)
axA.annotate(f"best epoch {best}\nCE = {ce_v[best]:.4f}", xy=(best, ce_v[best]), xytext=(-18, 34),
             textcoords="offset points", fontsize=14, ha="right",
             arrowprops=dict(arrowstyle="-", lw=0.8, color="k"))
axA.text(0.5, 0.9, "valid < train loss: dropout active in training", transform=axA.transAxes,
         fontsize=14, style="italic", va="bottom", ha="center")
axA.set_xlabel("Epoch"); axA.set_ylabel("Cross-Entropy Loss")
axA.legend(fontsize=12, loc="upper right")
_panel(axA, "A.", "Classification Loss")

axAi = axA.inset_axes([0.44, 0.30, 0.52, 0.46])
axAi.plot(ce_t, color=C_TRAIN, lw=1.3)
axAi.plot(ce_v, color=C_VALID, lw=1.3)
axAi.axvline(best, color="k", ls=":", lw=1.0)
axAi.plot([best], [ce_v[best]], "o", color="k", ms=5, zorder=5)
axAi.set_yscale("log")
axAi.set_xlim(0, len(ce_t) - 1)
axAi.tick_params(labelsize=11)
axAi.grid(True, alpha=0.25, lw=0.6, which="both")
axAi.set_title("all epochs, log scale", fontsize=12, pad=4)

# ---- B: reconstruction loss ----
axB = fig.add_subplot(gs[0, 3:6])
rc_t, rc_v = np.asarray(tl["dec_reconstruction"]), np.asarray(vl["dec_reconstruction"])
b2 = int(rc_v.argmin())
axB.plot(rc_t, color=C_TRAIN, lw=1.6, label="Train")
axB.plot(rc_v, color=C_VALID, lw=1.6, label="Valid")
axB.plot([b2], [rc_v[b2]], "o", color="k", ms=7, zorder=5)
axB.annotate(f"Best MSE = {rc_v[b2]:.4f}", xy=(b2, rc_v[b2]), xytext=(-10, 45),
             textcoords="offset points", fontsize=14, ha="right")
axB.set_xlabel("Epoch"); axB.set_ylabel("Reconstruction Mean-Squared\nError")
axB.legend(fontsize=12, loc="upper right")
_panel(axB, "B.", r"Reconstruction Loss from z$\mathregular{_2}$")

axBi = axB.inset_axes([0.44, 0.30, 0.52, 0.46])
axBi.plot(rc_t, color=C_TRAIN, lw=1.3)
axBi.plot(rc_v, color=C_VALID, lw=1.3)
axBi.plot([b2], [rc_v[b2]], "o", color="k", ms=5, zorder=5)
axBi.set_yscale("log")
axBi.set_xlim(0, len(rc_t) - 1)
axBi.tick_params(labelsize=11)
axBi.grid(True, alpha=0.25, lw=0.6, which="both")
axBi.set_title("all epochs, log scale", fontsize=12, pad=4)

# ---- C: macro metrics per evaluation set ----
axC = fig.add_subplot(gs[1, 0:2])
ov = overall.loc[["Held-out validation", "Secondary standards", "GEOROC",
                  "EDS map MH0811b_pool", "EDS map Bii_pool"]]
x = np.arange(len(ov)); w = 0.26
for i, (col, name, fc, ls) in enumerate([("macro_recall", "Recall", "white", "-"),
                                         ("macro_precision", "Precision", "white", "--"),
                                         ("macro_f1", "F1", "#929292", "-")]):
    axC.bar(x + (i - 1)*w, ov[col], w, facecolor=fc, label=name, edgecolor="k", lw=1.1, linestyle=ls)
for xi, nc, top in zip(x, ov["n_classes"], ov[["macro_precision", "macro_recall", "macro_f1"]].max(axis=1)):
    axC.text(xi, top + 2.0, f"{int(nc)}\nclasses", ha="center", va="bottom",
             fontsize=12, style="italic", linespacing=0.95)
axC.set_xticks(x)
axC.set_xticklabels(["Held-out\nValidation", "Secondary\nStandards", "GEOROC",
                     "EDS map\nMH0811b", "EDS map\nBii"], fontsize=12)
axC.set_ylabel("Macro Score (%)")
axC.set_ylim(0, 115)
axC.set_yticks([0, 20, 40, 60, 80, 100])
axC.legend(fontsize=12, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.155), frameon=False)
_panel(axC, "C.", "Validation Dataset Metrics")

# ---- D: GEOROC per class. Bar = recall, dashed outline spans to precision ----
axD = fig.add_subplot(gs[1, 2:4])
g = perclass[perclass["dataset"] == "GEOROC"].sort_values("mineral", ascending=False)
y = np.arange(len(g))
rD, pD = g["recall"].to_numpy(), g["precision"].to_numpy()
axD.barh(y, rD, 0.62, facecolor="white", edgecolor="k", lw=0.9, label="Recall", zorder=2)
m = pD != rD
axD.barh(y[m], abs(pD - rD)[m], 0.62*TAIL_FRAC, left=np.minimum(rD, pD)[m], facecolor="none",
         edgecolor="k", lw=1.0, ls="--", zorder=4, label="Precision")
for yi, rv, pv in zip(y, rD, pD):
    axD.text(max(rv, pv) + 1.5, yi, f"{rv:.0f} ({pv:.0f})", va="center", ha="left",
             fontsize=10, color="0.15", zorder=6)
axD.text(*HDR_XY, "Recall (Precision)", transform=axD.transAxes, va="center", ha="left",
         fontsize=12, color="0.15", style="italic")
axD.set_yticks(y)
axD.set_yticklabels([short(mn) + ("*" if e else "") for mn, e in zip(g["mineral"], g["empirical"])], fontsize=12)
axD.set_xlabel("Score (%)")
axD.set_xlim(0, 120)
h, l = axD.get_legend_handles_labels()
axD.legend(h, l, fontsize=12, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.155),
           frameon=False, handler_map={h[1]: HandlerThin(TAIL_FRAC)})
_panel(axD, "D.", "GEOROC - Per-Class Metrics")

# ---- E: EDS maps vs EBSD per class, feldspar pooled ----
axE = fig.add_subplot(gs[1, 4:6])
mp = perclass[perclass["dataset"].isin(["EDS map MH0811b_pool", "EDS map Bii_pool"])].copy()
mp["sample"] = mp["dataset"].str.replace("EDS map ", "").str.replace("_pool", "")
cols = ["MH0811b", "Bii"]
R = mp.pivot_table(index="mineral", columns="sample", values="recall")[cols].fillna(0)
R = R[R.max(axis=1) > 0].sort_index(ascending=False)
P = mp.pivot_table(index="mineral", columns="sample", values="precision")[cols].fillna(0).loc[R.index]
emp = set(mp.loc[mp["empirical"], "mineral"])
minerals = R.index
R, P = R.to_numpy(), P.to_numpy()
hgt = 0.4
y = np.arange(len(R))
Y = y[:, None] + np.array([hgt/2, -hgt/2])
for i, (c, fc) in enumerate(zip(cols, ["#BFBFBF", "#F0F0F0"])):
    axE.barh(Y[:, i], R[:, i], hgt, facecolor=fc, edgecolor="k", lw=1.0, label=c, zorder=2)
m = P != R
axE.barh(Y[m], abs(P - R)[m], hgt*TAIL_FRAC, left=np.minimum(R, P)[m], facecolor="none",
         edgecolor="k", lw=1.2, ls="--", zorder=4, label="Precision")
for yi, rv, pv in zip(Y[R > 0], R[R > 0], P[R > 0]):
    axE.text(max(rv, pv) + 1.5, yi, f"{rv:.0f} ({pv:.0f})", va="center", ha="left",
             fontsize=12, color="0.15", zorder=6)
axE.text(*HDR_XY, "Recall (Precision)", transform=axE.transAxes, va="center", ha="left",
         fontsize=12, color="0.15", style="italic")
axE.set_yticks(y)
axE.set_yticklabels([short(mn) + ("*" if mn in emp else "") for mn in minerals], fontsize=12)
axE.set_xlabel("Score vs. EBSD (%)")
axE.set_xlim(0, 120)
h, l = axE.get_legend_handles_labels()
axE.legend(h, l, fontsize=11, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.155),
           frameon=False, handler_map={h[2]: HandlerThin(TAIL_FRAC)})
_panel(axE, "E.", "EBSD-EDS maps - Per-Class Metrics")

plt.tight_layout()
fig.savefig("Model_Performance.pdf", dpi=300, bbox_inches="tight")
