# %%
"""
Figure 10 (main text): mineralML against the Random Forest and Gaussian Naive Bayes
baselines (grouped labels). A, agreement with the reference labels on each
evaluation dataset; B, per-class F1 on GEOROC (* resolved by the empirical rules).
Reads Baseline_Numbers.json and the per-class CSVs. Writes Figure_Baseline_Comparison_stacked.pdf.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

plt.style.use("../../style.mplstyle")
plt.rcParams["mathtext.default"] = "regular"
FS_TITLE, FS_AXIS, FS_TICK, FS_HEAT = 22, 21, 20, 20

ORDER = ["Gaussian Naive Bayes", "Random Forest", "mineralML"]
C = {"Gaussian Naive Bayes": "#eda100", "Random Forest": "#eb6834", "mineralML": "#2a78d6"}
EMPIRICAL = {"Zircon", "Carbonate", "SiO2_Polymorph"}
F1_CMAP = LinearSegmentedColormap.from_list("f1", ["#f7fbff", "#c6dcef", "#6aaed6", "#2a78d6", "#0b3d80"])
TWO_LINE = {"Random Forest": "Random\nForest", "Gaussian Naive Bayes": "Gaussian\nNaive Bayes"}


def star(m):
    m_ = {"Rhombohedral_Oxides": "Rhomb. Oxides"}.get(m, m.replace("_", " ").replace("SiO2", "SiO$_2$"))
    return m_ + ("*" if m in EMPIRICAL else "")


J = json.load(open("Baseline_Numbers.json"))
KEY = {"mineralML": "mineralML", "Random Forest": "RF grouped", "Gaussian Naive Bayes": "GNB grouped"}
GEO = {"mineralML": J["georoc"]["mineralML"],
       "Random Forest": J["georoc"]["Random Forest (tuned) grouped"],
       "Gaussian Naive Bayes": J["georoc"]["Gaussian Naive Bayes grouped"]}
DATASETS = ["Held-out\nvalidation", "Secondary\nstandards", "GEOROC", "EDS map\nMH0811b", "EDS map\nBii"]
agree = {m: [J["heldout"][KEY[m]]["accuracy"], J["standards"][KEY[m]]["accuracy"], GEO[m],
             J["maps"]["MH0811b"][KEY[m]]["agreement_pool"], J["maps"]["Bii"][KEY[m]]["agreement_pool"]]
         for m in ORDER}

nn = pd.read_csv("Model_Performance_perclass.csv").query("dataset == 'GEOROC'").assign(model="mineralML")
bl = pd.read_csv("Baseline_perclass.csv").replace({"model": {"Random Forest (tuned)": "Random Forest"}})
f1 = pd.concat([nn, bl]).pivot_table(index="mineral", columns="model", values="f1")
f1 = f1.reindex(index=sorted(f1.index))[ORDER[::-1]].T    # models down the side, minerals across

fig, ax = plt.subplots(2, 1, figsize=(19, 15), gridspec_kw={"height_ratios": [1, 0.42], "hspace": 0.30})

# ---- A: agreement on each dataset ----
x = np.arange(len(DATASETS)); w = 0.26
for i, m in enumerate(ORDER):
    ax[0].bar(x + (i - 1) * w, agree[m], w, color=C[m], edgecolor="k", lw=0.6, label=m, zorder=3)
    for xi, vi in zip(x + (i - 1) * w, agree[m]):
        ax[0].text(xi, vi + 1.5, f"{vi:.0f}", ha="center", va="bottom", fontsize=FS_TICK, rotation=90)
ax[0].set_xticks(x, DATASETS, fontsize=FS_TICK)
ax[0].set_ylabel("Agreement with reference (%)", fontsize=FS_AXIS)
ax[0].set_ylim(0, 128); ax[0].set_yticks([0, 20, 40, 60, 80, 100])
ax[0].tick_params(labelsize=FS_TICK)
ax[0].grid(axis="y", color="0.92", lw=0.8, zorder=0)
ax[0].spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False, loc="upper right", ncol=3, fontsize=FS_TICK, bbox_to_anchor=(1.0, 1.04),
             columnspacing=0.8, handlelength=1.2, handletextpad=0.4)
ax[0].set_title("A.  Performance on each dataset", fontsize=FS_TITLE, loc="left", pad=10)

# ---- B: per-class F1 on GEOROC ----
vmin = max(55.0, float(np.floor(f1.min().min() / 5) * 5))
ax[1].imshow(f1.values, aspect="auto", cmap=F1_CMAP, vmin=vmin, vmax=100)
for i in range(f1.shape[0]):
    for j in range(f1.shape[1]):
        v = f1.values[i, j]
        # numbers above the hatching, on a patch of the cell colour
        box = (dict(boxstyle="square,pad=0.15", facecolor=F1_CMAP((v - vmin) / (100 - vmin)), edgecolor="none")
               if f1.columns[j] in EMPIRICAL else None)
        ax[1].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=FS_HEAT, zorder=7, bbox=box,
                   color="w" if v > vmin + 0.62 * (100 - vmin) else "0.15")
for k, m in enumerate(f1.columns):
    if m in EMPIRICAL:
        ax[1].add_patch(plt.Rectangle((k - 0.5, -0.5), 1, f1.shape[0], fill=False, hatch="///",
                                      edgecolor="0.35", lw=0.0, zorder=5))
        ax[1].add_patch(plt.Rectangle((k - 0.5, -0.5), 1, f1.shape[0], fill=False, edgecolor="0.2", lw=1.2, zorder=6))
ax[1].set_xticks(range(f1.shape[1]), [star(m) for m in f1.columns], fontsize=FS_TICK,
                 rotation=45, ha="right", rotation_mode="anchor")
ax[1].set_yticks(range(f1.shape[0]), [TWO_LINE.get(m, m) for m in f1.index], fontsize=FS_TICK)
ax[1].tick_params(length=0)
ax[1].legend(handles=[mpatches.Patch(facecolor="white", edgecolor="0.2", hatch="///", lw=1.0, label="empirical filter")],
             loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=FS_TICK)
ax[1].set_title("B.  Per-class F1 on GEOROC (%)", fontsize=FS_TITLE, loc="left", pad=10)

fig.savefig("Figure_Baseline_Comparison_stacked.pdf", bbox_inches="tight")
