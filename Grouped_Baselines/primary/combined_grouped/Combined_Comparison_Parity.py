# %%
"""
Combine the overall-performance and per-class GEOROC panels into one figure.

The AGU SI class redefines \\caption with a plain \\def, so a float cannot
carry two captions, and as two separate floats these figures overflow a page.
Emitting them as a single file means the document needs one
\\includegraphics and one ordinary \\caption.

Both blocks are drawn natively onto the same canvas -- the per-class script is
executed with HOST_FIG/HOST_GS set, so it adds its axes to this figure rather
than creating its own. Nothing is rasterised, and the two blocks share one
coordinate system, so their widths and font sizes match.

  A      agreement on every evaluation dataset, ordered by independence
  B--E   the per-class GEOROC comparison, drawn by GEOROC_PerClass_Figure.py
"""

import json
import runpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("./style.mplstyle")
plt.rcParams["mathtext.default"] = "regular"


ORDER = ["Gaussian Naive Bayes", "Random Forest (tuned)", "mineralML"]
SHORT = {"Gaussian Naive Bayes": "Gaussian Naive Bayes (grouped)",
         "Random Forest (tuned)": "Random Forest (grouped)",
         "mineralML": "mineralML"}
C = {"Gaussian Naive Bayes": "#eda100",
     "Random Forest (tuned)": "#eb6834",
     "mineralML": "#2a78d6"}

# GROUPED-LABEL COPY (Grouped_Baselines/primary/combined_grouped). Panel A is
# computed from the grouped-baseline results instead of hardcoded. The same code
# run for the fine scheme must reproduce the values the repo version hardcodes.
_ORIG = {"Gaussian Naive Bayes": [91, 95, 82, 30, 33],
         "Random Forest (tuned)": [100, 100, 95, 74, 92],
         "mineralML": [100, 100, 97, 89, 96]}
GB = "../.."                                   # Grouped_Baselines/
_sc = pd.read_csv(f"{GB}/data/grouped_baselines_scores.csv")
_mp = pd.read_csv(f"{GB}/data/grouped_baselines_maps.csv")
_mp = _mp[_mp["maps"] == "renorm"]
_st = pd.read_csv(f"{GB}/primary/standards_accuracy.csv")
_nn = pd.read_csv("Model_Performance_overall.csv").set_index("dataset")["accuracy"]
_KEY = {"Random Forest (tuned)": "RF (tuned)", "Gaussian Naive Bayes": "GNB"}


def panel_a(scheme):
    out = {"mineralML": [_nn["Held-out validation"], _nn["Secondary standards"], _nn["GEOROC"],
                         _nn["EDS map MH0811b_pool"], _nn["EDS map Bii_pool"]]}
    ov = pd.read_csv(f"{GB}/primary/{scheme}/Model_Performance_RF_overall.csv")
    for m, k in _KEY.items():
        mk = f"{k} | {scheme}"
        out[m] = [ov.loc[(ov.model == m) & (ov.dataset == "Held-out validation"), "accuracy"].item(),
                  _st.loc[(_st.model == m) & (_st.scheme == scheme), "accuracy"].item(),
                  ov.loc[(ov.model == m) & (ov.dataset == "GEOROC"), "accuracy"].item(),
                  _mp.loc[(_mp.model == mk) & (_mp["sample"] == "MH0811b"), "agreement_fsp_pooled"].item(),
                  _mp.loc[(_mp.model == mk) & (_mp["sample"] == "Tuolumne_Bii"), "agreement_fsp_pooled"].item()]
    return out


_fine = panel_a("fine")
for m in _ORIG:
    assert [round(v) for v in _fine[m]] == _ORIG[m], (m, _fine[m], _ORIG[m])
_A = panel_a("grouped")
print("panel A, fine   :", {m: [round(v, 1) for v in x] for m, x in _fine.items()})
print("panel A, grouped:", {m: [round(v, 1) for v in x] for m, x in _A.items()})
_N = ["Held-out\nvalidation", "Secondary\nstandards", "GEOROC",
      "EDS map\nMH0811b", "EDS map\nBii"]
DSETS = [(n, (lambda i: lambda m: _A[m][i])(i)) for i, n in enumerate(_N)]

# %% ------------------------------------------------------------- canvas ----
# Height splits as the two blocks need: the overview is one axes, the
# per-class block is a 2 x 3 arrangement and needs roughly three times as much.
fig = plt.figure(figsize=(20, 16))
outer = fig.add_gridspec(2, 1, height_ratios=[0.34, 1.0], hspace=0.20,
                         left=0.06, right=0.97)

# ---- A: overview across datasets -------------------------------------------
axA = fig.add_subplot(outer[0])
x = np.arange(len(DSETS)); w = 0.26
for i, m in enumerate(ORDER):
    v = [g(m) for _, g in DSETS]
    axA.bar(x + (i - 1) * w, v, w, color=C[m], edgecolor="k", lw=0.6,
            label=SHORT[m], zorder=3)
    for xi, vi in zip(x + (i - 1) * w, v):
        axA.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", fontsize=11,
                 rotation=90, va="bottom")
axA.set_xticks(x)
axA.set_xticklabels([d for d, _ in DSETS], fontsize=13)
axA.set_ylabel("Agreement with reference (%)", fontsize=14)
axA.set_ylim(0, 120)
axA.tick_params(labelsize=13)
axA.grid(axis="y", color="0.92", lw=0.8, zorder=0)
axA.spines[["top", "right"]].set_visible(False)
axA.legend(frameon=False, loc="upper right", ncol=3, fontsize=12,
           bbox_to_anchor=(1.0, 1.02))
axA.set_title("A.", fontsize=15, loc="left", pad=10)

# ---- B-E: the per-class block, drawn into this same figure -----------------
runpy.run_path("GEOROC_PerClass_Parity.py",
               init_globals={"HOST_FIG": fig, "HOST_GS": outer[1],
                             "PANEL_LETTERS_OVERRIDE": ["B.", "C.", "D.", "E.", "F."]})

fig.savefig("Combined_Comparison_Parity.pdf", bbox_inches="tight")
fig.savefig("Combined_Comparison_Parity.png", bbox_inches="tight", dpi=140)
print("wrote Combined_Comparison_Parity.pdf / .png")

# %%
