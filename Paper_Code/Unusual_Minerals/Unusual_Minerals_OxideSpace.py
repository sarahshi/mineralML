# %%
"""
Where the Smithsonian standards that mineralML was not trained on sit in
oxide space, against the training classes they are called.

  Meionite  (Na-bearing scapolite; CO2 2.5, SO3 1.3 wt% not in the 11 oxides)
            -> called plagioclase: compare with training plagioclase
  Osumilite -> called glass by the current model; the manuscript said
            Na-pyroxene: compare with training glass, Na-rich clinopyroxene
            (Na2O > 8 wt%, jadeite/aegirine-like) and alkali feldspar
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
plt.style.use(os.path.join(REPO, "style.mplstyle"))
FS = 12

OX = ["SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO", "CaO", "Na2O", "K2O", "Cr2O3", "P2O5"]
t = pd.read_csv(os.path.join(REPO, "Training_Data/min_df_v4_synth.csv"), low_memory=False)
t[OX] = t[OX].fillna(0)
t["FeMg"] = t.FeOt + t.MgO

STD = {  # Smithsonian preferred values, 11 oxides (osumilite CaO "<0.03" -> 0)
    "Meionite": dict(SiO2=49.78, Al2O3=25.05, FeOt=0.17, CaO=13.58, Na2O=5.20, K2O=0.94),
    "Osumilite": dict(SiO2=60.20, TiO2=0.18, Al2O3=22.60, FeOt=6.38, MgO=5.83, Na2O=0.39, K2O=4.00),
}
std = pd.DataFrame(STD).T.reindex(columns=OX).fillna(0)
std["FeMg"] = std.FeOt + std.MgO

REF = {
    "Meionite": [("Plagioclase", t[t.Mineral == "Plagioclase"], "#3a9e9e")],
    "Osumilite": [("Glass", t[t.Mineral == "Glass"], "#d9a400"),
                  ("Na-rich clinopyroxene (Na$_2$O > 8)", t[(t.Mineral == "Clinopyroxene") & (t.Na2O > 8)], "#c0392b"),
                  ("Alkali feldspar", t[t.Mineral == "Alkali_Feldspar"], "#7b6fb0")],
}
PANELS = {
    "Meionite": [("SiO2", "Al2O3"), ("CaO", "Na2O"), ("Al2O3", "CaO")],
    "Osumilite": [("SiO2", "Al2O3"), ("Na2O", "K2O"), ("SiO2", "FeMg")],
}
LAB = {"SiO2": "SiO$_2$", "Al2O3": "Al$_2$O$_3$", "CaO": "CaO", "Na2O": "Na$_2$O",
       "K2O": "K$_2$O", "FeMg": "FeO$_t$ + MgO"}
CALL = {"Meionite": "called plagioclase (labradorite), score 0.96",
        "Osumilite": "called glass, score 0.91 (manuscript: Na-pyroxene, 0.97)"}

rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), gridspec_kw=dict(hspace=0.42, wspace=0.28))
for r, name in enumerate(["Meionite", "Osumilite"]):
    for c, (x, y) in enumerate(PANELS[name]):
        ax = axes[r, c]
        for lab, d, col in REF[name]:
            d = d.sample(min(len(d), 4000), random_state=0)
            ax.scatter(d[x], d[y], s=6, color=col, alpha=0.35, lw=0, zorder=1, rasterized=True)
        ax.scatter(std.loc[name, x], std.loc[name, y], s=320, marker="*", color="k",
                   ec="white", lw=1.2, zorder=5)
        ax.set_xlabel(f"{LAB[x]} (wt.%)", fontsize=FS)
        ax.set_ylabel(f"{LAB[y]} (wt.%)", fontsize=FS)
        ax.tick_params(labelsize=FS - 1)
        ax.grid(color="0.92", lw=0.8, zorder=0)
    tot = std.loc[name, OX].sum()
    axes[r, 0].set_title(f"{'AB'[r]}. {name}: {CALL[name]}\n11-oxide total {tot:.1f} wt.%",
                         loc="left", fontsize=FS + 1)
    handles = [Line2D([], [], ls="", marker="*", ms=16, mfc="k", mec="white", label=f"{name} (Smithsonian)")]
    handles += [Line2D([], [], ls="", marker="o", ms=7, mfc=col, mec="none", alpha=0.7,
                       label=f"training {lab.lower() if lab != 'Glass' else 'glass'}")
                for lab, _, col in REF[name]]
    axes[r, 2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                      frameon=False, fontsize=FS - 1)
with plt.rc_context({"pdf.compression": 0}):
    fig.savefig(os.path.join(HERE, "Unusual_Minerals_OxideSpace.pdf"), bbox_inches="tight", dpi=300)
print("wrote Unusual_Minerals_OxideSpace.pdf")
