# %%
"""
Figure S6: phase maps from mineralML and the baselines, and beneath
each, where it disagrees with EBSD (grain interiors vs grain boundaries),
feldspars pooled. Reads Baseline_labelmaps_<sample>.npz from Baseline_Numbers.py.
"""

import os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.style.use(os.path.join(HERE, "..", "..", "style.mplstyle"))

MODELS = ["mineralML", "Random Forest (tuned)", "Gaussian Naive Bayes"]
SHORT = {"Gaussian Naive Bayes": "Gaussian Naive Bayes",
         "Random Forest (tuned)": "Random Forest (tuned)",
         "mineralML": "mineralML"}

PHASE_COLORS = {
    "Plagioclase": "#66C4C4", "Feldspar": "#66C4C4",
    "Alkali_Feldspar": "#FEF7C2", "Feldspar_Miscibility_Gap": "#003d36",
    "Clinopyroxene": "#E57A7A", "Orthopyroxene": "#931d1d",
    "Amphibole": "#5E2910", "Glass": "#F9C300",
    "Spinel_Group": "#2E2DCE", "Rhombohedral_Oxides": "#6FB2E4",
    "SiO2_Polymorph": "#A8A5A5", "Apatite": "#000000",
    "Titanite": "#E1D6DF", "Epidote": "magenta", "Zircon": "cyan",
    "Unindexed": "white", "Vesicles": "white",
    "Olivine": "#7BC043", "Biotite": "#8B5A2B", "Muscovite": "#D6A5D6",
    "Garnet": "#B03060", "Serpentine": "#4F7942", "Chlorite": "#2E8B57",
    "Tourmaline": "#5D3FD3", "Na-Pyroxene": "#FF8C00",
    "Melilite": "#C0C0C0", "Rutile": "#8B0000", "Carbonate": "#F2B5D4",
    "Kalsilite": "#9ACD32", "Leucite": "#FFD8B1", "Nepheline": "#B0E0E6",
}
NON_CRYST = {"Unindexed", "Glass", "Vesicles", "nan", "None", "NaN"}

C_OK       = (0.93, 0.93, 0.93)   # correct
C_BOUND    = (0.36, 0.61, 0.84)   # error on a grain boundary
C_INTERIOR = (0.85, 0.20, 0.20)   # error in a grain interior


def label_map_to_rgb(lab2d, mask_out=None):
    """Render a 2D array of phase-name strings to an RGB image."""
    h, w = lab2d.shape
    rgb = np.ones((h, w, 3))
    for phase in pd.unique(lab2d.ravel()):
        m = lab2d == phase
        rgb[m] = plt.matplotlib.colors.to_rgb(
            PHASE_COLORS.get(str(phase), "#DDDDDD"))
    if mask_out is not None:
        rgb[mask_out] = 1.0
    return rgb


from sklearn.metrics import balanced_accuracy_score
KEY = {"mineralML": "mineralML", "Random Forest (tuned)": "RF grouped",
       "Gaussian Naive Bayes": "GNB grouped"}
FSP = {"Plagioclase": "Feldspar", "Alkali_Feldspar": "Feldspar",
       "Feldspar_Miscibility_Gap": "Feldspar"}
pool = lambda a: pd.Series(a.ravel()).replace(FSP).to_numpy().reshape(a.shape)

for sample, nice, fname in [("MH0811b", "MH0811b", "MH0811b"), ("Bii", "Tuolumne Bii", "Tuolumne_Bii")]:
    zz = np.load(os.path.join(HERE, f"Baseline_labelmaps_{sample}.npz"), allow_pickle=True)
    z = {f"pred__{m}": pool(zz[f"pred::{k}"].astype(str)) for m, k in KEY.items()}
    ebsd = pool(zz["ebsd"].astype(str))
    edge = zz["on_edge"]
    unscored = np.isin(ebsd, list(NON_CRYST))
    scoreable = ~unscored
    ebp = pool(ebsd)
    agree, err = {}, {}
    for m in MODELS:
        pm = pool(z[f"pred__{m}"].astype(str))
        s = scoreable & (z[f"pred__{m}"].astype(str) != "nan")
        ok = pm[s] == ebp[s]
        agree[(sample, m)] = {"agreement": 100 * ok.mean(),
                              "balanced_agreement": 100 * balanced_accuracy_score(ebp[s], pm[s])}
        err[(sample, m)] = {"error_rate": 100 * (1 - ok.mean()),
                            "interior_error_rate": 100 * (pm[s & ~edge] != ebp[s & ~edge]).mean()}
        print(sample, m, {k: round(v, 2) for k, v in {**agree[(sample, m)], **err[(sample, m)]}.items()})
    agree = pd.DataFrame(agree).T
    err = pd.DataFrame(err).T

    # Panel size follows the map's own aspect, so a wide sample such as
    # Tuolumne Bii (2:1) does not sit in tall panels padded with dead space.
    h, w = ebsd.shape
    pw = 5.6
    ph = pw * h / w
    ncol = len(MODELS) + 1
    fig, axg = plt.subplots(2, ncol, figsize=(pw * ncol, 2 * ph + 1.5),
                            constrained_layout=True)

    # ---- top row: EBSD reference, then each model's phase map -------------
    eb_disp = np.where(unscored, "Unindexed", ebsd)
    axg[0, 0].imshow(label_map_to_rgb(eb_disp), interpolation="nearest")
    axg[0, 0].set_title("EBSD (ground truth)", fontsize=13, loc="left")
    for j, m in enumerate(MODELS, start=1):
        axg[0, j].imshow(label_map_to_rgb(z[f"pred__{m}"].astype(str)),
                         interpolation="nearest")
        a = agree.loc[(sample, m), "agreement"]
        b = agree.loc[(sample, m), "balanced_agreement"]
        axg[0, j].set_title(f"{SHORT[m]}\n{a:.1f}% agreement ({b:.1f}% balanced)",
                            fontsize=13, loc="left")

    # ---- bottom row: where each model is wrong, same column ---------------
    axg[1, 0].axis("off")     # no error map for the reference itself
    for j, m in enumerate(MODELS, start=1):
        pm = z[f"pred__{m}"].astype(str)
        wrong = scoreable & (pool(pm) != ebp)
        rgb = np.ones(ebsd.shape + (3,))
        rgb[scoreable] = C_OK
        rgb[wrong & edge] = C_BOUND
        rgb[wrong & ~edge] = C_INTERIOR
        rgb[~scoreable] = (1.0, 1.0, 1.0)
        axg[1, j].imshow(rgb, interpolation="nearest")
        r = err.loc[(sample, m)]
        axg[1, j].set_title(
            f"{r['error_rate']:.1f}% wrong — "
            f"{r['interior_error_rate']:.1f}% of grain interiors",
            fontsize=13, loc="left")

    for ax in axg.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ---- legends: phases under the top row, error key under the bottom ----
    stack = np.concatenate([ebsd.ravel()] +
                           [z[f"pred__{m}"].astype(str).ravel() for m in MODELS])
    vc = pd.Series(stack).value_counts()
    counts = pd.Series(ebsd[~unscored].ravel()).value_counts()
    shown = [p for p in vc.index
             if p not in ("nan", "None", "NaN", "Vesicles", "Unindexed")
             and vc[p] >= 0.005 * ebsd.size and p in PHASE_COLORS]
    shown = sorted(shown, key=lambda p: -counts.get(p, 0))[:10]

    fig.canvas.draw()
    pos_eb = axg[0, 0].get_position()      # EBSD panel, top-left
    pos_e1 = axg[1, 1].get_position()      # first error map, sets the row top
    fig.set_layout_engine("none")

    ph_handles = [mpatches.Patch(color=PHASE_COLORS[p],
                                 label=p.replace("_", " ")) for p in shown]
    ph_handles.append(mpatches.Patch(facecolor="white", edgecolor="0.55",
                                     linewidth=0.8, label="Unindexed"))
    er_handles = [
        mpatches.Patch(color=C_INTERIOR, label="error in a grain interior"),
        mpatches.Patch(color=C_BOUND, label="error on a grain boundary"),
        mpatches.Patch(color=C_OK, label="agrees with EBSD"),
    ]
    # Both legends occupy the vacant lower-left cell, left-aligned with the
    # EBSD panel above. The phase key hangs from the top of the cell, directly
    # beneath the EBSD panel it describes; the error key is anchored by its
    # BOTTOM to the bottom of the error maps beside it, so it sits level with
    # the row it belongs to rather than floating mid-cell.
    fig.legend(handles=ph_handles, loc="upper left",
               bbox_to_anchor=(pos_eb.x0, pos_eb.y0 - 0.020),
               ncol=2, frameon=False, fontsize=11,
               handlelength=1.4, columnspacing=1.2, labelspacing=0.5)
    fig.legend(handles=er_handles, loc="lower left",
               bbox_to_anchor=(pos_eb.x0, pos_e1.y0),
               ncol=1, frameon=False, fontsize=11,
               handlelength=1.4, labelspacing=0.5)

    with plt.rc_context({"pdf.compression": 0}):   # Illustrator misreads compressed indexed images
        fig.savefig(os.path.join(HERE, f"Tuned_MapErrors_{fname}_grouped.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote Tuned_MapErrors_{fname}_grouped.png")

# %%
