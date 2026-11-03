# %%

"""
Natural vs synthetic distributions for the six minerals augmented with the
synthetic generator (SI Text S6). One figure per mineral:
  olivine, plagioclase: crossplots on top, then oxide and cation violins
                        (*_synth_combined.pdf)
  kalsilite, zircon:    oxide and cation violins (*_synth_violins.pdf)
  hematite, SiO2:       cation violins only (*_synth_violins.pdf)
Natural and synthetic compositions are passed through the same mineralML
calculator, so cation counts are recalculated identically for both.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

sys.path.insert(0, "../../src")
import mineralML as mm

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "Avenir",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "axes.linewidth": 0.8,
})

NAT, SYN = "#0C7BDC", "#E42211"
SEED = 42

# (oxide column, cation label, oxide title, cation title)
# cross: crossplots (x column, y column, x label, y label), olivine and plagioclase only
# rows: violin rows to draw; hematite and SiO2 polymorph show cations only
def sub(base, s):
    return base + r"$\mathregular{_{" + s + "}}$"


WT = " (wt.%)"
SIO2, AL2O3, NA2O, K2O = sub("SiO", "2"), sub("Al", "2") + sub("O", "3"), sub("Na", "2") + "O", sub("K", "2") + "O"
FEOT, ZRO2, TIO2, FE2O3 = sub("FeO", "t"), sub("ZrO", "2"), sub("TiO", "2"), sub("Fe", "2") + sub("O", "3")
FE3 = r"Fe$\mathregular{^{3+}}$"

MINERALS = {
    "Olivine": dict(calc=mm.OlivineCalculator, ox=4, fname="ol", elements=[
        ("SiO2", "Si", SIO2, "Si"), ("MgO", "Mg", "MgO", "Mg"),
        ("FeOt", "Fe2t", FEOT, sub("Fe", "t")), ("MnO", "Mn", "MnO", "Mn"),
        ("CaO", "Ca", "CaO", "Ca")],
        cross=[("FeOt", "MgO", FEOT + WT, "MgO" + WT),
               ("SiO2", "MgO", SIO2 + WT, "MgO" + WT),
               ("Fo", "M_site_expanded", sub("X", "Fo") + " (Mg/(Mg+Fe))", "M-site cations (apfu)")]),
    "Plagioclase": dict(calc=mm.FeldsparCalculator, ox=8, fname="plag", elements=[
        ("SiO2", "Si", SIO2, "Si"), ("Al2O3", "Al", AL2O3, "Al"),
        ("CaO", "Ca", "CaO", "Ca"), ("Na2O", "Na", NA2O, "Na"),
        ("K2O", "K", K2O, "K")],
        cross=[("Na2O", "CaO", NA2O + WT, "CaO" + WT),
               ("Al2O3", "SiO2", AL2O3 + WT, SIO2 + WT),
               ("An", "Ab", sub("X", "An"), sub("X", "Ab"))]),
    "Kalsilite": dict(calc=mm.KalsiliteCalculator, ox=4, fname="ks", elements=[
        ("SiO2", "Si", SIO2, "Si"), ("Al2O3", "Al", AL2O3, "Al"),
        ("K2O", "K", K2O, "K"), ("Na2O", "Na", NA2O, "Na")]),
    "Zircon": dict(calc=mm.ZirconCalculator, ox=4, fname="zr", elements=[
        ("SiO2", "Si", SIO2, "Si"), ("ZrO2", "Zr", ZRO2, "Zr"),
        ("FeOt", "Fe2t", FEOT, sub("Fe", "t")), ("MgO", "Mg", "MgO", "Mg")]),
    "SiO2_Polymorph": dict(calc=mm.QuartzCalculator, ox=2, fname="sio2", title=SIO2 + " polymorph",
                           elements=[("SiO2", "Si", SIO2, "Si"), ("Al2O3", "Al", AL2O3, "Al"),
                                     ("TiO2", "Ti", TIO2, "Ti")],
                           rows=["cation"]),
    "Hematite": dict(calc=mm.RhombohedralOxideCalculator, ox=3, fname="hem",
                     kw={"Fe_correction": "All_Fe3"},
                     elements=[("Fe2O3", "Fe3", FE2O3, FE3), ("Al2O3", "Al", AL2O3, "Al")],
                     rows=["cation"]),
}


# %% Regenerate the full olivine and plagioclase populations (settings from DataClean_Train.py)

def regenerate(mineral):
    np.random.seed(SEED)
    if mineral == "Olivine":
        gen = mm.SolidSolutionGenerator(
            endmembers={"Fo": {"Mg": 2, "Si": 1, "O": 4}, "Fa": {"Fe2t": 2, "Si": 1, "O": 4}},
            oxygen_basis=4, element_noise_scale=0.025, min_site_fraction=0.2,
            minor_elements={"Ca": {"distribution": "exponential", "scale": 0.01, "max_fraction": 0.01},
                            "Mn": {"distribution": "exponential", "scale": 0.01, "max_fraction": 0.01}},
            mixing_dist="beta", mixing_params={"a": 1, "b": 1})
        return gen.generate(2000)
    gen = mm.SolidSolutionGenerator(
        endmembers={"Ab": {"Na": 1, "Al": 1, "Si": 3, "O": 8}, "An": {"Ca": 1, "Al": 2, "Si": 2, "O": 8}},
        oxygen_basis=8, element_noise_scale=0.05, min_site_fraction=0.2,
        minor_elements={"K": {"distribution": "exponential", "scale": 0.01, "max_fraction": 0.02}},
        mixing_dist="beta", mixing_params={"a": 1, "b": 1})
    return gen.generate(5000)


# %% Load natural and synthetic compositions, recalculate cations identically

train = pd.read_csv("../../Training_Data/min_df_v4_synth.csv", low_memory=False)
is_syn = train["Source"] == "Synthetic"

data = {}
for mineral, cfg in MINERALS.items():
    nat = train[(train["Mineral"] == mineral) & ~is_syn]
    if mineral == "Plagioclase":
        # 32 analyses (T8_plag2_rtoc, one T8_plag5_rtoc) with SiO2 33.6-38.9 wt.% and
        # totals 72-83 wt.%: below pure anorthite (43.2 wt.% SiO2), so not plagioclase
        # analyses. Dropped from these figures only; they remain in the training set.
        nat = nat[nat["SiO2"] >= 40]
    syn = regenerate(mineral) if mineral in ("Olivine", "Plagioclase") \
        else train[(train["Mineral"] == mineral) & is_syn]
    kw = cfg.get("kw", {})
    data[mineral] = [cfg["calc"](df.reset_index(drop=True)).calculate_components(**kw)
                     for df in (nat, syn)]
    print(f"{mineral:<15} natural {len(nat):>6,}  synthetic {len(syn):>5,}")


# %% KS statistics

rows = []
for mineral, cfg in MINERALS.items():
    nat, syn = data[mineral]
    for oxide, el, *_ in cfg["elements"]:
        for kind, col in (("oxide", oxide), ("cation", f"{el}_cat_{cfg['ox']}ox")):
            a, b = nat[col].dropna().to_numpy(float), syn[col].dropna().to_numpy(float)
            ks = ks_2samp(a, b)
            rows.append(dict(Mineral=mineral, Variable=col, Kind=kind, n_natural=len(a),
                             n_synthetic=len(b), D=ks.statistic, p_value=ks.pvalue,
                             mean_natural=a.mean(), mean_synthetic=b.mean(),
                             sd_natural=a.std(ddof=1), sd_synthetic=b.std(ddof=1)))
ks_df = pd.DataFrame(rows)
ks_df.to_csv("synth_ks_stats.csv", index=False)
print(ks_df.pivot_table(index=["Mineral", "Variable"], values=["D", "p_value"]).round(3).to_string())


# %% Figures: crossplots on top, then oxide violins, then cation violins

for mineral, cfg in MINERALS.items():
    nat, syn = data[mineral]
    cross = cfg.get("cross", [])
    rows = cfg.get("rows", ["oxide", "cation"])
    ncol, nrow = len(cfg["elements"]), len(rows)
    h_cross, h_viol = (4.4 if cross else 0), 3.2 * nrow
    fig = plt.figure(figsize=(max(2.9 * ncol + 0.6, 4.6 * len(cross)), h_cross + h_viol),
                     layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[h_cross, h_viol]) if cross else (None, fig)

    letter_axes = []
    if cross:
        ax = np.atleast_1d(top.subplots(1, len(cross))).flatten()
        for k, (x, y, xl, yl) in enumerate(cross):
            ax[k].scatter(nat[x], nat[y], s=20, c=NAT, ec="k", lw=0.1, label="Natural", rasterized=True)
            ax[k].scatter(syn[x], syn[y], s=20, c=SYN, ec="k", lw=0.1, alpha=0.25, label="Synthetic",
                          rasterized=True)
            ax[k].set_xlabel(xl)
            ax[k].set_ylabel(yl)
        leg = ax[-1].legend(loc="best", fontsize=11, framealpha=0.95)
        for h in leg.legend_handles:
            h.set_alpha(1)
        letter_axes += list(ax)

    ax = np.atleast_1d(bottom.subplots(nrow, ncol)).flatten()
    for i, kind in enumerate(rows):
        for j, (oxide, el, ox_title, cat_title) in enumerate(cfg["elements"]):
            col, title = (oxide, ox_title) if kind == "oxide" else (f"{el}_cat_{cfg['ox']}ox", cat_title)
            a, b = nat[col].dropna().to_numpy(float), syn[col].dropna().to_numpy(float)
            k = i * ncol + j
            parts = ax[k].violinplot([a, b], positions=[0, 1], widths=0.75, showextrema=False)
            for body, color in zip(parts["bodies"], (NAT, SYN)):
                body.set_facecolor(color)
                body.set_edgecolor("k")
                body.set_linewidth(0.5)
                body.set_alpha(0.75)
            for x, v in ((0, a), (1, b)):
                ax[k].vlines(x, v.min(), v.max(), color="k", lw=0.8, zorder=2)
                ax[k].hlines(v.mean(), x - 0.18, x + 0.18, color="k", lw=1.6, zorder=3)
            ax[k].set_xticks([0, 1], ["Natural", "Synthetic"])
            ax[k].set_xlim(-0.6, 1.6)
            ax[k].margins(y=0.08)
            ax[k].set_title(title, pad=6)
        ax[i * ncol].set_ylabel("Oxide (wt.%)" if kind == "oxide" else f"Cations ({cfg['ox']} O basis)")

    # one letter per crossplot, then one per violin row; a lone violin row gets one per panel
    letter_axes += list(ax[::ncol]) if cross or nrow > 1 else list(ax)
    for letter, a in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", letter_axes):
        a.text(0, 1.02, f"{letter}.", transform=a.transAxes, ha="left", va="bottom",
               fontsize=16, fontweight="bold")

    fig.suptitle(f"{cfg.get('title', mineral)}: natural n = {len(nat):,}, "
                 f"synthetic n = {len(syn):,}", fontsize=14)
    fig.savefig(f"{cfg['fname']}_synth_{'combined' if cross else 'violins'}.pdf",
                bbox_inches="tight", pad_inches=0.025, dpi=300)
    plt.close(fig)

# %%
