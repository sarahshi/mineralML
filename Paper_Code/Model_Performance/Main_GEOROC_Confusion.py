# %%
"""
Figure 5 (main text): mineralML's GEOROC confusion matrix, row-normalised
(diagonal = recall). * resolved by the empirical rules.
Reads GEOROC_truth_pred.csv.gz. Writes Figure_GEOROC_Confusion.pdf.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.style.use("../../style.mplstyle")
plt.rcParams["mathtext.default"] = "regular"
FS_AXIS, FS_TICK, FS_CM = 21, 20, 19
EMPIRICAL = {"Zircon", "Carbonate", "SiO2_Polymorph"}


def star(m):
    m_ = {"Rhombohedral_Oxides": "Rhomb. Oxides"}.get(m, m.replace("_", " ").replace("SiO2", "SiO$_2$"))
    return m_ + ("*" if m in EMPIRICAL else "")


tp = pd.read_csv("GEOROC_truth_pred.csv.gz")
rows = sorted(tp["truth"].unique())
other = sorted(set(tp["pred"]) - set(rows))          # mineralML classes absent from GEOROC
cols = rows + other
cm = pd.crosstab(tp["truth"], tp["pred"]).reindex(index=rows, columns=cols, fill_value=0)
M = (100 * cm.div(cm.sum(axis=1), axis=0)).to_numpy(float)
print(f"GEOROC analyses scored: {len(tp)}")

fig, ax = plt.subplots(figsize=(17, 14))
im = ax.imshow(np.where(M > 0, M, np.nan), cmap="BuGn", norm=LogNorm(vmin=0.01, vmax=100), aspect="equal")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        if i == j or v >= 1:
            t = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
            ax.text(j, i, t, ha="center", va="center", color="white" if v >= 20 else "0.1",
                    fontsize=FS_CM if len(t) < 3 or "." in t else FS_CM - 3)   # "100" fits its cell
ax.set_xticks(range(len(cols)), [star(c) for c in cols], fontsize=FS_TICK, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(rows)), [star(r) for r in rows], fontsize=FS_TICK)
ax.tick_params(length=0)
ax.set_xlabel("mineralML prediction", fontsize=FS_AXIS)
ax.set_ylabel("GEOROC label", fontsize=FS_AXIS)
ax.axvline(len(rows) - 0.5, color="0.2", lw=1.4, ls="--")
ax.text((len(rows) - 0.5 + len(cols) - 0.5) / 2, -0.75, "not in GEOROC", ha="center", va="bottom",
        fontsize=FS_TICK, style="italic", color="0.3")
cax = ax.inset_axes([0.2, -0.27, 0.6, 0.028])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_ticks([0.01, 0.1, 1, 10, 100], labels=["0.01", "0.1", "1", "10", "100"])
cb.set_label("% of analyses with this GEOROC label (row)", fontsize=FS_AXIS)
cb.ax.tick_params(labelsize=FS_TICK)

fig.savefig("Figure_GEOROC_Confusion.pdf", bbox_inches="tight")
