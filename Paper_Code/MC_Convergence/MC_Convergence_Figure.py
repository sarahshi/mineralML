# %%
"""
Figure S (Text S6): convergence of mineralML's Monte Carlo prediction score with
the number of forward passes n, on GEOROC. Reads MC_Convergence_GEOROC_summary.csv
from MC_Convergence_GEOROC.py.

  A  error of the prediction score at n against the independent n = 1000 reference
     (median, 95th percentile), and RMS error to the converged value, with n^-1/2
  B  % of samples whose predicted mineral differs from the reference
  C  runtime of predict_class_prob on the full dataset
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "..", "..", "style.mplstyle"))
plt.rcParams["mathtext.default"] = "regular"
FS_T, FS_L, FS_K, FS_S = 15, 13, 12, 11
BLUE, INK, GRAY = "#2a78d6", "0.15", "0.55"
N_USED = 50
N_TICKS = [5, 10, 25, 50, 100, 250, 500, 1000]

t = pd.read_csv(os.path.join(HERE, "MC_Convergence_GEOROC_summary.csv"))
n = t["n"].to_numpy()
slope = np.polyfit(np.log(n[n < 1000]), np.log(t["rms_to_converged"][n < 1000]), 1)[0]
r50 = float(t.loc[t.n == N_USED, "rms_to_converged"].iloc[0])
ref_p95 = float(t["ref_noise_p95"].iloc[0])
ref_lab = float(t["ref_label_diff_pct"].iloc[0])

fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"wspace": 0.32})
ax = ax.flatten()

# A: error against n
ax[0].plot(n, t["rms_to_converged"], "-o", color=BLUE, lw=2, ms=7, mec="k", mew=0.6, zorder=3)
ax[0].plot(n, t["p95"], "-s", color=INK, lw=1.4, ms=6, mfc="white", mew=1.4, zorder=3)
ax[0].plot(n, t["median"], "-^", color=GRAY, lw=1.4, ms=6, mfc="white", mew=1.4, zorder=3)
nn = np.geomspace(4, 1300, 80)
ax[0].plot(nn, r50 * (nn / N_USED) ** -0.5, ls="--", color=BLUE, lw=1, alpha=0.7, zorder=2)
ax[0].axhspan(0, ref_p95, color="0.93", zorder=0)
ax[0].set_ylabel("Error in prediction score", fontsize=FS_L)
ax[0].set_title("A. Prediction score", fontsize=FS_T, loc="left", pad=8)
legend = [Line2D([], [], color=BLUE, marker="o", mec="k", mew=0.6, lw=2,
                 label=f"RMS, to converged value (n$^{{{slope:.3f}}}$)"),
          Line2D([], [], color=BLUE, ls="--", lw=1, label="n$^{-1/2}$"),
          Line2D([], [], color=INK, marker="s", mfc="white", lw=1.4, label="95th percentile"),
          Line2D([], [], color=GRAY, marker="^", mfc="white", lw=1.4, label="median"),
          plt.Rectangle((0, 0), 1, 1, fc="0.93", ec="none",
                        label="noise of the n = 1000 reference\n(95th percentile)")]
ax[0].legend(handles=legend, fontsize=FS_S, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.9))
ax[0].set_ylim(0, 0.072)

# B: predicted mineral changes
ax[1].plot(n, t["label_diff_pct"], "-o", color=BLUE, lw=2, ms=7, mec="k", mew=0.6, zorder=3)
ax[1].axhline(ref_lab, color=GRAY, ls=":", lw=1.2)
ax[1].text(5.5, ref_lab + 0.008, "two independent n = 1000 runs", fontsize=FS_S - 1, color=GRAY, va="bottom")
ax[1].set_ylabel("Predicted mineral differs (% of samples)", fontsize=FS_L)
ax[1].set_title("B. Predicted mineral", fontsize=FS_T, loc="left", pad=8)
ax[1].set_ylim(0, 0.5)

# C: runtime (grows linearly with n)
ax[2].plot(n, t["runtime_s"] / 60, "-o", color=BLUE, lw=2, ms=7, mec="k", mew=0.6, zorder=3)
ax[2].set_ylabel(f"Runtime, {t['n_rows'].iloc[0] / 1e6:.1f} million analyses (min)", fontsize=FS_L)
ax[2].set_title("C. Cost", fontsize=FS_T, loc="left", pad=8)
ax[2].set_ylim(0, 13)

for a in ax:
    a.set_xscale("log")
    a.axvline(N_USED, color=INK, ls="--", lw=1, zorder=1)
    a.set_xticks(N_TICKS, [str(v) for v in N_TICKS])
    a.minorticks_off()
    a.set_xlabel("Monte Carlo forward passes, n", fontsize=FS_L)
    a.tick_params(labelsize=FS_K)
    a.grid(color="0.92", lw=0.8, zorder=0)
    a.text(N_USED * 1.06, 0.97, "n = 50", transform=a.get_xaxis_transform(), fontsize=FS_S, color=INK, va="top")

fig.savefig(os.path.join(HERE, "MC_Convergence_Figure_linear.pdf"), bbox_inches="tight")
