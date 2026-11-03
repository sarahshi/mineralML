"""
Sweep of mineralML's Monte Carlo forward passes n on GEOROC (n = 5-1000, seed 42),
scored against an independent n = 1000 reference (seed 7). Writes
MC_Convergence_GEOROC_summary.csv for MC_Convergence_Figure.py and Text S6.
"""
import os, sys, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
REPO = os.path.expanduser("~/GitHub/mineralML"); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{REPO}/src"); os.chdir(REPO)
import numpy as np, pandas as pd, mineralML as mm

NS = [5, 10, 25, 50, 100, 250, 500, 1000]
df = mm.prep_df(pd.read_csv("Validation_Data/GEOROC_validationdata_Fe_202602.csv"))


def run(n, seed):
    t0 = time.time()
    o = mm.predict_class_prob(df, n_iterations=n, seed=seed, verbose=False)
    print(f"n={n:5d} seed={seed:3d}  {time.time() - t0:7.1f} s", flush=True)
    return dict(score=o["Prediction_Score"].to_numpy(float), sigma=o["Prediction_Score_Sigma"].to_numpy(float),
                label=o["Predict_Mineral"].astype(str).to_numpy(), runtime=time.time() - t0)


runs = {n: run(n, 42) for n in NS}
B = run(1000, 7)                                       # independent reference
A = runs[1000]
ok = ~np.isnan(A["score"]) & ~np.isnan(B["score"])     # rows the network classified
ref, sig = B["score"][ok], A["sigma"][ok]              # sig: per-pass SD of the top-class score

rows = []
for n, r in runs.items():
    e = np.abs(r["score"][ok] - ref)
    rows.append(dict(n=n, runtime_s=r["runtime"], median=np.median(e), p95=np.percentile(e, 95),
                     rms=np.sqrt(np.mean(e**2)),
                     rms_expected=np.sqrt(np.mean(sig**2 * (1/n + 1/1000))),   # independent runs
                     # error to the converged value: remove the reference's own variance
                     rms_to_converged=np.sqrt(max(np.mean(e**2) - np.mean(sig**2) / 1000, 0)),
                     label_diff_pct=100 * np.mean(r["label"][ok] != B["label"][ok])))
t = pd.DataFrame(rows)
t["ref_noise_p95"] = np.percentile(np.abs(A["score"][ok] - ref), 95)
t["ref_label_diff_pct"] = 100 * np.mean(A["label"][ok] != B["label"][ok])
t["n_rows"] = ok.sum()
print(t.round(5).to_string(index=False))
m = t.n < 1000
print(f"RMS to converged value ~ n^{np.polyfit(np.log(t.n[m]), np.log(t.rms_to_converged[m]), 1)[0]:.3f}")
t.to_csv(f"{HERE}/MC_Convergence_GEOROC_summary.csv", index=False)
