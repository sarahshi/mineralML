# %%

""" Created on November 3, 2026 // @author: Sarah Shi """

# Builds a long-format, pixel-by-pixel table pairing EDS chemistry with the
# co-registered EBSD phase label for MH0811b and Tuolumne Bii. Intended as a
# supplementary test dataset for benchmarking other classifiers (e.g., MIST).

import os
import re
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../src"))
import mineralML as mm
from mineralML.constants import OXIDES
from mineralML.stoichiometry import element_to_oxide

# %%
# Configurations — merge rules match EBSD_v_EDS.py

mh_merge_rules = {
    "Andesine": "Plagioclase", "Augite": "Clinopyroxene", "Enstatite": "Orthopyroxene",
    "Orthoclase": "Alkali_Feldspar", "Magnetite": "Oxide", "Ilmenite": "Oxide",
    "Spinel_Group": "Oxide", "Rhombohedral_Oxides": "Oxide", "Oxide": "Oxide",
    "Quartz-new": "SiO2_Polymorph", "Cristobalite": "SiO2_Polymorph",
}

bii_merge_rules = {
    "Anorthite": "Plagioclase", "Hornblende": "Amphibole", "Orthoclase": "Alkali_Feldspar",
    "Muscovite": "Unindexed", "Biotite": "Unindexed",
    "Magnetite": "Oxide", "Ilmenite": "Oxide",
    "Spinel_Group": "Oxide", "Rhombohedral_Oxides": "Oxide", "Oxide": "Oxide",
    "Quartz-new": "SiO2_Polymorph",
}

SAMPLES = {
    "MH0811b": dict(eds_dir="../../EDS_EBSD_Maps/MountHood_MH0811b", ctf="../../EDS_EBSD_Maps/EBSD_CTF/MH0811b_EBSD_EDS.ctf",
                    merge_rules=mh_merge_rules),
    "Tuolumne_Bii": dict(eds_dir="../../EDS_EBSD_Maps/Tuolumne_Bii", ctf="../../EDS_EBSD_Maps/EBSD_CTF/Bii_EBSD_EDS.ctf",
                         merge_rules=bii_merge_rules),
}

ADD_MINERALML = True  # append mineralML predictions for direct comparison
N_ITERATIONS = 50
OUT_DIR = os.path.join(HERE, "EDS_EBSD_LongTable")

# %%
# --- Helper Functions ---

def load_all_element_maps(path):
    """Load every '<El> Wt%' CSV in a folder (not just the mineralML input set)."""
    maps = {}
    for f in sorted(os.listdir(path)):
        m = re.search(r"-([A-Z][a-z]?) Wt%\.csv$", f)
        if m is None:
            continue
        arr = pd.read_csv(os.path.join(path, f), header=None).to_numpy(dtype=float)
        # AZtec exports end each row with a comma -> drop the empty last column
        if np.all(np.isnan(arr[:, -1])):
            arr = arr[:, :-1]
        maps[m.group(1)] = arr
    shapes = {v.shape for v in maps.values()}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent EDS map shapes in {path}: {shapes}")
    return maps


def load_ctf_table(filepath, merge_rules):
    """Read the full CTF data table with raw and merged EBSD phase names."""
    x_cells, y_cells, data_start, phase_mapping = mm.parse_ctf_header(filepath)
    raw_names = {k: re.sub(r"\s+", " ", v) for k, v in phase_mapping.items()}

    # Same substring-matching rename as mm.plot_ctf_phases
    merged_names = dict(phase_mapping)
    for pid, name in phase_mapping.items():
        for messy, clean in merge_rules.items():
            if messy in name:
                merged_names[pid] = clean

    ctf = pd.read_csv(filepath, sep="\t", skiprows=data_start)
    if len(ctf) != x_cells * y_cells:
        raise ValueError("Mismatch between CTF header grid size and data points.")

    ctf["EBSD_Phase_Raw"] = ctf["Phase"].map(raw_names)
    ctf["EBSD_Phase"] = ctf["Phase"].map(merged_names)
    return ctf, (y_cells, x_cells)


def build_long_table(sample, eds_dir, ctf, merge_rules):
    E = load_all_element_maps(os.path.join(HERE, eds_dir))
    ctf_df, ctf_shape = load_ctf_table(os.path.join(HERE, ctf), merge_rules)
    eds_shape = next(iter(E.values())).shape
    if eds_shape != ctf_shape:
        raise ValueError(f"{sample}: EDS grid {eds_shape} != EBSD grid {ctf_shape}")

    H, W = eds_shape
    rows, cols = np.divmod(np.arange(H * W), W)

    # CTF is written X-fastest, so row-major flattening of the EDS grid lines up
    step = ctf_df.loc[1, "X"] - ctf_df.loc[0, "X"]
    assert np.allclose(ctf_df["X"].to_numpy(), cols * step)
    assert np.allclose(ctf_df["Y"].to_numpy(), rows * step)

    df = pd.DataFrame({
        "Sample": sample, "Row": rows, "Col": cols,
        "X_um": ctf_df["X"].to_numpy(), "Y_um": ctf_df["Y"].to_numpy(),
    })

    # Raw EDS element wt% (all exported channels)
    el_df = pd.DataFrame({f"{el}_wt%": E[el].ravel(order="C") for el in E})
    el_df["Element_Total"] = el_df.sum(axis=1, min_count=1)

    # Oxide wt% (FeO as FeOt), same conversion as mm.run_map(units='element_wt%')
    ox_in = pd.DataFrame({el: E[el].ravel(order="C") for el in E})
    ox_df, _ = element_to_oxide(ox_in)
    ox_df = ox_df.reindex(columns=OXIDES + ["ZrO2"])
    ox_df["Oxide_Total"] = ox_df.sum(axis=1, min_count=1)

    ebsd_cols = ["EBSD_Phase", "EBSD_Phase_Raw", "Bands", "Error", "MAD", "BC", "BS"]
    ebsd_df = ctf_df[ebsd_cols].rename(columns={c: f"EBSD_{c}" for c in ebsd_cols[2:]})

    out = pd.concat([df, el_df, ox_df, ebsd_df], axis=1)

    if ADD_MINERALML:
        pred = mm.predict_class_prob(ox_df[OXIDES + ["ZrO2"]], n_iterations=N_ITERATIONS)
        out["mineralML_Phase"] = pred["Predict_Mineral"].to_numpy()
        out["mineralML_Score"] = pred["Prediction_Score"].to_numpy()

    return out

# %%
# --- Build tables ---

os.makedirs(OUT_DIR, exist_ok=True)
tables = {}
for sample, cfg in SAMPLES.items():
    tables[sample] = build_long_table(sample, **cfg)
    print(f"\n--- {sample}: {len(tables[sample])} pixels ---")
    print(tables[sample]["EBSD_Phase"].value_counts().to_string())

# %%
# --- Sanity check: median oxide chemistry per EBSD phase ---

for sample, t in tables.items():
    print(f"\n--- {sample} ---")
    print(t.groupby("EBSD_Phase")[["SiO2", "Al2O3", "FeOt", "MgO", "CaO", "Na2O", "K2O", "Oxide_Total"]]
          .median().round(1).to_string())

# %%
# --- Write outputs ---

long_df = pd.concat(tables.values(), ignore_index=True)

# Keep union of element channels together, ahead of the oxide block
el_cols = sorted({c for c in long_df.columns if c.endswith("_wt%")})
head = ["Sample", "Row", "Col", "X_um", "Y_um"]
long_df = long_df[head + el_cols + [c for c in long_df.columns if c not in head + el_cols]]

# Columns not measured in a given sample stay NaN
tag = f"PredTable_v{mm.__version__.removeprefix('0.')}"   # 0.0.3.23 -> v0.3.23
long_df.to_csv(os.path.join(OUT_DIR, f"EDS_EBSD_{tag}.csv.gz"), index=False, float_format="%.4f")
for sample, t in tables.items():
    t.dropna(axis=1, how="all").to_csv(
        os.path.join(OUT_DIR, f"{sample}_EDS_EBSD_{tag}.csv.gz"), index=False, float_format="%.4f")

print(long_df.shape)

# %%
