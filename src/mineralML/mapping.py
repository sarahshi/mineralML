# %%

import re, os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

from mineralML.stoichiometry import *

# %% 

# standard oxides for mineralML
EXPECTED_OXIDES = [
    "SiO2","TiO2","Al2O3","FeOt","MnO","MgO","CaO","Na2O","K2O","P2O5","Cr2O3"
]


def _ensure_columns(df, expected=EXPECTED_OXIDES):
    out = df.copy().rename(columns={"FeO":"FeOt"})
    for col in expected:
        if col not in out.columns:
            out[col] = np.nan
    return out[expected]


def _maps_to_df(E):
    """
    Convert a dictionary of 2D arrays into a flat DataFrame.

    Parameters
    ----------
    E (dict): Dictionary mapping element symbols to 2D numpy arrays (maps).

    Returns
    -------
    df (df): Flattened DataFrame with each element as a column.
    shape (tuple): Original 2D shape (H, W) of the maps.
    """
    if not E:
        raise ValueError("No element maps provided.")
    shapes = {arr.shape for arr in E.values()}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent map shapes: {shapes}")
    H, W = next(iter(shapes))
    flat = {k: v.ravel(order="C") for k, v in E.items()}
    return pd.DataFrame(flat), (H, W)


def _df_to_maps(df, shape):
    """
    Convert a flattened DataFrame back into dict of 2D arrays.

    Parameters
    ----------
    df: DataFrame with flattened values for each feature/element.
    shape (tuple):  Original 2D shape (H, W).

    Returns
    -------
    maps (dict): Dictionary mapping column names to 2D numpy array shaped(H, W).
    """
    H, W = shape
    return {k: df[k].to_numpy().reshape(H, W, order="C") for k in df.columns}


def _clean_labels_1d(arr):
    """
    Flatten labels and de-noise (drop NaN/empties, strip), returning clean strings.

    Parameters:
        arr (array-like): 1D/2D labels (e.g., (H,W) mineral map or flat vector).

    Returns:
        labels (pd.Series): Cleaned string labels (index not meaningful).
    """
    s = pd.Series(np.asarray(arr).ravel())
    s = s[~s.isna()].astype(str).str.strip()
    return s[~s.str.lower().isin({"", "nan", "none", "null"})]


def _make_palette(labels, cmap_name="tab20"):
    """
    Map labels to RGB tuples sampled from a matplotlib colormap.

    Parameters:
        labels (list[str]): Unique labels in display order.
        cmap_name (str): Matplotlib colormap name to sample.

    Returns:
        palette (dict[str, tuple]): {label: (r,g,b)} with values in [0,1].
    """
    cmap = plt.get_cmap(cmap_name, max(len(labels), 1))
    cols = []
    for i in range(len(labels)):
        r, g, b, _ = cmap(i)
        cols.append((min(r, 0.95), min(g, 0.95), min(b, 0.95)))
    return {lab: cols[i] for i, lab in enumerate(labels)}


def load_element_maps(path, drop_trailing_blank=False, verbose=True):
    """
    Load element maps from a directory of CSVs into a dictionary of 2D arrays.

    Parameters
    ----------
    path (str): Path to directory containing CSV files of element maps.
 
    Returns
    -------
    out (dict): Dictionary mapping element symbols (str) to 2D numpy arrays (float).
        NaNs are preserved. Empty trailing columns are automatically dropped.
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(path)

    ELEMENTS = {"Na","Mg","Al","Si","P","K","Ca","Ti","Cr","Mn","Fe","Ni"}
    files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    out = {}

    for f in files:
        name = os.path.splitext(f)[0]  # drop extension

        matched = None
        for el in ELEMENTS:
            pat = rf"(?<![A-Za-z0-9]){re.escape(el)}(?![A-Za-z0-9])"

            if re.search(pat, name, flags=re.IGNORECASE):
                matched = el
                break

        if matched is None:
            if verbose:
                print(f"[skip] no element token in: {f}")
            continue

        arr = np.genfromtxt(os.path.join(path, f), delimiter=",")
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if drop_trailing_blank and arr.shape[1] > 0:
            last = arr[:, -1]
            if np.all(~np.isfinite(last)) or np.allclose(last, 0, equal_nan=True):
                arr = arr[:, :-1]

        if matched in out:
            print(f"[warn] duplicate element '{matched}': {f} overwrites previous")

        out[matched] = arr.astype(float, copy=False)
        if verbose:
            print(f"[ok] {f}  →  {matched}  {arr.shape}")

    # final sanity check: consistent shapes
    shapes = {k: v.shape for k, v in out.items()}
    if len({s for s in shapes.values()}) > 1:
        print("[warn] inconsistent shapes:", shapes)

    return out


def convert_dir_to_oxide_maps(path):
    """
    Load per-element CSV maps from a directory, convert to oxide wt% maps.

    Parameters
    ----------
    path (str): Path to directory containing element CSV maps.

    Returns
    -------
    ox_maps (dict): Dictionary mapping oxide names (str) to 2D numpy arrays (float).
    """
    E = load_element_maps(path)
    df_el, shape = _maps_to_df(E)
    df_ox, _ = element_to_oxide(df_el)
    ox_maps = _df_to_maps(df_ox, shape)

    return ox_maps


def pick_common_phases(mineral_map, min_frac=0.025, top_k=None):
    """
    Select abundant phases by pixel fraction, optionally capped at top_k.

    Parameters:
        mineral_map (array-like): (H,W) or (N,) phase labels.
        min_frac (float): Keep phases with fraction ≥ min_frac (default 0.025).
        top_k (int|None): After filtering, keep only the top_k most abundant.

    Returns:
        keep (list[str]): Phase names in decreasing abundance.
    """
    labels = _clean_labels_1d(mineral_map)
    if labels.empty:
        return []
    freqs = labels.value_counts(normalize=True)
    keep = [p for p, f in freqs.items() if f >= min_frac] or [freqs.idxmax()]
    return [p for p in freqs.index if p in keep][:top_k] if top_k else keep


def _auto_figsize_from_array(shape, n_legend=0, legend_side="right",
                             base_long=10.0, min_long=6.0, max_long=16.0):
    """
    Compute (width,height) inches from map shape; preserve aspect; widen for legend.

    Parameters:
        shape (tuple[int,int]): (H,W) of the map.
        n_legend (int): Number of legend entries; adds width if >0 and right-sided.
        legend_side (str): "right" or "left"; only "right" adds extra width.
        base_long (float): Base inches for long side before scaling.
        min_long (float): Minimum inches for width/height after scaling.
        max_long (float): Maximum inches for width/height after scaling.

    Returns:
        size (tuple[float,float]): (width_in, height_in) for plt.subplots(figsize=...).
    """
    H, W = map(int, shape)
    if H <= 0 or W <= 0:
        return (8.0, 6.0)
    aspect = W / float(H)
    long_px = max(H, W)
    scale = np.clip(long_px / 1200.0, min_long / base_long, max_long / base_long)
    long_in = base_long * scale
    if W >= H:
        fig_w, fig_h = long_in, long_in / max(aspect, 1e-6)
    else:
        fig_h, fig_w = long_in, long_in * max(aspect, 1e-6)
    if legend_side == "right" and n_legend > 0:
        fig_w += float(np.clip(0.6 + 0.12 * n_legend, 0.8, 3.0))
    return (float(np.clip(fig_w, min_long, max_long + 4.0)),
            float(np.clip(fig_h, min_long * 0.8, max_long)))


def _auto_bar_width(n, min_w=6.0, max_w=22.0, per_cat=0.45):
    """
    Compute bar-chart width (inches) from number of categories.

    Parameters:
        n (int): Number of bars.
        min_w (float): Minimum width in inches.
        max_w (float): Maximum width in inches.
        per_cat (float): Incremental width per category.

    Returns:
        width (float): Figure width in inches.
    """
    return float(np.clip(min_w + per_cat * max(n, 1), min_w, max_w))


def plot_phase_map(mineral_map_2d, keep=None, title="Phase Map",
                   bg_color=(0.08, 0.08, 0.08), cmap_name="tab20", ax=None):
    """
    Render a phase map with auto-figsize; non-kept phases are background.

    Parameters:
        mineral_map_2d (array-like): (H,W) phase labels (objects/strings).
        keep (list[str]|None): Phases to color (None→auto via pick_common_phases).
        title (str): Axes title text.
        bg_color (tuple): Background RGB in [0,1].
        cmap_name (str): Colormap name for phase colors.
        ax (matplotlib.axes.Axes|None): Existing axes (None→create new).

    Returns:
        fig_ax (tuple): (fig, ax) with the rendered map.
    """
    mineral_map_2d = np.asarray(mineral_map_2d, dtype=object)
    keep = keep or pick_common_phases(mineral_map_2d, min_frac=0.025)
    phase_to_id = {p: i + 1 for i, p in enumerate(keep)}
    ids = np.zeros(mineral_map_2d.shape, dtype=int)
    for p, pid in phase_to_id.items():
        ids[mineral_map_2d == p] = pid
    phase_colors = _make_palette(keep, cmap_name=cmap_name)
    cmap = ListedColormap([bg_color] + [phase_colors[p] for p in keep])
    fig_w, fig_h = _auto_figsize_from_array(ids.shape, n_legend=len(keep), legend_side="right")
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    else:
        fig = ax.figure
        fig.set_size_inches(fig_w, fig_h, forward=True)
    ax.imshow(ids, cmap=cmap, interpolation="nearest", origin="upper")
    ax.set_title(title)
    ax.axis("off")
    handles = [Patch(facecolor=phase_colors[p], label=p) for p in keep]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.20, 1.0),
              frameon=False, title="Phases")
    return fig, ax


def plot_phase_counts(mineral_map_2d, title="Mineral Phases (count)"):
    """
    Bar chart of pixel counts per phase with auto figure width.

    Parameters:
        mineral_map_2d (array-like): (H,W) or (N,) labels.
        title (str): Axes title text.

    Returns:
        fig_ax (tuple): (fig, ax) with the bar chart.
    """
    labels = _clean_labels_1d(mineral_map_2d)
    if labels.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No valid labels", ha="center", va="center")
        ax.axis("off")
        return fig, ax
    counts = labels.value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(_auto_bar_width(len(counts)), 4.5), constrained_layout=True)
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Pixels")
    ax.tick_params(axis='x', rotation=90)
    return fig, ax


def plot_probability_histograms(prob_map_2d, mineral_map_2d,
                                phases=None, bins=50, share_y=True,
                                title="Prediction Probabilities"):
    """
    Horizontal histograms of per-phase predicted probabilities (auto grid).

    Parameters:
        prob_map_2d (array-like): (H,W) max class probabilities per pixel.
        mineral_map_2d (array-like): (H,W) predicted labels (NaN allowed).
        phases (list[str]|None): Subset of phases to plot (None→auto).
        bins (int): Histogram bins.
        share_y (bool): Share probability axis across panels.
        title (str): Figure suptitle text.

    Returns:
        fig_axes (tuple): (fig, axes) with a 1-D array of axes.
    """
    mineral_map_2d = np.asarray(mineral_map_2d, dtype=object)
    prob_map_2d = np.asarray(prob_map_2d, dtype=float)
    phases = phases or pick_common_phases(mineral_map_2d, min_frac=0.025)
    if not phases:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No phases to plot", ha="center", va="center")
        ax.axis("off")
        return fig, ax
    per_row = min(5, len(phases))
    rows = int(np.ceil(len(phases) / per_row))
    fig, axes = plt.subplots(rows, per_row, figsize=(2.8*per_row, 2.2*rows),
                             sharey=share_y, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    finite = prob_map_2d[np.isfinite(prob_map_2d)]
    low = 0.0 if finite.size == 0 else float(np.nanpercentile(finite, 5))
    ylim = (max(0.0, min(low, 0.95)), 1.0)
    total = float(np.isfinite(prob_map_2d).sum() + 1e-12)
    for i, phase in enumerate(phases):
        ax = axes[i]
        vals = prob_map_2d[mineral_map_2d == phase]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            ax.text(0.5, 0.5, f"{phase}\n(no data)", ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.hist(vals, bins=bins, orientation="horizontal")
        ax.set_ylim(ylim)
        ax.set_title(f"{phase}\n{100.0*vals.size/total:.1f} %", fontsize=9)
        ax.set_xlabel("Pixels", fontsize=8)
        if i % per_row == 0:
            ax.set_ylabel("Pred. Prob.", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(title, y=1.02, fontsize=11)
    return fig, axes


def run_sample(sample_dir, n_iterations=50, prob_threshold=0.75,
               min_frac_to_show=0.025, top_k=None, phases=None,
               return_everything=False, show=True):
    """
    Load → convert → predict → plot for one folder of CSV maps.

    Parameters:
        sample_dir (str): Directory containing element CSV maps.
        n_iterations (int): MC forward passes for prediction.
        prob_threshold (float): Set label to NaN where max probability < threshold.
        min_frac_to_show (float): Keep phases with fraction ≥ this value.
        top_k (int|None): Cap displayed phases after filtering.
        phases (list[str]|None): Explicit phases to plot (None→auto).
        return_everything (bool): If True, return dict of intermediates.
        show (bool): If True, call plt.show().

    Returns:
        figs (tuple): (fig_map, fig_counts, fig_hists) if return_everything=False.
        data (dict): Full outputs (figs, maps, frames) if return_everything=True.
    """
    ox_maps = convert_dir_to_oxide_maps(sample_dir)
    if not ox_maps:
        raise ValueError(f"No oxide maps found in: {sample_dir}")
    df_ox_flat, shape = _maps_to_df(ox_maps)
    df_ordered = _ensure_columns(df_ox_flat)

    df_pred, prob_matrix = predict_class_prob_nn(df_ordered, n_iterations=n_iterations)
    labels = df_pred["Predict_Mineral"].astype(object)
    probs  = df_pred["Predict_Probability"].astype(float)
    labels = labels.mask(probs < prob_threshold)
    labels_flat, probs_flat = labels.to_numpy(), probs.to_numpy()
    H, W = shape
    mineral_map = labels_flat.reshape(H, W)
    prob_map = probs_flat.reshape(H, W)
    kept = list(phases) if phases else pick_common_phases(mineral_map, min_frac=min_frac_to_show, top_k=top_k)
    if not kept:
        raw = df_pred["Predict_Mineral"].to_numpy().reshape(H, W)
        kept = pick_common_phases(raw, min_frac=min_frac_to_show, top_k=top_k)
    fig_map, _    = plot_phase_map(mineral_map, keep=kept, title=f"Phase Map: {os.path.basename(sample_dir)}")
    fig_counts, _ = plot_phase_counts(mineral_map, title=f"Mineral Phases: {os.path.basename(sample_dir)}")
    fig_hists, _  = plot_probability_histograms(prob_map, mineral_map, phases=kept,
                                                title=f"Prediction Probabilities: {os.path.basename(sample_dir)}")
    if show:
        plt.show()
    if not return_everything:
        return fig_map, fig_counts, fig_hists
    return {
        "figs": (fig_map, fig_counts, fig_hists),
        "shape": shape,
        "oxide_maps": ox_maps,
        "df_ordered": df_ordered,
        "df_pred": df_pred,
        "prob_matrix": prob_matrix,
        "mineral_map": mineral_map,
        "prob_map": prob_map,
        "kept_phases": kept,
    }

