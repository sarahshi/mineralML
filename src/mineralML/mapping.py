# %%

import re, os
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from mineralML.stoichiometry import *
from mineralML.supervised import *
from mineralML.constants import *


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


def load_dir_to_oxide_maps(path):
    """
    Load per-oxide CSV maps from a directory, no conversion needed.

    Parameters
    ----------
    path (str): Path to directory containing oxide CSV maps.

    Returns
    -------
    ox_maps (dict): Dictionary mapping oxide names (str) to 2D numpy arrays (float).
    """
    O = load_element_maps(path)
    df_ox, shape = _maps_to_df(O)
    ox_maps = _df_to_maps(df_ox, shape)

    return ox_maps


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


def pick_common_phases(mineral_map, min_frac=0.01, top_k=None):
    """
    Select abundant phases by pixel fraction, optionally capped at top_k.

    Parameters:
        mineral_map (array-like): (H,W) or (N,) phase labels.
        min_frac (float): phases phases with fraction ≥ min_frac (default 0.025).
        top_k (int|None): After filtering, phases only the top_k most abundant.

    Returns:
        phases (list[str]): Phase names in decreasing abundance.
    """
    labels = _clean_labels_1d(mineral_map)
    if labels.empty:
        return []
    freqs = labels.value_counts(normalize=True)
    phases = [p for p, f in freqs.items() if f >= min_frac] or [freqs.idxmax()]
    return [p for p in freqs.index if p in phases][:top_k] if top_k else phases


def _auto_figsize_from_array(shape, n_legend, legend_side="right", legend_cols=1, 
                            base_width=6, base_height=6, legend_width_ratio=0.3):
    """
    Automatically calculate figure size based on array shape and legend requirements.
    
    Parameters:
    shape: tuple - Shape of the mineral map array (height, width)
    n_legend: int - Number of legend entries
    legend_side: str - "right", "left", "top", "bottom"
    legend_cols: int - Number of columns for legend
    base_width: float - Base width for square-ish maps
    base_height: float - Base height for square-ish maps
    legend_width_ratio: float - Ratio of width dedicated to legend
    
    Returns:
    tuple - (width, height) in inches
    """
    height, width = shape
    aspect_ratio = width / height
    
    if legend_side in ("right", "left"):
        # For side legends, adjust width to accommodate legend
        if aspect_ratio > 1:  # Wider than tall
            fig_width = base_width * aspect_ratio + base_width * legend_width_ratio
            fig_height = base_height
        else:  # Taller than wide or square
            fig_width = base_width + base_width * legend_width_ratio
            fig_height = base_height / aspect_ratio
        
        # Adjust for multi-column legends
        if legend_cols > 1:
            fig_width += base_width * legend_width_ratio * 0.5
        
    elif legend_side in ("top", "bottom"):
        # For top/bottom legends, adjust height
        if aspect_ratio > 1:  # Wider than tall
            fig_width = base_width * aspect_ratio
            fig_height = base_height + base_height * 0.4
        else:  # Taller than wide or square
            fig_width = base_width
            fig_height = base_height / aspect_ratio + base_height * 0.4
        
        # Adjust for legend rows
        legend_rows = (n_legend + legend_cols - 1) // legend_cols
        fig_height += legend_rows * 0.2
    
    else:
        # Default to square-ish
        fig_width = base_width
        fig_height = base_height
    
    return fig_width, fig_height


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


def plot_phase_map(
    mineral_map_2d,
    phases=None,
    title="Phase Map",
    bg_color=(0.08, 0.08, 0.08),
    cmap_name="tab20",
    legend_side="right", 
    legend_cols=1,
    ax=None,
    dpi=100
):
    mineral_map_2d = np.asarray(mineral_map_2d, dtype=object)

    # map to integer ids
    phases = phases or pick_common_phases(mineral_map_2d, min_frac=0.025)
    phase_to_id = {p: i + 1 for i, p in enumerate(phases)}
    ids = np.zeros(mineral_map_2d.shape, dtype=int)
    for p, pid in phase_to_id.items():
        ids[mineral_map_2d == p] = pid

    # palette & cmap
    phase_colors = _make_palette(phases, cmap_name=cmap_name)
    cmap = ListedColormap([bg_color] + [phase_colors[p] for p in phases])

    # figure size computed once; DO NOT fudge after
    fig_w, fig_h = _auto_figsize_from_array(
        ids.shape, n_legend=len(phases),
        legend_side=legend_side, legend_cols=legend_cols
    )

    # compute legend thickness (inches) for robust layout ratios
    # side legends: width ~ per-column width; top/bottom: height ~ rows * per-item
    per_item_h = 0.22  # inches per legend entry vertically
    per_col_w = 1.2 # inches per legend column (label + swatch)
    ncols = max(1, int(legend_cols))
    nrows = int(np.ceil(len(phases) / ncols)) if len(phases) else 1

    if legend_side in ("right", "left"):
        legend_w_in = ncols * per_col_w
        map_w_in    = max(1e-6, fig_w - legend_w_in)
        # convert to relative GridSpec ratios
        width_ratios = [map_w_in, legend_w_in] if legend_side == "right" else [legend_w_in, map_w_in]
        fig = plt.figure(figsize=(map_w_in + legend_w_in, fig_h), dpi=dpi, layout="constrained")
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=width_ratios, wspace=0.02)
        ax_map   = fig.add_subplot(gs[0, 0] if legend_side == "right" else gs[0, 1])
        ax_legend= fig.add_subplot(gs[0, 1] if legend_side == "right" else gs[0, 0])
    elif legend_side in ("top", "bottom"):
        legend_h_in = max(per_item_h * nrows, 0.5)
        map_h_in    = max(1e-6, fig_h - legend_h_in)
        height_ratios = [legend_h_in, map_h_in] if legend_side == "top" else [map_h_in, legend_h_in]
        fig = plt.figure(figsize=(fig_w, map_h_in + legend_h_in), dpi=dpi, layout="constrained")
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=height_ratios, hspace=0.02)
        ax_legend= fig.add_subplot(gs[0, 0] if legend_side == "top" else gs[1, 0])
        ax_map   = fig.add_subplot(gs[1, 0] if legend_side == "top" else gs[0, 0])
    else:
        # default: right
        legend_w_in = ncols * per_col_w
        map_w_in    = max(1e-6, fig_w - legend_w_in)
        fig = plt.figure(figsize=(map_w_in + legend_w_in, fig_h), dpi=dpi, layout="constrained")
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[map_w_in, legend_w_in], wspace=0.02)
        ax_map   = fig.add_subplot(gs[0, 0])
        ax_legend= fig.add_subplot(gs[0, 1])

    # draw map
    ax_map.imshow(ids, cmap=cmap, interpolation="none", origin="upper")
    ax_map.set_title(title, pad=8)
    ax_map.axis("off")

    # draw legend on other axes
    handles = [mpatches.Patch(facecolor=phase_colors[p], label=p) for p in phases]
    ax_legend.axis("off")
    # place legend fully inside legend axes; no bbox_to_anchor; no overlap possible
    ax_legend.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        title="Phases",
        ncol=ncols,
        borderaxespad=0.0,
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=1.0
    )

    return fig, ax_map


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


def plot_phase_counts(mineral_map_2d, title="Mineral Phases (count)",
                      phases=None, normalize=True):
    """
    Bar chart of pixel counts (or fractions) per phase with auto figure width.

    Parameters:
        mineral_map_2d (array-like): (H,W) or (N,) labels.
        title (str): Axes title text.
        phases (list[str]|None): Subset of phases to plot (None→auto).
        normalize (bool): True, plot fraction of total pixels.

    Returns:
        fig_ax (tuple): (fig, ax) with the bar chart.
    """
    labels = _clean_labels_1d(mineral_map_2d)
    if labels.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No valid labels", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    counts = labels.value_counts()

    if phases is not None:
        wanted, seen = [], set()
        for p in phases:
            if p not in seen:
                wanted.append(p)
                seen.add(p)
        counts = counts.reindex(wanted, fill_value=0)
        counts = counts.sort_values(ascending=False)
    else:
        counts = counts.sort_values(ascending=False)

    if normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total
        ylabel = "Fraction of Pixels"
    else:
        ylabel = "Pixels"

    fig, ax = plt.subplots(figsize=(_auto_bar_width(len(counts)), 4.5),
                           constrained_layout=True)
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45, pad=1)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')

    return fig, ax


def plot_probability_histograms(prob_map_2d, mineral_map_2d, prob_threshold,
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
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No phases to plot", ha="center", va="center")
        ax.axis("off")
        return fig, ax
    per_row = min(5, len(phases))
    rows = int(np.ceil(len(phases) / per_row))
    fig, axes = plt.subplots(rows, per_row, figsize=(2.8*per_row, 2.2*rows),
                             sharey=share_y, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    ylim = (prob_threshold, 1.0)
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
        ax.set_title(f"{phase}\n{100.0*vals.size/total:.1f} %", fontsize=12)
        ax.set_xlabel("Pixels", fontsize=12)
        if i % per_row == 0:
            ax.set_ylabel("Prediction Probability", fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(title, y=1.04, fontsize=14)
    return fig, axes


def run_sample(sample_input, n_iterations=50, prob_threshold=0.6,
               min_frac_to_show=0.01, units="element_wt%",
               top_k=None, phases=None,
               return_everything=True, show=True, 
               compute_components=False, components_spec=None):
    """
    Load, convert, predict, plot for one folder of CSV maps.

    Parameters:
        sample_input (str | Path | dict): Either a directory path containing element CSV maps
            or a preloaded dict of oxide maps.
        n_iterations (int): MC forward passes for prediction.
        prob_threshold (float): Set label to NaN where max probability < threshold.
        min_frac_to_show (float): phases phases with fraction ≥ this value.
        top_k (int|None): Cap displayed phases after filtering.
        phases (list[str]|None): Explicit phases to plot (None->auto).
        return_everything (bool): If True, return dict of intermediates.
        show (bool): If True, call plt.show().

    Returns:
        figs (tuple): (fig_map, fig_counts, fig_hists) if return_everything=False.
        data (dict): Full outputs (figs, maps, frames) if return_everything=True.
    """
    # Accept either a path or a preloaded oxide map dict
    if isinstance(sample_input, (str, os.PathLike)):
        sample_dir = sample_input
        if units == "element_wt%":
            ox_maps = convert_dir_to_oxide_maps(sample_dir)
        elif units == "oxide_wt%":
            ox_maps = load_dir_to_oxide_maps(sample_dir)
        else:
            raise ValueError("units must be 'element_wt%' or 'oxide_wt%'")
    elif isinstance(sample_input, dict):
        ox_maps = sample_input
        sample_dir = "Provided ox_maps"
    else:
        raise TypeError("sample_input must be a directory path or a dict of oxide maps")

    if not ox_maps:
        raise ValueError(f"No oxide maps found or provided in: {sample_dir}")

    df_ox_flat, shape = _maps_to_df(ox_maps)
    df_ordered = _ensure_columns(df_ox_flat)

    df_pred, prob_matrix = predict_class_prob_nn(df_ordered, n_iterations=n_iterations)
    labels = df_pred["Predict_Mineral"].astype(object)
    probs = df_pred["Predict_Probability"].astype(float)
    labels = labels.mask(probs < prob_threshold)
    labels_flat, probs_flat = labels.to_numpy(), probs.to_numpy()
    H, W = shape
    mineral_map = labels_flat.reshape(H, W)
    prob_map = probs_flat.reshape(H, W)
    kept = list(phases) if phases else pick_common_phases(mineral_map, min_frac=min_frac_to_show, top_k=top_k)
    if not kept:
        raw = df_pred["Predict_Mineral"].to_numpy().reshape(H, W)
        kept = pick_common_phases(raw, min_frac=min_frac_to_show, top_k=top_k)
    fig_map, _ = plot_phase_map(mineral_map, phases=kept, title=f"Phase Map: {os.path.basename(sample_dir)}")
    fig_counts, _ = plot_phase_counts(mineral_map, phases=kept, title=f"Mineral Phases: {os.path.basename(sample_dir)}")
    fig_hists, _  = plot_probability_histograms(prob_map, mineral_map, phases=kept, prob_threshold=prob_threshold,
                                                title=f"Prediction Probabilities: {os.path.basename(sample_dir)}")

    component_maps, component_frames = {}, {}
    if compute_components:
        # default spec for feldspar, pyroxene, olivine; extend as needed
        default_spec = {
            "Feldspar": {
                "labels": ["Feldspar", "Plagioclase", "KFeldspar"],
                "calculator": FeldsparClassifier,
                "method": "classify",
                "cols": ["An", "Ab", "Or"],
                "kwargs": {"subclass": False},
                "transforms": {}
            },
            "Clinopyroxene": {
                "labels": ["Clinopyroxene"],
                "calculator": PyroxeneClassifier,
                "method": "calculate_components",
                "cols": ["XMg", "En", "Fs", "Wo"],
                "kwargs": {},
                "transforms": {}
            },
            "Orthopyroxene": {
                "labels": ["Orthopyroxene"],
                "calculator": PyroxeneClassifier,
                "method": "calculate_components",
                "cols": ["XMg", "En", "Fs", "Wo"],
                "kwargs": {},
                "transforms": {}
            },
            "Olivine": {
                "labels": ["Olivine"],
                "calculator": OlivineCalculator,
                "method": "calculate_components",
                "cols": ["XFo"],
                "kwargs": {},
                "transforms": {}
            },
        }

        spec = components_spec or default_spec
        component_maps, component_frames = _compute_component_maps(
            df_ordered=df_ordered,
            df_pred=df_pred,
            shape=shape,
            prob_threshold=prob_threshold,
            components_spec=spec,
            oxide_list=OXIDES,
        )

    if show:
        plt.show()
    if not return_everything:
        return fig_map, fig_counts, fig_hists

    out = {
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
    if compute_components:
        out["component_maps"] = component_maps
        out["component_frames"] = component_frames
    return out


def _compute_component_maps(df_ordered, df_pred, shape, prob_threshold,
                            components_spec, oxide_list):
    """
    components_spec: dict like
      {
        "Feldspar": {
          "labels": ["Feldspar", "Plagioclase", "KFeldspar"],
          "calculator": FeldsparClassifier,
          "method": "classify",                 # or "calculate_components"
          "cols": ["An", "Ab", "Or"],           # columns to rasterize
          "kwargs": {"subclass": False},        # optional kwargs for method
          "transforms": {"An": None}            # optional post transforms per col
        },
        "Pyroxene": {
          "labels": ["Pyroxene","Orthopyroxene","Clinopyroxene","Na-Pyroxene"],
          "calculator": PyroxeneClassifier,
          "method": "calculate_components",
          "cols": ["XMg", "En", "Fs", "Wo"],
          "kwargs": {},
          "transforms": {"XMg": None}
        },
        ...
      }
    """
    maps = {}
    frames = {}
    H, W = shape
    probs = pd.to_numeric(df_pred["Predict_Probability"], errors="coerce").fillna(0.0).to_numpy()
    labels = df_pred["Predict_Mineral"].astype(str).to_numpy()
    valid = probs >= prob_threshold

    oxide_cols = [c for c in df_ordered.columns if c in oxide_list]

    for phase_name, spec in components_spec.items():
        phase_labels = set(spec["labels"])
        mask = valid & np.isin(labels, list(phase_labels))
        if not mask.any():
            continue

        sub = df_ordered.loc[mask, oxide_cols].copy()
        sub["Predict_Mineral"] = labels[mask]

        calc = spec["calculator"](sub)
        method = getattr(calc, spec.get("method", "calculate_components"))
        out = method(**spec.get("kwargs", {}))  # DataFrame with requested columns

        # stash the full frame for inspection
        frames[phase_name] = out.copy()

        # rasterize requested columns to image-shaped maps
        for col in spec["cols"]:
            if col not in out.columns:
                continue
            arr = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
            # optional post-transform
            tf = (spec.get("transforms") or {}).get(col)
            if tf is not None:
                arr = tf(arr)

            m = np.full((H, W), np.nan, dtype=float)
            m.reshape(-1)[np.where(mask)[0]] = arr
            maps[f"{phase_name}.{col}"] = m

    return maps, frames


def plot_component_composite(
    res,
    name="Composite",
    save_path=None,
    # which component maps to draw (keys must exist in res["component_maps"])
    feld_key="Feldspar.An",
    cpx_key="Clinopyroxene.XMg",
    opx_key="Orthopyroxene.XMg",
    ol_key="Olivine.XFo",
    glass_labels=("Glass",),
    spinel_labels=("Spinels",),
    smooth_sigma=0.0,
    limits_mode="meanstd",
    percentile=(2, 98),
    legend_side="right", 
    legend_cols=1,
    dpi=100,
):
    """
    Layer An (plagioclase), XMg (pyroxenes), XFo (olivine) with class masks (glass, spinel).
    Expects:
        res["component_maps"]: dict of 2D arrays with NaNs where not present, e.g. "Feldspar.An"
        res["mineral_map"]: 2D array of predicted coarse labels (strings)
    """

    comp_maps = res.get("component_maps", {})
    mineral_map = res.get("mineral_map", None)

    An  = comp_maps.get(feld_key, None)
    XMg_cpx = comp_maps.get(cpx_key, None)
    XMg_opx = comp_maps.get(opx_key, None)
    XFo = comp_maps.get(ol_key, None)
    glass = mineral_map == glass_labels
    spinel = mineral_map == spinel_labels

    if An is None and XMg_cpx is None and XMg_opx is None and XFo is None:
        raise ValueError("No component maps found. Check your keys in res['component_maps'].")

    H, W = None, None
    for a in (An, XMg_cpx, XMg_opx, XFo):
        if a is not None:
            H, W = a.shape
            break

    def _nanmasked(arr):
        """Return a masked array with NaNs masked (transparent under imshow with set_bad)."""
        return np.ma.masked_invalid(arr.astype(float))

    def _auto_limits(a, mode="meanstd", p=(2, 98)):
        """Compute display range for imshow."""
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 0.0, 1.0
        if mode == "percentile":
            return np.percentile(a, p[0]), np.percentile(a, p[1])
        mu, sigma = np.mean(a), np.std(a)
        return (mu - 2*sigma, mu + 2*sigma)

    # custom colormaps (transparent NaNs)
    N = 256
    glasscmap = ListedColormap(['#FFFFFF00', '#F9C300'])
    spinelcmap = ListedColormap(['#FFFFFF00', '#2E2DCE'])

    # teal plag blend
    plag_cmap_arr = np.ones((N, 4))
    plag_cmap_arr[:, 0] = np.linspace(204/256, 0/256, N)
    plag_cmap_arr[:, 1] = np.linspace(238/256, 153/256, N)
    plag_cmap_arr[:, 2] = np.linspace(255/256, 136/256, N)
    plag_cmap = ListedColormap(plag_cmap_arr)

    # light red clinopyroxene
    cpx_cmap_arr = np.ones((N, 4))
    cpx_cmap_arr[:, 0] = np.linspace(255/256, 200/256, N) 
    cpx_cmap_arr[:, 1] = np.linspace(230/256, 60/256, N)
    cpx_cmap_arr[:, 2] = np.linspace(230/256, 80/256, N)
    cpx_cmap = ListedColormap(cpx_cmap_arr)

    # maroon for orthopyroxene
    opx_cmap_arr = np.ones((N, 4))
    opx_cmap_arr[:, 0] = np.linspace(180/256, 90/256, N)
    opx_cmap_arr[:, 1] = np.linspace(50/256, 0/256, N)
    opx_cmap_arr[:, 2] = np.linspace(50/256, 0/256, N)
    opx_cmap = ListedColormap(opx_cmap_arr)

    # green ol blend
    ol_cmap_arr = np.ones((N, 4))
    ol_cmap_arr[:, 0] = np.linspace(239/256, 102/256, N)
    ol_cmap_arr[:, 1] = np.linspace(238/256, 102/256, N)
    ol_cmap_arr[:, 2] = np.linspace(187/256, 51/256, N)
    ol_cmap = ListedColormap(ol_cmap_arr)

    # Make NaNs fully transparent
    plag_cmap.set_bad(color='white', alpha=0)
    cpx_cmap.set_bad(color='white', alpha=0)
    opx_cmap.set_bad(color='white', alpha=0)
    ol_cmap.set_bad(color='white', alpha=0)
    glasscmap.set_bad(color='white', alpha=0)
    spinelcmap.set_bad(color='white', alpha=0)

    # optional smoothing (on masked data -> preserve NaNs)
    def _smooth(m):
        if m is None or not smooth_sigma or smooth_sigma <= 0:
            return m
        data = m.copy()
        mask = ~np.isfinite(data)
        data[mask] = 0.0
        w = (~mask).astype(float)
        data_s = gaussian_filter(data, smooth_sigma, mode="nearest")
        w_s = gaussian_filter(w, smooth_sigma, mode="nearest")
        with np.errstate(invalid='ignore'):
            out = data_s / np.maximum(w_s, 1e-12)
        out[mask] = np.nan
        return out

    An_s = _smooth(An)
    XMg_s_cpx = _smooth(XMg_cpx)
    XMg_s_opx = _smooth(XMg_opx)
    XFo_s = _smooth(XFo)

    vmin_plag, vmax_plag = _auto_limits(An_s, limits_mode, percentile) if An_s is not None else (0, 1)
    vmin_cpx, vmax_cpx = _auto_limits(XMg_s_cpx, limits_mode, percentile) if XMg_s_cpx is not None else (0, 1)
    vmin_opx, vmax_opx = _auto_limits(XMg_s_opx, limits_mode, percentile) if XMg_s_opx is not None else (0, 1)
    vmin_ol, vmax_ol = _auto_limits(XFo_s, limits_mode, percentile) if XFo_s is not None else (0, 1)

    # plot
    legend_entries = []
    if An is not None: 
        legend_entries.append(("Plagioclase (higher An darker)", "#009988"))
    if XMg_cpx is not None: 
        legend_entries.append(("Clinopyroxene (higher Mg# darker)","#E57A7A"))
    if XMg_opx is not None: 
        legend_entries.append(("Orthopyroxene (higher Mg# darker)","#5A0F0F"))
    if XFo is not None: 
        legend_entries.append(("Olivine (higher Fo darker)", "#666633"))
    if glass is not None: 
        legend_entries.append(("Glass", "#F9C300"))
    if spinel is not None:
        legend_entries.append(("Spinel", "#2E2DCE"))

    n_legend = len(legend_entries)

    # figure size computed once; same helper as phase map
    fig_w, fig_h = _auto_figsize_from_array(
        mineral_map.shape,
        n_legend=n_legend,
        legend_side=legend_side,
        legend_cols=legend_cols
    )

    # consistent legend sizing
    per_item_h = 0.22  # in
    per_col_w = 1.20  # in
    ncols = max(1, int(legend_cols))
    nrows = int(np.ceil(max(n_legend,1) / ncols))

    if legend_side in ("right", "left"):
        legend_w_in = ncols * per_col_w
        map_w_in = max(1e-6, fig_w - legend_w_in)
        width_ratios = [map_w_in, legend_w_in] if legend_side == "right" else [legend_w_in, map_w_in]
        fig = plt.figure(figsize=(map_w_in + legend_w_in, fig_h), dpi=dpi, layout="constrained")
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=width_ratios, wspace=0.02)
        ax_map = fig.add_subplot(gs[0, 0] if legend_side == "right" else gs[0, 1])
        ax_legend = fig.add_subplot(gs[0, 1] if legend_side == "right" else gs[0, 0])
    elif legend_side in ("top", "bottom"):
        legend_h_in = max(per_item_h * nrows, 0.5)
        map_h_in = max(1e-6, fig_h - legend_h_in)
        height_ratios = [legend_h_in, map_h_in] if legend_side == "top" else [map_h_in, legend_h_in]
        fig = plt.figure(figsize=(fig_w, map_h_in + legend_h_in), dpi=dpi, layout="constrained")
        gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=height_ratios, hspace=0.02)
        ax_legend = fig.add_subplot(gs[0, 0] if legend_side == "top" else gs[1, 0])
        ax_map = fig.add_subplot(gs[1, 0] if legend_side == "top" else gs[0, 0])
    else: # default right
        legend_w_in = ncols * per_col_w
        map_w_in = max(1e-6, fig_w - legend_w_in)
        fig = plt.figure(figsize=(map_w_in + legend_w_in, fig_h), dpi=dpi, layout="constrained")
        gs  = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[map_w_in, legend_w_in], wspace=0.02)
        ax_map = fig.add_subplot(gs[0, 0])
        ax_legend = fig.add_subplot(gs[0, 1])

    if An_s is not None:
        ax_map.imshow(_nanmasked(An_s), cmap=plag_cmap, interpolation="none",
                      vmin=vmin_plag, vmax=vmax_plag)
    if XMg_s_cpx is not None:
        ax_map.imshow(_nanmasked(XMg_s_cpx), cmap=cpx_cmap, interpolation="none",
                      vmin=vmin_cpx, vmax=vmax_cpx)
    if XMg_s_opx is not None:
        ax_map.imshow(_nanmasked(XMg_s_opx), cmap=opx_cmap, interpolation="none",
                      vmin=vmin_opx, vmax=vmax_opx)
    if XFo_s is not None:
        ax_map.imshow(_nanmasked(XFo_s), cmap=ol_cmap, interpolation="none",
                      vmin=vmin_ol, vmax=vmax_ol)
    if glass is not None:
        ax_map.imshow(_nanmasked(glass), cmap=glasscmap, interpolation="none")
    if spinel is not None:
        ax_map.imshow(_nanmasked(spinel), cmap=spinelcmap, interpolation="none")

    ax_map.set_title(name, pad=8)
    ax_map.set_aspect('equal', adjustable='box')
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.set_frame_on(False)
    for sp in ax_map.spines.values():
        sp.set_visible(False)

    ax_legend.axis("off")
    handles = [mpatches.Patch(facecolor=c, label=lab) for lab, c in legend_entries]
    ax_legend.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        title="Layers",
        ncol=ncols,
        borderaxespad=0.0,
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=1.0,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


# %%
