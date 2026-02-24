# %%

import os
import re
import warnings
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes, generate_binary_structure

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, to_rgb
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from mineralML.stoichiometry import *
from mineralML.hybrid import *
from mineralML.supervised import *
from mineralML.constants import *
from .constants import OXIDES

# %% 


def _ensure_columns(df, expected=OXIDES):
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

    ELEMENTS = {"Na","Mg","Al","Si","P","K","Ca","Ti","Cr","Mn","Fe","Ni","Zr"}
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
    df_ox, _ = oxide_to_oxide(df_ox)
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


def pick_common_phases(mineral_map, top_k=None):
    """
    Select abundant phases by pixel fraction, optionally capped at top_k.

    Parameters:
        mineral_map (array-like): (H,W) or (N,) phase labels.
        top_k (int|None): Only keep the top_k most abundant phases.

    Returns:
        phases (list[str]): Phase names in decreasing abundance.
    """
    labels = _clean_labels_1d(mineral_map)
    if labels.empty:
        return []
    
    # Get all phases sorted by frequency
    freqs = labels.value_counts(normalize=True)
    phases = list(freqs.index)
    
    return phases[:top_k] if top_k else phases


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


def _auto_limits(data, mode="percentile", percentile=(5, 95)):
    """
    Calculate dynamic vmin/vmax for a map based on its valid values.
    """
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return 0.0, 1.0
    if mode == "percentile":
        return np.percentile(vals, percentile[0]), np.percentile(vals, percentile[1])
    else:
        # Mean +/- 2SD for contrast amplification
        mu, sigma = np.mean(vals), np.std(vals)
        return mu - 2*sigma, mu + 2*sigma


def remove_islands(mineral_map, min_size=2, connectivity=1, fill_val="nan"):
    """
    Removes isolated specks of valid minerals by converting them to 'nan'. 

    Parameters: 
        mineral_map: 2D object array of mineral strings.
        min_size: remove connected components smaller than this (per phase).
        connectivity: 1=4-connected, 2=8-connected (2D).
        fill_val: the background string to replace removed pixels with.
    """
    # Copy the map so we don't alter the original array in memory
    cleaned_map = mineral_map.astype(object)
    
    # Define background values we DO NOT want to process
    bad_vals = {"nan", "NaN"}
    
    # Find all unique phases present in the map
    unique_phases = np.unique(cleaned_map.astype(str))
    valid_phases = set(unique_phases) - bad_vals
    
    for phase in valid_phases:
        # Create a boolean mask of just this one phase
        mask = (cleaned_map == phase)
        
        # Clean the mask (this returns a new boolean array where small specks are False)
        cleaned_mask = remove_small_objects(mask, min_size=min_size, connectivity=connectivity)
        
        # Identify pixels that were True in the original mask but False in the cleaned mask
        removed_pixels = mask & ~cleaned_mask
        
        # Set those tiny isolated specks to your background value
        cleaned_map[removed_pixels] = fill_val
        
    return cleaned_map


def fill_islands(mineral_map, fill_val="nan"):
    """
    Fills enclosed topological islands with the nearest valid neighbor.
    
    Parameters:
        mineral_map: 2D array of mineral IDs or strings.
        fill_val: The background value representing holes/islands to be filled.
    """
    filled_map = mineral_map.copy()
    
    # Safely identify invalid/background pixels regardless of data type
    if str(fill_val).lower() == "nan" or pd.isna(fill_val):
        # Catches string "nan", "NaN", and actual float np.nan
        is_string_nan = np.isin(filled_map.astype(str), ["nan", "NaN"])
        is_actual_nan = pd.isna(filled_map)
        invalid_mask = is_string_nan | is_actual_nan
    else:
        # Catches specific integers (e.g., -1) or specific strings
        invalid_mask = (filled_map == fill_val)
        
    valid_mask = ~invalid_mask
    
    # Identify topological holes
    filled_valid_mask = binary_fill_holes(valid_mask)
    islands_mask = filled_valid_mask & invalid_mask
    
    # Fill the islands with the nearest valid neighbor
    if np.any(islands_mask):
        _, indices = distance_transform_edt(invalid_mask, return_indices=True)
        nearest_y = indices[0, islands_mask]
        nearest_x = indices[1, islands_mask]
        filled_map[islands_mask] = filled_map[nearest_y, nearest_x]
        
    return filled_map


def plot_phase_map(
    mineral_map_2d,
    phases=None,
    title="Phase Map",
    bg_color=(0.08, 0.08, 0.08),
    cmap_name="tab20",
    legend_side="right",
    legend_cols=1,
    ax=None,
    dpi=300,
    remove_islands_flag=False,
    fill_islands_flag=False,
    scalebar_um=None,
    pixel_size_um=1.0,
    scalebar_color="black",
    phase_colors=None,
    legend_on=True,
):
    mineral_map_2d = np.asarray(mineral_map_2d, dtype=object)

    # Map string phases to integer IDs
    phases = phases or pick_common_phases(mineral_map_2d)
    phase_to_id = {p: i + 1 for i, p in enumerate(phases)}
    id_to_phase = {i + 1: p for i, p in enumerate(phases)} # Reverse dictionary
    id_to_phase[0] = "Unknown" # Ensure background/removed pixels have a name
    
    ids = np.zeros(mineral_map_2d.shape, dtype=int)
    for p, pid in phase_to_id.items():
        ids[mineral_map_2d == p] = pid

    # Apply morphology cleanup to the IDs
    if remove_islands_flag:
        ids = remove_islands(ids, min_size=1, connectivity=1, fill_val=-1)
    if fill_islands_flag:
        ids = fill_islands(ids, fill_val=-1)

    # REVERSE MAP: Update the string map so the data matches the visual!
    cleaned_mineral_map = np.vectorize(id_to_phase.get)(ids)

    # Set up palette & cmap
    palette = _make_palette(phases, cmap_name=cmap_name)
    if phase_colors:
        for phase_name, color in phase_colors.items():
            if phase_name in palette:
                palette[phase_name] = color
                
    cmap = ListedColormap([bg_color] + [palette[p] for p in phases])

    # Figure sizing and grid setup
    fig_w, fig_h = _auto_figsize_from_array(
        ids.shape, n_legend=len(phases),
        legend_side=legend_side, legend_cols=legend_cols
    )

    per_item_h = 0.22  
    per_col_w = 1.2 
    ncols = max(1, int(legend_cols))
    nrows = int(np.ceil(len(phases) / ncols)) if len(phases) else 1

    if legend_side in ("right", "left"):
        legend_w_in = ncols * per_col_w
        map_w_in    = max(1e-6, fig_w - legend_w_in)
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
        legend_w_in = ncols * per_col_w
        map_w_in    = max(1e-6, fig_w - legend_w_in)
        fig = plt.figure(figsize=(map_w_in + legend_w_in, fig_h), dpi=dpi, layout="constrained")
        gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[map_w_in, legend_w_in], wspace=0.02)
        ax_map   = fig.add_subplot(gs[0, 0])
        ax_legend= fig.add_subplot(gs[0, 1])

    # Draw Map
    ax_map.imshow(ids, cmap=cmap, interpolation="none", origin="upper")
    ax_map.set_title(title, pad=8)
    ax_map.axis("off")

    # Add Scale Bar
    if scalebar_um is not None:
        scalebar_pixels = scalebar_um / pixel_size_um
        fontprops = fm.FontProperties(size=12, weight='bold')
        scalebar = AnchoredSizeBar(
            ax_map.transData, scalebar_pixels, f'{scalebar_um} µm', 'lower left',                 
            pad=0.5, color=scalebar_color, frameon=False, size_vertical=1, 
            sep=3, label_top=True, fontproperties=fontprops
        )
        ax_map.add_artist(scalebar)

    # Make Legend
    handles = [mpatches.Patch(facecolor=palette[p], label=p) for p in phases]
    ax_legend.axis("off")
    if legend_on: 
        ax_legend.legend(
            handles=handles, loc="upper left", frameon=False, title="Phases",
            ncol=ncols, borderaxespad=0.0, handlelength=1.2, handletextpad=0.6,
            columnspacing=1.0, fontsize=8
        )

    return fig, ax_map, cleaned_mineral_map


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
                                phases=None, bins=50, min_frac=0.0001,
                                share_y=True, title="Prediction Probabilities"):
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
    phases = phases or pick_common_phases(mineral_map_2d)
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


def run_sample(sample_input, 
               n_iterations=50, 
               prob_threshold=0.6,
            #    min_frac_to_show=0.0001,
               units="element_wt%",
               top_k=None, 
               phases=None, 
               exclude_phases=None,
               phase_colors=None,
               components_spec=None,
               remove_islands_flag=False,
               fill_islands_flag=False,
               scalebar_um=None,
               pixel_size_um=None,
               show=True
):
    """
    Load, convert, predict, and plot for one folder of CSV maps.
    Always computes mineral components and returns a full results dictionary.

    Parameters:
        sample_input (str | Path | dict): Directory path or a dict of oxide maps.
        n_iterations (int): MC forward passes for prediction.
        prob_threshold (float): Set label to NaN where max probability < threshold.
        min_frac_to_show (float): Ignore phases with area fraction < this value.
        top_k (int|None): Cap displayed phases after filtering.
        phases (list[str]|None): Explicit phases to plot (overrides auto-pick).
        exclude_phases (list[str]|None): Phases to remove from auto-pick.
        phase_colors (dict|None): Manual color mapping {PhaseName: HexColor}.
        components_spec (dict|None): Custom mineral formula logic.
        show (bool): If True, calls plt.show().
    """

    # Load and prepare data
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
    expected_with_zr = list(OXIDES) + ["ZrO2"]
    df_ordered = _ensure_columns(df_ox_flat, expected=expected_with_zr)
    # df_ordered = _ensure_columns(df_ox_flat)

    # Prediction logic
    df_pred, prob_matrix = predict_class_prob_nnwr(df_ordered, n_iterations=n_iterations)
    labels = df_pred["Predict_Mineral"].astype(object)
    probs = df_pred["Predict_Probability"].astype(float)
    
    # Mask low confidence pixels
    labels = labels.mask(probs < prob_threshold)
    labels_flat, probs_flat = labels.to_numpy(), probs.to_numpy()
    H, W = shape
    mineral_map = labels_flat.reshape(H, W)
    prob_map = probs_flat.reshape(H, W)

    # Determine which phases to keep for the dashboard
    if phases and exclude_phases:
        warnings.warn(
            "Both 'phases' and 'exclude_phases' were provided. "
            "'phases' will take priority and 'exclude_phases' will be ignored.",
            UserWarning
        )

    if phases:
        kept = list(phases)
    else:
        kept = pick_common_phases(mineral_map, top_k=top_k)
        if not kept:
            # Fallback to raw labels if thresholding wiped everything out
            raw = df_pred["Predict_Mineral"].to_numpy().reshape(H, W)
            kept = pick_common_phases(raw, top_k=top_k)
            
        if exclude_phases:
            if isinstance(exclude_phases, str):
                exclude_phases = [exclude_phases]
            kept = [p for p in kept if p not in exclude_phases]

    # Plotting (Pass phase_colors to all three)
    title_suffix = os.path.basename(sample_dir) if isinstance(sample_dir, str) else "Sample"
    
    fig_map, _, _ = plot_phase_map(
        mineral_map, phases=kept, title=f"Phase Map: {title_suffix}",
        remove_islands_flag=remove_islands_flag, fill_islands_flag=fill_islands_flag,
        phase_colors=phase_colors, scalebar_um=scalebar_um, pixel_size_um=pixel_size_um
    )
    
    fig_counts, _ = plot_phase_counts(
        mineral_map, phases=kept, 
        title=f"Mineral Phases: {title_suffix}"
    )
    
    fig_hists, _  = plot_probability_histograms(
        prob_map, mineral_map, phases=kept, prob_threshold=prob_threshold,
        title=f"Prediction Probabilities: {title_suffix}"
    )

    # Mineral Component Calculations (Now Always Active)
    default_spec = {
        "Feldspar": {
            "labels": ["Feldspar", "Plagioclase", "KFeldspar"],
            "calculator": FeldsparClassifier, "method": "classify",
            "cols": ["An", "Ab", "Or"], "kwargs": {"subclass": False}, "transforms": {}
        },
        "Clinopyroxene": {
            "labels": ["Clinopyroxene"],
            "calculator": PyroxeneClassifier, "method": "calculate_components",
            "cols": ["XMg", "En", "Fs", "Wo"], "kwargs": {}, "transforms": {}
        },
        "Orthopyroxene": {
            "labels": ["Orthopyroxene"],
            "calculator": PyroxeneClassifier, "method": "calculate_components",
            "cols": ["XMg", "En", "Fs", "Wo"], "kwargs": {}, "transforms": {}
        },
        "Olivine": {
            "labels": ["Olivine"],
            "calculator": OlivineCalculator, "method": "calculate_components",
            "cols": ["XFo"], "kwargs": {}, "transforms": {}
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

    # Always return the full results dictionary
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
        "component_maps": component_maps,
        "component_frames": component_frames
    }


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


def plot_component_composite(res,
                             title="Composite",
                             save_path=None,
                             fill_missing=True,
                             min_speck_size=1,
                             # Component configuration
                             feld_key="Feldspar.An",
                             cpx_key="Clinopyroxene.XMg",
                             opx_key="Orthopyroxene.XMg",
                             ol_key="Olivine.XFo",
                             phases=None,
                             component_labels={
                                "Plagioclase": ("Feldspar", "Plagioclase", "Anorthite", "Albite"),
                                "Clinopyroxene": ("Clinopyroxene", "Augite", "Diopside"),
                                "Orthopyroxene": ("Orthopyroxene", "Enstatite", "Hypersthene"),
                                "Olivine": ("Olivine", "Forsterite", "Fayalite"),
                             },
                             mask_config=None,
                             phase_colors=None,
                             smooth_sigma=0.0,
                             limits_mode="percentile",
                             percentile=(5, 95),
                             legend_on=True,
                             legend_cols=1,
                             ax=None,
                             scalebar_um=None,
                             scalebar_loc="lower left",
                             pixel_size_um=1.0,
                             scalebar_color="black",
                             dpi=300,
                             ):
    
    # Handle NoneType mask_config
    if mask_config is None:
        mask_config = {
            "Glass": {"labels": ("Glass", "Melt"), "color": "#F9C300"},
            "Spinel": {"labels": ("Spinel"), "color": "#2E2DCE"},
        }
    
    comp_maps = res.get("component_maps", {})
    # Use .copy() so we don't destructively modify the original dictionary data
    raw_mineral_map = res.get("mineral_map")
    if raw_mineral_map is None:
        raise ValueError("mineral_map is required in res.")

    # 2. Now it is safe to cast to an object array and copy
    mineral_map = np.asarray(raw_mineral_map, dtype=object).copy()

    bad_vals = {"nan", "NaN"}

    # 3. Standardize NaNs
    is_actual_nan = pd.isna(mineral_map)
    mineral_map[is_actual_nan] = "nan"

    # =====================
    # Remove small speckles
    # =====================
    if min_speck_size > 0:
        unique_phases_raw = set(np.unique(mineral_map.astype(str))) - bad_vals
        for phase in unique_phases_raw:
            mask = (mineral_map == phase)
            cleaned_mask = remove_small_objects(mask, min_size=min_speck_size, connectivity=1)
            # Replace removed tiny specks with "nan"
            mineral_map[mask & ~cleaned_mask] = "nan"

    # ============
    # Fill islands
    # ============
    if fill_missing:
        invalid_mask = np.isin(mineral_map.astype(str), list(bad_vals))
        valid_mask = ~invalid_mask
        
        # Identify topological holes (invalid pixels completely surrounded by valid pixels)
        filled_valid_mask = binary_fill_holes(valid_mask)
        islands_mask = filled_valid_mask & invalid_mask
        
        if np.any(islands_mask):
            _, indices = distance_transform_edt(invalid_mask, return_indices=True)
            nearest_y = indices[0, islands_mask]
            nearest_x = indices[1, islands_mask]
            mineral_map[islands_mask] = mineral_map[nearest_y, nearest_x]

    # Identify unique phases for plotting
    unique_str_array = np.unique(mineral_map.astype(str))
    unique_phases = set(unique_str_array) - bad_vals
    
    if phases is not None:
        if isinstance(phases, str):
            phases = [phases]
        unique_phases = {p for p in unique_phases if p in phases}

    consumed_phases = set()
    active_components = []
    active_masks = []
    
    # Dictionary to store the processed continuous data we return at the end
    processed_comp_maps = {}

    # Process Components
    comp_defs = [
        {"id": "Plagioclase", "key": feld_key, "ramp": "teal", "leg": "Plagioclase (An)", "col": "#009988"},
        {"id": "Clinopyroxene", "key": cpx_key, "ramp": "red", "leg": "Clinopyroxene (Mg#)", "col": "#e7b3b1"},
        {"id": "Orthopyroxene", "key": opx_key, "ramp": "maroon", "leg": "Orthopyroxene (Mg#)", "col": "#5A0F0F"},
        {"id": "Olivine", "key": ol_key, "ramp": "green", "leg": "Olivine (Fo)", "col": "#666633"},
    ]

    for cd in comp_defs:
        labels = component_labels.get(cd["id"], (cd["id"],))
        present_labels = [l for l in labels if l in unique_phases]
        if present_labels:
            raw_data = comp_maps.get(cd["key"], None)
            if raw_data is not None and np.any(np.isfinite(raw_data)):
                # Copy data so we don't modify the raw array
                data = raw_data.copy()
                
                # ==========================================
                # Fill islands and components
                # ==========================================
                if fill_missing:
                    invalid_data_mask = ~np.isfinite(data)
                    valid_data_mask = ~invalid_data_mask
                    
                    filled_data_mask = binary_fill_holes(valid_data_mask)
                    data_islands_mask = filled_data_mask & invalid_data_mask
                    
                    if np.any(data_islands_mask):
                        _, indices = distance_transform_edt(invalid_data_mask, return_indices=True)
                        nearest_y = indices[0, data_islands_mask]
                        nearest_x = indices[1, data_islands_mask]
                        data[data_islands_mask] = data[nearest_y, nearest_x]

                if smooth_sigma > 0:
                    valid_mask = np.isfinite(data)
                    d_copy = np.where(valid_mask, data, 0.0)
                    w = valid_mask.astype(float)
                    d_s = gaussian_filter(d_copy, smooth_sigma)
                    w_s = gaussian_filter(w, smooth_sigma)
                    data = np.where(valid_mask, d_s / np.maximum(w_s, 1e-12), np.nan)
                
                # Save the processed data map to hand back to the user
                processed_comp_maps[cd["key"]] = data
                
                # NOTE: Ensure _auto_limits is defined in your script
                vmin, vmax = _auto_limits(data, mode=limits_mode, percentile=percentile)                
                active_components.append({**cd, "data": data, "vmin": vmin, "vmax": vmax})
                consumed_phases.update(present_labels)

    # Process Configured Masks
    for m_name, cfg in mask_config.items():
        present_labels = [l for l in cfg['labels'] if l in unique_phases]
        if present_labels:
            mask = np.isin(mineral_map, present_labels)
            active_masks.append({
                "name": m_name, "mask": mask, "color": cfg['color'],
                "cmap": ListedColormap(['#FFFFFF00', cfg['color']])
            })
            consumed_phases.update(present_labels)

    # Process Leftovers
    leftover = unique_phases - consumed_phases
    if leftover:
        if phase_colors is None:
            from matplotlib import cm
            palette = cm.get_cmap("tab20", len(leftover))
            phase_colors = {p: palette(i) for i, p in enumerate(leftover)}

        for p in sorted(list(leftover)):
            mask = (mineral_map == p)
            color = phase_colors.get(p, "#808080")
            active_masks.append({
                "name": p, "mask": mask, "color": color,
                "cmap": ListedColormap(['#FFFFFF00', color])
            })

    # --- Setup Plotting Helpers ---
    def _make_ramp(c1, c2):
        # Convert hex strings to normalized (0.0 to 1.0) RGB tuples
        c1_rgb = to_rgb(c1)
        c2_rgb = to_rgb(c2)
        
        arr = np.ones((256, 4))
        for i in range(3):
            # No need to divide by 255 anymore since to_rgb already does it
            arr[:, i] = np.linspace(c1_rgb[i], c2_rgb[i], 256)
            
        cmap = ListedColormap(arr)
        cmap.set_bad(alpha=0)
        return cmap


    ramps = {
        "teal":   _make_ramp('#CCEEFF', '#009988'),
        "red":    _make_ramp('#FFE6E6', '#C83C50'),
        "maroon": _make_ramp('#CB4545', '#5A0000'), #[180, 50, 50], [90, 0, 0]),
        "green":  _make_ramp('#EFEEBB', '#666633') #[239, 238, 187], [102, 102, 51])
    }

    legend_entries = [(c['leg'], c['col']) for c in active_components] + \
                     [(m['name'], m['color']) for m in active_masks]

    # --- Setup Figure and Axes ---
    if ax is None:
        # NOTE: Ensure _auto_figsize_from_array is defined in your script
        fig_w, fig_h = _auto_figsize_from_array(mineral_map.shape, n_legend=len(legend_entries))
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
        
        if legend_on:
            gs = fig.add_gridspec(1, 2, width_ratios=[fig_w-1.5, 1.5])
            ax_map = fig.add_subplot(gs[0, 0])
            ax_legend = fig.add_subplot(gs[0, 1])
        else:
            ax_map = fig.add_subplot(111)
            ax_legend = None
    else:
        ax_map = ax
        fig = ax.get_figure()
        ax_legend = None

    # Render layers
    for c in active_components:
        ax_map.imshow(np.ma.masked_invalid(c['data']), cmap=ramps[c['ramp']], 
                      vmin=c['vmin'], vmax=c['vmax'], interpolation="none")
    
    for m in active_masks:
        display = np.where(m['mask'], 1.0, np.nan)
        ax_map.imshow(display, cmap=m['cmap'], vmin=0, vmax=1, interpolation="none")

    ax_map.set_title(title, pad=8)
    ax_map.axis("off")

    # Add Legend only if legend_on is True
    if legend_on:
        handles = [mpatches.Patch(facecolor=c, label=lab) for lab, c in legend_entries]
        if ax_legend is not None:
            ax_legend.axis("off")
            ax_legend.legend(handles=handles, loc="upper left", frameon=False, 
                             title="Layers", ncol=legend_cols, fontsize=8)
        else:
            ax_map.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), 
                          frameon=False, title="Layers", fontsize=8)

    # Add scalebar if requested
    if scalebar_um is not None:
        scalebar_pixels = scalebar_um / pixel_size_um
        fontprops = fm.FontProperties(size=12, weight='bold')
        scalebar = AnchoredSizeBar(
            ax_map.transData, scalebar_pixels, f'{scalebar_um} µm', loc=scalebar_loc,                 
            pad=0.5, color=scalebar_color, frameon=False, size_vertical=1, 
            sep=3, label_top=True, fontproperties=fontprops
        )
        ax_map.add_artist(scalebar)

    if save_path: 
        fig.savefig(save_path, bbox_inches='tight')
        
    # Now returns the figure, the cleaned string map, and a dict of the cleaned float maps!
    return fig, mineral_map, processed_comp_maps


# %% EBSD mapping


def parse_ctf_header(filepath: str):
    """
    Parses the header of a .ctf file to extract grid dimensions and phase mappings.
    The file is expected to contain 'XCells', 'YCells', 'Phases', and a data table 
    starting with 'Phase\\tX\\tY'.

    Args:
        filepath (str): The path to the .ctf file.

    Returns:
        x_cells (int): The number of cells in the X direction.
        y_cells (int): The number of cells in the Y direction.
        data_start (int): The line index where the actual data table begins.
        phase_mapping (dict): A dictionary that maps each integer ID to its phase name.
    """
    x_cells = y_cells = n_phases = data_start = None
    phase_mapping = {0: "Unindexed"}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith("XCells"):
            x_cells = int(line.split("\t")[1])
        elif line.startswith("YCells"):
            y_cells = int(line.split("\t")[1])
        elif line.startswith("Phases"):
            n_phases = int(line.split("\t")[1])
            
            # Extract phase names immediately using the n_phases count
            for j in range(1, n_phases + 1):
                parts = lines[i + j].strip().split("\t")
                if len(parts) >= 3 and parts[2].strip():
                    phase_mapping[j] = parts[2].strip()
                else:
                    phase_mapping[j] = f"Phase{j}"
                    
        elif line.startswith("Phase\tX\tY"):
            data_start = i
            break

    # Verify all required header components were found
    if None in (x_cells, y_cells, n_phases, data_start):
        raise ValueError("Missing required header information in CTF file.")

    return x_cells, y_cells, data_start, phase_mapping


def plot_ctf_phases(filepath: str, max_legend=25, rename_dict=None, phase_colors=None,
                    ax=None, title="default", scalebar_um=None, scalebar_loc="lower left", legend_on=True):
    """
    Loads phase data from a .ctf file and generates a 2D categorical phase map.
    It maps raw phase IDs to their corresponding names, optionally renames them,
    and orders the legend by phase abundance.

    Args:
        filepath (str): The path to the .ctf file.
        max_legend (int, optional): The maximum number of phases to display in 
        the legend. Defaults to 25.
        rename_dict (dict, optional): A dictionary mapping messy phase names 
        (or partial matches) to clean names. Defaults to None.
        phase_colors (dict, optional): A dictionary mapping clean phase names 
        to specific matplotlib colors (e.g., {'Quartz': 'red', 'Enstatite': '#00FF00'}). 
        Defaults to None.
        ax (matplotlib.axes.Axes, optional): An existing axes object to plot on. 
        If None, a new figure and axes will be created.
        title (str or None, optional): The title for the plot. If "default", creates an 
        auto-generated title with dimensions. If None, no title is shown.
        scalebar_um (float, optional): If provided, draws a scale bar of this many microns 
        in the lower left corner.

    Returns:
        phase_map (ndarray): A 2D array of the mapped phase names as strings.
        raw_ids (ndarray): A 2D array of the raw numeric phase IDs from the file.
        phase_mapping (dict): A dictionary mapping raw IDs to phase names.
        unique_names (ndarray): An array of the unique phase names sorted by abundance.
    """
    x_cells, y_cells, data_start, phase_mapping = parse_ctf_header(filepath)

    # Extract step size from the file to calculate scale
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("XStep"):
                step_size = float(line.split()[1])
                break
            elif line.startswith("Phases"):  # Stop searching once we hit the phases list
                break

    # --- Apply clean names to the phase mapping ---
    if rename_dict:
        for phase_id, original_name in phase_mapping.items():
            for messy_name, clean_name in rename_dict.items():
                if messy_name in original_name:
                    phase_mapping[phase_id] = clean_name

    # Read only the Phase column to save memory
    df = pd.read_csv(filepath, sep="\t", skiprows=data_start, usecols=["Phase"])
    raw_ids = df["Phase"].values

    if len(raw_ids) != (x_cells * y_cells):
        raise ValueError("Mismatch between header grid size and data points.")

    # Map integer IDs to phase names 
    phase_names = df["Phase"].map(phase_mapping).fillna(df["Phase"].astype(str) + "_Unknown").values

    # Get unique names and their frequencies
    unique_names, counts = np.unique(phase_names, return_counts=True)

    # Sort arrays descending by abundance
    order = np.argsort(counts)[::-1]
    unique_names = unique_names[order]
    counts = counts[order]

    # Create a 2D map of categorical codes purely for plotting
    name_to_code = {name: i for i, name in enumerate(unique_names)}
    numeric_ids = np.array([name_to_code[name] for name in phase_names]).reshape((y_cells, x_cells))
    
    # Create the 2D string-based map to return
    phase_map = phase_names.reshape((y_cells, x_cells))

    # --- Set up the custom colormap ---
    base_colors = list(plt.get_cmap("tab20").colors)
    colors = []
    color_idx = 0
    
    for name in unique_names:
        if phase_colors and name in phase_colors:
            colors.append(phase_colors[name])
        else:
            colors.append(base_colors[color_idx % len(base_colors)])
            color_idx += 1
            
    cmap = ListedColormap(colors)

    # --- Handle Axes ---
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True  

    ax.imshow(numeric_ids, cmap=cmap, interpolation="none")
    ax.axis("off")

    # Title
    if title == "default":
        ax.set_title(f"Phase Map — {x_cells} x {y_cells}")
    elif title is not None:
        ax.set_title(title)

    # Add Scale Bar
    if scalebar_um is not None:
        # Convert the physical size (µm) to image pixels
        scalebar_pixels = scalebar_um / step_size
        
        # Configure the bar
        fontprops = fm.FontProperties(size=12, weight='bold')
        scalebar = AnchoredSizeBar(
            ax.transData,
            scalebar_pixels,
            f'{scalebar_um} µm',
            loc=scalebar_loc,
            pad=0.5,
            color='black',
            frameon=False,
            size_vertical=1,
            sep=3,
            label_top=True,
            fontproperties=fontprops
        )
        ax.add_artist(scalebar)

    # Build the legend handles and labels
    n_show = min(max_legend, len(unique_names))
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=10, color=cmap(name_to_code[name]))
        for name in unique_names[:n_show]
    ]
    labels = [f"{name} ({count})" for name, count in zip(unique_names[:n_show], counts[:n_show])]

    if len(unique_names) > n_show:
        handles.append(plt.Line2D([0], [0], linestyle=""))
        labels.append(f"... +{len(unique_names) - n_show} more")

    if legend_on:
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    
    if show_plot:
        plt.tight_layout()
        plt.show()

    return phase_map, raw_ids.reshape((y_cells, x_cells)), phase_mapping, unique_names


# %% 