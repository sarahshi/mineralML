# %%

__author__ = "Sarah Shi"

import ast
import os
import re
import warnings
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.morphology import remove_small_objects, remove_small_holes

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap, is_color_like, to_hex, to_rgb
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from mineralML.core import *
from mineralML.hybrid import *
from mineralML.stoichiometry import *
from mineralML.constants import *
from .constants import OXIDES

# %%


def _legacy_remove_small_max_size(size):
    """
    Translate pre-0.26 skimage thresholds to the new ``max_size`` semantics.

    Older ``min_size`` / ``area_threshold`` parameters removed objects or holes
    strictly smaller than the given size. New ``max_size`` removes sizes less
    than or equal to its value, so we subtract 1 to preserve legacy behavior.
    """
    return max(int(size) - 1, 0)


def _remove_small_objects_compat(mask, size, **kwargs):
    try:
        return remove_small_objects(mask, max_size=_legacy_remove_small_max_size(size), **kwargs)
    except TypeError:
        return remove_small_objects(mask, min_size=size, **kwargs)


def _remove_small_holes_compat(mask, size, **kwargs):
    try:
        return remove_small_holes(mask, max_size=_legacy_remove_small_max_size(size), **kwargs)
    except TypeError:
        return remove_small_holes(mask, area_threshold=size, **kwargs)


def _coerce_profile_color(color, fallback=None):
    """
    Normalize saved profile colors into a Matplotlib-compatible value.

    Handles RGBA tuples, hex strings, named colors, and tuple-like strings
    produced by DataFrame serialization.
    """
    if fallback is None:
        fallback = plt.get_cmap("tab10")(0)

    if color is None or (isinstance(color, float) and pd.isna(color)):
        return fallback

    if is_color_like(color):
        return color

    if isinstance(color, str):
        try:
            parsed = ast.literal_eval(color)
        except (ValueError, SyntaxError):
            parsed = None
        if parsed is not None and is_color_like(parsed):
            return parsed

    return fallback


def _profile_value_column_names(key):
    """
    Canonical output column names for a specific profile variable.
    """
    return {
        "raw": key,
        "smoothed": f"{key}_smoothed",
    }


def _profile_table_for_key(profile_df, key):
    """
    Convert a generic profile dataframe into a key-named table.
    """
    out = profile_df.copy()
    cols = _profile_value_column_names(key)
    rename_map = {
        "value_smoothed": cols["smoothed"],
        "value": cols["raw"],
    }
    out = out.rename(columns=rename_map)
    drop_cols = [c for c in ("bin", "key", "n_pixels") if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    preferred = [
        "profile_id",
        "distance_px",
        "distance_um",
        key,
        f"{key}_smoothed",
    ]
    ordered = [c for c in preferred if c in out.columns]
    tail_cols = [
        c for c in ("x0", "y0", "x1", "y1")
        if c in out.columns
    ]
    ordered += [c for c in out.columns if c not in ordered and c not in tail_cols]
    ordered += tail_cols
    return out[ordered]


def _resolve_profile_value_columns(profile_df):
    """
    Infer the raw and smoothed value columns from either generic or key-named
    profile tables.
    """
    columns = list(profile_df.columns)
    if "value" in columns:
        raw_col = "value"
    else:
        excluded = {
            "profile_id",
            "distance_px",
            "distance_um",
            "bin",
            "key",
            "source",
            "n_pixels",
            "x0",
            "y0",
            "x1",
            "y1",
            "width_px",
            "length_px",
            "length_um",
            "pixel_size_um",
            "method",
            "smooth_window",
            "color",
            "x",
            "y",
            "perp_distance_px",
        }
        candidates = [
            c for c in columns if c not in excluded and not c.endswith("_smoothed")
        ]
        raw_col = candidates[0] if candidates else None

    if "value_smoothed" in columns:
        smoothed_col = "value_smoothed"
    elif raw_col is not None and f"{raw_col}_smoothed" in columns:
        smoothed_col = f"{raw_col}_smoothed"
    else:
        smoothed_candidates = [c for c in columns if c.endswith("_smoothed")]
        smoothed_col = smoothed_candidates[0] if smoothed_candidates else raw_col

    if raw_col is None or smoothed_col is None:
        raise KeyError(
            "Could not determine profile value columns. Expected either "
            "'value'/'value_smoothed' or key-named columns like "
            "'SiO2'/'SiO2_smoothed'."
        )

    return raw_col, smoothed_col


def maps_to_df(E):
    """
    Convert a dictionary of 2D arrays into a flat DataFrame.

    Parameters:
        E (dict): Dictionary mapping element symbols to 2D numpy arrays (maps).

    Returns:
        df (pd.DataFrame): Flattened DataFrame with each element as a column.
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


def df_to_maps(df, shape):
    """
    Convert a flattened DataFrame back into dict of 2D arrays.
 
    Parameters:
        df (pd.DataFrame): DataFrame with flattened values for each feature/element.
        shape (tuple): Original 2D shape (H, W).
 
    Returns:
        maps (dict): Dictionary mapping column names to 2D numpy arrays shaped (H, W).
    """
    H, W = shape
    return {k: df[k].to_numpy().reshape(H, W, order="C") for k in df.columns}


def renormalize_maps(ox_maps):
    """
    Scale each pixel so oxide totals sum to 100 wt%.
 
    Parameters:
        ox_maps (dict): Dictionary mapping oxide names to 2D numpy arrays.
 
    Returns:
        ox_maps (dict): Renormalized dictionary with the same keys.
    """
    keys = list(ox_maps.keys())
    stack = np.stack([ox_maps[k] for k in keys], axis=0) # (n_oxides, rows, cols)
    totals = np.nansum(stack, axis=0, keepdims=True) # (1, rows, cols)
    totals[totals == 0] = np.nan # avoid div-by-zero
    stack = stack / totals * 100.0
    return {k: stack[i] for i, k in enumerate(keys)}


def load_element_maps(path, drop_trailing_blank=False, verbose=True):
    """
    Load element maps from a directory of CSVs into a dictionary of 2D arrays.
 
    Parameters:
        path (str): Path to directory containing CSV files of element maps.
        drop_trailing_blank (bool): If True, drops the last column when it is
            entirely NaN or zero. Defaults to False.
        verbose (bool): If True, prints status messages for each loaded file.
 
    Returns:
        out (dict): Dictionary mapping element symbols (str) to 2D numpy arrays (float).
            NaNs are preserved. Trailing blank columns are dropped when
            drop_trailing_blank is True.
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(path)


    ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "Cr", "Ni", "Zr", "P", "K"]
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

        filepath = os.path.join(path, f)
        arr = pd.read_csv(filepath, header=None).to_numpy(dtype=float)

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if drop_trailing_blank and arr.shape[1] > 0:
            last = arr[:, -1]
            if np.all(~np.isfinite(last)) or np.allclose(last, 0, equal_nan=True):
                arr = arr[:, :-1]

        if matched in out:
            print(f"[warn] duplicate element '{matched}': {f} overwrites previous")

        out[matched] = arr
        if verbose:
            print(f"[ok] {f}  →  {matched}  {arr.shape}")

    # final sanity check: consistent shapes
    shapes = {k: v.shape for k, v in out.items()}
    if len({s for s in shapes.values()}) > 1:
        print("[warn] inconsistent shapes:", shapes)

    return out


def load_maps_from_dir(path, units="element_wt%", renormalize=False):
    """
    Load per-element CSV maps from a directory and return oxide wt% maps.
 
    Parameters:
        path (str): Path to directory containing element CSV maps.
        units (str): Interpretation of the input map values. Use
            ``"element_wt%"`` if the CSV maps contain elemental wt% values
            that should be stoichiometrically converted to oxide wt%, or
            ``"oxide_wt%"`` if the CSV maps already contain oxide wt% values
            and only need to be relabeled from element-style names to oxide
            names.
        renormalize (bool): If True, rescale each pixel so oxides sum to 100 wt%.
 
    Returns:
        ox_maps (dict): Dictionary mapping oxide names (str) to 2D numpy arrays (float).
    """

    E = load_element_maps(path)
    df_in, shape = maps_to_df(E)

    if units == "element_wt%":
        df_ox, _ = element_to_oxide(df_in)
    elif units == "oxide_wt%":
        df_ox, _ = element_to_oxide_identity(df_in)
    else:
        raise ValueError(
            "units must be one of {'element_wt%', 'oxide_wt%'}"
        )

    ox_maps = df_to_maps(df_ox, shape)

    if renormalize:
        ox_maps = renormalize_maps(ox_maps)

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


# %% 


def _ensure_columns(df, expected=OXIDES):
    """
    Aligns DataFrame columns to the expected list in one fast operation.
 
    Parameters:
        df (pd.DataFrame): Input DataFrame (may have extra or missing columns).
        expected (list[str]): Target column names in desired order.
 
    Returns:
        out (pd.DataFrame): Reindexed DataFrame with columns matching expected.
            Missing columns are filled with NaN; extra columns are dropped.
    """
    out = df.copy()
    if "FeO" in out.columns:
        out.rename(columns={"FeO": "FeOt"}, inplace=True)

    # Reindex aligns columns and fills missing ones with NaN
    return out.reindex(columns=expected)


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
    n_labels = len(labels)
    cmap = plt.get_cmap(cmap_name, max(n_labels, 1))
    cols = []
    for i in range(len(labels)):
        r, g, b, _ = cmap(i)
        cols.append((min(r, 0.95), min(g, 0.95), min(b, 0.95)))
    return {lab: cols[i] for i, lab in enumerate(labels)}


def _annotate_stacked_bar(
    ax, y, phases, props, phase_colors=None, fmt="{:.1f}%", fs=10,
    min_inside=0.03, dy_inside=0.0, dy_out=0.35, alternate=True,
    x_jitter=1.0, force_outside=None, force_dx=None,
):
    """
    Label each segment of a stacked horizontal bar with its percentage.
 
    Values above ``min_inside`` are placed inside the bar; smaller slices get
    staggered callout annotations above/below.
 
    Parameters:
        ax (matplotlib.axes.Axes): Axes containing the stacked bar.
        y (float): Vertical position of the bar in data coordinates.
        phases (list[str]): Phase names in bar-segment order.
        props (list[float]): Proportions (0-1) corresponding to each phase.
        phase_colors (dict|None): {phase: color} used to tint annotation boxes.
        fmt (str): Format string for the percentage label.
        fs (int): Font size for annotations.
        min_inside (float): Minimum proportion to place the label inside the bar.
        dy_inside (float): Vertical offset for inside labels.
        dy_out (float): Vertical offset magnitude for outside callouts.
        alternate (bool): If True, alternate callout direction (above/below).
        x_jitter (float): Horizontal jitter scale for forced-outside labels.
        force_outside (set|None): Phase names always placed outside.
        force_dx (dict|None): {phase: signed_offset} for manual x-nudging.
    """
    force_outside = set(force_outside or [])
    force_dx = dict(force_dx or {})
    phase_colors = dict(phase_colors or {})
    bbox = dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=1, lw=1.0)

    left, out_i = 0.0, 0
    for p, prop in zip(phases, props):
        prop_pct = prop * 100
        x = left + prop_pct / 2.0
        left += prop_pct
        txt = fmt.format(prop_pct)

        ec_color = phase_colors.get(p, "none")
        bb = {**bbox, "ec": ec_color}

        if prop >= min_inside and p not in force_outside:
            ax.text(
                x, y + dy_inside, txt, ha="center", va="center",
                fontsize=fs, color="black", bbox=bb, clip_on=False, zorder=10,
            )
            continue

        sgn_y = 1 if (not alternate or out_i % 2 == 0) else -1
        dx = (
            float(np.sign(force_dx[p])) * x_jitter * float(abs(force_dx[p]))
            if (p in force_outside and x_jitter and p in force_dx)
            else 0.0
        )
        x_txt = float(np.clip(x + dx, 0.0, 100.0))

        ax.annotate(
            txt, xy=(x, y), xycoords="data",
            xytext=(x_txt, y + sgn_y * dy_out), textcoords="data",
            ha="center", va="bottom" if sgn_y > 0 else "top",
            fontsize=fs, color="black", bbox=bb,
            arrowprops=dict(arrowstyle="-", lw=0.9, color="k", shrinkA=0, shrinkB=0),
            clip_on=False, zorder=20,
        )
        out_i += 1


def _auto_figsize_from_array(
    shape,
    n_legend,
    legend_side="right",
    legend_cols=1,
    base_width=6,
    base_height=6,
    legend_width_ratio=0.3,
):
    """
    Automatically calculate figure size based on array shape and legend requirements.

    Parameters:
        shape (tuple): Shape of the mineral map array (height, width).
        n_legend (int): Number of legend entries.
        legend_side (str): "right", "left", "top", "bottom".
        legend_cols (int): Number of columns for legend.
        base_width (float): Base width for square-ish maps.
        base_height (float): Base height for square-ish maps.
        legend_width_ratio (float): Ratio of width dedicated to legend.

    Returns:
        figsize (tuple): (width, height) in inches.
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


def _auto_limits(data, mode="std", percentile=(5, 95)):
    """
    Calculate dynamic vmin/vmax for a map based on its valid values.
 
    Parameters:
        data (ndarray): 2D array of values (NaN/inf pixels are ignored).
        mode (str): 'percentile' or 'std' (mean ± 2 SD).
        percentile (tuple): (low, high) percentile bounds when mode='percentile'.
 
    Returns:
        vmin (float): Lower color-scale limit.
        vmax (float): Upper color-scale limit.
    """
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return 0.0, 1.0
    if mode == "percentile":
        return np.percentile(vals, percentile[0]), np.percentile(vals, percentile[1])
    else:
        # Mean +/- 2SD for contrast amplification
        mu, sigma = np.mean(vals), np.std(vals)
        return mu - 2 * sigma, mu + 2 * sigma


def _add_scalebar(
    ax,
    scalebar_um=None,
    pixel_size_um=None,
    *,
    scalebar_loc="lower left",
    scalebar_col="black",
    fontsize=12,
    size_vertical=1,
    pad=0.5,
    sep=3,
    label_top=True,
    warn=True,
):
    """
    Adds a scale bar to an image axis.

    This helper centralizes scale-bar creation for spatial plotting functions.
    If no scale bar length is provided, nothing is added. If a scale bar length
    is provided but the pixel size is unknown, the scale bar is skipped and a
    warning can optionally be emitted.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to which the scale bar will be added.
        scalebar_um (float, optional): Desired scale bar length in micrometers.
            If None, no scale bar is added.
        pixel_size_um (float, optional): Physical size of one pixel in
            micrometers. Required to convert the requested scale bar length into
            data units.
        scalebar_loc (str, optional): Location of the scale bar on the axis.
            Passed to `AnchoredSizeBar`. Defaults to "lower left".
        scalebar_col (str, optional): Color of the scale bar and label text.
            Defaults to "black".
        fontsize (float, optional): Font size for the scale bar label.
            Defaults to 12.
        size_vertical (float, optional): Thickness of the scale bar in data
            units. Defaults to 1.
        pad (float, optional): Padding around the scale bar box. Defaults to 0.5.
        sep (float, optional): Separation between the bar and its label.
            Defaults to 3.
        label_top (bool, optional): Whether to place the label above the bar.
            Defaults to True.
        warn (bool, optional): Whether to emit a warning when `scalebar_um` is
            provided but `pixel_size_um` is missing. Defaults to True.

    Returns:
        scalebar (AnchoredSizeBar or None): The added scale bar artist, or None
            if no scale bar was added.
    """
    if scalebar_um is None:
        return None

    if pixel_size_um is None:
        if warn:
            warnings.warn(
                "scalebar_um provided without pixel_size_um; skipping scale bar."
            )
        return None

    scalebar_pixels = scalebar_um / pixel_size_um
    fontprops = fm.FontProperties(size=fontsize)

    scalebar = AnchoredSizeBar(
        ax.transData,
        scalebar_pixels,
        f"{scalebar_um:g} µm",
        loc=scalebar_loc,
        pad=pad,
        color=scalebar_col,
        frameon=False,
        size_vertical=size_vertical,
        sep=sep,
        label_top=label_top,
        fontproperties=fontprops,
    )
    ax.add_artist(scalebar)

    return scalebar


# %% 

def remove_islands(
    mineral_map,
    min_size=2,
    connectivity=1,
    fill_val=0,
    phase_min_sizes=None,
    grouped_phases=None,
    ignore_vals=None,
):
    """
    Removes isolated islands of minerals using morphological size filtering.
 
    This works with both string and integer arrays. Small objects below the
    specified area threshold are overwritten with `fill_val`.
 
    Parameters:
        mineral_map (ndarray): 2D array of phase labels or IDs.
        min_size (int): Default minimum area (in pixels) for an object to be kept.
        connectivity (int): Neighborhood definition (1 for 4-connected, 2 for 8-connected).
        fill_val (any): The value to insert where pixels are removed (e.g., "nan", 0).
        phase_min_sizes (dict, optional): Custom minimum sizes mapped per phase.
        grouped_phases (list[tuple], optional): Lists of phases that should be
            treated as a single continuous unit when evaluating size
            (e.g., grouped clinopyroxene and orthopyroxene).
        ignore_vals (set, optional): Values that are considered background and
            skipped (e.g., 'NaN', 'Unindexed').
 
    Returns:
        cleaned_map (ndarray): A new 2D array with small islands removed.
    """
    # Create a copy to modify
    cleaned_map = np.copy(mineral_map)

    # Setup ignore list (combining default nan/background inputs)
    ignore_vals = set(ignore_vals) if ignore_vals else set()
    ignore_vals.update({"nan", "NaN", "None", 0, fill_val})

    # Get unique phases without forcing a global string cast
    unique_phases = pd.unique(cleaned_map.ravel())

    valid_phases = set()
    for p in unique_phases:
        # Ignore NaNs, Nones, and designated ignore values (like 0)
        if pd.notna(p) and str(p) not in ignore_vals and p not in ignore_vals:
            valid_phases.add(p)

    phase_min_sizes = phase_min_sizes or {}
    grouped_phases = grouped_phases or []
    processed_phases = set()

    # -----------------------------------------
    # Process Grouped Phases
    # -----------------------------------------
    for group in grouped_phases:
        present_in_group = [p for p in group if p in valid_phases]
        if not present_in_group:
            continue

        # Create a combined boolean mask
        mask = np.isin(cleaned_map, present_in_group)

        # Pick the largest min_size among the grouped phases (or use default)
        group_min_size = max(
            [phase_min_sizes.get(p, min_size) for p in present_in_group]
        )
        cleaned_mask = _remove_small_objects_compat(
            mask, group_min_size - 1, connectivity=connectivity
        )
        removed_pixels = mask & ~cleaned_mask
        cleaned_map[removed_pixels] = fill_val
        processed_phases.update(present_in_group)

    # -----------------------------------------
    # Process Individual Phases
    # -----------------------------------------
    remaining_phases = valid_phases - processed_phases
    for phase in remaining_phases:
        # Type match natively, so 1 == 1 works, and 'Quartz' == 'Quartz' works
        mask = cleaned_map == phase
        current_min_size = phase_min_sizes.get(phase, min_size)
        cleaned_mask = _remove_small_objects_compat(
            mask, current_min_size, connectivity=connectivity
        )
        removed_pixels = mask & ~cleaned_mask

        cleaned_map[removed_pixels] = fill_val

    return cleaned_map


def fill_phase_holes(mineral_map, max_hole_size=10, exclude_phases=None):
    """
    Fills holes strictly within individual continuous phases.
 
    This function prevents accidental "spillover" by ensuring that it only fills
    empty spaces (NaNs or Unindexed) completely enclosed by a single phase.
    It intentionally preserves natural interstitial networks.
 
    Parameters:
        mineral_map (ndarray): 2D array of phase labels (strings or IDs).
        max_hole_size (int): The maximum area (in pixels) of an enclosed empty space
            allowed to be filled.
        exclude_phases (list[str], optional): Phases that naturally exist as
            interstitial material and should NOT be artificially expanded.
            Defaults to ["Glass", "Vesicles"].
 
    Returns:
        filled_map (ndarray): A new 2D array with enclosed holes filled.
    """

    if exclude_phases is None:
        exclude_phases = ["Glass", "Vesicles"]

    filled_map = mineral_map.astype(object).copy()

    # Define what counts as an empty space
    bad_vals = {"nan", "NaN", "None", "unindexed"}
    is_invalid = np.isin(filled_map.astype(str), list(bad_vals)) | pd.isna(filled_map)

    unique_phases = set(np.unique(filled_map.astype(str))) - bad_vals

    for phase in unique_phases:
        # Skip interstitial phases that should not be expanded
        if phase in exclude_phases:
            continue

        # Create a boolean mask for only this phase
        phase_mask = filled_map == phase

        # Fill holes that are completely surrounded by phase
        filled_phase_mask = _remove_small_holes_compat(phase_mask, max_hole_size)

        # Identify pixels that were just filled in
        new_pixels = filled_phase_mask & ~phase_mask

        # Only overwrite pixels that were originally invalid/nan
        # (Prevents accidental overwrite of real mineral inclusions)
        safe_to_fill = new_pixels & is_invalid
        filled_map[safe_to_fill] = phase

    return filled_map


def plot_phase_map(
    mineral_map_2d,
    phases=None,
    title="Phase Map",
    bg_color=(0.08, 0.08, 0.08),
    cmap_name="tab20",
    legend_side="right",
    legend_cols=1,
    remove_islands_flag=False,
    fill_holes_flag=False,
    cleanup_min_size=2,
    hole_size=10,
    min_frac=0.00001,
    scalebar_um=None,
    pixel_size_um=None,
    scalebar_loc="lower left",
    scalebar_col="black",
    phase_colors=None,
    legend_on=True,
    ax=None,
    dpi=300,
):
    """
    Generates a 2D categorical phase map colored by mineral type.
 
    Applies optional morphological cleaning (removing small islands and filling
    small holes) before rendering. Automatically handles legend placement, figure
    sizing, and scale bar generation.
 
    Parameters:
        mineral_map_2d (array-like): 2D array of phase labels (strings or ints).
        phases (list[str], optional): Explicit list of phases to include in the legend.
            If None, common phases are automatically extracted.
        title (str): The title of the plot.
        bg_color (tuple): RGB tuple for the background (unindexed/NaN) color.
        cmap_name (str): Name of the matplotlib colormap to sample from.
        legend_side (str): Placement of the legend ('right', 'left', 'top', 'bottom').
        legend_cols (int): Number of columns for the legend text.
        remove_islands_flag (bool): If True, removes isolated pixels.
        fill_holes_flag (bool): If True, fills small holes within continuous phases.
        cleanup_min_size (int): Minimum pixel area to keep if remove_islands_flag is True.
        hole_size (int): Maximum hole area (in pixels) to fill if fill_holes_flag is True.
        min_frac (float): Minimum pixel fraction (default 0.00001, i.e., 0.001%) 
            required to keep a phase. Rare phases below this are grouped into 'Unknown'.
        scalebar_um (float, optional): Length of the scale bar in micrometers.
        pixel_size_um (float): Physical size of a single pixel in micrometers.
        scalebar_loc (str): Location of the scale bar (e.g., 'lower left').
        scalebar_col (str): Color of the scale bar text/line.
        phase_colors (dict, optional): Custom mapping of {PhaseName: (R,G,B)}.
        legend_on (bool): If True, displays the legend.
        ax (matplotlib.axes.Axes, optional): Pre-existing axes to plot on.
        dpi (int): Resolution for the generated figure.
 
    Returns:
        fig (matplotlib.figure.Figure): The generated matplotlib figure.
        ax_map (matplotlib.axes.Axes): The axes containing the image map.
        cleaned_mineral_map (ndarray): The processed 2D mineral map after cleanup.
    """

    mineral_map_2d = np.asarray(mineral_map_2d, dtype=object)

    if fill_holes_flag:
        mineral_map_2d = fill_phase_holes(mineral_map_2d, max_hole_size=hole_size)

    phases = phases or pick_common_phases(mineral_map_2d)
    
    # ----------------------------------------------------
    # Filter Rare Phases (< min_frac)
    # ----------------------------------------------------
    if min_frac > 0:
        total_pixels = mineral_map_2d.size
        filtered_phases = []
        for p in phases:
            frac = np.sum(mineral_map_2d == p) / total_pixels
            if frac >= min_frac:
                filtered_phases.append(p)
        phases = filtered_phases
        
        # Fallback in case the threshold was set too high and wiped everything out
        if not phases:
            print(f"[warn] All phases were below the min_frac={min_frac} threshold.")
            phases = ["Unknown"]

    # Mapping phase names to integer IDs
    phase_to_id = {p: i + 1 for i, p in enumerate(phases)}
    id_to_phase = {i + 1: p for i, p in enumerate(phases)} 
    id_to_phase[0] = None

    # Any phase filtered out above won't be in `phase_to_id`, remaining 0 (NaN)
    ids = np.zeros(mineral_map_2d.shape, dtype=int)
    for p, pid in phase_to_id.items():
        ids[mineral_map_2d == p] = pid

    if remove_islands_flag:
        ids = remove_islands(ids, min_size=cleanup_min_size, connectivity=1, fill_val=0)

    palette = _make_palette(phases, cmap_name=cmap_name)
    if phase_colors:
        for phase_name, color in phase_colors.items():
            if phase_name in palette:
                palette[phase_name] = color
                
    cmap = ListedColormap([bg_color] + [palette[p] for p in phases])

    # ----------------------------------------------------
    # Figure and Axis Setup
    # ----------------------------------------------------
    if ax is not None:
        ax_map = ax
        fig = ax.get_figure()
        ax_legend = None
        ax_map.clear()
        ax_map.set_axis_off()   
    else:
        fig_w, fig_h = _auto_figsize_from_array(
            ids.shape, n_legend=len(phases), legend_side=legend_side, legend_cols=legend_cols
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

    # ----------------------------------------------------
    # Rendering Data
    # ----------------------------------------------------
    ax_map.imshow(ids, cmap=cmap, interpolation="none", origin="upper")
    ax_map.set_title(title, pad=8)
    ax_map.axis("off")

    _add_scalebar(
        ax_map,
        scalebar_um=scalebar_um,
        pixel_size_um=pixel_size_um,
        scalebar_loc=scalebar_loc,
        scalebar_col=scalebar_col,
    )
    handles = [mpatches.Patch(facecolor=palette[p], label=p) for p in phases]
    
    if legend_on: 
        if ax_legend is not None:
            ax_legend.axis("off")
            ax_legend.legend(
                handles=handles, loc="upper left", frameon=False, title="Phases",
                ncol=ncols, borderaxespad=0.0, handlelength=1.2, handletextpad=0.6,
                columnspacing=1.0, fontsize=8
            )
        else:
            # Attach legend to the map axis if no separate legend axis was built (e.g., custom `ax`)
            ax_map.legend(
                handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), 
                frameon=False, title="Phases", fontsize=8
            )

    cleaned_mineral_map = np.vectorize(lambda x: id_to_phase.get(x, None))(ids)
    return fig, ax_map, cleaned_mineral_map


def plot_phase_counts(
    mineral_map_2d, title="Mineral Phases (count)", phases=None, normalize=True, 
    min_frac=0.0001, ax=None
):
    """
    Bar chart of pixel counts (or fractions) per phase with auto figure width.
 
    Parameters:
        mineral_map_2d (array-like): (H,W) or (N,) labels.
        title (str): Axes title text.
        phases (list[str]|None): Subset of phases to plot (None→auto).
        min_frac (float): Minimum pixel fraction required to include a phase.
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

    # Drop phases below min_frac
    total = counts.sum()
    if total > 0 and min_frac is not None:
        counts = counts[counts / total >= min_frac]

    if normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total
        ylabel = "Fraction of Pixels"
    else:
        ylabel = "Pixels"

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(_auto_bar_width(len(counts)), 4.5), constrained_layout=True
        )
    else:
        fig = ax.get_figure()

    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Phase")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45, pad=1)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    return fig, ax


def plot_phase_proportions(
    mineral_map,
    title="Phase Proportions",
    phases=None,
    min_frac=0.0001,
    phase_colors=None,
    cmap_name="tab20",
    annotate=True,
    annotate_kw=None,
    ax=None,
):
    """
    Stacked horizontal bar of phase area proportions.
    Proportions are normalized to classified pixels only. The fraction of
    unclassified pixels (NaN, epoxy, low-confidence) is printed below the
    x-axis as a note.
 
    Parameters:
        mineral_map (array-like): (H,W) or (N,) phase labels.
        title (str): Axes title / y-label for the bar.
        phases (list[str]|None): Subset of phases in display order (None→auto).
        min_frac (float): Minimum pixel fraction required to include a phase.
        phase_colors (dict|None): {PhaseName: color}. Falls back to cmap_name.
        cmap_name (str): Matplotlib colormap used when phase_colors is incomplete.
        annotate (bool): If True, label each segment with its percentage.
        annotate_kw (dict|None): Extra keyword arguments forwarded to
            ``_annotate_stacked_bar``.
        ax (matplotlib.axes.Axes|None): Axes to plot on (None→create new).
 
    Returns:
        fig_ax (tuple): (fig, ax) with the stacked bar chart.
    """
    arr = np.asarray(mineral_map, dtype=object)

    labels = _clean_labels_1d(arr)
    if labels.empty:
        fig, ax = plt.subplots(figsize=(14, 2))
        ax.text(0.5, 0.5, "No valid labels", ha="center", va="center")
        ax.axis("off")
        return fig, ax

    counts = labels.value_counts()
    total_classified = counts.sum()

    # Proportions relative to classified pixels only
    props = counts / total_classified

    # Filter by min_frac
    if min_frac is not None:
        props = props[props >= min_frac]

    # Restrict / reorder to requested phases
    if phases is not None:
        seen = set()
        ordered = [p for p in phases if p not in seen and not seen.add(p)]
        props = props.reindex(ordered).dropna()
    else:
        props = props.sort_values(ascending=False)

    phase_list = list(props.index)
    prop_vals = list(props.values)

    remainder = 1.0 - sum(prop_vals)
    remainder_label = "Other"
    if remainder > 1e-6:
        phase_list.append(remainder_label)
        prop_vals.append(remainder)

    # Build color mapping
    palette = _make_palette(phase_list, cmap_name=cmap_name)
    if phase_colors:
        for p, c in phase_colors.items():
            if p in palette:
                palette[p] = c
    # Always force remainder to grey
    if remainder_label in palette:
        palette[remainder_label] = "#cccccc"

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 2), constrained_layout=True)
    else:
        fig = ax.get_figure()

    left = 0.0
    for p, prop in zip(phase_list, prop_vals):
        c = palette.get(p, "#999999")
        ax.barh(
            y=title, width=prop * 100, left=left, color=c,
            edgecolor="white", height=0.5,
        )
        left += prop * 100

    ax.set_xlim(0, 100)
    ax.set_xlabel("Area %")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Legend
    handles = [
        mpatches.Patch(facecolor=palette[p], label=f"{p} ({v * 100:.1f}%)")
        for p, v in zip(phase_list, prop_vals)
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, frameon=False, title="Phases",
    )

    # Annotations
    if annotate:
        kw = dict(phase_colors=palette)
        if annotate_kw:
            kw.update(annotate_kw)
        _annotate_stacked_bar(ax, y=0, phases=phase_list, props=prop_vals, **kw)

    return fig, ax


def plot_pred_score_histograms(
    pred_score_map,
    mineral_map,
    pred_score_threshold,
    phases=None,
    bins=50,
    min_frac=0.0001,
    share_y=True,
    title="Prediction Scores",
    empirical_phases=("Zircon", "SiO2_Polymorph", "Carbonate"),
):
    """
    Horizontal histograms of per-phase prediction scores (auto grid).

    Parameters:
        pred_score_map (array-like): (H,W) max class prediction scores per pixel.
        mineral_map (array-like): (H,W) predicted labels (NaN allowed).
        pred_score_threshold (float): Lower bound for the y-axis.
        phases (list[str] | None): Subset of phases to plot (None -> auto).
        bins (int): Histogram bins.
        min_frac (float): Minimum pixel fraction required to plot a phase.
        share_y (bool): Share prediction score axis across panels.
        title (str): Figure suptitle text.
        empirical_phases (iterable[str]): Phases assigned empirically and therefore
            lacking prediction scores.

    Returns:
        tuple: (fig, axes)
    """
    mineral_map = np.asarray(mineral_map, dtype=object)
    pred_score_map = np.asarray(pred_score_map, dtype=float)
    phases = phases or pick_common_phases(mineral_map)
    empirical_phases = set(empirical_phases)

    valid_phase_pixels = pd.notna(mineral_map)
    total = float(valid_phase_pixels.sum() + 1e-12)

    filtered_phases = []
    phase_data = {}
    phase_is_empirical = {}

    for p in phases:
        phase_mask = mineral_map == p
        phase_frac = phase_mask.sum() / total

        if phase_frac < min_frac:
            continue

        filtered_phases.append(p)

        if p in empirical_phases:
            phase_data[p] = None
            phase_is_empirical[p] = True
        else:
            vals = pred_score_map[phase_mask]
            vals = vals[np.isfinite(vals)]
            phase_data[p] = vals
            phase_is_empirical[p] = False

    phases = filtered_phases

    if not phases:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No phases meet min_frac", ha="center", va="center")
        ax.axis("off")
        return fig, np.array([ax])

    per_row = min(5, len(phases))
    rows = int(np.ceil(len(phases) / per_row))
    fig, axes = plt.subplots(
        rows,
        per_row,
        figsize=(2.8 * per_row, 2.2 * rows),
        sharey=share_y,
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    ylim = (pred_score_threshold, 1.0)

    for i, phase in enumerate(phases):
        ax = axes_flat[i]
        phase_mask = mineral_map == phase
        phase_pct = 100.0 * phase_mask.sum() / total

        if phase_is_empirical[phase]:
            ax.text(
                0.5,
                0.5,
                "No prediction score\n(empirical)",
                ha="center",
                va="center",
                fontsize=11,
                transform=ax.transAxes,
            )
            ax.set_title(f"{phase}\n{phase_pct:.2f} %", fontsize=12)
            ax.set_xlabel("Pixels", fontsize=12)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelleft=False, length=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
        else:
            vals = phase_data[phase]
            ax.hist(vals, bins=bins, orientation="horizontal")
            ax.set_ylim(ylim)
            ax.set_title(f"{phase}\n{phase_pct:.2f} %", fontsize=12)
            ax.set_xlabel("Pixels", fontsize=12)

            if i % per_row == 0:
                ax.set_ylabel("Prediction Score", fontsize=12)

            ax.tick_params(axis="both", labelsize=12)

    for j in range(len(phases), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.suptitle(title, y=1.04, fontsize=14)
    return fig, axes_flat


def run_map(
    sample_input,
    renormalize=False,
    total_threshold=None,
    n_iterations=50,
    min_frac=0.00001,
    pred_score_threshold=0.6,
    units="element_wt%",
    top_k=None,
    phases=None,
    exclude_phases=None,
    phase_colors=None,
    bar_style="vertical",
    components_spec=None,
    remove_islands_flag=False,
    fill_holes_flag=False,
    hole_size=10,
    scalebar_um=None,
    pixel_size_um=None,
    scalebar_loc="lower left",
    scalebar_col="black",
    show=True,
):
    """
    Load, convert, predict, and plot for one folder of CSV maps.
    Always computes mineral components and returns a full results dictionary.
 
    Parameters:
        sample_input (str | Path | dict): Directory path or a dict of oxide maps.
        renormalize (bool): If True, scale each pixel so oxides sum to 100 wt%.
            Applied after total masking.
        total_threshold (float | None): Pixels with oxide total below this
            value (wt%) are set to NaN before renormalization and prediction.
            Use this to mask epoxy/background.
        n_iterations (int): MC forward passes for prediction.
        min_frac (float): Minimum pixel fraction required to keep a phase.
        pred_score_threshold (float): Label NaN where max prediction score < 
            threshold.
        units (str): Input format — 'element_wt%' or 'oxide_wt%'.
        top_k (int|None): Cap displayed phases after filtering.
        phases (list[str]|None): Explicit phases to plot (overrides auto-pick).
        exclude_phases (list[str]|None): Phases to remove from auto-pick.
        phase_colors (dict|None): Manual color mapping {PhaseName: HexColor}.
        bar_style (str): "vertical" for the default bar chart
            (``plot_phase_counts``), or "stacked" for a stacked horizontal
            bar (``plot_phase_proportions``).
        components_spec (dict|None): Custom mineral formula logic.
        remove_islands_flag (bool): If True, removes isolated pixel clusters
            smaller than 2 pixels (4-connected) from the phase map. Useful
            for cleaning up salt-and-pepper noise in the epoxy region.
        fill_holes_flag (bool): If True, fills enclosed background holes within
            continuous phase regions up to ``hole_size`` pixels. Useful for
            patching small gaps inside large mineral grains.
        hole_size (int): Maximum hole area (in pixels) to fill when
            ``fill_holes_flag`` is True.
        scalebar_um (float, optional): Length of the scale bar in micrometers.
        pixel_size_um (float): Physical size of a single pixel in micrometers.
        scalebar_loc (str): Location of the scale bar (e.g., 'lower left').
        scalebar_col (str): Color of the scale bar text/line.
        show (bool): If True, calls plt.show().
 
    Returns:
        result (dict): Dictionary with keys 'figs', 'shape', 'oxide_maps',
            'df_pred', 'mineral_map', 'pred_score_map', 'kept_phases',
            'component_maps', 'component_frames'. ``oxide_maps`` includes a
            ``'Total'`` key with the per-pixel oxide sum.
    """

    # Load and prepare data
    if isinstance(sample_input, (str, os.PathLike)):
        sample_dir = sample_input
        ox_maps = load_maps_from_dir(
            sample_dir,
            units=units,
            renormalize=False,
        )
    elif isinstance(sample_input, dict):
        ox_maps = sample_input
        sample_dir = "Provided ox_maps"
    else:
        raise TypeError("sample_input must be a directory path or a dict of oxide maps")

    if not ox_maps:
        raise ValueError(f"No oxide maps found or provided in: {sample_dir}")

    if total_threshold is not None:
        raw_total = np.nansum(np.stack(list(ox_maps.values())), axis=0)
        total_mask = raw_total < total_threshold
        for key in ox_maps:
            ox_maps[key] = np.where(total_mask, np.nan, ox_maps[key])

    # Do not apply renormalization above, but here after total filter.
    if renormalize:
        ox_maps = renormalize_maps(ox_maps)

    df_ox_flat, shape = maps_to_df(ox_maps)
    expected_with_zr = list(OXIDES) + ["ZrO2"]
    df_ordered = _ensure_columns(df_ox_flat, expected=expected_with_zr)

    # Prediction logic
    df_pred = predict_class_prob(
        df_ordered, n_iterations=n_iterations
    )

    # Mask low-prediction score predictions
    low_score_mask = df_pred["Prediction_Score"] < pred_score_threshold
    df_pred.loc[low_score_mask, "Predict_Mineral"] = np.nan
    df_pred.loc[low_score_mask, "Prediction_Score"] = np.nan

    H, W = shape
    mineral_map = df_pred["Predict_Mineral"].to_numpy().reshape(H, W)
    pred_score_map = df_pred["Prediction_Score"].to_numpy(dtype=float).reshape(H, W)

    # Determine which phases to keep for the dashboard
    if phases and exclude_phases:
        warnings.warn(
            "Both 'phases' and 'exclude_phases' were provided. "
            "'phases' will take priority and 'exclude_phases' will be ignored.",
            UserWarning,
        )

    if phases:
        kept = list(phases)
    else:
        kept = pick_common_phases(mineral_map, top_k=top_k)
        if not kept:
            raw = df_pred["Predict_Mineral"].to_numpy().reshape(H, W)
            kept = pick_common_phases(raw, top_k=top_k)

        if exclude_phases:
            if isinstance(exclude_phases, str):
                exclude_phases = [exclude_phases]
            kept = [p for p in kept if p not in exclude_phases]

    # Plotting (Pass phase_colors to all three)
    title_suffix = (
        os.path.basename(sample_dir) if isinstance(sample_dir, str) else "Sample"
    )

    fig_map, _, mineral_map = plot_phase_map(
        mineral_map,
        phases=kept,
        min_frac=min_frac,
        title=f"Phase Map: {title_suffix}",
        remove_islands_flag=remove_islands_flag,
        fill_holes_flag=fill_holes_flag,
        hole_size=hole_size,
        phase_colors=phase_colors,
        scalebar_um=scalebar_um,
        pixel_size_um=pixel_size_um,
        scalebar_col=scalebar_col,
        scalebar_loc=scalebar_loc,
    )

    if bar_style == "stacked":
        fig_counts, _ = plot_phase_proportions(
            mineral_map,
            phases=kept,
            min_frac=min_frac,
            phase_colors=phase_colors,
            title=f"Phase\nProportions:\n{title_suffix}",
        )
    else:
        fig_counts, _ = plot_phase_counts(
            mineral_map,
            phases=kept,
            min_frac=min_frac,
            title=f"Mineral Phases: {title_suffix}",
        )

    fig_hists, _ = plot_pred_score_histograms(
        pred_score_map,
        mineral_map,
        phases=kept,
        pred_score_threshold=pred_score_threshold,
        min_frac=min_frac,
        title=f"Prediction Scores: {title_suffix}",
    )

    default_spec = {
        "Feldspar": {
            "labels": ["Feldspar", "Plagioclase", "KFeldspar"],
            "calculator": FeldsparClassifier,
            "method": "classify",
            "cols": ["An", "Ab", "Or"],
            "kwargs": {"subclass": False},
            "transforms": {},
        },
        "Clinopyroxene": {
            "labels": ["Clinopyroxene"],
            "calculator": PyroxeneClassifier,
            "method": "calculate_components",
            "cols": ["XMg", "En", "Fs", "Wo"],
            "kwargs": {},
            "transforms": {},
        },
        "Orthopyroxene": {
            "labels": ["Orthopyroxene"],
            "calculator": PyroxeneClassifier,
            "method": "calculate_components",
            "cols": ["XMg", "En", "Fs", "Wo"],
            "kwargs": {},
            "transforms": {},
        },
        "Olivine": {
            "labels": ["Olivine"],
            "calculator": OlivineCalculator,
            "method": "calculate_components",
            "cols": ["XFo"],
            "kwargs": {},
            "transforms": {},
        },
        "Amphibole": {
            "labels": ["Amphibole"],
            "calculator": AmphiboleCalculator,
            "method": "calculate_components",
            "cols": ["XMg"],
            "kwargs": {},
            "transforms": {},
        },
    }

    spec = components_spec or default_spec
    component_maps, component_frames = _compute_component_maps(
        df_ordered=df_ordered,
        df_pred=df_pred,
        shape=shape,
        pred_score_threshold=pred_score_threshold,
        components_spec=spec,
        oxide_list=OXIDES,
    )

    oxides_plus_zr = list(OXIDES) + ["ZrO2"]
    total_cols = df_ordered.columns.intersection(oxides_plus_zr)
    ox_maps["Total"] = (
        df_ordered[total_cols].sum(axis=1, skipna=True).to_numpy().reshape(H, W)
    )

    if show:
        plt.show()

    return {
        "figs": (fig_map, fig_counts, fig_hists),
        "shape": shape,
        "oxide_maps": ox_maps,
        "df_pred": df_pred,
        "mineral_map": mineral_map,
        "pred_score_map": pred_score_map,
        "kept_phases": kept,
        "component_maps": component_maps,
        "component_frames": component_frames,
    }


def _compute_component_maps(
    df_ordered, df_pred, shape, pred_score_threshold, components_spec, oxide_list
):
    """
    Computes stoichiometric components and maps them back into 2D arrays
    based on a provided specification dictionary.
 
    Parameters:
        df_ordered (pd.DataFrame): Flat dataframe of oxide weight percentages.
        df_pred (pd.DataFrame): Flat dataframe of predicted mineral classes and prediction scores.
        shape (tuple): The (H, W) shape of the original 2D map.
        pred_score_threshold (float): Minimum confidence threshold to process a pixel.
        components_spec (dict): Configuration specifying which calculators and methods
            to use for each phase group (e.g., Feldspar, Pyroxene).
        oxide_list (list[str]): The subset of dataframe columns containing valid oxides.
 
    Returns:
        maps (dict): A dictionary mapping "Phase.Component" strings to 2D numpy arrays
            (e.g., 'Feldspar.An': array(...)).
        frames (dict): A dictionary mapping phase names to their corresponding
            flattened stoichiometry DataFrames.
    """

    maps, frames = {}, {}
    H, W = shape
    probs = (
        pd.to_numeric(df_pred["Prediction_Score"], errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )
    labels = df_pred["Predict_Mineral"].astype(str).to_numpy()
    valid = probs >= pred_score_threshold
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
    title="Composite",
    save_path=None,
    fill_missing=True,
    max_hole_size=10,
    min_speck_size=5,
    min_frac=0.0,
    feld_key="Feldspar.An",
    cpx_key="Clinopyroxene.XMg",
    opx_key="Orthopyroxene.XMg",
    ol_key="Olivine.XFo",
    amp_key="Amphibole.XMg",
    phases=None,
    component_labels={
        "Plagioclase": ("Feldspar", "Plagioclase", "Anorthite", "Albite"),
        "Clinopyroxene": ("Clinopyroxene", "Augite", "Diopside"),
        "Orthopyroxene": ("Orthopyroxene", "Enstatite", "Hypersthene"),
        "Olivine": ("Olivine", "Forsterite", "Fayalite"),
        "Amphibole": ("Amphibole", "Tremolite", "Actinolite", "Anthrophyllite", "Grunerite"),
    },
    mask_config=None,
    phase_colors=None,
    smooth_sigma=0.0,
    limits_mode="std",
    percentile=(5, 95),
    legend_on=True,
    legend_cols=1,
    ax=None,
    scalebar_um=None,
    pixel_size_um=None,
    scalebar_loc="lower left",
    scalebar_col="black",
    cbar_hgap=0.015,
    cbar_vgap=-0.05,
    cbar_height=0.03,
    dpi=300,
):
    """
    Renders a composite map overlaying continuous solid-solution compositions
    (e.g., Plagioclase An%, Olivine Fo%) on top of a categorical phase mask.
 
    Parameters:
        res (dict): The result dictionary returned by ``run_map()``,
            containing 'mineral_map' and 'component_maps'.
        title (str): Title of the composite plot.
        save_path (str, optional): Filepath to save the figure (e.g., 'plot.png').
        fill_missing (bool): If True, interpolates small holes in the continuous data.
        max_hole_size (int): Maximum hole area (in pixels) to interpolate.
        min_speck_size (int): Minimum pixel area for a phase to be kept.
        min_frac (float): Minimum pixel fraction required for a phase to be
            included in the composite. Fractions are computed from the cleaned
            mineral map after island removal / hole filling.
        feld_key (str): Dictionary key in res['component_maps'] for feldspar data.
        cpx_key (str): Dictionary key for clinopyroxene data.
        opx_key (str): Dictionary key for orthopyroxene data.
        ol_key (str): Dictionary key for olivine data.
        amp_key (str): Dictionary key for amphibole data.
        phases (list[str], optional): Explicit list of phases to plot.
        component_labels (dict): Mapping to group sub-phases into overarching component logic.
        mask_config (dict, optional): Custom layer masking configuration (e.g., mapping Glass).
        phase_colors (dict, optional): Custom categorical colors for leftover phases.
        smooth_sigma (float): Gaussian blur sigma for smoothing compositional data.
        limits_mode (str): 'percentile' or 'std' for auto-scaling color ramps.
        percentile (tuple): (min, max) percentiles for color limits.
        legend_on (bool): If True, display the legend.
        legend_cols (int): Number of columns in the legend.
        ax (matplotlib.axes.Axes, optional): Pre-existing axes to plot onto.
        scalebar_um (float, optional): Length of the scale bar in micrometers.
        pixel_size_um (float): Physical size of a single pixel in micrometers.
        scalebar_loc (str): Location of the scale bar (e.g., 'lower left').
        scalebar_col (str): Color of the scale bar text/line.
        cbar_hgap (float): Horizontal gap between adjacent colorbars (axes fraction).
        cbar_vgap (float): Vertical offset of the colorbar row below the map
            (axes fraction; negative = below the axes).
        cbar_height (float): Height of each colorbar (axes fraction).
        dpi (int): Resolution of the figure.
 
    Returns:
        fig (matplotlib.figure.Figure): The generated composite figure.
        mineral_map (ndarray): The cleaned 2D categorical map used as the base.
        processed_comp_maps (dict): The smoothed/filled 2D continuous component data.
    """

    plt.close("all")

    if mask_config is None:
        mask_config = {
            "Glass": {"labels": ("Glass", "Melt",), "color": "#F9C300"},
            "Oxide": {"labels": ("Oxide",), "color": "#2E2DCE"},
            "Feldspar_Miscibility_Gap": {"labels": ("Feldspar_Miscibility_Gap",), "color": "#003d36"},
        }

    comp_maps = res.get("component_maps", {})
    raw_mineral_map = res.get("mineral_map")
    if raw_mineral_map is None:
        raise ValueError("mineral_map is required in res.")

    mineral_map = np.asarray(raw_mineral_map, dtype=object).copy()
    bad_vals = {"nan", "NaN"}

    # Standardize NaNs
    is_actual_nan = pd.isna(mineral_map)
    mineral_map[is_actual_nan] = "nan"

    # Clean up map by removing small islands
    if min_speck_size > 0:
        mineral_map = remove_islands(
            mineral_map,
            min_size=min_speck_size,
            fill_val="nan",
            phase_min_sizes={"Glass": 20},
            grouped_phases=[
                ("Spinel_Group", "Rhombohedral_Oxides"),
                ("Alkali_Feldspar", "Plagioclase"),
                ("Clinopyroxene", "Orthopyroxene"),
            ],
        )

    if fill_missing:
        mineral_map = fill_phase_holes(mineral_map, max_hole_size=max_hole_size)

    # Identify unique phases for plotting
    unique_str_array = np.unique(mineral_map.astype(str))
    unique_phases = set(unique_str_array) - bad_vals

    if phases is not None:
        if isinstance(phases, str):
            phases = [phases]
        unique_phases = {p for p in unique_phases if p in phases}

    if min_frac and min_frac > 0:
        total_pixels = mineral_map.size
        kept_by_frac = {
            p for p in unique_phases
            if (np.sum(mineral_map == p) / total_pixels) >= min_frac
        }
        unique_phases = kept_by_frac

    consumed_phases, active_components, active_masks, processed_comp_maps = (
        set(),
        [],
        [],
        {},
    )

    comp_defs = [
        {
            "id": "Plagioclase",
            "key": feld_key,
            "ramp": "teal",
            "leg": "Plagioclase\n(An)",
            "col": "#009988",
        },
        {
            "id": "Clinopyroxene",
            "key": cpx_key,
            "ramp": "red",
            "leg": "Clinopyroxene\n(Mg#)",
            "col": "#e7b3b1",
        },
        {
            "id": "Orthopyroxene",
            "key": opx_key,
            "ramp": "maroon",
            "leg": "Orthopyroxene\n(Mg#)",
            "col": "#5A0F0F",
        },
        {
            "id": "Olivine",
            "key": ol_key,
            "ramp": "green",
            "leg": "Olivine\n(Fo)",
            "col": "#666633",
        },
        {
            "id": "Amphibole",
            "key": amp_key,
            "ramp": "brown",
            "leg": "Amphibole\n(Mg#)",
            "col": "#5E2910",
        },
    ]

    for cd in comp_defs:
        labels = component_labels.get(cd["id"], (cd["id"],))
        present_labels = [l for l in labels if l in unique_phases]
        if present_labels:
            raw_data = comp_maps.get(cd["key"], None)
            if raw_data is not None and np.any(np.isfinite(raw_data)):
                data = raw_data.copy()

                # ------------------------------------------
                # Fill islands in continuous data
                # ------------------------------------------
                if fill_missing:
                    invalid_data_mask = ~np.isfinite(data)
                    valid_data_mask = ~invalid_data_mask
                    filled_data_mask = _remove_small_holes_compat(valid_data_mask, max_hole_size)
                    data_islands_mask = filled_data_mask & invalid_data_mask

                    if np.any(data_islands_mask):
                        _, indices = distance_transform_edt(
                            invalid_data_mask, return_indices=True
                        )
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

                processed_comp_maps[cd["key"]] = data
                vmin, vmax = _auto_limits(data, mode=limits_mode, percentile=percentile)
                active_components.append(
                    {**cd, "data": data, "vmin": vmin, "vmax": vmax}
                )
                consumed_phases.update(present_labels)

    # Process Configured Masks
    for m_name, cfg in mask_config.items():
        present_labels = [l for l in cfg["labels"] if l in unique_phases]
        if present_labels:
            mask = np.isin(mineral_map, present_labels)
            active_masks.append(
                {
                    "name": m_name,
                    "mask": mask,
                    "color": cfg["color"],
                    "cmap": ListedColormap(["#FFFFFF00", cfg["color"]]),
                }
            )
            consumed_phases.update(present_labels)

    # Process Leftovers
    leftover = unique_phases - consumed_phases
    if leftover:
        if phase_colors is None:
            palette = plt.get_cmap("tab20", len(leftover))
            phase_colors = {p: palette(i) for i, p in enumerate(leftover)}

        for p in sorted(list(leftover)):
            mask = mineral_map == p
            color = phase_colors.get(p, "#808080")
            active_masks.append(
                {
                    "name": p,
                    "mask": mask,
                    "color": color,
                    "cmap": ListedColormap(["#FFFFFF00", color]),
                }
            )

    # --- Setup Plotting Helpers ---
    def _make_ramp(c1, c2):
        c1_rgb = to_rgb(c1)
        c2_rgb = to_rgb(c2)
        arr = np.ones((256, 4))
        for i in range(3):
            arr[:, i] = np.linspace(c1_rgb[i], c2_rgb[i], 256)
        cmap = ListedColormap(arr)
        cmap.set_bad(alpha=0)
        return cmap

    ramps = {
        "teal": _make_ramp("#CCEEFF", "#009988"),
        "red": _make_ramp("#FFE6E6", "#C83C50"),
        "maroon": _make_ramp("#CB4545", "#5A0000"),
        "green": _make_ramp("#EFEEBB", "#666633"),
        "brown": _make_ramp("#843916", "#473127"),
    }

    # Only include discrete masks/phases in the patch legend
    legend_entries = [(m["name"], m["color"]) for m in active_masks]

    # --- Setup Figure and Axes ---
    if ax is None:
        fig_w, fig_h = _auto_figsize_from_array(
            mineral_map.shape, n_legend=len(legend_entries) + len(active_components)
        )
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")

        if legend_on and legend_entries:
            gs = fig.add_gridspec(1, 2, width_ratios=[fig_w - 1.5, 1.5])
            ax_map, ax_legend = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        else:
            ax_map, ax_legend = fig.add_subplot(111), None
    else:
        ax_map, fig, ax_legend = ax, ax.get_figure(), None

    # Store the image objects so we can attach colorbars to them
    comp_images = []
    for c in active_components:
        im = ax_map.imshow(
            np.ma.masked_invalid(c["data"]),
            cmap=ramps[c["ramp"]],
            vmin=c["vmin"],
            vmax=c["vmax"],
            interpolation="none",
        )
        comp_images.append((im, c["leg"]))

    for m in active_masks:
        display = np.where(m["mask"], 1.0, np.nan)
        ax_map.imshow(display, cmap=m["cmap"], vmin=0, vmax=1, interpolation="none")

    ax_map.set_title(title, pad=8)
    ax_map.axis("off")

    # Add colorbars for each continuous component
    n_bars = len(comp_images)
    if n_bars > 0:
        gap_ = cbar_hgap
        fill = 0.95   # fraction of axis width the whole group should occupy
        total_w = fill
        bar_w = (total_w - (n_bars - 1) * gap_) / n_bars
        x0 = 0.5 - total_w / 2.0

        for i, (im, label) in enumerate(comp_images):
            x = x0 + i * (bar_w + gap_)
            cax = ax_map.inset_axes([x, cbar_vgap, bar_w, cbar_height])

            cbar = fig.colorbar(im, cax=cax, orientation="horizontal", format="%.2f")
            cbar.set_label(label, size=9, labelpad=4)
            cbar.ax.tick_params(labelsize=10, axis="x")

    # Add Legend
    if legend_on and legend_entries:
        handles = [mpatches.Patch(facecolor=c, label=lab) for lab, c in legend_entries]
        if ax_legend is not None:
            ax_legend.axis("off")
            ax_legend.legend(
                handles=handles,
                loc="upper left",
                frameon=False,
                title="Categorical Phases",
                ncol=legend_cols,
                fontsize=8,
            )
        else:
            ax_map.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=False,
                title="Categorical Phases",
                fontsize=8,
            )

    _add_scalebar(
        ax_map,
        scalebar_um=scalebar_um,
        pixel_size_um=pixel_size_um,
        scalebar_loc=scalebar_loc,
        scalebar_col=scalebar_col,
    )
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, mineral_map, processed_comp_maps


def _plot_continuous_map(
    data,
    title,
    cmap,
    vmin,
    vmax,
    cbar_label,
    bg_value=np.nan,
    min_speck_size=0,
    scalebar_um=None,
    pixel_size_um=1.0,
    scalebar_col="black",
    scalebar_loc="lower left",
    ax=None,
    dpi=300,
):
    """
    Core renderer for any 2-D continuous-value map.

    Parameters:
        data (array-like): 2-D array of values to plot.
        title (str): Plot title.
        cmap (str or Colormap): Colormap name or object.
        vmin (float): Lower colour-scale limit.
        vmax (float): Upper colour-scale limit.
        cbar_label (str): Label for the colourbar.
        bg_value (float): Value used for background pixels.
            These are masked so the background stays transparent.
        min_speck_size (int): Minimum pixel area for a phase to be kept.
        scalebar_um (float, optional): Length of the scale bar in micrometres.
        pixel_size_um (float): Physical size of a single pixel in micrometres.
        scalebar_col (str): Colour of the scale bar text/line.
        scalebar_loc (str): Location of the scale bar (e.g., 'lower left').
        ax (matplotlib.axes.Axes, optional): Pre-existing axes to draw on.
        dpi (int): Figure resolution when creating a new figure.

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
        ax_map (matplotlib.axes.Axes): The axes containing the map.
    """
    data = np.asarray(data, dtype=float)

    if min_speck_size > 0:
        valid = np.isfinite(data)
        if np.isfinite(bg_value):
            valid &= (data != bg_value)
        cleaned = _remove_small_objects_compat(valid, min_speck_size)
        data = np.where(cleaned, data, np.nan)

    mask = np.isnan(data)
    if np.isfinite(bg_value):
        mask |= (data == bg_value)
    masked = np.ma.masked_where(mask, data)

    if ax is not None:
        ax_map = ax
        fig = ax.get_figure()
        ax_map.clear()
    else:
        h, w = data.shape
        aspect = w / h
        fig_h = 6
        fig_w = fig_h * aspect + 0.8
        fig, ax_map = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    im = ax_map.imshow(
        masked, cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="none", origin="upper",
    )
    ax_map.set_title(title, pad=8)
    ax_map.axis("off")

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    _add_scalebar(
        ax_map,
        scalebar_um=scalebar_um,
        pixel_size_um=pixel_size_um,
        scalebar_loc=scalebar_loc,
        scalebar_col=scalebar_col,
    )

    fig.tight_layout()
    return fig, ax_map


def plot_score_map(
    res,
    phases=None,
    title="Prediction Score Map",
    cmap="magma",
    vmin=0,
    vmax=1,
    cbar_label="Prediction Score",
    **kwargs,
):
    """
    Plots a continuous prediction-score map with a colourbar.

    Parameters:
        res (dict): The result dictionary returned by ``run_map()``,
            containing 'mineral_map' and 'component_maps'.
        phases (list[str] | None): If provided, only show prediction scores for
            these phases; all other pixels are masked to background.
        title (str): Plot title.
        cmap (str or Colormap): Colourmap name (default 'magma').
        vmin (float): Lower colour-scale limit.
        vmax (float): Upper colour-scale limit.
        cbar_label (str): Label for the colourbar.
        **kwargs: Forwarded to ``_plot_continuous_map`` (bg_value, scalebar_um,
            pixel_size_um, scalebar_col, scalebar_loc, ax, dpi).

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
        ax_map (matplotlib.axes.Axes): The axes containing the score map.
    """
    if not isinstance(res, dict):
        raise TypeError("`res` must be a result dictionary.")

    if "pred_score_map" not in res:
        raise KeyError("'pred_score_map' not found in result dictionary.")
    if "mineral_map" not in res:
        raise KeyError("'mineral_map' not found in result dictionary.")

    pred_score_map = np.asarray(res["pred_score_map"], dtype=float)

    if phases is not None:
        mineral_map = np.asarray(res["mineral_map"], dtype=object)
        phase_mask = np.isin(mineral_map, phases)

        pred_score_map = pred_score_map.copy()
        pred_score_map[~phase_mask] = np.nan

    return _plot_continuous_map(
        pred_score_map,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label,
        **kwargs,
    )


def plot_oxide_map(
    res,
    oxide_name,
    title=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    cbar_label=None,
    **kwargs,
):
    """
    Plots a 2-D oxide-concentration map with a colourbar.

    Parameters:
        res (dict): The result dictionary returned by ``run_map()``,
            containing ``'oxide_maps'``.
        oxide_name (str): Oxide name to plot from ``res['oxide_maps']``
            (e.g., ``'SiO2'``, ``'FeOt'``).
        title (str, optional): Plot title. Defaults to ``'{oxide_name} Map'``.
        cmap (str or Colormap, optional): Colormap name. Defaults to
            ``'magma'``.
        vmin (float, optional): Lower colour-scale limit. Defaults to the
            data minimum, ignoring background values.
        vmax (float, optional): Upper colour-scale limit. Defaults to the
            data maximum, ignoring background values.
        cbar_label (str, optional): Colourbar label. Defaults to
            ``'{oxide_name} (wt.%)'``.
        **kwargs: Forwarded to ``_plot_continuous_map`` (e.g., ``bg_value``,
            ``scalebar_um``, ``pixel_size_um``, ``scalebar_col``,
            ``scalebar_loc``, ``ax``, ``dpi``).

    Returns:
        fig (matplotlib.figure.Figure): The generated figure.
        ax_map (matplotlib.axes.Axes): The axes containing the oxide map.
    """
    if not isinstance(res, dict):
        raise TypeError("res must be the result dictionary returned by run_map().")

    if "oxide_maps" not in res:
        raise KeyError("'oxide_maps' not found in result dictionary.")

    oxide_maps = res["oxide_maps"]

    if oxide_name not in oxide_maps:
        available = ", ".join(sorted(oxide_maps.keys()))
        raise KeyError(
            f"{oxide_name!r} not found in res['oxide_maps']. "
            f"Available oxides: {available}"
        )

    oxide_map = np.asarray(oxide_maps[oxide_name], dtype=float)

    if title is None:
        title = f"{oxide_name} Map"
    if cbar_label is None:
        cbar_label = f"{oxide_name} (wt.%)"

    bg_value = kwargs.get("bg_value", np.nan)
    valid = oxide_map[np.isfinite(oxide_map)]
    if np.isfinite(bg_value):
        valid = valid[valid != bg_value]

    if vmin is None:
        vmin = float(valid.min()) if valid.size else 0
    if vmax is None:
        vmax = float(valid.max()) if valid.size else 1

    return _plot_continuous_map(
        oxide_map,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label,
        **kwargs,
    )


# %% EBSD mapping


def parse_ctf_header(filepath):
    """
    Parses the header of a .ctf file to extract grid dimensions and phase mappings.
    The file is expected to contain 'XCells', 'YCells', 'Phases', and a data table
    starting with 'Phase\\tX\\tY'.
 
    Parameters:
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
            for j in range(1, n_phases + 1):
                parts = lines[i + j].strip().split("\t")
                if len(parts) >= 3 and parts[2].strip():
                    phase_mapping[j] = parts[2].strip()
                else:
                    phase_mapping[j] = f"Phase{j}"

        elif line.startswith("Phase\tX\tY"):
            data_start = i
            break

    if None in (x_cells, y_cells, n_phases, data_start):
        raise ValueError("Missing required header information in CTF file.")

    return x_cells, y_cells, data_start, phase_mapping


def plot_ctf_phases(
    filepath: str,
    max_legend=25,
    rename_dict=None,
    phase_colors=None,
    ax=None,
    title="default",
    scalebar_um=None,
    scalebar_loc="lower left",
    scalebar_col='black',
    legend_on=True,
):
    """
    Loads phase data from a .ctf file and generates a 2D categorical phase map.
    It maps raw phase IDs to their corresponding names, optionally renames them,
    and orders the legend by phase abundance.
 
    Parameters:
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
        scalebar_um (float, optional): Length of the scale bar in micrometers.
        scalebar_loc (str): Location of the scale bar (e.g., 'lower left').
        scalebar_col (str): Color of the scale bar text/line.
        legend_on (bool): If True, displays the legend. Defaults to True.

    Returns:
        fig (matplotlib.figure.Figure): The figure object.
        phase_map (ndarray): A 2D array of the mapped phase names as strings.
        raw_ids (ndarray): A 2D array of the raw numeric phase IDs from the file.
        phase_mapping (dict): A dictionary mapping raw IDs to phase names.
        unique_names (ndarray): An array of the unique phase names sorted by abundance.
    """
    x_cells, y_cells, data_start, phase_mapping = parse_ctf_header(filepath)
    step_size = None

    # Extract step size from the file to calculate scale
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("XStep"):
                step_size = float(line.split()[1])
                break
            elif line.startswith(
                "Phases"
            ):  # Stop searching once we hit the phases list
                break

    # --- Apply clean names to the phase mapping ---
    if rename_dict:
        for phase_id, original_name in phase_mapping.items():
            for messy_name, clean_name in rename_dict.items():
                if messy_name in original_name:
                    phase_mapping[phase_id] = clean_name

    # Read only the Phase column to save memory
    df = pd.read_csv(filepath, sep="\t", skiprows=data_start, usecols=["Phase"])
    # raw_ids = df["Phase"].values
    raw_ids = df["Phase"].to_numpy()


    if len(raw_ids) != (x_cells * y_cells):
        raise ValueError("Mismatch between header grid size and data points.")

    # Map integer IDs to phase names
    phase_names = (
        df["Phase"]
        .map(phase_mapping)
        .fillna(df["Phase"].astype(str) + "_Unknown")
        .values
    )

    # Get unique names and their frequencies
    unique_names, counts = np.unique(phase_names, return_counts=True)

    # Sort arrays descending by abundance
    order = np.argsort(counts)[::-1]
    unique_names = unique_names[order]
    counts = counts[order]

    # Create the 2D string-based map to return
    # phase_map = phase_names.reshape((y_cells, x_cells))
    phase_map = np.asarray(phase_names, dtype=object).reshape((y_cells, x_cells))


    # --- Handle Axes ---
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True

    ax.axis("off")

    # --- Plot EACH phase as its own distinct layer ---
    base_colors = list(plt.get_cmap("tab20").colors)
    cmap_dict = {} # We'll store this to build the legend later

    for i, name in enumerate(unique_names):
        # Create a mask: 1 where the phase exists, NaN everywhere else
        phase_mask = np.where(phase_map == name, 1.0, np.nan)
        
        # Determine the color
        if phase_colors and name in phase_colors:
            color = phase_colors[name]
        else:
            color = base_colors[i % len(base_colors)]
            
        # Create a single-color colormap for this specific phase
        single_cmap = ListedColormap([color])
        cmap_dict[name] = color # Save for the legend
        
        # Plot just this phase. 
        # Illustrator will now read this as an individual object!
        ax.imshow(phase_mask, cmap=single_cmap, vmin=0, vmax=1, interpolation="none")

    # Title
    if title == "default":
        ax.set_title(f"Phase Map — {x_cells} x {y_cells}")
    elif title is not None:
        ax.set_title(title)

    # Add Scale Bar
    if scalebar_um is not None and step_size is None:
        warnings.warn("XStep not found in CTF header; cannot draw scale bar.")
    else:
        _add_scalebar(
            ax,
            scalebar_um=scalebar_um,
            pixel_size_um=step_size,
            scalebar_loc=scalebar_loc,
            scalebar_col=scalebar_col,
            warn=False,
        )

    # Build the legend handles and labels
    n_show = min(max_legend, len(unique_names))
    handles = [
        plt.Line2D(
            [0], [0],
            marker="s",
            linestyle="",
            markersize=10,
            color=cmap_dict[name],
        )
        for name in unique_names[:n_show]
    ]
    labels = [
        f"{name} ({count})"
        for name, count in zip(unique_names[:n_show], counts[:n_show])
    ]

    if len(unique_names) > n_show:
        handles.append(plt.Line2D([0], [0], linestyle=""))
        labels.append(f"... +{len(unique_names) - n_show} more")

    if legend_on:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        show_plot = True
    else:
        fig = ax.get_figure()

    return fig, phase_map, raw_ids.reshape((y_cells, x_cells)), phase_mapping, unique_names


# %%


def interactive_pixels(
    result,
    cmap_name="tab20",
    phase_colors=None,
    phase=None,
    oxide_key=None,
    oxide_cmap="viridis",
    vmin=None,
    vmax=None,
):
    """
    Display the phase map and collect oxide compositions by clicking pixels.

    Uses the same colormap and legend style as ``plot_phase_map``. Each click
    places a marker on the map, prints the oxide values, and appends a row to
    ``controller["picks"]``. Requires an interactive Matplotlib backend
    (``%matplotlib widget`` in a notebook).

    Keybindings:
        r/u: undo the last picked pixel
        c: clear all picks
        q: quit (disconnect click and key handlers)

    Parameters:
        result (dict): Result dictionary returned by ``run_map()``.
        cmap_name (str): Matplotlib colormap for the phase map display.
        phase_colors (dict|None): Optional manual color overrides {PhaseName: color}.
        phase (str|list[str]|None): If provided, only pixels matching this phase
            are shown and clickable. Others are rendered as background.
        oxide_key (str|None): If provided, display this oxide or component as a
            heatmap instead of the phase map (e.g. ``"SiO2"``). The phase map
            legend is replaced by a colorbar.
        oxide_cmap (str): Colormap for the oxide heatmap when ``map_key`` is set.
        vmin (float|None): Lower display limit for the oxide heatmap.
        vmax (float|None): Upper display limit for the oxide heatmap.

    Returns:
        controller (dict): Dict with keys ``'fig'`` and ``'picks'``.
            ``picks`` is a ``pd.DataFrame`` that grows with each click,
            with columns ``x``, ``y``, ``phase``, and one column per oxide.
    """
    plt.close("all")
    backend = plt.get_backend().lower()
    _non_interactive = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template",
                        "module://matplotlib_inline.backend_inline"}
    if backend in _non_interactive or backend.endswith("inline"):
        warnings.warn(
            "interactive_pixels() needs an interactive Matplotlib backend. "
            f"The current backend is {plt.get_backend()!r}. In a notebook, "
            "try `%matplotlib widget` (requires ipympl).",
            UserWarning,
        )

    mineral_map = result["mineral_map"]
    oxide_maps = result["oxide_maps"]
    H, W = mineral_map.shape

    # Derive phases from what actually exists in the cleaned mineral_map,
    # preserving the order from kept_phases where possible.
    present = {v for v in mineral_map.ravel() if v is not None and v != "nan"}
    kept_phases = [p for p in result["kept_phases"] if p in present]

    # Optionally restrict to a single phase or list of phases.
    if phase is not None:
        phase_filter = {phase} if isinstance(phase, str) else set(phase)
        kept_phases = [p for p in kept_phases if p in phase_filter]

    phase_to_id = {p: i + 1 for i, p in enumerate(kept_phases)}
    ids = np.zeros((H, W), dtype=int)
    for p, pid in phase_to_id.items():
        ids[mineral_map == p] = pid

    fig, (ax_map, ax_leg) = plt.subplots(
        1, 2, figsize=(10, 7),
        gridspec_kw={"width_ratios": [5, 1]},
    )

    if oxide_key is not None:
        data = get_profile_map(result, oxide_key, source="auto")
        if phase is not None:
            phase_mask = np.isin(mineral_map, list(kept_phases))
            data = np.where(phase_mask, data, np.nan)
        valid = data[np.isfinite(data)]
        _vmin = float(valid.min()) if vmin is None and valid.size else (vmin or 0.0)
        _vmax = float(valid.max()) if vmax is None and valid.size else (vmax or 1.0)
        im = ax_map.imshow(np.ma.masked_invalid(data), cmap=oxide_cmap,
                           vmin=_vmin, vmax=_vmax, interpolation="none", origin="upper")
        ax_leg.axis("off")
        fig.colorbar(im, ax=ax_leg, fraction=0.8, pad=0.05, label=oxide_key)
    else:
        palette = _make_palette(kept_phases, cmap_name=cmap_name)
        if phase_colors:
            for p, c in phase_colors.items():
                if p in palette:
                    palette[p] = c
        bg_color = (1.0, 1.0, 1.0, 1.0)
        listed_cmap = ListedColormap([bg_color] + [palette[p] for p in kept_phases])
        ax_map.imshow(ids, cmap=listed_cmap, vmin=0, vmax=len(kept_phases),
                      interpolation="none", origin="upper")
        handles = [mpatches.Patch(facecolor=palette[p], label=p) for p in kept_phases]
        ax_leg.axis("off")
        ax_leg.legend(handles=handles, loc="upper left", frameon=False, title="Phases",
                      borderaxespad=0.0, handlelength=1.2, handletextpad=0.6, fontsize=8)

    fig.suptitle(
        "Click pixels to sample oxide composition\n"
        "'r'/'u': undo last  |  'c': clear all  |  'q'/Esc: done",
        fontsize=11,
    )
    ax_map.axis("off")

    oxides = [k for k in oxide_maps if k != "Total"]
    state = {"rows": [], "markers": []}
    controller = {"fig": fig, "picks": pd.DataFrame()}

    try:
        from IPython.display import display, clear_output, HTML
        import ipywidgets as widgets
        _out = widgets.Output()
        _btn = widgets.Button(description="Copy to clipboard", icon="clipboard",
                              layout=widgets.Layout(width="160px"))

        def _on_copy(_):
            if controller["picks"].empty:
                return
            tsv = controller["picks"].to_csv(sep="\t", index=False)
            escaped = tsv.replace("`", "\\`")
            script = (
                "<script>"
                f"navigator.clipboard.writeText(`{escaped}`);"
                "</script>"
            )
            with _out:
                clear_output(wait=True)
                display(HTML(script))
                display(controller["picks"])

        _btn.on_click(_on_copy)
        _use_widget = True
    except ImportError:
        _use_widget = False

    def _update_picks():
        controller["picks"] = (
            pd.DataFrame(state["rows"]).reset_index(drop=True)
            if state["rows"] else pd.DataFrame()
        )
        if _use_widget:
            with _out:
                clear_output(wait=True)
                if not controller["picks"].empty:
                    display(controller["picks"])

    def _on_click(event):
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if not (0 <= x < W and 0 <= y < H):
            return

        clicked_phase = mineral_map[y, x]
        if phase is not None:
            phase_filter = {phase} if isinstance(phase, str) else set(phase)
            if clicked_phase not in phase_filter:
                return
        row = {"x": x, "y": y, "phase": clicked_phase}
        for ox in oxides:
            row[ox] = oxide_maps[ox][y, x]
        row["Total"] = oxide_maps["Total"][y, x]
        state["rows"].append(row)

        pick_num = len(state["rows"])
        marker = ax_map.scatter([x], [y], c="white", s=30, edgecolors="black",
                                linewidths=0.6, zorder=5)
        label_artist = ax_map.text(x + 3, y - 3, str(pick_num), fontsize=7,
                                   color="white", zorder=6)
        state["markers"].append((marker, label_artist))

        _update_picks()
        fig.canvas.draw_idle()

        print(f"\n#{pick_num}  Pixel ({x}, {y})  —  Phase: {phase}")
        print(f"{'Oxide':<10} {'wt%':>8}")
        print("-" * 20)
        for ox in oxides:
            print(f"{ox:<10} {row[ox]:>8.2f}")
        print(f"{'Total':<10} {row['Total']:>8.2f}")

    def _on_key(event):
        if event.key in ("q", "escape"):
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)
            fig.suptitle("Inactive — picks saved in controller['picks']", fontsize=10)
            fig.canvas.draw_idle()
        elif event.key in ("u", "r") and state["rows"]:
            state["rows"].pop()
            marker, label_artist = state["markers"].pop()
            marker.remove()
            label_artist.remove()
            _update_picks()
            fig.canvas.draw_idle()
            print(f"Undid last pick. {len(state['rows'])} picks remaining.")
        elif event.key == "c":
            state["rows"].clear()
            for marker, label_artist in state["markers"]:
                marker.remove()
                label_artist.remove()
            state["markers"].clear()
            _update_picks()
            fig.canvas.draw_idle()
            print("Cleared all picks.")

    cid_click = fig.canvas.mpl_connect("button_press_event", _on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", _on_key)

    def _on_close(_event):
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)

    fig.canvas.mpl_connect("close_event", _on_close)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

    if _use_widget:
        display(widgets.HBox([_btn]))
        display(_out)

    return controller


# %%


def get_profile_map(res, key, source="auto"):
    """
    Resolve a 2-D map for line-profile extraction from a ``run_map()`` result
    or a plain oxide-map dict returned by ``load_maps_from_dir()``.

    Parameters:
        res (dict): Result dictionary returned by ``run_map()``, or a plain
            dict of ``{oxide: 2D array}`` as returned by
            ``load_maps_from_dir()``.
        key (str): Map key to extract, e.g. ``"SiO2"`` or ``"Feldspar.An"``.
        source (str): One of ``"auto"``, ``"oxide"``, or ``"component"``.

    Returns:
        data (ndarray): 2-D float array for the requested map.
    """
    if not isinstance(res, dict):
        raise TypeError("res must be a dict returned by run_map() or load_maps_from_dir().")

    # Plain flat dict from load_maps_from_dir — treat as oxide_maps directly.
    if "oxide_maps" not in res and "component_maps" not in res:
        oxide_maps = res
        component_maps = {}
    else:
        oxide_maps = res.get("oxide_maps", {}) or {}
        component_maps = res.get("component_maps", {}) or {}

    if source not in {"auto", "oxide", "component"}:
        raise ValueError("source must be one of {'auto', 'oxide', 'component'}")

    if source == "oxide":
        if key not in oxide_maps:
            available = ", ".join(sorted(oxide_maps))
            raise KeyError(f"{key!r} not found in oxide_maps. Available: {available}")
        return np.asarray(oxide_maps[key], dtype=float)

    if source == "component":
        if key not in component_maps:
            available = ", ".join(sorted(component_maps))
            raise KeyError(
                f"{key!r} not found in component_maps. Available: {available}"
            )
        return np.asarray(component_maps[key], dtype=float)

    if key in oxide_maps:
        return np.asarray(oxide_maps[key], dtype=float)
    if key in component_maps:
        return np.asarray(component_maps[key], dtype=float)

    available = sorted(set(oxide_maps) | set(component_maps))
    raise KeyError(f"{key!r} not found. Available profile maps: {', '.join(available)}")


def _line_strip_geometry(start, end, width_px):
    """
    Compute line-direction vectors and strip-outline coordinates.

    Parameters:
        start (array-like): (x, y) start coordinate in pixel space.
        end (array-like): (x, y) end coordinate in pixel space.
        width_px (float): Total strip width in pixels.

    Returns:
        geom (dict): Geometry fields for projection and plotting.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("start and end must each be length-2 coordinates.")

    vec = end - start
    length_px = float(np.hypot(vec[0], vec[1]))
    if length_px == 0:
        raise ValueError("start and end cannot be identical.")

    direction = vec / length_px
    normal = np.array([-direction[1], direction[0]], dtype=float)
    half_width = max(float(width_px), 0.0) / 2.0
    offset = normal * half_width

    outline = np.vstack(
        [
            start + offset,
            end + offset,
            end - offset,
            start - offset,
            start + offset,
        ]
    )

    return {
        "start": start,
        "end": end,
        "vec": vec,
        "length_px": length_px,
        "direction": direction,
        "normal": normal,
        "half_width": half_width,
        "outline": outline,
    }


def extract_line_profile(
    data,
    start,
    end,
    width_px=1.0,
    n_bins=None,
    pixel_size_um=None,
    method="none",
    smooth_window=1,
):
    """
    Extract a line profile from a 2-D map with a finite-width strip.
    Pixels with centers inside the strip are projected onto the transect axis.
    With ``method="none"`` every pixel is returned as its own row (no binning).
    Otherwise pixels are aggregated into distance bins.

    Parameters:
        data (array-like): 2-D map to sample.
        start (array-like): (x, y) start coordinate in pixel space.
        end (array-like): (x, y) end coordinate in pixel space.
        width_px (float): Total projection width in pixels.
        n_bins (int|None): Number of bins along the profile. Ignored when
            ``method="none"``. Defaults to roughly one bin per pixel of line
            length.
        pixel_size_um (float|None): Micrometres per pixel, used to populate
            physical-distance columns.
        method (str): Aggregation method, one of ``"mean"``, ``"median"``, or
            ``"none"``. Use ``"none"`` to skip binning and return each pixel
            as an individual data point.
        smooth_window (int): Optional rolling window, in bins (or pixels when
            ``method="none"``).

    Returns:
        profile_df (pd.DataFrame): Aggregated profile table, or one row per
            pixel when ``method="none"``.

        samples (pd.DataFrame): Raw projected strip pixels inside the strip.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array.")

    if method not in {"mean", "median", "none"}:
        raise ValueError("method must be one of {'mean', 'median', 'none'}")

    geom = _line_strip_geometry(start, end, width_px)

    yy, xx = np.indices(data.shape, dtype=float)
    values = data.ravel()
    points = np.column_stack([xx.ravel(), yy.ravel()])

    finite = np.isfinite(values)
    rel = points[finite] - geom["start"]
    dist_along = rel @ geom["direction"]
    dist_perp = rel @ geom["normal"]

    in_strip = (
        (dist_along >= 0.0)
        & (dist_along <= geom["length_px"])
        & (np.abs(dist_perp) <= geom["half_width"])
    )

    samples = pd.DataFrame(
        {
            "x": points[finite, 0][in_strip],
            "y": points[finite, 1][in_strip],
            "distance_px": dist_along[in_strip],
            "perp_distance_px": dist_perp[in_strip],
            "value": values[finite][in_strip],
        }
    ).sort_values("distance_px", kind="mergesort")

    if method == "none":
        profile_df = samples[["x", "y", "distance_px", "perp_distance_px", "value"]].copy()
        profile_df["distance_um"] = (
            profile_df["distance_px"] * pixel_size_um
            if pixel_size_um is not None
            else np.nan
        )
        smooth_window = int(max(smooth_window, 1))
        if smooth_window > 1:
            profile_df["value_smoothed"] = (
                profile_df["value"]
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
            )
        else:
            profile_df["value_smoothed"] = profile_df["value"]

        profile_df.attrs["start"] = tuple(np.asarray(start, dtype=float))
        profile_df.attrs["end"] = tuple(np.asarray(end, dtype=float))
        profile_df.attrs["width_px"] = float(width_px)
        profile_df.attrs["length_px"] = geom["length_px"]
        profile_df.attrs["pixel_size_um"] = pixel_size_um
        profile_df.attrs["method"] = method
        profile_df.attrs["smooth_window"] = smooth_window

        return profile_df, profile_df

    if n_bins is None:
        n_bins = max(int(np.ceil(geom["length_px"])), 1)
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    bin_edges = np.linspace(0.0, geom["length_px"], n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if samples.empty:
        profile_df = pd.DataFrame(
            {
                "bin": np.arange(n_bins, dtype=int),
                "distance_px": bin_centres,
                "value": np.full(n_bins, np.nan),
                "n_pixels": np.zeros(n_bins, dtype=int),
                "distance_um": (
                    bin_centres * pixel_size_um
                    if pixel_size_um is not None
                    else np.full(n_bins, np.nan)
                ),
            }
        )
    else:
        samples["bin"] = np.minimum(
            np.digitize(samples["distance_px"], bin_edges, right=False) - 1,
            n_bins - 1,
        ).astype(int)

        grouped = samples.groupby("bin", sort=True)["value"]
        if method == "median":
            values_by_bin = grouped.median()
        else:
            values_by_bin = grouped.mean()
        counts_by_bin = grouped.size()

        profile_df = pd.DataFrame(
            {
                "bin": np.arange(n_bins, dtype=int),
                "distance_px": bin_centres,
            }
        )
        profile_df["value"] = profile_df["bin"].map(values_by_bin).astype(float)
        profile_df["n_pixels"] = (
            profile_df["bin"].map(counts_by_bin).fillna(0).astype(int)
        )
        profile_df["distance_um"] = (
            profile_df["distance_px"] * pixel_size_um
            if pixel_size_um is not None
            else np.nan
        )

    smooth_window = int(max(smooth_window, 1))
    if smooth_window > 1:
        profile_df["value_smoothed"] = (
            profile_df["value"]
            .rolling(window=smooth_window, center=True, min_periods=1)
            .mean()
        )
    else:
        profile_df["value_smoothed"] = profile_df["value"]

    profile_df.attrs["start"] = tuple(np.asarray(start, dtype=float))
    profile_df.attrs["end"] = tuple(np.asarray(end, dtype=float))
    profile_df.attrs["width_px"] = float(width_px)
    profile_df.attrs["length_px"] = geom["length_px"]
    profile_df.attrs["pixel_size_um"] = pixel_size_um
    profile_df.attrs["method"] = method
    profile_df.attrs["smooth_window"] = smooth_window

    samples.attrs["start"] = profile_df.attrs["start"]
    samples.attrs["end"] = profile_df.attrs["end"]
    samples.attrs["width_px"] = float(width_px)

    return profile_df, samples


def plot_line_profile(
    profile_df,
    ax=None,
    label=None,
    color="black",
    show_counts=False,
):
    """
    Plot a line-profile dataframe produced by ``extract_line_profile()``.

    Parameters:
        profile_df (pd.DataFrame): Output profile table from
            ``extract_line_profile()``.
        ax (matplotlib.axes.Axes|None): Existing axis to draw on.
        label (str|None): Series label.
        color (str): Line colour.
        show_counts (bool): If True, add a secondary count histogram.

    Returns:
        ax (matplotlib.axes.Axes): Axis containing the profile.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    xcol = "distance_um"
    xlabel = "Distance (µm)"
    if np.all(~np.isfinite(pd.to_numeric(profile_df[xcol], errors="coerce"))):
        xcol = "distance_px"
        xlabel = "Distance (px)"

    _, ycol = _resolve_profile_value_columns(profile_df)
    x = pd.to_numeric(profile_df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(profile_df[ycol], errors="coerce").to_numpy(dtype=float)

    ax.plot(x, y, color=color, lw=2, label=label)
    ax.scatter(x, y, color=color, s=12, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)

    if label:
        ax.legend(frameon=False)

    if show_counts:
        ax2 = ax.twinx()
        counts = pd.to_numeric(
            profile_df["n_pixels"], errors="coerce"
        ).to_numpy(dtype=float)
        ax2.fill_between(x, counts, color="0.85", alpha=0.4)
        ax2.set_ylabel("Pixels / bin")

    return ax


def interactive_line_profile(
    res,
    key,
    method="none",
    source="auto",
    *,
    phase=None,
    width_px=3.0,
    n_bins=None,
    pixel_size_um=None,
    smooth_window=1,
    cmap="viridis",
    vmin=None,
    vmax=None,
    title=None,
    cbar_label=None,
    layout="vertical",
    multi=True,
    figsize=None,
):
    """
    Launch a clickable Jupyter transect tool for oxide or component maps.

    Intended for notebook use with an interactive Matplotlib backend such as
    ``%matplotlib widget``. Click once for the profile start and a second time
    for the profile end. Press ``r`` to clear and redraw.

    Parameters:
        res (dict): Result dictionary returned by ``run_map()``.
        key (str): Oxide or component key, e.g. ``"SiO2"`` or ``"Feldspar.An"``.
        method (str): Aggregation method passed to ``extract_line_profile()``.
        source (str): One of ``"auto"``, ``"oxide"``, or ``"component"``.
        phase (str|list[str]|None): If provided, mask the map so only pixels
            matching this phase are shown; all others are set to NaN.
        width_px (float): Transect-strip width in pixels.
        n_bins (int|None): Number of distance bins.
        pixel_size_um (float|None): Micrometers per pixel.
        smooth_window (int): Rolling smoothing window, in bins.
        cmap (str): Colormap for the source map.
        vmin (float|None): Lower display limit.
        vmax (float|None): Upper display limit.
        title (str|None): Title for the map panel.
        cbar_label (str|None): Colorbar label for the map panel.
        layout (str): ``"vertical"`` for stacked axes or ``"horizontal"``
            for side-by-side axes.
        multi (bool): If True, each completed click-pair is retained as a new
            profile. If False, a new transect replaces the previous one.
        figsize (tuple): Figure size.

    Returns:
        controller (dict): Dictionary with keys ``fig``, ``profiles``
            (list of per-transect DataFrames), ``profiles_df`` (all transects
            concatenated), ``samples`` (list of per-transect raw pixel
            DataFrames), ``samples_df`` (all concatenated),
            ``coordinates_df`` (transect metadata), and helper accessors
            ``get_profile``, ``get_samples``, ``get_coordinates``.
    """
    plt.close("all")
    backend = plt.get_backend().lower()
    _non_interactive = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template",
                        "module://matplotlib_inline.backend_inline"}
    if backend in _non_interactive or backend.endswith("inline"):
        warnings.warn(
            "interactive_line_profile() needs an interactive Matplotlib backend. "
            f"The current backend is {plt.get_backend()!r}, so the figure will "
            "render as static and clicks will not register. In a notebook, try "
            "`%matplotlib notebook`, or install `ipympl` and use "
            "`%matplotlib widget`.",
            UserWarning,
        )

    data = get_profile_map(res, key, source=source)

    if phase is not None and "mineral_map" in res:
        phase_filter = {phase} if isinstance(phase, str) else set(phase)
        phase_mask = np.isin(res["mineral_map"], list(phase_filter))
        data = np.where(phase_mask, data, np.nan)

    valid = data[np.isfinite(data)]
    if vmin is None:
        vmin = float(valid.min()) if valid.size else 0.0
    if vmax is None:
        vmax = float(valid.max()) if valid.size else 1.0
    if title is None:
        title = f"Interactive Line Profile: {key}"
    if cbar_label is None:
        cbar_label = key

    if layout not in {"vertical", "horizontal"}:
        raise ValueError("layout must be one of {'vertical', 'horizontal'}")

    if layout == "vertical":
        fig, (ax_map, ax_profile) = plt.subplots(
            2,
            1,
            figsize=(5, 10) if figsize is None else figsize,
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
    else:
        fig, (ax_map, ax_profile) = plt.subplots(
            1,
            2,
            figsize=(10, 5) if figsize is None else figsize,
            gridspec_kw={"width_ratios": [1.15, 1.0]},
        )

    masked = np.ma.masked_invalid(data)
    im = ax_map.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        origin="upper",
    )
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label=cbar_label)
    ax_map.set_title(title)
    ax_map.set_xlabel("X (px)")
    ax_map.set_ylabel("Y (px)")

    ax_profile.set_title("Profile")
    ax_profile.set_xlabel("Distance")
    ax_profile.set_ylabel("Value")
    ax_profile.grid(alpha=0.25)

    fig.suptitle(
        "Click to delineate start and end.\n"
        "'r': reset clicks | 'u': undo last | 'c': clear all | 'q'/Esc: done",
        y=0.96,
        fontsize=11,
    )

    state = {
        "points": [],
        "current_artists": [],
        "saved_artists": [],
        "profiles_long": [],
        "profiles": [],
        "samples": [],
        "records": [],
        "colors": [],
    }
    controller = {
        "fig": fig,
        "profiles": [],
        "profiles_df": None,
        "samples": [],
        "samples_df": None,
        "coordinates_df": None,
    }

    def _get_profile(index=-1):
        if not controller["profiles"]:
            return None
        return controller["profiles"][index]

    def _get_samples(index=-1):
        if not controller["samples"]:
            return None
        return controller["samples"][index]

    def _get_coordinates():
        return controller["coordinates_df"]

    controller["get_profile"] = _get_profile
    controller["get_samples"] = _get_samples
    controller["get_coordinates"] = _get_coordinates

    def _clear_artist_list(artists):
        while artists:
            artist = artists.pop()
            artist.remove()

    def _update_controller_tables():
        controller["profiles"] = list(state["profiles"])
        controller["samples"] = list(state["samples"])

        controller["profiles_df"] = (
            pd.concat(state["profiles"], ignore_index=True) if state["profiles"] else None
        )
        controller["samples_df"] = (
            pd.concat(state["samples"], ignore_index=True) if state["samples"] else None
        )
        controller["coordinates_df"] = (
            pd.DataFrame(state["records"]) if state["records"] else None
        )

    def _reset_profile_axis():
        ax_profile.clear()
        ax_profile.set_title("Profile")
        ax_profile.set_xlabel("Distance")
        ax_profile.set_ylabel("Value")
        ax_profile.grid(alpha=0.25)

    def _redraw_saved_profiles():
        _reset_profile_axis()
        for profile_df, color in zip(state["profiles"], state["colors"]):
            label = f"{key} #{int(profile_df['profile_id'].iat[0])}"
            plot_line_profile(profile_df, ax=ax_profile, label=label, color=color)
        if state["profiles"]:
            ax_profile.set_title(f"{key} Profiles ({len(state['profiles'])})")

    def _color_for_profile(index):
        cmap_obj = plt.get_cmap("tab10")
        return cmap_obj(index % 10)

    def _reset_current():
        state["points"].clear()
        _clear_artist_list(state["current_artists"])
        fig.canvas.draw_idle()

    def _clear_all():
        _reset_current()
        state["profiles_long"].clear()
        state["profiles"].clear()
        state["samples"].clear()
        state["records"].clear()
        state["colors"].clear()
        _clear_artist_list(state["saved_artists"])
        _update_controller_tables()
        _reset_profile_axis()
        fig.canvas.draw_idle()

    def _draw_profile():
        start, end = state["points"]
        geom = _line_strip_geometry(start, end, width_px)
        profile_df, samples_df = extract_line_profile(
            data,
            start=start,
            end=end,
            method=method,
            width_px=width_px,
            n_bins=n_bins,
            pixel_size_um=pixel_size_um,
            smooth_window=smooth_window,
        )

        if not multi:
            _clear_artist_list(state["saved_artists"])
            state["profiles_long"].clear()
            state["profiles"].clear()
            state["samples"].clear()
            state["records"].clear()
            state["colors"].clear()

        profile_id = len(state["profiles_long"]) + 1
        profile_color = _color_for_profile(profile_id - 1)

        profile_df = profile_df.copy()
        samples_df = samples_df.copy()

        profile_df["profile_id"] = profile_id
        profile_df["key"] = key
        profile_df["source"] = source
        profile_df["x0"] = int(round(start[0]))
        profile_df["y0"] = int(round(start[1]))
        profile_df["x1"] = int(round(end[0]))
        profile_df["y1"] = int(round(end[1]))
        profile_df["width_px"] = float(width_px)
        profile_df["length_px"] = float(geom["length_px"])
        profile_df["length_um"] = (
            float(geom["length_px"] * pixel_size_um)
            if pixel_size_um is not None
            else np.nan
        )
        profile_df["color"] = to_hex(profile_color)

        samples_df["profile_id"] = profile_id
        samples_df["key"] = key
        samples_df["source"] = source
        samples_df["x0"] = int(round(start[0]))
        samples_df["y0"] = int(round(start[1]))
        samples_df["x1"] = int(round(end[0]))
        samples_df["y1"] = int(round(end[1]))
        samples_df["width_px"] = float(width_px)
        samples_df["length_px"] = float(geom["length_px"])
        samples_df["length_um"] = (
            float(geom["length_px"] * pixel_size_um)
            if pixel_size_um is not None
            else np.nan
        )
        samples_df["color"] = to_hex(profile_color)

        profile_df_clean = _profile_table_for_key(profile_df, key)

        state["profiles_long"].append(profile_df)
        state["profiles"].append(profile_df_clean)
        state["samples"].append(samples_df)
        state["colors"].append(profile_color)
        state["records"].append(
            {
                "profile_id": profile_id,
                "key": key,
                "source": source,
                "x0": int(round(start[0])),
                "y0": int(round(start[1])),
                "x1": int(round(end[0])),
                "y1": int(round(end[1])),
                "width_px": float(width_px),
                "length_px": float(geom["length_px"]),
                "length_um": (
                    float(geom["length_px"] * pixel_size_um)
                    if pixel_size_um is not None
                    else np.nan
                ),
                "n_bins": int(len(profile_df)),
                "pixel_size_um": pixel_size_um,
                "method": method,
                "smooth_window": int(max(smooth_window, 1)),
                "color": to_hex(profile_color),
            }
        )
        _update_controller_tables()

        _clear_artist_list(state["current_artists"])
        state["saved_artists"].append(
            ax_map.plot(
                [geom["start"][0], geom["end"][0]],
                [geom["start"][1], geom["end"][1]],
                color=profile_color,
                lw=2,
            )[0]
        )
        state["saved_artists"].append(
            ax_map.scatter(
                [geom["start"][0], geom["end"][0]],
                [geom["start"][1], geom["end"][1]],
                c=[profile_color, profile_color],
                s=36,
                zorder=5,
                edgecolors="black",
                linewidths=0.6,
            )
        )

        if width_px > 0:
            poly = mpatches.Polygon(
                geom["outline"][:-1],
                closed=True,
                fill=False,
                ec=profile_color,
                lw=1.2,
                ls="--",
                alpha=0.9,
            )
            ax_map.add_patch(poly)
            state["saved_artists"].append(poly)

        _redraw_saved_profiles()
        state["points"].clear()
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return

        xy = np.array([event.xdata, event.ydata], dtype=float)
        if len(state["points"]) >= 2:
            _reset_current()
        if not multi and controller["profiles"]:
            _clear_all()

        state["points"].append(xy)
        marker = ax_map.scatter(
            [xy[0]],
            [xy[1]],
            c="white",
            s=30,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        state["current_artists"].append(marker)

        if len(state["points"]) == 2:
            _draw_profile()
        else:
            fig.canvas.draw_idle()

    def _on_key(event):
        if event.key in ("q", "escape"):
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)
            fig.suptitle("Inactive (press saved)", fontsize=11)
            fig.canvas.draw_idle()
            return
        if event.key == "r":
            _reset_current()
        elif event.key == "c":
            _clear_all()
        elif event.key == "u" and state["profiles"]:
            _reset_current()
            last_saved = len(state["profiles"]) > 0
            if last_saved:
                state["profiles_long"].pop()
                state["profiles"].pop()
                state["samples"].pop()
                state["records"].pop()
                state["colors"].pop()
                if width_px > 0:
                    n_art = 3
                else:
                    n_art = 2
                for _ in range(n_art):
                    artist = state["saved_artists"].pop()
                    artist.remove()
                _update_controller_tables()
                _redraw_saved_profiles()
                fig.canvas.draw_idle()

    cid_click = fig.canvas.mpl_connect("button_press_event", _on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", _on_key)

    if layout == "vertical":
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0)
    plt.show()
    return controller


def batch_extract_line_profiles(
    res,
    transects,
    keys=None,
    source="auto",
    *,
    pixel_size_um=None,
    method="mean",
    smooth_window=1,
    return_long=False,
):
    """
    Batch-extract profiles for one or more map keys from saved transect coordinates.

    Parameters:
        res (dict): Result dictionary returned by ``run_map()``.
        transects (pd.DataFrame|list[dict]): Table with at least
            ``x0``, ``y0``, ``x1``, ``y1``. If ``width_px`` or
            ``n_bins`` are present they are reused per transect.
        keys (str|list[str]|None): Oxide/component keys to extract. If None,
            defaults to available oxide maps in the canonical ``OXIDES`` order.
        source (str): One of ``"auto"``, ``"oxide"``, or ``"component"``.
        pixel_size_um (float|None): Override physical pixel size. If None,
            uses the value from each transect row when present.
        method (str): Aggregation method passed to ``extract_line_profile()``.
            Use ``"none"`` to keep individual pixels, ``"mean"`` or
            ``"median"`` to bin along the transect.
        smooth_window (int): Rolling smoothing window, in bins (or pixels when
            ``method="none"``).
        return_long (bool): If True, also return the long-format profile table
            as a second output.

    Returns:
        profiles_df (pd.DataFrame): One row per pixel (``method="none"``) or
            per bin, with each key as its own column.
        profiles_long_df (pd.DataFrame, optional): Long-format table returned
            only when ``return_long=True``.
    """
    if keys is None:
        if isinstance(res, dict):
            oxide_maps = res.get("oxide_maps", None)
            oxide_maps = oxide_maps if oxide_maps is not None else res
        else:
            oxide_maps = {}
        keys = [k for k in OXIDES if k in oxide_maps]
        if not keys:
            raise ValueError(
                "No oxide keys were found; pass keys explicitly."
            )
    elif isinstance(keys, str):
        keys = [keys]
    else:
        keys = list(keys)

    if not keys:
        raise ValueError("keys must contain at least one map key.")

    transects_df = pd.DataFrame(transects).copy()
    required = {"x0", "y0", "x1", "y1"}
    missing = required - set(transects_df.columns)
    if missing:
        raise KeyError(
            f"transects is missing required columns: {', '.join(sorted(missing))}"
        )

    if "profile_id" not in transects_df.columns:
        transects_df["profile_id"] = np.arange(1, len(transects_df) + 1)
    if "width_px" not in transects_df.columns:
        transects_df["width_px"] = 1.0
    if "n_bins" not in transects_df.columns:
        transects_df["n_bins"] = np.nan

    profiles_out = []

    for key in keys:
        data = get_profile_map(res, key, source=source)
        for row in transects_df.itertuples(index=False):
            row_pixel_size_um = pixel_size_um
            if row_pixel_size_um is None and hasattr(row, "pixel_size_um"):
                val = getattr(row, "pixel_size_um")
                row_pixel_size_um = val if pd.notna(val) else None

            row_n_bins = None
            if hasattr(row, "n_bins") and pd.notna(getattr(row, "n_bins")):
                row_n_bins = int(getattr(row, "n_bins"))

            profile_df, _ = extract_line_profile(
                data,
                start=(float(row.x0), float(row.y0)),
                end=(float(row.x1), float(row.y1)),
                width_px=float(row.width_px),
                n_bins=row_n_bins,
                pixel_size_um=row_pixel_size_um,
                method=method,
                smooth_window=smooth_window,
            )

            profile_df = profile_df.copy()
            profile_df["profile_id"] = int(row.profile_id)
            profile_df["key"] = key
            profile_df["source"] = source
            profile_df["x0"] = float(row.x0)
            profile_df["y0"] = float(row.y0)
            profile_df["x1"] = float(row.x1)
            profile_df["y1"] = float(row.y1)
            profile_df["width_px"] = float(row.width_px)
            profiles_out.append(profile_df)

    profiles_long_df = (
        pd.concat(profiles_out, ignore_index=True)
        if profiles_out
        else pd.DataFrame()
    )

    if profiles_long_df.empty:
        profiles_df = pd.DataFrame()
    elif method == "none":
        join_cols = ["profile_id", "x", "y", "distance_px", "perp_distance_px",
                     "distance_um", "x0", "y0", "x1", "y1", "width_px", "source"]

        wide = None
        for key in keys:
            sub = profiles_long_df[profiles_long_df["key"] == key][
                join_cols + ["value", "value_smoothed"]
            ].rename(columns={"value": key, "value_smoothed": f"{key}_smoothed"})
            if wide is None:
                wide = sub
            else:
                wide = wide.merge(sub, on=join_cols, how="outer")

        if wide is None:
            profiles_df = pd.DataFrame()
        else:
            value_cols = []
            for key in keys:
                if key in wide.columns:
                    value_cols.append(key)
                if f"{key}_smoothed" in wide.columns:
                    value_cols.append(f"{key}_smoothed")
            ordered = (
                ["profile_id", "x", "y", "distance_px", "perp_distance_px", "distance_um"]
                + value_cols
                + ["x0", "y0", "x1", "y1", "width_px", "source"]
            )
            ordered = [c for c in ordered if c in wide.columns]
            profiles_df = wide[ordered].sort_values(
                ["profile_id", "distance_px"]
            ).reset_index(drop=True)
    else:
        meta_cols = [
            "profile_id",
            "distance_px",
            "distance_um",
            "x0",
            "y0",
            "x1",
            "y1",
            "width_px",
        ]

        wide_tables = []
        for key in keys:
            sub = profiles_long_df[profiles_long_df["key"] == key].copy()
            if sub.empty:
                continue

            sub = _profile_table_for_key(sub, key)
            keep_cols = meta_cols + [
                key,
                f"{key}_smoothed",
            ]
            wide_tables.append(sub[keep_cols])

        if not wide_tables:
            profiles_df = pd.DataFrame()
        else:
            profiles_df = wide_tables[0]
            for tbl in wide_tables[1:]:
                profiles_df = profiles_df.merge(tbl, on=meta_cols, how="outer")
            profiles_df = profiles_df.sort_values(["profile_id", "distance_px"]).reset_index(
                drop=True
            )
            preferred = ["profile_id", "distance_px", "distance_um"]
            value_cols = []
            for key in keys:
                if key in profiles_df.columns:
                    value_cols.append(key)
                smoothed_col = f"{key}_smoothed"
                if smoothed_col in profiles_df.columns:
                    value_cols.append(smoothed_col)
            tail_cols = [
                c for c in ("x0", "y0", "x1", "y1")
                if c in profiles_df.columns
            ]
            ordered = [c for c in preferred if c in profiles_df.columns]
            ordered += [c for c in value_cols if c not in ordered]
            ordered += [
                c for c in profiles_df.columns
                if c not in ordered and c not in tail_cols
            ]
            ordered += tail_cols
            profiles_df = profiles_df[ordered]

    if return_long:
        return profiles_df, profiles_long_df
    return profiles_df


def plot_locations(
    res,
    transects,
    map_key=None,
    source="auto",
    *,
    cmap="viridis",
    vmin=None,
    vmax=None,
    title=None,
    cbar_label=None,
    show_width=True,
    annotate=True,
    annotate_offset_px=4.0,
    ax=None,
    figsize=(7, 7),
):
    """
    Plot transect lines or pixel pick locations on top of a map or blank canvas.

    Accepts either a transects table (from ``interactive_line_profile`` or
    ``batch_line_profiles``) with columns ``x0``, ``y0``, ``x1``, ``y1``, or a
    pixel picks table (from ``extract_pixel_comp``) with columns ``x``, ``y``.
    The input type is detected automatically.

    Parameters:
        res (dict): Result dictionary returned by ``run_map()``.
        transects (pd.DataFrame|list[dict]): Transect table with ``x0``, ``y0``,
            ``x1``, ``y1``, or pixel picks table with ``x``, ``y``.
        map_key (str|None): Background oxide/component key. If None, uses a
            blank pixel-space canvas based on ``res['shape']``.
        source (str): One of ``"auto"``, ``"oxide"``, or ``"component"``.
        cmap (str): Background colormap when ``map_key`` is provided.
        vmin (float|None): Lower display limit for the background map.
        vmax (float|None): Upper display limit for the background map.
        title (str|None): Plot title.
        cbar_label (str|None): Colorbar label for the background map.
        show_width (bool): If True, draw the finite-width strip outline when
            ``width_px`` is available (transect mode only).
        annotate (bool): If True, label each point or transect with its index.
        annotate_offset_px (float): Pixel offset for transect annotation labels.
        ax (matplotlib.axes.Axes|None): Existing axis to draw on.
        figsize (tuple): Figure size when creating a new figure.

    Returns:
        fig (matplotlib.figure.Figure): Figure containing the overlay.
        ax (matplotlib.axes.Axes): Axis containing the map and overlay.
    """
    transects_df = pd.DataFrame(transects).copy()
    cols = set(transects_df.columns)

    plt.close("all")

    # Detect mode: pixel picks have x/y but no x0/y0/x1/y1
    _is_picks = {"x", "y"}.issubset(cols) and not {"x0", "x1"}.issubset(cols)

    if not _is_picks:
        required = {"x0", "y0", "x1", "y1"}
        missing = required - cols
        if missing:
            raise KeyError(
                f"transects is missing required columns: {', '.join(sorted(missing))}"
            )

    if not _is_picks:
        if "profile_id" not in transects_df.columns:
            transects_df["profile_id"] = np.arange(1, len(transects_df) + 1)
        if "width_px" not in transects_df.columns:
            transects_df["width_px"] = np.nan

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if map_key is not None:
        data = get_profile_map(res, map_key, source=source)
        valid = data[np.isfinite(data)]
        if vmin is None:
            vmin = float(valid.min()) if valid.size else 0.0
        if vmax is None:
            vmax = float(valid.max()) if valid.size else 1.0
        if cbar_label is None:
            cbar_label = map_key

        im = ax.imshow(
            np.ma.masked_invalid(data),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
            origin="upper",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
        h, w = data.shape
    else:
        if not isinstance(res, dict) or "shape" not in res:
            raise KeyError("res['shape'] is required when map_key is None.")
        h, w = res["shape"]
        ax.set_facecolor("#f5f5f5")
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)

    if _is_picks:
        for i, row in enumerate(transects_df.itertuples(index=False)):
            color = plt.get_cmap("tab10")(i % 10)
            ax.scatter([row.x], [row.y], c=[color], s=40, edgecolors="black",
                       linewidths=0.7, zorder=5)
            if annotate:
                ax.text(
                    row.x + float(annotate_offset_px),
                    row.y - float(annotate_offset_px),
                    str(i + 1),
                    color="black", fontsize=9, ha="left", va="bottom",
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=color, alpha=0.9),
                )
    else:
        for i, row in enumerate(transects_df.itertuples(index=False)):
            raw_color = (
                getattr(row, "color")
                if hasattr(row, "color") and pd.notna(getattr(row, "color"))
                else None
            )
            color = _coerce_profile_color(raw_color, fallback=plt.get_cmap("tab10")(i % 10))

            start = np.array([float(row.x0), float(row.y0)], dtype=float)
            end = np.array([float(row.x1), float(row.y1)], dtype=float)
            pid = int(row.profile_id)
            geom = _line_strip_geometry(start, end, 0.0)
            label_offset = geom["normal"] * float(annotate_offset_px)

            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=2, alpha=0.95)
            ax.scatter([start[0]], [start[1]], c=[color], s=40, marker="o",
                       edgecolors="black", linewidths=0.7, zorder=5)
            ax.scatter([end[0]], [end[1]], c=[color], s=48, marker="s",
                       edgecolors="black", linewidths=0.7, zorder=5)

            width_px = getattr(row, "width_px", np.nan)
            if show_width and pd.notna(width_px) and float(width_px) > 0:
                geom_w = _line_strip_geometry(start, end, float(width_px))
                poly = mpatches.Polygon(
                    geom_w["outline"][:-1], closed=True, fill=False,
                    ec=color, lw=1.0, ls="--", alpha=0.8,
                )
                ax.add_patch(poly)

            if annotate:
                mid = 0.5 * (start + end)
                mid_label_xy = mid + 1.25 * label_offset
                ax.text(
                    mid_label_xy[0], mid_label_xy[1], f"{pid}",
                    color="black", fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=color, alpha=0.9),
                )

    if title is None:
        if _is_picks:
            title = "Pixel Pick Locations" if map_key is None else f"Pixel Pick Locations: {map_key}"
        elif map_key is None:
            title = "Profile Locations"
        else:
            title = f"Profile Locations: {map_key}"

    ax.set_title(title)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_aspect("equal")

    if map_key is not None:
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)

    fig.tight_layout()
    return fig, ax


# %%
