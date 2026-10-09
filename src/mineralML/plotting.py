# %%

""" Shared point styling for mineralML's classification diagrams: color by any column (categorical
or numeric), shape by any column, and per-category color and symbol choices. // @author: Sarah Shi """

import re
import math

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle

from .constants import OXIDES

__all__ = ["PALETTE", "SYMBOLS", "MAX_CATEGORIES", "category_slots", "is_continuous",
           "scatter_points", "scatter_ternary", "add_legend", "plot_columns"]

# Categorical slots in fixed order (validated for color-vision deficiency); symbols are the
# second cue, so identity never rests on color alone. No hues are generated beyond these 8:
# when symbols follow the colors, categories 9-24 reuse the colors with open, then extra,
# symbols, so every category keeps a unique color + symbol pair. Beyond that, "Other".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#9a9893"
UNCLASSIFIED_COLOR = "#52514e"
SYMBOL_INK = "#3d3d3a"  # symbol legend entries when color comes from another column
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "mineralML_blues", ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

# Symbols users can pick: name -> (matplotlib marker, filled), in default slot order.
SYMBOLS = {
    "● Circle": ("o", True), "■ Square": ("s", True), "▲ Triangle": ("^", True), "◆ Diamond": ("D", True),
    "▼ Triangle down": ("v", True), "✚ Plus": ("P", True), "✖ Cross": ("X", True), "⬢ Hexagon": ("h", True),
    "○ Open circle": ("o", False), "□ Open square": ("s", False), "△ Open triangle": ("^", False),
    "◇ Open diamond": ("D", False), "▽ Open triangle down": ("v", False), "Open plus": ("P", False),
    "Open cross": ("X", False), "⬡ Open hexagon": ("h", False),
    "◀ Triangle left": ("<", True), "▶ Triangle right": (">", True), "⬟ Pentagon": ("p", True),
    "★ Star": ("*", True), "♦ Thin diamond": ("d", True), "Octagon": ("8", True),
    "☆ Open star": ("*", False), "⬠ Open pentagon": ("p", False),
}
SYMBOL_ORDER = list(SYMBOLS)
MAX_CATEGORIES = len(SYMBOL_ORDER)  # 24
OTHER_MARK, UNCLASSIFIED_MARK = ("+", None), ("x", None)  # line markers, never user choices
SAME = "same"  # symbol="same": shapes follow the color categories
# Matplotlib draws these smaller than a circle of the same size; scale them to look equal (marker size).
SYMBOL_SCALE = {"^": 1.25, "v": 1.25, "<": 1.25, ">": 1.25, "*": 1.55, "P": 1.15, "X": 1.15,
                "d": 1.15, "p": 1.1, "+": 1.2, "x": 1.1}
# Labels the classifiers give to points outside every field; drawn as grey crosses.
UNCLASSIFIED = {"Unclassified", "Unlabeled", "OOD", "Feldspar_Miscibility_Gap", "nan", "None", ""}
RASTERIZE_ABOVE = 5000  # points; keeps PDFs and SVGs small while lines and text stay vector
OXIDE_NAMES = set(OXIDES) | {"FeO", "Fe2O3", "Fe2O3t", "ZrO2"}


def _plain(name):
    """'○ Open circle' -> 'open circle', so symbols can be named without the glyph."""
    return re.sub(r"^[^A-Za-z]+\s*", "", name).strip().lower()


_SYMBOL_LOOKUP = {**{_plain(k): v for k, v in SYMBOLS.items()}, **{k: v for k, v in SYMBOLS.items()}}
_MARKERS = {m for m, _ in SYMBOLS.values()}


def symbol_spec(symbol):
    """
    (marker, filled) for a symbol given as a SYMBOLS name ('○ Open circle'), the same name
    without its glyph ('open circle'), or a matplotlib marker ('o', '^', 's', ...).
    """
    if isinstance(symbol, str):
        if symbol in _SYMBOL_LOOKUP:
            return _SYMBOL_LOOKUP[symbol]
        if _plain(symbol) in _SYMBOL_LOOKUP:
            return _SYMBOL_LOOKUP[_plain(symbol)]
        try:
            MarkerStyle(symbol)
            return (symbol, True)
        except ValueError:
            pass
    raise ValueError(f"Unknown symbol {symbol!r}. Use a matplotlib marker ({', '.join(sorted(_MARKERS))}) "
                     f"or one of: {', '.join(_plain(k) for k in SYMBOLS)}.")


def pretty(category):
    return str(category).replace("_", " ")


def chem(label):
    """Subscripts oxide formulas for matplotlib, e.g. 'Na2O+K2O' -> Na$_2$O+K$_2$O."""
    terms = [t.strip() for t in str(label).split("+")]
    if not all(t in OXIDE_NAMES for t in terms):
        return str(label)
    return " + ".join(re.sub(r"(?<=[A-Za-z])(\d+)", r"$\\mathregular{_{\1}}$", t) for t in terms)


def category_slots(series, limit=len(PALETTE)):
    """
    Categories for a column, most frequent first. Pass the unfiltered column so a filter
    never repaints the survivors. Beyond `limit` categories, the most frequent limit - 1
    keep their slots and the rest fold into "Other".
    """
    counts = series.astype(str).value_counts()
    counts = counts[~counts.index.isin(UNCLASSIFIED)]
    cats = list(counts.index)
    return tuple(cats if len(cats) <= limit else cats[: limit - 1])


def folds(series, limit=len(PALETTE)):
    """Whether some categories in series are grouped as "Other"."""
    return len(set(series.astype(str)) - UNCLASSIFIED) > limit


def is_continuous(series):
    """Numbers get a colorbar unless they are a few whole numbers (e.g. group 1, 2, 3)."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = series.dropna()
    return vals.nunique() > len(PALETTE) or bool((vals % 1 != 0).any())


def color_limit(symbol):
    """How many color categories fit: composite color + symbol pairs, or the palette alone."""
    return MAX_CATEGORIES if symbol == SAME else len(PALETTE)


def slot_color(i):
    return PALETTE[i % len(PALETTE)]


def plot_columns(df_class, *sources):
    """
    df_class plus any columns it lacks from `sources` (e.g. a calculator's metadata and oxides),
    matched on the index, so points can be colored by a user's own columns such as a volcano name.
    """
    out = df_class
    for src in sources:
        if src is None:
            continue
        extra = [c for c in src.columns if c not in out.columns]
        if extra:
            out = out.join(src[extra], how="left")
    return out


def _check_column(data, col, what):
    if col not in data.columns:
        raise KeyError(f"{what}={col!r} is not a column of the plotted data. "
                       f"Available columns: {', '.join(map(str, data.columns))}.")


def _points(ax, x, y, rgba, mark, size, alpha, zorder, n, scatter_kw):
    marker, filled = mark
    size = size * SYMBOL_SCALE.get(marker, 1) ** 2  # scatter sizes are areas
    kw = dict(s=size, alpha=alpha, marker=marker, rasterized=n > RASTERIZE_ABOVE, zorder=zorder)
    if filled is None:  # line marker
        kw.update(c=rgba, linewidths=0.8, s=size * 0.7)
    elif filled:
        kw.update(c=rgba, edgecolors="white", linewidths=0.4 if n <= 2000 else 0)
    else:
        kw.update(facecolors="none", edgecolors=rgba, linewidths=1.2)
    ax.scatter(x, y, **{**kw, **scatter_kw})


def _handle(mark, color, size, label):
    marker, filled = mark
    ms = math.sqrt(size) * (0.9 if filled is None else 1.1) * SYMBOL_SCALE.get(marker, 1)
    if filled is False:
        return Line2D([], [], ls="", marker=marker, mfc="none", mec=color, mew=1.2, ms=ms, label=label)
    if filled is None:
        return Line2D([], [], ls="", marker=marker, color=color, mew=0.8, ms=ms, label=label)
    return Line2D([], [], ls="", marker=marker, color=color, mec="white", mew=0.4, ms=ms, label=label)


def scatter_points(ax, x, y, data, color=None, symbol=SAME, colors=None, symbols=None, size=30,
                   alpha=0.85, categories=None, symbol_categories=None, scatter_kw=None):
    """
    Draws points on ax, colored by one column of `data` and shaped by another.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to draw on.
        x, y (array-like): Point coordinates, one per row of `data`.
        data (pd.DataFrame): The plotted rows, holding the color and symbol columns.
        color (str|None): Column to color by. Categorical columns get a color per category;
            numeric columns (e.g. 'TiO2') get a colorbar. None draws one color.
        symbol (str|None): Column to shape by, "same" for shapes that follow the color categories,
            or None for one symbol.
        colors (dict|str|None): {category: color} choices overriding the default colors, or one
            color for every point when `color` is None.
        symbols (dict|str|None): {category: symbol} choices overriding the default symbols, or one
            symbol for every point. Symbols are matplotlib markers ('o', '^') or names
            ('open circle', 'star'); see mineralML.plotting.SYMBOLS.
        size (float): Marker size (area, as in matplotlib's scatter `s`).
        alpha (float): Marker opacity.
        categories, symbol_categories (sequence|None): Category order, most important first.
            Defaults to most frequent first; pass the unfiltered column's order to keep colors
            fixed across subsets.
        scatter_kw (dict|None): Extra keyword arguments for every ax.scatter call.

    Returns:
        dict: Legend entries for add_legend: {"colors": [...], "symbols": [...], "mappable": ...}.
    """
    scatter_kw = dict(scatter_kw or {})
    if {"c", "color"} & set(scatter_kw):
        raise TypeError("Point colors come from `color=` (a column) and `colors=` (choices), "
                        "not scatter keyword arguments.")
    n = len(data)
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = {"colors": [], "symbols": [], "mappable": None}
    one_color = colors if isinstance(colors, str) else PALETTE[0]
    # Categories are compared as text, so {1: "red"} matches a group column of 1, 2, 3.
    color_of = {str(k): v for k, v in colors.items()} if isinstance(colors, dict) else {}
    symbol_of = {str(k): v for k, v in symbols.items()} if isinstance(symbols, dict) else {}
    categories = [str(c) for c in categories] if categories else None
    symbol_categories = [str(c) for c in symbol_categories] if symbol_categories else None
    one_symbol = symbols if isinstance(symbols, str) else "● Circle"
    rgba = np.tile(to_rgba(one_color), (n, 1))
    zorder = np.full(n, 20)

    # Color
    cats, labels, unclassified = [], None, np.zeros(n, bool)
    hue = None
    if color is not None:
        _check_column(data, color, "color")
        hue = data[color]
    if hue is not None and is_continuous(hue):
        vals = pd.to_numeric(hue, errors="coerce").to_numpy(float)
        ok = np.isfinite(vals)
        norm = Normalize(*(np.nanmin(vals), np.nanmax(vals)) if ok.any() else (0, 1))
        rgba[ok] = SEQUENTIAL(norm(vals[ok]))
        rgba[~ok] = to_rgba(OTHER_COLOR)
        zorder[~ok] = 15
        out["mappable"] = ScalarMappable(norm=norm, cmap=SEQUENTIAL)
    elif hue is not None:
        labels = hue.astype(str).to_numpy()
        cats = [c for c in (categories or ()) if c not in UNCLASSIFIED] or list(category_slots(hue, color_limit(symbol)))
        unclassified = np.isin(labels, list(UNCLASSIFIED))
        other = ~np.isin(labels, cats) & ~unclassified
        rgba[other], zorder[other] = to_rgba(OTHER_COLOR), 15
        rgba[unclassified], zorder[unclassified] = to_rgba(UNCLASSIFIED_COLOR), 18
        for i, cat in enumerate(cats):
            rgba[labels == cat] = to_rgba(color_of.get(cat, slot_color(i)))

    # Symbol: each point gets a key into `specs`
    specs = {"other": OTHER_MARK, "unclassified": UNCLASSIFIED_MARK, "one": symbol_spec(one_symbol)}
    marks = np.full(n, "one", dtype=object)
    symbol_cats, sym_labels = [], None
    if symbol == SAME and labels is not None:
        marks[unclassified] = "unclassified"
        for i, cat in enumerate(cats):
            key = f"c{i}"
            specs[key] = symbol_spec(symbol_of.get(cat, SYMBOL_ORDER[i]))
            marks[labels == cat] = key
    elif symbol not in (SAME, None):
        _check_column(data, symbol, "symbol")
        col = data[symbol]
        sym_labels = col.astype(str).to_numpy()
        symbol_cats = ([c for c in (symbol_categories or ()) if c not in UNCLASSIFIED]
                       or list(category_slots(col, MAX_CATEGORIES)))
        marks[:] = "other"
        for i, cat in enumerate(symbol_cats):
            key = f"s{i}"
            specs[key] = symbol_spec(symbol_of.get(cat, SYMBOL_ORDER[i]))
            marks[sym_labels == cat] = key

    for m in pd.unique(marks):
        for z in np.unique(zorder):
            sel = (marks == m) & (zorder == z)
            if sel.any():
                _points(ax, x[sel], y[sel], rgba[sel], specs[m], size, alpha, z, n, scatter_kw)

    # Legend entries, for categories present in this panel
    if labels is not None:
        for i, cat in enumerate(cats):
            if (labels == cat).any():
                mark = specs[f"c{i}"] if symbol == SAME else specs["one"]
                out["colors"].append(_handle(mark, color_of.get(cat, slot_color(i)), size, pretty(cat)))
        if (zorder == 15).any():
            out["colors"].append(_handle(specs["one"], OTHER_COLOR, size, "Other"))
        if unclassified.any():
            mark = UNCLASSIFIED_MARK if symbol == SAME else specs["one"]
            out["colors"].append(_handle(mark, UNCLASSIFIED_COLOR, size, "Unclassified"))
    if sym_labels is not None:
        for i, cat in enumerate(symbol_cats):
            if (sym_labels == cat).any():
                out["symbols"].append(_handle(specs[f"s{i}"], SYMBOL_INK, size, pretty(cat)))
        if (marks == "other").any():
            out["symbols"].append(_handle(OTHER_MARK, SYMBOL_INK, size, "Other"))
    return out


def ternary_xy(tax, points):
    """Projects (right, top, left) triples, in a python-ternary axis's order, to its x, y."""
    import ternary

    pts = np.asarray(points, float).reshape(-1, 3)
    if not len(pts):
        return np.array([]), np.array([])
    xs, ys = ternary.helpers.project_sequence(pts, permutation=tax._permutation)
    return np.asarray(xs), np.asarray(ys)


def finish_ternary(tax):
    """Keeps a ternary triangle equilateral and its corner labels in place after drawing points."""
    tax.get_axes().set_aspect("equal", adjustable="box")
    tax._redraw_labels()


def scatter_ternary(tax, points, data, **kwargs):
    """
    scatter_points for a python-ternary axis. `points` are (right, top, left) triples in the
    axis's order, one per row of `data`. Returns the legend entries.
    """
    xs, ys = ternary_xy(tax, points)
    entries = scatter_points(tax.get_axes(), xs, ys, data, **kwargs)
    finish_ternary(tax)
    return entries


def add_legend(fig, ax, entries, color_label=None, symbol_label=None, color_column=None, fontsize=10,
               panel=None):
    """
    Colorbar for a numeric color; one legend outside the axis for categories, with a bold
    section per column when color and symbol come from different columns.

    Parameters:
        panel (matplotlib.axes.Axes|None): An empty axis to hold the colorbar and legend, e.g. a
            fixed column beside the plot, so the plot keeps its size whatever it is colored by.
            If None, the colorbar takes its space from `ax` and the legend sits outside `ax`.
    """
    cb_ax = None
    if entries["mappable"] is not None:
        if panel is not None:
            cb = fig.colorbar(entries["mappable"], cax=panel.inset_axes([0, 0.15, 0.06, 0.7]))
        else:
            cb = fig.colorbar(entries["mappable"], ax=ax, shrink=0.7, pad=0.03)
        cb.set_label(f"{chem(color_column)} (wt%)" if color_column in OXIDE_NAMES
                     else pretty(color_label or color_column or ""))
        cb.outline.set_linewidth(0.5)
        cb_ax = cb.ax
    sections = [(pretty(color_label or color_column or ""), entries["colors"]),
                (pretty(symbol_label or ""), entries["symbols"])]
    sections = [(t, h) for t, h in sections if h]
    if not sections:
        return None
    kw = dict(frameon=False, fontsize=fontsize, title_fontsize=fontsize, alignment="left")
    if len(sections) == 1:
        title, handles = sections[0]
        kw.update(title=title, ncols=1 if len(handles) <= 20 else 2)
    else:
        handles, headers = [], []
        for title, hs in sections:
            if handles:
                handles.append(Line2D([], [], ls="", label=" "))
            headers.append(len(handles))
            handles += [Line2D([], [], ls="", label=title)] + hs
    if panel is not None:  # top of the column, right of the colorbar's tick labels if there is one
        n = sum(len(h) for _, h in sections)
        if n > 12:  # long legends get smaller text to stay within the column
            kw.update(fontsize=8, title_fontsize=8)
        x = 0
        if cb_ax is not None:
            box = cb_ax.get_tightbbox(fig.canvas.get_renderer()).transformed(panel.transAxes.inverted())
            x = box.x1 + 0.04
        leg = panel.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, 1), borderaxespad=0, **kw)
    elif cb_ax is None:
        leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), **kw)
    else:  # beside the colorbar, clear of its tick labels; anchored to it so tight_layout keeps them together
        fig.canvas.draw()
        box = cb_ax.get_tightbbox().transformed(cb_ax.transAxes.inverted())
        leg = cb_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(box.x1 + 0.4, 1), **kw)
    if len(sections) > 1:
        for i in headers:
            leg.get_texts()[i].set_fontweight("bold")
    return leg


def style_points(ax, x, y, data, default_color, color=None, symbol=SAME, colors=None, symbols=None,
                 size=None, alpha=None, scatter_kw=None, legend=True, color_label=None, fig=None,
                 default_size=30, default_alpha=0.85):
    """
    What each classifier's plot() calls: resolves color=None to the plot's default column
    (or one color if it is missing) and color=False to one color, draws, and adds the legend.
    """
    # Older keyword arguments for the points map onto the named options, so legends match.
    # Explicit size/alpha/symbols win over s/alpha/marker in scatter_kw.
    scatter_kw = dict(scatter_kw or {})
    s, a, marker = scatter_kw.pop("s", None), scatter_kw.pop("alpha", None), scatter_kw.pop("marker", None)
    size = s if size is None else size
    alpha = a if alpha is None else alpha
    if marker is not None and symbols is None:
        symbols, symbol = marker, None
    scatter_kw.pop("label", None)  # legends are built from the categories
    # Singular aliases would clash with the plural names set per symbol (e.g. edgecolors="k" still works).
    for single, plural in (("edgecolor", "edgecolors"), ("linewidth", "linewidths")):
        if single in scatter_kw:
            scatter_kw.setdefault(plural, scatter_kw.pop(single))
    if color is None:
        color = default_color if default_color in data.columns else None
        label = color_label if color is not None else None
    else:
        label = None
        if color is False:
            color = None
    if symbol is not None and symbol != SAME and symbol is not False:
        _check_column(data, symbol, "symbol")
    if symbol is False:
        symbol = None
    entries = scatter_points(ax, x, y, data, color=color, symbol=symbol, colors=colors, symbols=symbols,
                             size=default_size if size is None else size,
                             alpha=default_alpha if alpha is None else alpha, scatter_kw=scatter_kw)
    if legend:
        add_legend(fig or ax.get_figure(), ax, entries, color_label=label, color_column=color,
                   symbol_label=symbol if symbol != SAME else None)
    return entries
