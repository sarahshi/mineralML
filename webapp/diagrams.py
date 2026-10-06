""" Composition diagrams for the mineralML web app. Field boundaries come from mineralML's
classifier plots; the points are redrawn here so they can be colored by any column. // @author: Sarah Shi """

import io
import re
import math
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

import ternary
import mineralML as mm

# Categorical slots in fixed order (validated for color-vision deficiency); markers are the
# second cue, so identity never rests on color alone. A 9th category folds into "Other".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
OTHER_COLOR = "#9a9893"
UNCLASSIFIED_COLOR = "#52514e"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "mineralML_blues", ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
# Labels the classifiers give to points outside every field; drawn as grey crosses.
UNCLASSIFIED = {"Unclassified", "Unlabeled", "OOD", "Feldspar_Miscibility_Gap", "nan", "None", ""}
RASTERIZE_ABOVE = 5000  # points; keeps PDFs and SVGs small while lines and text stay vector

OXIDE_NAMES = set(mm.OXIDES) | {"FeO", "Fe2O3", "Fe2O3t", "ZrO2"}


@dataclass(frozen=True)
class Diagram:
    label: str
    minerals: tuple  # Predict_Mineral values included by default
    field_label: str = ""  # legend title for the classification field; "" if none
    ternary: bool = False
    label_choices: tuple = ()  # field-label options shown to the user
    figsize: tuple = (8.0, 6.0)  # default size, in inches, matching the package plots


DIAGRAMS = {
    "tas": Diagram("TAS (total alkali–silica)", ("Glass",), "TAS field",
                   label_choices=("Volcanic names", "Plutonic names", "None"), figsize=(9.0, 6.0)),
    "feldspar": Diagram("Feldspar ternary (An–Ab–Or)", ("Plagioclase", "Alkali_Feldspar"),
                        "Feldspar", ternary=True, label_choices=("Short", "Full", "None"), figsize=(8.0, 7.0)),
    "pyroxene": Diagram("Pyroxene quadrilateral (En–Wo–Fs)", ("Clinopyroxene", "Orthopyroxene", "Na-Pyroxene"),
                        "Pyroxene", ternary=True, label_choices=("Short", "Full", "None"), figsize=(9.0, 5.0)),
    "napyroxene": Diagram("Na-pyroxene ternary (Jd–Aeg–Quad)", ("Clinopyroxene", "Orthopyroxene", "Na-Pyroxene"),
                          "Na-pyroxene", ternary=True, label_choices=("Short", "Full", "None"), figsize=(8.0, 7.0)),
    "amphibole": Diagram("Calcic amphibole (Si vs Mg#)", ("Amphibole",), "Amphibole", figsize=(10.0, 6.0)),
    "fetioxide": Diagram("Fe–Ti oxide ternary (FeO–Fe₂O₃–TiO₂)", ("Oxide",), "Oxide", ternary=True, figsize=(8.0, 8.0)),
    "spinel": Diagram("Spinel (Fe²⁺# vs Fe³⁺#)", ("Oxide",), "Spinel", figsize=(9.0, 6.0)),
    "ternary": Diagram("Custom ternary", (), ternary=True, figsize=(8.0, 7.0)),
    "xy": Diagram("Custom x–y (Harker)", (), figsize=(10.0, 7.0)),
}
CLASSIFICATION = [k for k, d in DIAGRAMS.items() if d.field_label]


@dataclass
class Style:
    hue: str | None = "Field"  # column to color by; None for a single color
    categories: tuple = ()  # colored categories in slot order (see category_slots)
    hue_label: str = ""
    labels: str = "Short"  # field labels, from Diagram.label_choices
    quad_only: bool = True  # pyroxene: zoom to the quadrilateral
    size: float = 30
    alpha: float = 0.85
    width: float = 8.0
    height: float = 6.0
    title: str = ""
    logx: bool = False
    logy: bool = False


def chem(label):
    """Subscripts oxide formulas for matplotlib, e.g. 'Na2O+K2O' -> Na$_2$O+K$_2$O."""
    terms = [t.strip() for t in str(label).split("+")]
    if not all(t in OXIDE_NAMES for t in terms):
        return str(label)
    return " + ".join(re.sub(r"(?<=[A-Za-z])(\d+)", r"$\\mathregular{_{\1}}$", t) for t in terms)


def pretty(category):
    return str(category).replace("_", " ")


# %% ----------------------------------------------------------------
# preparing data: classify each subset and compute plot coordinates

def _quiet(fn, *args, **kwargs):
    """Runs fn, returning (result, sorted unique UserWarning messages)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    msgs = sorted({str(w.message) for w in caught if issubclass(w.category, UserWarning)})
    return out, msgs


def _finish(df, coords, names, field=None):
    """Attaches coordinate columns c0.. (and Field) to df and drops rows that cannot be plotted."""
    out = df.copy()
    for i, c in enumerate(coords):
        out[f"c{i}"] = np.asarray(c, dtype=float)
    out["Field"] = (pd.Series(field, index=df.index).astype(object) if field is not None else np.nan)
    out["Field"] = out["Field"].where(out["Field"].notna(), "Unclassified").astype(str)
    cols = [f"c{i}" for i in range(len(coords))]
    return out[np.isfinite(out[cols]).all(axis=1)], list(names)


def _prep_tas(df, labels="Volcanic names", anhydrous=True, **_):
    ox = df.reindex(columns=[c for c in df.columns if c in OXIDE_NAMES]).astype(float).fillna(0)
    scale = 100 / ox.sum(axis=1).replace(0, np.nan) if anhydrous else 1.0
    sio2 = ox.get("SiO2", 0) * scale
    alk = (ox.get("Na2O", 0) + ox.get("K2O", 0)) * scale
    clf = mm.TASClassifier()
    which = "intrusive" if labels.startswith("Plutonic") else "volcanic"
    field = clf.predict(sio2, alk).apply(lambda f: clf.get_rock_name(f, which=which))
    suffix = " (anhydrous)" if anhydrous else ""
    return _finish(df, [sio2, alk], [f"SiO2{suffix}", f"Na2O+K2O{suffix}"], field)


def _prep_feldspar(df, **_):
    dc = mm.FeldsparClassifier(df).classify()
    return _finish(df, [dc["An"], dc["Or"], dc["Ab"]], ["An", "Or", "Ab"], dc["Submineral"])


def _prep_pyroxene(df, **_):
    dc = mm.PyroxeneClassifier(df).classify()
    keep = dc["Mineral"].isin(["Clinopyroxene", "Orthopyroxene"])
    sub = df[keep]
    dc = dc[keep]
    return _finish(sub, [dc["Fs"], dc["Wo"], dc["En"]], ["Fs", "Wo", "En"], dc["Submineral"])


def _prep_napyroxene(df, **_):
    dc = mm.PyroxeneClassifier(df).classify()
    keep = dc["Mineral"] == "Na-Pyroxene"
    sub = df[keep]
    dc = dc[keep]
    # Same normalization as PyroxeneClassifier.plot: Q = En + Di + Hd, then Q + Jd + Aeg = 1
    parts = {k: pd.to_numeric(dc[f"{k}_h"], errors="coerce").fillna(0).clip(lower=0)
             for k in ("Jd", "En", "Aeg", "Di", "Hd")}
    quad = parts["En"] + parts["Di"] + parts["Hd"]
    total = (quad + parts["Jd"] + parts["Aeg"]).replace(0, np.nan)
    return _finish(sub, [parts["Aeg"] / total, quad / total, parts["Jd"] / total],
                   ["Aeg", "Quad", "Jd"], dc["Submineral"])


def _prep_amphibole(df, **_):
    dc = mm.AmphiboleClassifier(df).classify()
    return _finish(df, [dc["Si_T_leake"], dc["Mgno_leake"]], ["Si (apfu)", "Mg#"], dc["Submineral"])


def _has_oxide_labels(df):
    """OxideClassifier routes rows by name, so it needs a mineral label column."""
    return any(c in df.columns and df[c].notna().any() for c in ["Submineral", "Mineral", "Predict_Mineral"])


def _prep_fetioxide(df, **_):
    if not _has_oxide_labels(df):
        return _finish(df.iloc[0:0], [[], [], []], ["XR3", "XTi", "XR2"])
    dc = mm.OxideClassifier(df).classify()
    if "XR3" not in dc.columns:  # no oxides among the rows
        return _finish(df.iloc[0:0], [[], [], []], ["XR3", "XTi", "XR2"])
    valid = dc[["XR2", "XR3", "XTi"]].sum(axis=1) > 0
    sub, dc = df[valid], dc[valid]
    return _finish(sub, [dc["XR3"], dc["XTi"], dc["XR2"]], ["XR3", "XTi", "XR2"], dc["Suboxide"])


def _prep_spinel(df, **_):
    if not _has_oxide_labels(df):
        return _finish(df.iloc[0:0], [[], []], ["Fe2+/(Fe2++Mg)", "Fe3+/(Fe3++Al)"])
    oc = mm.OxideClassifier(df)
    dc = oc.classify()
    sp = oc._name_masks(dc)[1]  # spinel-group names, including magnetite
    sub, dc = df[sp], dc[sp]
    x, y = oc._spinel_axes(dc) if len(dc) else ([], [])
    return _finish(sub, [x, y], ["Fe2+/(Fe2++Mg)", "Fe3+/(Fe3++Al)"],
                   dc["Subspinel"] if len(dc) else None)


def _prep_ternary(df, apices=(), labels_text=(), **_):
    sums = [df.reindex(columns=list(cols)).apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
            for cols in apices]
    total = sum(sums).replace(0, np.nan)
    top, left, right = (s / total for s in sums)
    # python-ternary points are (right, top, left), as in FeldsparClassifier.plot
    out, names = _finish(df, [right, top, left], [labels_text[2], labels_text[0], labels_text[1]])
    return out[(out[["c0", "c1", "c2"]] >= 0).all(axis=1)], names


def _prep_xy(df, x=None, ys=(), **_):
    num = df.reindex(columns=[x, *ys]).apply(pd.to_numeric, errors="coerce")
    out = df.copy()
    for i, c in enumerate([x, *ys]):
        out[f"c{i}"] = num[c].to_numpy()
    out["Field"] = "Unclassified"
    ycols = [f"c{i + 1}" for i in range(len(ys))]
    keep = np.isfinite(out["c0"]) & np.isfinite(out[ycols]).any(axis=1)  # panels drop their own NaNs
    return out[keep], [x, *ys]


PREPARE = {
    "tas": _prep_tas, "feldspar": _prep_feldspar, "pyroxene": _prep_pyroxene,
    "napyroxene": _prep_napyroxene, "amphibole": _prep_amphibole,
    "fetioxide": _prep_fetioxide, "spinel": _prep_spinel,
    "ternary": _prep_ternary, "xy": _prep_xy,
}


def prepare(key, df, **opts):
    """
    Classifies df (one mineral group's analyses) for diagram `key` and adds plot coordinates.

    Returns:
        data (pd.DataFrame): df rows that can be plotted, with coordinate columns c0, c1, ...
            and a Field column holding each row's classification field.
        names (list[str]): what c0, c1, ... are, e.g. ["An", "Or", "Ab"].
        msgs (list[str]): warnings raised by the classifier.
    """
    df = df.reset_index(drop=True)
    if df.empty:
        return df.assign(Field=pd.Series(dtype=str)), [], []
    (out, names), msgs = _quiet(PREPARE[key], df, **opts)
    return out.reset_index(drop=True), names, msgs


def plotted_table(data, names, style):
    """The plotted rows as a table for download, with readable coordinate column names."""
    coords = {f"c{i}": n for i, n in enumerate(names)}
    meta = [c for c in ["Sample Name", "SampleID", "Sample", "Sample ID", "Mineral", "Predict_Mineral",
                        "Submineral", "Prediction_Score"] if c in data.columns]
    if style.hue and style.hue not in meta + ["Field"] and style.hue in data.columns:
        meta.append(style.hue)
    out = data[meta + (["Field"] if (data["Field"] != "Unclassified").any() else []) + list(coords)]
    return out.rename(columns={**coords, "Field": style.hue_label or "Field"})


# %% ----------------------------------------------------------------
# coloring

def category_slots(series):
    """
    Colored categories for a column, most frequent first. Computed on the full column so a
    filter never repaints the survivors. Beyond 8 categories, the 7 most frequent keep
    their slots and the rest fold into "Other".
    """
    counts = series.astype(str).value_counts()
    counts = counts[~counts.index.isin(UNCLASSIFIED)]
    cats = list(counts.index)
    return tuple(cats if len(cats) <= len(PALETTE) else cats[: len(PALETTE) - 1])


def folds(series):
    """Whether some categories in series are grouped as "Other"."""
    return len(set(series.astype(str)) - UNCLASSIFIED) > len(PALETTE)


def is_continuous(series):
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > len(PALETTE)


def _scatter(ax, x, y, data, style, legend=True):
    """Draws points colored by style.hue; returns legend handles (or the colorbar mappable)."""
    n = len(data)
    kw = dict(s=style.size, alpha=style.alpha, edgecolors="white",
              linewidths=0.4 if n <= 2000 else 0, rasterized=n > RASTERIZE_ABOVE, zorder=20)
    x, y = np.asarray(x, float), np.asarray(y, float)

    if style.hue is None or style.hue not in data.columns:
        ax.scatter(x, y, color=PALETTE[0], marker="o", **kw)
        return []

    hue = data[style.hue]
    if is_continuous(hue):
        vals = pd.to_numeric(hue, errors="coerce").to_numpy(float)
        ok = np.isfinite(vals)
        if (~ok).any():
            ax.scatter(x[~ok], y[~ok], color=OTHER_COLOR, marker="o", **{**kw, "zorder": 15})
        return ax.scatter(x[ok], y[ok], c=vals[ok], cmap=SEQUENTIAL, marker="o", **kw)

    labels = hue.astype(str).to_numpy()
    cats = [c for c in style.categories if c not in UNCLASSIFIED] or list(category_slots(hue))
    handles = []
    unclassified = np.isin(labels, list(UNCLASSIFIED))
    other = ~np.isin(labels, cats) & ~unclassified
    if other.any():
        ax.scatter(x[other], y[other], color=OTHER_COLOR, marker="o", **{**kw, "zorder": 15})
    for i, cat in enumerate(cats):
        m = labels == cat
        if not m.any():
            continue
        ax.scatter(x[m], y[m], color=PALETTE[i], marker=MARKERS[i], **kw)
        handles.append(Line2D([], [], ls="", marker=MARKERS[i], color=PALETTE[i], mec="white",
                              mew=0.4, ms=math.sqrt(style.size) * 1.1, label=pretty(cat)))
    if other.any():
        handles.append(Line2D([], [], ls="", marker="o", color=OTHER_COLOR, mec="white",
                              ms=math.sqrt(style.size) * 1.1, label="Other"))
    if unclassified.any():
        ax.scatter(x[unclassified], y[unclassified], color=UNCLASSIFIED_COLOR, marker="x",
                   s=style.size * 0.7, alpha=style.alpha, linewidths=0.8,
                   rasterized=n > RASTERIZE_ABOVE, zorder=18)
        handles.append(Line2D([], [], ls="", marker="x", color=UNCLASSIFIED_COLOR,
                              ms=math.sqrt(style.size) * 0.9, label="Unclassified"))
    return handles


def _legend(fig, ax, handles, style):
    if isinstance(handles, PathCollection):
        cb = fig.colorbar(handles, ax=ax, shrink=0.7, pad=0.03)
        cb.set_label(pretty(style.hue_label or style.hue))
        cb.outline.set_linewidth(0.5)
        return
    if not handles:
        return
    ax.legend(handles=handles, title=pretty(style.hue_label or style.hue), loc="upper left",
              bbox_to_anchor=(1.02, 1), frameon=False, fontsize=10, title_fontsize=10,
              alignment="left", ncols=1 if len(handles) <= 20 else 2)


# %% ----------------------------------------------------------------
# drawing: package plot for the fields, then our own points

def _strip_points(ax):
    """Removes the package plot's scatter points and legend, keeping field boundaries and labels."""
    for coll in list(ax.collections):
        if isinstance(coll, PathCollection):
            coll.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def _ternary_scatter(tax, data, style):
    ax = tax.get_axes()
    pts = data[["c0", "c1", "c2"]].to_numpy(float)
    if len(pts):
        xs, ys = ternary.helpers.project_sequence(pts, permutation=tax._permutation)
    else:
        xs, ys = [], []
    handles = _scatter(ax, xs, ys, data, style)
    ax.set_aspect("equal", adjustable="box")
    tax._redraw_labels()
    return ax, handles


def _labels_arg(style):
    return {"Short": "short", "Full": "long"}.get(style.labels)


def _one(data):
    """One row is enough for the package plots to draw their fields."""
    return data.head(1)


def _draw_tas(data, names, style):
    fig, ax = plt.subplots()
    which = "intrusive" if style.labels.startswith("Plutonic") else "volcanic"
    mm.TASClassifier().add_to_axes(ax, add_labels=style.labels != "None", which_labels=which,
                                   label_fontsize=8)
    suffix = ", anhydrous" if "anhydrous" in names[0] else ""
    ax.set_xlabel(f"SiO$\\mathregular{{_2}}$ (wt%{suffix})")
    ax.set_ylabel(f"Na$\\mathregular{{_2}}$O + K$\\mathregular{{_2}}$O (wt%{suffix})")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, length=5)
    return fig, ax, _scatter(ax, data["c0"], data["c1"], data, style)


def _draw_feldspar(data, names, style):
    fig, tax = mm.FeldsparClassifier(_one(data)).plot(labels=_labels_arg(style))
    _strip_points(tax.get_axes())
    ax, handles = _ternary_scatter(tax, data, style)
    return fig, ax, handles


def _draw_pyroxene(data, names, style):
    one = _one(data)
    pc = mm.PyroxeneClassifier(one)
    fig, tax = pc.plot(df_class=pc.classify(), labels=_labels_arg(style), quad_only=style.quad_only)
    _strip_points(tax.get_axes())
    ax, handles = _ternary_scatter(tax, data, style)
    return fig, ax, handles


def _draw_napyroxene(data, names, style):
    one = _one(data)
    pc = mm.PyroxeneClassifier(one)
    fig, tax = pc.plot(df_class=pc.classify(), labels=_labels_arg(style))
    _strip_points(tax.get_axes())
    ax, handles = _ternary_scatter(tax, data, style)
    return fig, ax, handles


def _draw_amphibole(data, names, style):
    ac = mm.AmphiboleClassifier(_one(data))
    fig, ax = ac.plot(df_class=ac.classify())
    _strip_points(ax)
    return fig, ax, _scatter(ax, data["c0"], data["c1"], data, style)


def _draw_fetioxide(data, names, style):
    figs = mm.OxideClassifier(_one(data)).plot()
    if figs["spinel"][0] is not None:
        plt.close(figs["spinel"][0])
    fig, tax = figs["ternary"]
    _strip_points(tax.get_axes())
    ax, handles = _ternary_scatter(tax, data, style)
    return fig, ax, handles


def _draw_spinel(data, names, style):
    oc = mm.OxideClassifier(_one(data))
    fig, ax = oc.plot_spinel(df=oc.classify())
    _strip_points(ax)
    ax.set_xlabel("Fe$\\mathregular{^{2+}}$ / (Fe$\\mathregular{^{2+}}$ + Mg)")
    ax.set_ylabel("Fe$\\mathregular{^{3+}}$ / (Fe$\\mathregular{^{3+}}$ + Al)")
    return fig, ax, _scatter(ax, data["c0"], data["c1"], data, style)


def _draw_ternary(data, names, style):
    fig, tax = ternary.figure(scale=1)
    tax.boundary(linewidth=1.5, zorder=0)
    tax.gridlines(multiple=0.2, ls=":", lw=0.5, c="k", alpha=0.25, zorder=0)
    tax.gridlines(multiple=0.05, lw=0.25, c="lightgrey", alpha=0.25, zorder=0)
    tax.ticks(axis="lbr", linewidth=0.5, multiple=0.2, offset=0.02, tick_formats="%.1f")
    right, top, left = names
    tax.top_corner_label(chem(top), fontsize=14, offset=0.22)
    tax.left_corner_label(chem(left), fontsize=14, offset=0.1)
    tax.right_corner_label(chem(right), fontsize=14, offset=0.1)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    ax, handles = _ternary_scatter(tax, data, style)
    return fig, ax, handles


def _draw_xy(data, names, style):
    x_name, y_names = names[0], names[1:]
    n = max(len(y_names), 1)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    handles = []
    for i, ax in enumerate(axes.flat):
        if i >= len(y_names):
            ax.set_visible(False)
            continue
        y = data[f"c{i + 1}"]
        ok = np.isfinite(y.to_numpy(float))
        h = _scatter(ax, data["c0"][ok], y[ok], data[ok], style)
        if i == 0:
            handles = h
        ax.set_xlabel(chem(x_name))
        ax.set_ylabel(chem(y_names[i]))
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, length=5)
        if style.logx:
            ax.set_xscale("log")
        if style.logy:
            ax.set_yscale("log")
    return fig, axes.flat[min(ncols, len(y_names)) - 1], handles


DRAW = {
    "tas": _draw_tas, "feldspar": _draw_feldspar, "pyroxene": _draw_pyroxene,
    "napyroxene": _draw_napyroxene, "amphibole": _draw_amphibole,
    "fetioxide": _draw_fetioxide, "spinel": _draw_spinel,
    "ternary": _draw_ternary, "xy": _draw_xy,
}


def make_figure(key, data, names, style):
    """Builds the matplotlib figure for diagram `key` from prepared data. Caller closes it."""
    hue = data.get(style.hue) if style.hue else None
    if hue is not None and not style.categories and not is_continuous(hue):
        style = replace(style, categories=category_slots(hue))  # same slots in every panel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax, handles = DRAW[key](data, names, style)
    fig.set_size_inches(style.width, style.height)
    _legend(fig, ax, handles, style)
    if style.title:
        fig.suptitle(style.title, fontsize=14)
    return fig


def save(fig, fmt, dpi=300):
    buf = io.BytesIO()
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):  # editable text in Illustrator
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    return buf.getvalue()


def render(key, data, names, style, fmt="png", dpi=300):
    """Figure bytes in fmt ('png', 'pdf' or 'svg')."""
    fig = make_figure(key, data, names, style)
    try:
        return save(fig, fmt, dpi)
    finally:
        plt.close(fig)


def render_pdf_pages(pages):
    """One multi-page PDF from [(key, data, names, style), ...]."""
    buf = io.BytesIO()
    with plt.rc_context({"pdf.fonttype": 42}), PdfPages(buf) as pdf:
        for key, data, names, style in pages:
            fig = make_figure(key, data, names, style)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.1, facecolor="white")
            plt.close(fig)
    return buf.getvalue()
