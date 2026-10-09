""" mineralML web app: upload oxide compositions, get mineral classifications. // @author: Sarah Shi """

import io
import warnings
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import streamlit as st
import mineralML as mm

import diagrams as dg  # webapp/diagrams.py; streamlit puts the script's folder on sys.path

st.set_page_config(page_title="mineralML", page_icon="💎", initial_sidebar_state=350,  # sidebar width, px (default 300)
                   layout="wide")  # full browser width, so diagrams and tables render large
st.set_page_config(initial_sidebar_state="expanded")  # additive: keeps the width, and opens the sidebar on every screen size

OXIDES = mm.OXIDES + ["ZrO2"]
SAMPLE_COLS = ["SampleID", "Sample", "Sample Name", "Sample ID"]
MAX_ROWS = 200_000
SOURCES = ["Upload a file", "Use the example dataset", "Type in analyses"]
MODES = {
    "Classify and plot": "Predict each analysis's mineral, with prediction scores, then draw composition diagrams.",
    "Plot only": "Skip classification and draw diagrams from your compositions and your own Mineral labels, "
                 "e.g. for glass or whole-rock data.",
}
# 100 analyses of each of 28 minerals from the training data, also used in the docs notebooks.
EXAMPLE_FILE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "TabularData" / "training_hundred.csv"

EXAMPLE = pd.DataFrame(
    [
        ["Olivine_1", 40.2, 0.02, 0.05, 12.1, 0.18, 47.5, 0.25, 0.0, 0.0, 0.03, 0.02, 0.0],
        ["Plag_1", 52.8, 0.05, 29.4, 0.6, 0.0, 0.1, 12.3, 4.4, 0.2, 0.0, 0.0, 0.0],
        ["Cpx_1", 50.9, 0.9, 3.8, 7.6, 0.2, 15.1, 20.6, 0.4, 0.0, 0.0, 0.3, 0.0],
        ["Magnetite_1", 0.1, 12.5, 2.9, 78.4, 0.5, 2.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
    ],
    columns=["Sample Name"] + OXIDES,
)

# %% ----------------------------------------------------------------
# helpers 

def read_upload(uploaded):
    """Reads an uploaded CSV/Excel file, dropping any unnamed pandas index column."""
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded, encoding="utf-8-sig")  # utf-8-sig strips Excel's byte-order mark
    else:
        df = pd.read_excel(uploaded)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = df.columns.astype(str).str.strip()
    # Stripping can create duplicates (e.g. "Total" and " Total "); keep the first of each.
    dupes = df.columns[df.columns.duplicated()].unique().tolist()
    if dupes:
        st.warning(f"Duplicate column names {dupes}: using the first column of each.")
        df = df.loc[:, ~df.columns.duplicated()]
    return df


@st.cache_data(show_spinner=False)
def load_example():
    return pd.read_csv(EXAMPLE_FILE, index_col=0)


def use_example():
    st.session_state["source"] = "Use the example dataset"


@st.cache_data(show_spinner=False, max_entries=8)
def classify(df, convert_fe, renormalize, drop_empty_rows, predict=True):
    """Runs prep_df (+ predict_class_prob if predict), returning results and any warnings raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepped = mm.prep_df(
            df.copy(), convert_fe=convert_fe, renormalize=renormalize,
            drop_empty_rows=drop_empty_rows, verbose=False,
        )
        results = mm.predict_class_prob(prepped, verbose=False) if predict else prepped
    # predict_class_prob keeps only oxides + known metadata; carry the user's other columns through.
    extra = [c for c in prepped.columns if c not in results.columns]
    results = pd.concat([results, prepped[extra]], axis=1)
    # Missing oxide columns are expected (treated as zero, as stated on the page); surface everything else.
    msgs = sorted({
        str(w.message) for w in caught
        if issubclass(w.category, UserWarning) and "columns were missing" not in str(w.message)
    })
    return results, msgs


@st.cache_data(show_spinner=False, max_entries=4)
def to_excel(results, include_stoich):
    buf = io.BytesIO()
    mm.export_predictions_to_excel(results, filename=buf, stoichiometry=include_stoich)
    return buf.getvalue()


def latent_png(results):
    classified = results[results["Predict_Mineral"].notna()]
    buf = io.BytesIO()
    mm.plot_latent_space(classified, filename=buf)
    return buf.getvalue()


# Options each diagram is prepared with before the user changes anything.
DEFAULT_OPTS = {"tas": {"labels": "Volcanic names", "anhydrous": True}}
PREDICTION_COLS = ["Predict_Mineral", "Submineral", "Prediction_Score", "Prediction_Score_Sigma",
                   "Second_Predict_Mineral", "Second_Prediction_Score"]


# How rows are chosen for a diagram: label shown to the user -> column (None for every row).
SELECT_BY = {"Predicted mineral": "Predict_Mineral", "Your Mineral column": "Mineral", "All analyses": None}
OXIDE_WORDS = ("oxide", "spinel", "magnetite", "ilmenite", "hematite")  # names OxideClassifier routes


def label_values(df, by):
    return sorted(df[by].dropna().astype(str).unique()) if by else []


# Common names in users' own Mineral columns, beyond mineralML's labels (compared in lowercase).
ALIASES = {
    "Glass": ("melt", "matrix glass", "melt inclusion", "whole rock", "whole-rock", "bulk rock", "liquid"),
    "Plagioclase": ("plag", "pl", "albite", "anorthite", "labradorite", "andesine", "bytownite", "oligoclase"),
    "Alkali_Feldspar": ("kfeldspar", "k-feldspar", "k feldspar", "kfs", "sanidine", "orthoclase",
                        "anorthoclase", "microcline", "alkali feldspar", "feldspar"),
    "Clinopyroxene": ("cpx", "augite", "diopside", "hedenbergite", "pigeonite", "pyroxene"),
    "Orthopyroxene": ("opx", "enstatite", "hypersthene", "ferrosilite", "bronzite"),
    "Na-Pyroxene": ("omphacite", "aegirine", "jadeite", "aegirine-augite"),
    "Amphibole": ("amph", "amp", "hornblende", "hbl", "pargasite", "edenite", "kaersutite", "tremolite",
                  "actinolite", "tschermakite"),
}


def matching_values(d, values):
    """Labels that belong on diagram d, matched case-insensitively."""
    if not d.minerals:
        return values
    wanted = {m.lower() for m in d.minerals}
    wanted |= {a for m in d.minerals for a in ALIASES.get(m, ())}
    match = [v for v in values if v.lower() in wanted]
    if "Oxide" in d.minerals:  # user labels: Magnetite, Ilmenite...
        match += [v for v in values if v not in match and any(w in v.lower() for w in OXIDE_WORDS)]
    return match


def default_values(d, values, by):
    """Labels selected for diagram d by default. Only users' own labels, which can be any name,
    fall back to every label when none match; predicted labels always use mineralML's names."""
    match = matching_values(d, values)
    return match or (values if by == "Mineral" else [])


def off_diagram_warning(d, labels):
    """Warning text when labels outside diagram d's mineral groups are plotted on it, else None."""
    stray = [v for v in labels if v not in matching_values(d, labels)]
    if not d.minerals or not stray:
        return None
    meant = " and ".join(dg.pretty(m) for m in d.minerals) if len(d.minerals) < 3 else (
        ", ".join(dg.pretty(m) for m in d.minerals[:-1]) + f" and {dg.pretty(d.minerals[-1])}")
    shown = ", ".join(dg.pretty(v) for v in stray[:5]) + (f" and {len(stray) - 5} more" if len(stray) > 5 else "")
    return (f"The {d.label} diagram should only be used for **{meant}** analyses. "
            f"Also included: {shown}. Their positions and fields on this diagram are not meaningful.")


def _subset(df, by, values):
    return df if by is None else df[df[by].astype(str).isin(values)]


@st.cache_data(show_spinner=False, max_entries=4)
def default_diagrams(results, by):
    """Every classification diagram at its default options: {key: (data, names, msgs)}."""
    values = label_values(results, by)
    return {
        k: dg.prepare(k, _subset(results, by, default_values(dg.DIAGRAMS[k], values, by)),
                      **DEFAULT_OPTS.get(k, {}))
        for k in dg.CLASSIFICATION
    }


@st.cache_data(show_spinner=False, max_entries=32)
def prepare_diagram(key, df, by, values, opts):
    return dg.prepare(key, _subset(df, by, values), **opts)


@st.cache_data(show_spinner=False, max_entries=4)
def with_stoichiometry(results):
    """Results plus stoichiometry columns (Fo, An, Mg#, cations...) for the custom plots."""
    # Without classification, the user's own labels pick the calculator (names as in mineralML).
    col = next((c for c in ["Predict_Mineral", "Mineral"] if c in results.columns), None)
    if col is None:
        return results
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mm.append_stoichiometry(results, mineral_col=col)


@st.fragment
def show_diagrams(results, header=True):
    """Composition diagrams. A fragment, so changing a plot option reruns only this section."""
    if header:
        st.header("Composition diagrams")
    st.markdown(
        "Classification diagrams for your analyses, with fields drawn by the mineralML classifiers. "
        "Choose a diagram and adjust it, then download it as PDF, SVG or PNG."
    )
    by_opts = [k for k, c in SELECT_BY.items() if c is None or (c in results.columns and results[c].notna().any())]
    by_name = st.segmented_control(
        "Select analyses by", by_opts, default=by_opts[0], key=f"dg_by_{'|'.join(by_opts)}", required=True,
        help="Which rows go on each diagram. Your Mineral column is matched to each diagram by name "
             "(e.g. Plagioclase, Clinopyroxene, Glass), ignoring case.",
    )
    by = SELECT_BY[by_name]
    with st.spinner("Classifying for diagrams..."):
        defaults = default_diagrams(results, by)
    counts = {k: len(v[0]) for k, v in defaults.items()}
    # Empty diagrams sort below the custom plots, except that the two pyroxene diagrams stay together.
    pyx = counts["pyroxene"] or counts["napyroxene"]
    listed = {k: counts[k] or (k in ("pyroxene", "napyroxene") and pyx) for k in dg.CLASSIFICATION}
    keys = ([k for k in dg.CLASSIFICATION if listed[k]] + ["ternary", "xy"]
            + [k for k in dg.CLASSIFICATION if not listed[k]])

    def describe(k):
        if k not in counts:
            return dg.DIAGRAMS[k].label
        return f"{dg.DIAGRAMS[k].label} · " + (f"{counts[k]:,} analyses" if counts[k] else "no analyses")

    key = st.selectbox("Diagram", keys, format_func=describe, key="dg_key")
    d = dg.DIAGRAMS[key]
    custom = not d.field_label
    source = with_stoichiometry(results) if custom else results
    all_values = label_values(results, by)
    scored = "Prediction_Score" in results.columns

    left, right = st.columns([1, 2.8], gap="large")
    with left:
        default_vals = default_values(d, all_values, by)
        values = all_values
        if by:
            values = st.multiselect("Include analyses " + ("predicted as" if by == "Predict_Mineral" else "labeled"),
                                    all_values, default=default_vals, key=f"dg_{key}_{by}_values",
                                    help=None if custom else "Starts with the mineral groups this diagram is for.")
        # With "All analyses", judge the rows by whichever label column the data has.
        label_col = by or next((c for c in ["Predict_Mineral", "Mineral"]
                                if c in results.columns and results[c].notna().any()), None)
        off_warning = off_diagram_warning(d, list(values) if by else label_values(results, label_col))
        min_score = 0.0
        if scored:
            # Slider and number box share one value: drag for a rough cut, type for an exact one (e.g. 0.99).
            slide_k, type_k = f"dg_{key}_score", f"dg_{key}_score_typed"
            st.session_state.setdefault(slide_k, 0.0)
            st.session_state.setdefault(type_k, 0.0)

            def copy_score(src, dst):
                st.session_state[dst] = st.session_state[src]

            st.slider("Minimum prediction score", 0.0, 1.0, step=0.005, format="%.3f", key=slide_k,
                      on_change=copy_score, args=(slide_k, type_k),
                      help="Hide ambiguous analyses. Drag, or type an exact value below.")
            min_score = st.number_input("Minimum prediction score (typed)", 0.0, 1.0, step=0.01, format="%.3f",
                                        key=type_k, on_change=copy_score, args=(type_k, slide_k),
                                        label_visibility="collapsed")

        opts, labels = {}, "Short"
        if d.label_choices:
            labels = st.selectbox("Field labels", d.label_choices, key=f"dg_{key}_labels")
        if key == "tas":
            opts["labels"] = labels
            opts["anhydrous"] = st.checkbox(
                "Recalculate to 100% volatile-free", value=True, key="dg_tas_anhydrous",
                help="IUGS convention (Le Bas et al., 1986), using the oxides in your file with iron as FeOt.",
            )
        numeric = [c for c in source.columns if c not in PREDICTION_COLS
                   and pd.api.types.is_numeric_dtype(source[c]) and source[c].notna().any()]
        if key == "ternary":
            st.caption("Each apex sums the columns you choose; the three sums are renormalized to 1.")
            apices, texts = [], []
            for apex, dflt in zip(["Top", "Left", "Right"], [["Na2O", "K2O"], ["FeOt"], ["MgO"]]):
                cols = st.multiselect(f"{apex} apex", numeric, default=[c for c in dflt if c in numeric],
                                      key=f"dg_tern_{apex}")
                text = st.text_input(f"{apex} label", value="+".join(cols), key=f"dg_tern_{apex}_{'+'.join(cols)}")
                apices.append(tuple(cols))
                texts.append(text)
            if not all(apices):
                st.info("Choose at least one column for each apex.")
                return
            opts.update(apices=tuple(apices), labels_text=tuple(texts))
        if key == "xy":
            x = st.selectbox("x axis", numeric, index=numeric.index("SiO2") if "SiO2" in numeric else 0,
                             key="dg_xy_x")
            harker = [c for c in ["Al2O3", "FeOt", "MgO", "CaO", "Na2O", "K2O"] if c in numeric]
            ys = st.multiselect("y axes (one panel each)", numeric, default=harker, key="dg_xy_ys")
            if not ys:
                st.info("Choose at least one column for the y axis.")
                return
            opts.update(x=x, ys=tuple(ys))

        # Color by: only columns worth coloring by for the analyses on this diagram.
        included = _subset(results, by, values)

        def worth_listing(col):
            vals = included[col].dropna()
            if vals.nunique() <= 1:
                return False
            # Text with a different value on nearly every row (e.g. sample names) makes no useful legend.
            return dg.is_continuous(vals) or not (vals.nunique() > dg.MAX_CATEGORIES
                                                  and vals.nunique() > 0.5 * len(vals))

        hue_opts = {}
        if not custom:
            hue_opts["Field on this diagram"] = "Field"
        if scored:
            if worth_listing("Predict_Mineral"):
                hue_opts["Predicted mineral"] = "Predict_Mineral"
            hue_opts["Prediction score"] = "Prediction_Score"
        # Oxides share one entry, with the oxide chosen below; all-zero ones were not analyzed.
        oxides = [c for c in OXIDES if c in included.columns
                  and pd.to_numeric(included[c], errors="coerce").fillna(0).ne(0).any()]
        if oxides:
            hue_opts["Oxide (wt%)"] = "__oxide__"
        # The user's own columns (e.g. Volcano, Arc segment) share one entry, with the column chosen below.
        user_cols = [c for c in results.columns
                     if c not in PREDICTION_COLS and c not in OXIDES + ["FeO", "Fe2O3", "Fe2O3t"] and worth_listing(c)]
        if user_cols:
            hue_opts["Column from your file"] = "__column__"
        hue_opts["Single color"] = None
        hue_name = st.selectbox("Color by", list(hue_opts), key=f"dg_{key}_hue",
                                help="Pick a column to color the points by. Point shapes, and the colors and "
                                     "shapes of each category, are under **Colors and shapes** below.")
        hue = hue_opts[hue_name]
        if hue == "__column__":
            hue = st.selectbox("Column", user_cols, key=f"dg_{key}_column",
                               help="Text columns get a color per category; numeric columns get a colorbar.")
            hue_name = hue
        if hue == "__oxide__":
            hue = st.selectbox("Oxide", oxides, index=oxides.index("TiO2") if "TiO2" in oxides else 0,
                               key=f"dg_{key}_oxide")
            hue_name = f"{hue} (wt%)"

        def categorical(col):
            return col == "Field" or (col is not None and not dg.is_continuous(results[col]))

        quad_only = True
        if key == "pyroxene":
            quad_only = st.checkbox("Zoom to the quadrilateral", value=True, key="dg_px_quad")
        with st.expander("Size and style"):
            size = st.slider("Marker size", 4, 120, 30, key=f"dg_{key}_size")
            alpha = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05, key=f"dg_{key}_alpha")
            w, h = d.figsize
            c1, c2 = st.columns(2)
            width = c1.number_input("Width (in)", 3.0, 20.0, w, 0.5, key=f"dg_{key}_w",
                                    help="Width of the plot. The legend or colorbar gets its own column to the "
                                         "right, so the plot stays this size whatever it is colored by.")
            height = c2.number_input("Height (in)", 3.0, 20.0, h, 0.5, key=f"dg_{key}_h")
            if d.ternary:
                st.caption("Ternaries keep their shape; the plot fits inside this size.")
            title = st.text_input("Title", key=f"dg_{key}_title")
            logx = logy = False
            if key == "xy":
                logx = st.checkbox("Log x axis", key="dg_xy_logx")
                logy = st.checkbox("Log y axes", key="dg_xy_logy")

    opts = {**DEFAULT_OPTS.get(key, {}), **opts}
    if not custom and values == default_vals and opts == DEFAULT_OPTS.get(key, {}):
        data, names, msgs = defaults[key]
    else:
        with st.spinner("Classifying..."):
            data, names, msgs = prepare_diagram(key, source, by, tuple(values), opts)
    # Slots come from this diagram's analyses before the score filter, so moving the slider never
    # repaints the survivors, and the editor lists only categories that are on the diagram.
    def full_column(col):
        return data[col]

    hue_label = d.field_label if hue == "Field" else hue_name
    colors, symbols = {}, {}
    with left:
        with st.expander("Colors and shapes"):
            # Shapes follow the colors, or come from a second column (e.g. color by volcano, shape by mineral).
            shape_opts = {"Match colors": dg.SAME} if categorical(hue) else {}
            shape_opts["All circles"] = None
            def shapeable(col):
                return col not in (hue, None) and categorical(col) and 1 < full_column(col).nunique() <= dg.MAX_CATEGORIES

            shape_opts.update({name: col for name, col in hue_opts.items()
                               if col not in ("__column__", "__oxide__") and shapeable(col)})
            shape_cols = [c for c in user_cols if shapeable(c)]
            if shape_cols:
                shape_opts["Column from your file"] = "__column__"
            shape_name = st.selectbox(
                "Shape by", list(shape_opts), key=f"dg_{key}_shape_{hue}",
                help="**Match colors**: each color group also gets its own shape. Or pick a second column, "
                     "e.g. color by volcano and shape by predicted mineral.",
            )
            symbol = shape_opts[shape_name]
            if symbol == "__column__":
                symbol = st.selectbox("Shape column", shape_cols, key=f"dg_{key}_shape_column_{hue}")
                shape_name = symbol
            symbol_label = d.field_label if symbol == "Field" else shape_name

            color_limit = dg.MAX_CATEGORIES if symbol == dg.SAME else len(dg.PALETTE)
            categories = dg.category_slots(full_column(hue), color_limit) if categorical(hue) else ()
            symbol_categories = (dg.category_slots(full_column(symbol), dg.MAX_CATEGORIES)
                                 if symbol not in (dg.SAME, None) else ())

            if categories or symbol_categories:
                prefix = f"dg_{key}_cs_{hue}_{symbol}_"
                if st.button("Reset to defaults", key=f"dg_{key}_cs_reset", icon=":material/restart_alt:"):
                    for k in [k for k in st.session_state if str(k).startswith(prefix)]:
                        del st.session_state[k]
                if categories:
                    st.markdown(f"**{dg.pretty(hue_label)}**")
                for i, cat in enumerate(categories):
                    swatch, rest = st.columns([1, 5], vertical_alignment="bottom")
                    colors[cat] = swatch.color_picker(f"{dg.pretty(cat)} color", dg.slot_color(i),
                                                      key=f"{prefix}c_{cat}", label_visibility="collapsed")
                    if symbol == dg.SAME:
                        symbols[cat] = rest.selectbox(dg.pretty(cat), dg.SYMBOL_ORDER, index=i, key=f"{prefix}s_{cat}")
                    else:
                        rest.markdown(dg.pretty(cat))
                if symbol_categories:
                    st.markdown(f"**{dg.pretty(symbol_label)}**")
                for i, cat in enumerate(symbol_categories):
                    symbols[cat] = st.selectbox(dg.pretty(cat), dg.SYMBOL_ORDER, index=i, key=f"{prefix}s_{cat}")
                if len(categories) == color_limit or len(symbol_categories) == dg.MAX_CATEGORIES:
                    st.caption("Less common categories are grouped as Other (grey).")

    n_in = len(_subset(source, by, values))
    if min_score > 0:
        data = data[data["Prediction_Score"] >= min_score]

    style = dg.Style(hue=hue, categories=categories, hue_label=hue_label, symbol=symbol,
                     symbol_categories=symbol_categories, symbol_label=symbol_label,
                     colors=tuple(colors.items()), symbols=tuple(symbols.items()),
                     labels=labels, quad_only=quad_only, size=size, alpha=alpha, width=width, height=height,
                     title=title, logx=logx, logy=logy)

    with right:
        if off_warning:
            st.warning(off_warning, icon=":material/warning:")
        for m in msgs:
            st.warning(m)
        if data.empty:
            if key in ("fetioxide", "spinel") and not scored:
                st.info("The oxide diagrams need to know which analyses are spinels and which are "
                        "ilmenite–hematite. Add a Mineral column (e.g. Magnetite, Spinel, Ilmenite, Hematite) "
                        "or choose **Classify and plot** in the menu at left.")
            elif by and not custom and not matching_values(d, all_values):
                meant = ", ".join(dg.pretty(m) for m in d.minerals)
                st.info(f"None of your analyses are {'predicted as' if by == 'Predict_Mineral' else 'labeled'} "
                        f"{meant}, so this diagram is empty. You can still add other minerals above.")
            else:
                st.info("No analyses to plot with these settings. Check the analyses included above.")
            return
        st.image(dg.render(key, data, names, style, "png", dpi=200), width="stretch")
        dropped = n_in - len(data)
        st.caption(
            f"{len(data):,} of {n_in:,} analyses plotted"
            + (f" ({dropped:,} hidden: outside the diagram, failed the score filter, or classified "
               "onto another diagram)." if dropped else ".")
            + (f" Colors beyond the {color_limit - 1} most common categories are grouped as Other."
               if categories and dg.folds(full_column(hue), color_limit) else "")
            + (f" Symbols beyond the {dg.MAX_CATEGORIES - 1} most common categories are grouped as Other."
               if symbol_categories and dg.folds(full_column(symbol), dg.MAX_CATEGORIES) else "")
            + (" To tell more than 8 categories apart by color, set **Shape by** to *Match colors* (under Colors and shapes)."
               if categories and symbol != dg.SAME and dg.folds(full_column(hue), color_limit) else "")
        )
        b = st.columns(4)
        stem = f"mineralML_{key}"
        for col, fmt, mime in zip(b, ["pdf", "svg", "png"], ["application/pdf", "image/svg+xml", "image/png"]):
            col.download_button(
                fmt.upper(), lambda fmt=fmt: dg.render(key, data, names, style, fmt),
                file_name=f"{stem}.{fmt}", mime=mime, on_click="ignore", key=f"dg_dl_{fmt}",
                icon=":material/download:",
            )
        b[3].download_button(
            "Data CSV", lambda: dg.plotted_table(data, names, style).to_csv(index=False).encode(),
            file_name=f"{stem}.csv", mime="text/csv", on_click="ignore", key="dg_dl_csv",
            icon=":material/download:", help="The plotted analyses with their diagram coordinates.",
        )

    pages = [(k, *defaults[k][:2], dg.Style(hue="Field", categories=dg.category_slots(defaults[k][0]["Field"], dg.MAX_CATEGORIES),
                                            hue_label=dg.DIAGRAMS[k].field_label, labels=(dg.DIAGRAMS[k].label_choices or ("Short",))[0],
                                            width=dg.DIAGRAMS[k].figsize[0], height=dg.DIAGRAMS[k].figsize[1]))
             for k in dg.CLASSIFICATION if counts[k]]
    if pages:
        st.download_button(
            f"Download all {len(pages)} classification diagrams (one PDF)",
            lambda: dg.render_pdf_pages(pages), file_name="mineralML_diagrams.pdf", mime="application/pdf",
            on_click="ignore", key="dg_dl_all", icon=":material/picture_as_pdf:",
            help="Every diagram that has analyses, with default settings, one per page.",
        )


def show_results(results, msgs, include_stoich, show_latent):
    for m in msgs:
        st.warning(m)

    # prep_df adds an empty Mineral column when the user gave no labels; hide it.
    if "Mineral" in results.columns and results["Mineral"].isna().all():
        results = results.drop(columns="Mineral")

    tabs = st.tabs(["Predictions", "Composition diagrams"] + (["Latent space"] if show_latent else []))
    with tabs[0]:
        show_predictions(results, include_stoich)
    with tabs[1]:
        show_diagrams(results, header=False)
    if show_latent:
        with tabs[2]:
            st.markdown("Your analyses (circles) projected onto the training data (faint crosses).")
            with st.spinner("Projecting..."):
                st.image(latent_png(results))


def show_predictions(results, include_stoich):
    st.subheader("Summary")
    n_class = int(results["Predict_Mineral"].notna().sum())
    low = int((results["Prediction_Score"] < 0.8).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Analyses", f"{len(results):,}")
    c2.metric("Classified", f"{n_class:,}")
    c3.metric("Prediction score < 0.8", f"{low:,}", help="Worth a second look: ambiguous or unusual compositions.")
    counts = results["Predict_Mineral"].fillna("Unclassified").value_counts()
    st.bar_chart(counts, horizontal=True, x_label="", y_label="Number of analyses")

    st.subheader("Predictions")
    show_cols = [c for c in SAMPLE_COLS + ["Mineral"] if c in results.columns] + [
        "Predict_Mineral", "Submineral", "Prediction_Score", "Prediction_Score_Sigma",
        "Second_Predict_Mineral", "Second_Prediction_Score",
    ]
    show_cols += [c for c in results.columns if c not in show_cols]
    st.dataframe(
        results[show_cols],
        column_config={
            "Prediction_Score": st.column_config.NumberColumn(format="%.3f"),
            "Prediction_Score_Sigma": st.column_config.NumberColumn(format="%.3f"),
            "Second_Prediction_Score": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button(
        "Download Excel",
        lambda: to_excel(results, include_stoich),  # built on click; large files take a while
        file_name="mineralML_predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
        help="One sheet with all rows, plus one sheet per predicted mineral (with stoichiometry appended, if selected).",
    )
    d2.download_button(
        "Download CSV", results.to_csv(index=False).encode(),
        file_name="mineralML_predictions.csv", mime="text/csv",
        help="The predictions table above only. No per-mineral sheets and no stoichiometry.",
    )
    # A CSV holds a single table, so the per-mineral sheets can only come in the Excel file.
    extra = "a sheet for each mineral with its stoichiometry" if include_stoich else "a sheet for each mineral"
    st.info(
        f"**Want the stoichiometry? Download the Excel file.** It has an \"All\" sheet plus {extra} "
        "(moles, cations, site assignments, end-members such as Fo or An). "
        "A CSV can only hold one table, so **the CSV contains just the predictions table above**, "
        "and opening it in Excel will not show the other sheets.",
        icon=":material/info:",
    )


# %% ---------------------------------------------------------------- 
# sidebar

with st.sidebar:
    st.header("What do you want to do?")
    mode = st.radio("What do you want to do?", list(MODES), captions=list(MODES.values()), key="mode",
                    label_visibility="collapsed")
    classify_on = mode == "Classify and plot"

    st.header("Input data")
    source = st.selectbox("How do you want to enter data?", SOURCES, key="source")
    uploaded = None
    if source == "Upload a file":
        uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        st.download_button(
            "Download template CSV", EXAMPLE.to_csv(index=False).encode(),
            file_name="mineralML_template.csv", mime="text/csv",
        )

    st.header("Options")
    convert_fe = st.checkbox("Convert FeO / Fe2O3 to FeOt", value=True)
    renormalize = st.checkbox("Renormalize to 100 wt%", value=False)
    drop_empty = st.checkbox("Drop rows with fewer than 2 oxides", value=True)
    include_stoich = show_latent = False
    if classify_on:
        include_stoich = st.checkbox("Add stoichiometry to Excel download", value=True,
                                     help="Appends cations, site assignments and end-members to each mineral "
                                          "sheet. Excel only: the CSV download never includes stoichiometry.")
        show_latent = st.checkbox("Show latent space plot", value=True)

    st.header("About")
    st.markdown(
        "[Documentation](https://mineralml.readthedocs.io) · "
        "[GitHub](https://github.com/sarahshi/mineralML) · "
        "[Paper](https://doi.org/10.31223/X53J2M)"
    )
    st.caption(
        f"mineralML v{mm.__version__}. Questions, bugs, ideas, or high-quality analyses "
        "to add to the training data? "
        "Email [sarahshi@berkeley.edu](mailto:sarahshi@berkeley.edu) or open a "
        "[GitHub issue](https://github.com/sarahshi/mineralML/issues)."
    )

# %% ----------------------------------------------------------------
# main

intro = st.container()  # intro and help text span the full page width
intro.title("mineralML")
intro.markdown(
    "Probabilistic classification of common igneous minerals from oxide compositions, "
    "with stoichiometry and crystallographic sites calculated for each classified analysis. "
    "Use it to label new EPMA or quantitative EDS data, or to catch misclassified phases and "
    "poor-quality analyses in existing compilations. Composition diagrams (TAS, feldspar and pyroxene "
    "ternaries, amphibole, oxides, and custom ternaries and Harker plots) can be drawn with or without "
    "classifying, and downloaded as PDF."
)
intro.markdown(
    "Working with quantitative EDS maps? This site handles point analyses. For maps, follow the "
    "[mapping notebook example](https://mineralml.readthedocs.io/en/latest/examples/mineralML_mapping.html), "
    "which runs the full workflow with `mm.run_map` (phase maps, proportions, prediction score maps, "
    "and interactive pixel, transect, and region tools). Try it with the "
    "[example maps](https://github.com/sarahshi/mineralML/tree/main/docs/examples/Maps)."
)

intro.subheader("How to use")
intro.markdown(
    """
1. In the menu at left, choose **Classify and plot**, or **Plot only** to skip classification
   (e.g. for glass or whole-rock data, or phases you have already identified).
2. Upload a CSV or Excel file, choose **Use the example dataset**, or **Type in analyses**.
3. Adjust the options if needed. The defaults suit most data.
4. Results appear below: predictions with Excel/CSV downloads, composition diagrams to download as
   PDF, SVG or PNG, and the latent space. Stoichiometry is in the **Excel** download only, not the CSV.
"""
)

with intro.expander("Input format"):
    st.markdown(
        "One analysis per row, with oxides in wt%: "
        "SiO₂, TiO₂, Al₂O₃, FeOₜ, MnO, MgO, CaO, Na₂O, K₂O, Cr₂O₃, P₂O₅ (and ZrO₂ for zircon).\n\n"
        "* Oxides that were not analyzed or not detected can be left blank; they are treated as 0.\n"
        "* Iron should be total FeO (FeOt). FeO, Fe₂O₃ or Fe₂O₃t columns are converted automatically "
        "(see Options).\n"
        "* A `Sample Name` column is optional but recommended. A `Mineral` column with your own label is "
        "optional and is shown next to the prediction for comparison.\n"
        "* Any other columns are carried through to the output unchanged."
    )
    st.dataframe(EXAMPLE, hide_index=True)
    st.download_button(
        "Download template CSV", EXAMPLE.to_csv(index=False).encode(),
        file_name="mineralML_template.csv", mime="text/csv", key="template_main",
    )

with intro.expander("Minerals classified"):
    st.markdown(
        "The neural network is trained on a curated dataset of 128k analyses of 23 mineral groups and glass:"
    )
    m1, m2, m3 = st.columns(3)
    m1.markdown(
        "* Amphibole\n* Apatite\n* Biotite\n* Carbonate (calcite)\n* Chlorite\n* Epidote\n"
        "* Feldspar: Plagioclase, Alkali_Feldspar\n* Garnet"
    )
    m2.markdown(
        "* Glass\n* Kalsilite\n* Leucite\n* Melilite\n* Muscovite\n* Nepheline\n* Olivine\n"
        "* Oxide: Spinel_Group, Rhombohedral_Oxides"
    )
    m3.markdown(
        "* Pyroxene: Clinopyroxene, Orthopyroxene, Na-Pyroxene\n* Rutile\n* Serpentine\n"
        "* SiO₂ polymorphs\n* Titanite\n* Tourmaline\n* Zircon"
    )

with intro.expander("Reading the results"):
    st.markdown(
        """
* **Predict_Mineral**: the most likely mineral. **Submineral** refines pyroxenes (e.g. Augite),
  feldspars (e.g. Labradorite) and oxides (Spinel_Group or Rhombohedral_Oxides).
* **Prediction_Score**: the probability of the predicted mineral, averaged over 50 Monte Carlo passes.
  **Prediction_Score_Sigma** is its standard deviation across passes.
* **Second_Predict_Mineral** and **Second_Prediction_Score**: the next most likely mineral.
  A low score, or a close second, flags ambiguous compositions, mixed analyses or poor-quality data.
* **Excel download**: an "All" sheet, plus one sheet per mineral with stoichiometry appended
  (moles, cations, site assignments, end-members such as Fo or An).
* **CSV download**: the predictions table only. A CSV file holds a single table, so it has no
  per-mineral sheets and **no stoichiometry**. Download the Excel file if you need either.
* **Latent space**: your analyses projected onto the training data. Points far from any cluster
  are unusual for their assigned mineral.
"""
    )

with intro.expander("How to cite"):
    st.markdown("If you use mineralML in your work, please cite:")
    st.code(
        "Shi, S., Wieser, P., Gordon, C., Toth, N., Antoshechkina, P., Gleeson, M., & Lehnert, K. (2026). "
        "mineralML: Leveraging Machine Learning for Probabilistic Mineral Classification. "
        "EarthArXiv. doi:10.31223/X53J2M",
        language=None, wrap_lines=True,
    )
    st.markdown("BibTeX:")
    st.code(
        "@article{Shietal2026,\n"
        "  doi     = {10.31223/X53J2M},\n"
        "  url     = {https://doi.org/10.31223/X53J2M},\n"
        "  year    = {2026},\n"
        "  author  = {Shi, Sarah C and Wieser, Penny E and Gordon, Charlotte and Toth, Norbert and "
        "Antoshechkina, Paula M and Gleeson, Matthew LM and Lehnert, Kerstin},\n"
        "  title   = {mineralML: Leveraging Machine Learning for Probabilistic Mineral Classification},\n"
        "  journal = {Earth ArXiv},\n"
        "}",
        language="bibtex",
    )

df_in = None
if source == "Upload a file":
    if uploaded is None:
        st.info("Upload a CSV or Excel file in the menu at left to get started. See **Input format** above.")
        st.button("Or try the example dataset", on_click=use_example, icon=":material/science:")
    else:
        try:
            df_in = read_upload(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
elif source == "Use the example dataset":
    try:
        df_in = load_example()
    except OSError as e:
        st.error(f"Could not load the example dataset: {e}")
        st.stop()
    st.header("Example dataset")
    st.markdown(
        f"{len(df_in):,} analyses: 100 each of {df_in['Mineral'].nunique()} minerals from the mineralML "
        "training data, with their published labels (`Mineral`) and sources (`Source`). Because these "
        "analyses were used to train the model, nearly all are classified correctly; your own data will "
        "show more spread in prediction scores. Download it to see the expected input format."
    )
    st.download_button(
        "Download example CSV", lambda: df_in.to_csv(index=False).encode(),
        file_name="mineralML_example.csv", mime="text/csv", on_click="ignore", icon=":material/download:",
    )
else:
    st.header("Your analyses")
    st.markdown("Edit the table or paste rows from a spreadsheet. Add rows with the + at the bottom.")
    df_in = st.data_editor(EXAMPLE, num_rows="dynamic", key="editor").dropna(how="all")

if df_in is not None:
    if df_in.empty:
        st.error("Enter at least one analysis.")
        st.stop()
    if len(df_in) > MAX_ROWS:
        st.error(f"{len(df_in):,} rows exceeds the {MAX_ROWS:,}-row web limit. "
                 "Use `pip install mineralML` for larger datasets.")
        st.stop()
    if not any(c in df_in.columns for c in mm.OXIDES + ["FeO", "Fe2O3", "Fe2O3t"]):
        st.error(f"No oxide columns found. Columns in your file: {list(df_in.columns)}")
        st.stop()
    try:
        with st.spinner(f"Classifying {len(df_in):,} analyses..." if classify_on else "Reading analyses..."):
            res, msgs = classify(df_in, convert_fe, renormalize, drop_empty, predict=classify_on)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    if classify_on:
        show_results(res, msgs, include_stoich, show_latent)
    else:
        for m in msgs:
            st.warning(m)
        if res["Mineral"].isna().all():
            res = res.drop(columns="Mineral")
        st.caption(f"{len(res):,} analyses loaded, not classified. For predictions and prediction scores, "
                   "choose **Classify and plot** in the menu at left.")
        show_diagrams(res)
