""" mineralML web app: upload oxide compositions, get mineral classifications. // @author: Sarah Shi """

import io
import warnings

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import streamlit as st
import mineralML as mm

st.set_page_config(page_title="mineralML", page_icon="🌋", initial_sidebar_state=200)  # sidebar width, px (default 300)
st.set_page_config(initial_sidebar_state="expanded")  # additive: keeps the width, and opens the sidebar on every screen size

OXIDES = mm.OXIDES + ["ZrO2"]
SAMPLE_COLS = ["SampleID", "Sample", "Sample Name", "Sample ID"]
MAX_ROWS = 200_000

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


@st.cache_data(show_spinner=False, max_entries=8)
def classify(df, convert_fe, renormalize, drop_empty_rows):
    """Runs prep_df + predict_class_prob, returning results and any warnings raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepped = mm.prep_df(
            df.copy(), convert_fe=convert_fe, renormalize=renormalize,
            drop_empty_rows=drop_empty_rows, verbose=False,
        )
        results = mm.predict_class_prob(prepped, verbose=False)
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


def show_results(results, msgs, include_stoich, show_latent):
    for m in msgs:
        st.warning(m)

    # prep_df adds an empty Mineral column when the user gave no labels; hide it.
    if "Mineral" in results.columns and results["Mineral"].isna().all():
        results = results.drop(columns="Mineral")

    st.header("Summary")
    n_class = int(results["Predict_Mineral"].notna().sum())
    low = int((results["Prediction_Score"] < 0.8).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Analyses", f"{len(results):,}")
    c2.metric("Classified", f"{n_class:,}")
    c3.metric("Prediction score < 0.8", f"{low:,}", help="Worth a second look: ambiguous or unusual compositions.")
    counts = results["Predict_Mineral"].fillna("Unclassified").value_counts()
    st.bar_chart(counts, horizontal=True, x_label="", y_label="Number of analyses")

    st.header("Predictions")
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
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="One sheet with all rows, plus one sheet per predicted mineral (with stoichiometry appended, if selected).",
    )
    d2.download_button(
        "Download CSV", results.to_csv(index=False).encode(),
        file_name="mineralML_predictions.csv", mime="text/csv",
    )

    if show_latent:
        st.header("Latent space")
        st.markdown("Your analyses (circles) projected onto the training data (faint crosses).")
        with st.spinner("Projecting..."):
            st.image(latent_png(results))


# %% ---------------------------------------------------------------- 
# sidebar

with st.sidebar:
    st.header("Input data")
    source = st.selectbox("How do you want to enter data?", ["Upload a file", "Type in analyses"])
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
    include_stoich = st.checkbox("Add stoichiometry to Excel download", value=True,
                                 help="Appends cations, site assignments and end-members to each mineral sheet.")
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

st.title("mineralML")
st.markdown(
    "Probabilistic classification of common igneous minerals from oxide compositions, "
    "with stoichiometry and crystallographic sites calculated for each classified analysis. "
    "Use it to label new EPMA or quantitative EDS data, or to catch misclassified phases and "
    "poor-quality analyses in existing compilations."
)
st.markdown(
    "Working with quantitative EDS maps? This site handles point analyses. For maps, follow the "
    "[mapping notebook example](https://mineralml.readthedocs.io/en/latest/examples/mineralML_mapping.html), "
    "which runs the full workflow with `mm.run_map` (phase maps, proportions, prediction score maps, "
    "and interactive pixel, transect, and region tools). Try it with the "
    "[example maps](https://github.com/sarahshi/mineralML/tree/main/docs/examples/Maps)."
)

st.subheader("How to use")
st.markdown(
    """
1. In the menu at left, upload a CSV or Excel file, or choose **Type in analyses**.
2. Adjust the options if needed. The defaults suit most data.
3. Predictions, prediction scores, and Excel/CSV downloads appear below.
"""
)

with st.expander("Input format"):
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

with st.expander("Minerals classified"):
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

with st.expander("Reading the results"):
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
* **Latent space**: your analyses projected onto the training data. Points far from any cluster
  are unusual for their assigned mineral.
"""
    )

with st.expander("How to cite"):
    st.markdown("If you use mineralML in your work, please cite:")
    st.code(
        "Shi, S., Wieser, P., Gordon, C., Toth, N., Antoshechkina, P., Gleeson, M., & Lehnert, K. (2026). "
        "mineralML: Leveraging Machine Learning for Probabilistic Mineral Classification. "
        "EarthArXiv. doi:10.31223/X53J2M",
        language=None, wrap_lines=True,
    )

df_in = None
if source == "Upload a file":
    if uploaded is None:
        st.info("Upload a CSV or Excel file in the menu at left to get started. See **Input format** above.")
    else:
        try:
            df_in = read_upload(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
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
        with st.spinner(f"Classifying {len(df_in):,} analyses..."):
            res, msgs = classify(df_in, convert_fe, renormalize, drop_empty)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    show_results(res, msgs, include_stoich, show_latent)
