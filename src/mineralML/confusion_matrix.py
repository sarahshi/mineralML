# %%

__author__ = "Sarah Shi"

import re
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

__all__ = ["MINERAL_LABELS", "PARENT_LABELS", "LABEL_ALIASES", "harmonize_labels",
           "confusion_matrix_df", "pp_matrix", "insert_totals", "config_cell_text_and_colors"]

# %%

# Labels that given and predicted minerals are compared at in classification
# reports and confusion matrices. These are the Predict_Mineral labels, with
# oxides split into their Submineral group (Rhombohedral_Oxides, Spinel_Group).
MINERAL_LABELS = [
    "Alkali_Feldspar",
    "Amphibole",
    "Apatite",
    "Biotite",
    "Carbonate",
    "Chlorite",
    "Clinopyroxene",
    "Epidote",
    "Garnet",
    "Glass",
    "Kalsilite",
    "Leucite",
    "Melilite",
    "Muscovite",
    "Nepheline",
    "Olivine",
    "Orthopyroxene",
    "Plagioclase",
    "Rhombohedral_Oxides",
    "Rutile",
    "Serpentine",
    "SiO2_Polymorph",
    "Spinel_Group",
    "Titanite",
    "Tourmaline",
    "Zircon",
]

# Parent groups: when either array uses the parent label, its children are
# merged into the parent so both arrays are compared at the same level.
PARENT_LABELS = {
    "Feldspar": ("Alkali_Feldspar", "Plagioclase"),
    "Pyroxene": ("Clinopyroxene", "Orthopyroxene"),
    "Oxide": ("Rhombohedral_Oxides", "Spinel_Group"),
}

# Other names that map onto each label in MINERAL_LABELS (or a parent label),
# in alphabetical order. Each list starts with the names mineralML's own
# classifiers return (Submineral, AmphiboleClassifier, OxideClassifier), so
# they roll up to their Predict_Mineral, then species, varieties, and
# synonyms, including the MINERAL names used in GEOROC. Matching ignores case
# and punctuation (see _label_key), so "Magnesio-Hornblende" matches
# "Magnesiohornblende" and "(Al)kalifeldspar" matches "Alkali_Feldspar".
# Any label containing "spinel" also maps to Spinel_Group.
LABEL_ALIASES = {
    "Alkali_Feldspar": [
        "Sanidine", "Anorthoclase",                          # Submineral
        "K-Feldspar", "Orthoclase", "Microcline", "Adularia", "Perthite",
    ],
    "Amphibole": [
        "Tremolite", "Actinolite", "Ferroactinolite",        # AmphiboleClassifier
        "Magnesiohornblende", "Ferrohornblende", "Tschermakite",
        "Ferrotschermakite",
        "Hornblende", "Pargasite", "Edenite", "Hastingsite",  # calcic
        "Magnesiohastingsite", "Kaersutite",
        "Richterite", "Ferrorichterite", "Katophorite",      # sodic-calcic
        "Arfvedsonite", "Riebeckite", "Glaucophane",         # sodic
        "Cummingtonite", "Anthophyllite",                    # Mg-Fe
    ],
    "Apatite": ["Fluorapatite", "Chlorapatite", "Hydroxylapatite"],
    "Biotite": ["Phlogopite", "Annite", "Siderophyllite", "Eastonite"],
    "Carbonate": [
        "Calcite", "Aragonite", "Dolomite", "Ankerite", "Magnesite", "Siderite",
        "Rhodochrosite", "Strontianite", "Witherite",
    ],
    "Clinopyroxene": [
        "Augite", "Diopside", "Hedenbergite", "Pigeonite",   # Submineral
        "Wollastonite", "Na-Pyroxene", "Jadeite", "Aegirine", "Aegirine-Augite",
        "Omphacite", "Ca-Mg-Fe Pyroxene",
        "Salite", "Cr-Diopside", "Titanaugite", "Ferroaugite",
        "Ferrohedenbergite",
    ],
    "Garnet": [                                              # end-members
        "Almandine", "Pyrope", "Spessartine", "Grossular", "Andradite",
    ],
    "Muscovite": ["Phengite", "Sericite"],
    "Olivine": ["Forsterite", "Fayalite"],
    "Orthopyroxene": [
        "Enstatite", "Ferrosilite",                          # Submineral
        "Hypersthene", "Ferrohypersthene",
    ],
    "Oxide": ["Oxides", "Fe-Ti Oxide", "Fe-Ti Oxides"],
    "Plagioclase": [                                         # Submineral
        "Albite", "Oligoclase", "Andesine", "Labradorite", "Bytownite",
        "Anorthite",
    ],
    "Rhombohedral_Oxides": [
        "Rhombohedral_Oxide", "Hematite", "Ilmenite",        # Submineral
        "Titanohematite", "Hemoilmenite", "Ilmenite-Hematite",
    ],
    "SiO2_Polymorph": [
        "SiO2", "Quartz", "Tridymite", "Cristobalite", "Coesite", "Stishovite",
        "Chalcedony", "Agate",
    ],
    "Spinel_Group": [
        "Magnetite", "Al-Magnetite", "Hercynite", "Pleonaste",  # OxideClassifier
        "Ferrian-Pleonaste", "Ferrian-Picotite", "Magnesioferrite",
        "Titanomagnetite", "Ti-Magnetite", "Ulvospinel", "Chromite",
        "Fe-Chromite", "Picotite",
    ],
    "Titanite": ["Sphene", "Titanite (Sphene)"],
}


def _label_key(label):
    """Normalizes a label for lookup: lowercase, with punctuation and spaces removed."""
    return re.sub(r"[^a-z0-9]+", "", str(label).strip().lower())


_LABEL_LOOKUP = {_label_key(m): m for m in MINERAL_LABELS + list(PARENT_LABELS)}
for _canonical, _aliases in LABEL_ALIASES.items():
    for _alias in _aliases:
        _LABEL_LOOKUP[_label_key(_alias)] = _canonical


def _map_label(x):
    if pd.isna(x):
        return x
    key = _label_key(x)
    if key in _LABEL_LOOKUP:
        return _LABEL_LOOKUP[key]
    if "spinel" in key:
        return "Spinel_Group"
    return x


def _resolve_oxide(pred, pred_submineral):
    """Replaces 'Oxide' predictions with their oxide group from Submineral."""
    if pred_submineral is None:
        return pred
    sub = pd.Series(np.asarray(pred_submineral, dtype=object), index=pred.index)
    is_oxide = pred.map(lambda x: not pd.isna(x) and _label_key(x) == "oxide")
    return pred.where(~is_oxide | sub.isna(), sub)


def _harmonize(given, pred):
    """Maps labels and applies parent merges. Unrecognized labels are kept as is."""
    given = given.map(_map_label)
    pred = pred.map(_map_label)

    all_labels = set(given) | set(pred)
    for parent, children in PARENT_LABELS.items():
        if parent in all_labels:
            def _merge_parent(x, _p=parent, _c=children):
                if pd.isna(x):
                    return x
                return _p if x in _c else x
            given = given.map(_merge_parent)
            pred = pred.map(_merge_parent)

    return given, pred


def harmonize_labels(given_min, pred_min=None, pred_submineral=None):
    """

    Maps mineral labels onto the names mineralML returns in 'Predict_Mineral',
    so that published and predicted labels can be compared directly (e.g., with
    sklearn's classification_report). Matching ignores case, spaces, and
    punctuation, so GEOROC names such as "TITANO-MAGNETITE" and
    "(AL)KALIFELDSPAR" match. For example, Hematite and Ilmenite map to
    "Rhombohedral_Oxides"; Magnetite and Spinel map to "Spinel_Group"; Quartz
    and Tridymite map to "SiO2_Polymorph"; Calcite maps to "Carbonate";
    Augite maps to "Clinopyroxene"; Hornblende maps to "Amphibole"; and
    Phlogopite maps to "Biotite". See LABEL_ALIASES for the full list.

    predict_class_prob returns oxides as "Oxide" in 'Predict_Mineral', with the
    oxide group in 'Submineral'. Pass the 'Submineral' column as
    pred_submineral to compare oxides by group.

    When "Feldspar", "Pyroxene", or "Oxide" is present in either array, child
    labels (e.g., "Alkali_Feldspar", "Plagioclase") are merged into the parent
    label in both arrays.

    Labels that do not match MINERAL_LABELS after mapping are returned
    unchanged and trigger a UserWarning. NaN values are kept.

    Parameters:
        given_min (array-like): The given (e.g., published) mineral labels.
        pred_min (array-like, optional): The predicted mineral labels, such as
            the 'Predict_Mineral' column returned by predict_class_prob.
        pred_submineral (array-like, optional): The predicted 'Submineral'
            column. Where pred_min is "Oxide", the oxide group in
            pred_submineral is used instead.

    Returns:
        published_harmonized (pd.Series): given_min mapped onto the
            'Predict_Mineral' names. If pred_min is provided, returns a tuple
            (published_harmonized, predict_mineral_harmonized), where
            predict_mineral_harmonized is pred_min mapped the same way, with
            "Oxide" replaced by its pred_submineral group where given.

    Example:
        published_harmonized, predict_mineral_harmonized = mm.harmonize_labels(
            df_pred['Mineral'], df_pred['Predict_Mineral'],
            pred_submineral=df_pred['Submineral'])

    """

    given = pd.Series(given_min)
    pred = pd.Series(pred_min) if pred_min is not None else pd.Series(dtype=object)
    pred = _resolve_oxide(pred, pred_submineral)

    given, pred = _harmonize(given, pred)

    known = set(MINERAL_LABELS) | set(PARENT_LABELS)
    labels = pd.concat([given, pred]).dropna()
    unrecognized = sorted(set(labels) - known, key=str)
    if unrecognized:
        warnings.warn(
            f"Unrecognized label(s) not in the canonical mineral list: "
            f"{unrecognized}. These labels are returned unchanged.",
            UserWarning,
            stacklevel=2,
        )

    if pred_min is None:
        return given
    return given, pred


def confusion_matrix_df(given_min, pred_min, pred_submineral=None):
    """

    Constructs a confusion matrix as a pandas DataFrame for easy visualization and
    analysis. Labels are first mapped onto the names mineralML returns in
    'Predict_Mineral' (see harmonize_labels), so that, e.g., Hematite is
    counted as "Rhombohedral_Oxides", Magnetite as "Spinel_Group", and
    Tridymite as "SiO2_Polymorph". Then,
    it uses these mappings to construct the confusion matrix, which compares
    the given and predicted classes.

    When parent labels such as "Feldspar", "Pyroxene", or "Oxide" are present in either
    the given or predicted arrays, child labels (e.g., "Alkali_Feldspar",
    "Plagioclase") are automatically merged into the parent label so the
    confusion matrix dimensions remain consistent.

    Labels that do not match any entry in the canonical mineral list after
    all merges are applied will trigger a UserWarning and the corresponding
    rows will be excluded from the confusion matrix.

    Parameters:
        given_min (array-like): The true class labels.
        pred_min (array-like): The predicted class labels.
        pred_submineral (array-like, optional): The predicted 'Submineral'
            column. Where pred_min is "Oxide", the oxide group in
            pred_submineral is used instead.

    Returns:
        cm_df (DataFrame): A DataFrame representing the confusion matrix, with rows
                           and columns labeled by the unique mineral names found in
                           the given and predicted class arrays.

    """

    minerals = MINERAL_LABELS
    parent_map = PARENT_LABELS

    given = pd.Series(given_min)
    pred = _resolve_oxide(pd.Series(pred_min), pred_submineral)

    given_nans = given.isna().sum()
    pred_nans = pred.isna().sum()
    if given_nans > 0 or pred_nans > 0:
        warnings.warn(
            f"Missing data detected: {given_nans} NaN(s) in given_min, "
            f"{pred_nans} NaN(s) in pred_min. "
            f"These rows will be excluded from the confusion matrix.",
            UserWarning,
            stacklevel=2,
        )
        mask = given.notna() & pred.notna()
        given = given[mask]
        pred = pred[mask]

    # --- Map aliases and merge children into parent labels ---
    given, pred = _harmonize(given, pred)
    all_labels = set(given) | set(pred)

    # Build label list, swapping children for parent where needed
    active_minerals = []
    for m in minerals:
        # Skip children that were merged into a parent
        skip = False
        for parent, children in parent_map.items():
            if m in children and parent in all_labels:
                skip = True
                break
        if not skip:
            active_minerals.append(m)
 
    # Insert parent labels at the position of their first child
    for parent, children in parent_map.items():
        if parent in all_labels and parent not in active_minerals:
            idx = next(
                (i for i, m in enumerate(minerals) if m in children),
                len(active_minerals),
            )
            # Translate the index in `minerals` to the corresponding
            # position in `active_minerals`
            insert_pos = len(active_minerals)
            for i, m in enumerate(active_minerals):
                if minerals.index(m) >= idx:
                    insert_pos = i
                    break
            active_minerals.insert(insert_pos, parent)
 
    # --- Warn and drop labels not in active_minerals ---
    active_set = set(active_minerals)
    post_merge_labels = set(given) | set(pred)
    unrecognized = sorted(post_merge_labels - active_set)
 
    if unrecognized:
        warnings.warn(
            f"Unrecognized label(s) not in the canonical mineral list: "
            f"{unrecognized}. These rows will be excluded from the "
            f"confusion matrix.",
            UserWarning,
            stacklevel=2,
        )
        mask = given.isin(active_set) & pred.isin(active_set)
        given = given[mask]
        pred = pred[mask]
 
    # Build the confusion matrix
    cm_matrix = confusion_matrix(given, pred, labels=active_minerals)
    cm_df = pd.DataFrame(cm_matrix, index=active_minerals, columns=active_minerals)
 
    return cm_df


def pp_matrix(
    df_cm,
    annot=True,
    cmap="BuGn",
    fmt=".2f",
    fz=12,
    lw=0.5,
    cbar=False,
    figsize=[14, 14],
    show_null_values=0,
    pred_val_axis="x",
    savefig=None
):
    """

    Creates and displays a confusion matrix visualization using Seaborn's heatmap function.

    Parameters:
        df_cm (pd.DataFrame): DataFrame containing the confusion matrix without totals.
        annot (bool, optional): If True, display the text in each cell. Default is True.
        cmap (str, optional): Color map for the heatmap. Default is 'BuGn'.
        fmt (str, optional): String format for annotating. Default is '.2f'.
        fz (int, optional): Font size for text annotations. Default is 12.
        lw (float, optional): Line width for cell borders. Default is 0.5.
        cbar (bool, optional): If True, display the color bar. Default is False.
        figsize (list, optional): Figure size. Default is [10.5, 10.5].
        show_null_values (int, optional): Show null values, 0 or 1. Default is 0.
        pred_val_axis (str, optional): Axis to show prediction values ('x' or 'y'). Default is 'x'.
        savefig (str, optional): If provided, saves the plot to the specified path with a '.pdf' extension.

    Returns:
        None. The function creates and displays the heatmap of the confusion matrix.

    Note:
        The function modifies the input DataFrame to include total counts and adjusts text and color configurations.
        The source of the original code is from: 
        https://github.com/wcipriano/pretty-print-confusion-matrix/blob/master/pretty_confusion_matrix/pretty_confusion_matrix.py\
        
    """

    from matplotlib.collections import QuadMesh

    if pred_val_axis in ("col", "x"):
        xlbl = "Predicted"
        ylbl = "Published [True]"
    else:
        xlbl = "Published [True]"
        ylbl = "Predicted"
        df_cm = df_cm.T

    # create "Total" column
    df_cm = df_cm.copy()
    insert_totals(df_cm)
    df_cm = df_cm.astype(int)

    fig1 = plt.figure("Conf matrix default", figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot

    ax = sns.heatmap(
        df_cm,
        annot=annot,
        annot_kws={"size": fz},
        linewidths=lw,
        ax=ax1,
        cbar=cbar,
        cmap=cmap,
        linecolor="w",
        fmt=fmt,
    )

    # force one tick per column
    n_cols = df_cm.shape[1]
    ax.set_xticks(np.arange(n_cols) + 0.5)                 # centre ticks in each cell
    ax.set_xticklabels(df_cm.columns, rotation=45,         # use all column names
                       fontsize=13, ha="right")

    # force one tick per row
    n_rows = df_cm.shape[0]
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(df_cm.index, rotation=35,          # use all index names
                       fontsize=13, va="top")

    # # set ticklabels rotation
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=13, ha="right")
    # ax.set_yticklabels(ax.get_yticklabels(), rotation=35, fontsize=13, va="top")

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = []
    text_del = []
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        posi += 1

        # set text
        txt_res = config_cell_text_and_colors(
            array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values
        )

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item["x"], item["y"], item["text"], **item["kw"])

    # titles and legends
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  # set layout slim

    if savefig:
        plt.savefig(savefig + '.pdf')


def insert_totals(df_cm):
    """

    Inserts total sums for each row and column into the confusion matrix DataFrame.

    This function adds a 'sum_row' column and a 'sum_col' row to the DataFrame, representing
    the total counts across each row and column, respectively. It also sets the bottom-right
    cell to the grand total.

    Parameters:
        df_cm (pd.DataFrame): DataFrame representing the confusion matrix.

    Returns:
        None: The function modifies the DataFrame in place.

    Note:
        If 'sum_row' or 'sum_col' already exist in the DataFrame, they will be recalculated.

    """

    # Check if 'sum_row' and 'sum_col' already exist and remove them if they do
    if "sum_row" in df_cm.columns:
        df_cm.drop("sum_row", axis=1, inplace=True)
    if "sum_col" in df_cm.index:
        df_cm.drop("sum_col", axis=0, inplace=True)

    # Calculate the sum of each column to create 'sum_row'
    sum_col = df_cm.sum(axis=0).astype(int)  # sum columns
    sum_lin = df_cm.sum(axis=1).astype(int)  # sum rows

    # Add 'sum_row' and 'sum_col' to the dataframe
    df_cm["sum_row"] = sum_lin
    df_cm.loc["sum_col"] = sum_col
    df_cm.at[
        "sum_col", "sum_row"
    ] = sum_lin.sum()  # Set the bottom right cell to the grand total


def config_cell_text_and_colors(
    array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0
):
    """

    Configures cell text and colors for confusion matrix visualization.

    Adjusts the text and background colors of cells in the confusion matrix based on their values.
    Totals and percentages are calculated for the last row and column cells.

    Parameters:
        array_df (np.ndarray): 2D numpy array of the confusion matrix.
        lin (int): Row index of the cell to configure.
        col (int): Column index of the cell to configure.
        oText (matplotlib.text.Text): Text object of the cell.
        facecolors (np.ndarray): Array of facecolors for the cells.
        posi (int): Position index in the flattened array of cells.
        fz (int): Font size for cell text.
        fmt (str): Format string for cell text.
        show_null_values (int, optional): Flag to show null values. Default is 0.

    Returns:
        tuple: A tuple containing two lists: text elements to add and to delete.

    Note:
        The function modifies text and background colors based on the value in each cell.

    """

    import matplotlib.font_manager as fm

    text_add = []
    text_del = []
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if cell_val != 0:
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif col == ccl - 1:
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif lin == ccl - 1:
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = "100%" if per_ok == 100 else f"{per_ok:.1f}%"

        # text to DEL
        text_del.append(oText)

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # text to ADD
        font_prop = fm.FontProperties(weight="bold", size=fz)
        text_kwargs = dict(
            color="k",
            ha="center",
            va="center",
            gid="sum",
            fontproperties=font_prop,
        )
        lis_txt = [f"{int(cell_val)}", per_ok_s, f"{per_err:.1f}%"]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic["color"] = "g"
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic["color"] = "r"
        lis_kwa.append(dic)
        lis_pos = [
            (oText._x, oText._y - 0.3),
            (oText._x, oText._y),
            (oText._x, oText._y + 0.3),
        ]
        for i in range(len(lis_txt)):
            newText = dict(
                x=lis_pos[i][0],
                y=lis_pos[i][1],
                text=lis_txt[i],
                kw=lis_kwa[i],
            )
            text_add.append(newText)

        # set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if per > 0:
            txt = "%s\n%.1f%%" % (cell_val, per)
        else:
            if show_null_values == 0:
                txt = ""
            elif show_null_values == 1:
                txt = "0"
            else:
                txt = "0\n0.0%"
        oText.set_text(txt)

        # main diagonal
        if col == lin:
            # set color of the textin the diagonal to white
            oText.set_color("k")
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color("r")

    return text_add, text_del
