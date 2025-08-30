# %%

import numpy as np
import pandas as pd
import warnings
import seaborn as sns

import matplotlib
from matplotlib import pyplot as plt

# %%


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

    if savefig != None:
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

        per_ok_s = ["%.1f%%" % (per_ok), "100%"][per_ok == 100]

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
        lis_txt = ["%d" % (cell_val), per_ok_s, "%.1f%%" % (per_err)]
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
