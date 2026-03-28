# %% 

import os, re, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit

sys.path.append('../src')
import mineralML as mm

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.rcParams.update({
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'pdf.fonttype': 42,
    'font.family': 'Avenir',
    'font.size': 20,
    'xtick.direction': 'in',  # Set x-tick direction to 'in'
    'ytick.direction': 'in',  # Set y-tick direction to 'in'
    'xtick.major.size': 5,    # Set x-tick length
    'ytick.major.size': 5,    # Set y-tick length
    'xtick.major.pad': 6.5,   # Set x-tick padding
    'ytick.major.pad': 6.5    # Set y-tick padding
})
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['image.composite_image'] = False


# %% 

sorted_df=pd.read_excel('MH0811b_EDS_SpotAnalysesforUncertainties.xlsx')

def extract_time(sample_name):
    # Match number followed by either 's' or 'ms'
    match = re.search(r'_(\d*\.?\d+)(ms|s)_', sample_name)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        
        if unit == 'ms':
            return value / 1000  # convert milliseconds to seconds
        else:
            return value  # already in seconds
    return np.nan  # if no match found

sorted_df['time'] = sorted_df['Sample Name'].apply(extract_time)
sorted_df['time'].unique()
oxides = ['SiO2', 'MgO', 'CaO']
for oxide in oxides:
    sorted_df[f'{oxide}_err'] = 100 * (
        sorted_df[f'Oxide % Sigma_{oxide}'] / sorted_df[oxide]
)

# %% 

## The original EBSD geometry that produced the 'bad' map: 
# Flat at 21na and 25 mm WD
Flat_21nA_25mm=(sorted_df['Sample Name'].str.contains('21nA'))& (sorted_df['Sample Name'].str.contains('Flat'))
# New acquisition conditions: 
# Tilted at 21nA and 25 mm WD
Tilted_21nA_25mm=(sorted_df['Sample Name'].str.contains('21nA'))& (sorted_df['Sample Name'].str.contains('Tilted'))
Tilted_7nA_12mm=(sorted_df['Sample Name'].str.contains('6.8nA'))& (sorted_df['Sample Name'].str.contains('Tilted'))

# Tilted_0.01s_Opx_12mm_6.8nA. Need to get this - this is the most important
# using the EBSD mode to allow us to get to lower dwell times
# Using The nice EBSD settings
EBSD_Mode_12mm_6nA_2ms=(sorted_df['Sample Name'].str.contains('EBSDMode'))& (sorted_df['Sample Name'].str.contains('2.56ms'))& (sorted_df['Sample Name'].str.contains('12mm'))
EBSD_Mode_12mm_6nA_7ms=(sorted_df['Sample Name'].str.contains('EBSDMode'))& (sorted_df['Sample Name'].str.contains('7.65ms'))& (sorted_df['Sample Name'].str.contains('12mm'))
EBSD_Mode_12mm_6nA_15ms=(sorted_df['Sample Name'].str.contains('EBSDMode'))& (sorted_df['Sample Name'].str.contains('15.3ms'))& (sorted_df['Sample Name'].str.contains('12mm'))

## Same for 25 mm EBSDMode_2.56ms_25mm_20.5nA
EBSD_Mode_25mm_6nA_2ms=(sorted_df['Sample Name'].str.contains('EBSDMode'))& (sorted_df['Sample Name'].str.contains('2.56ms'))& (sorted_df['Sample Name'].str.contains('25mm')&(sorted_df['Sample Name'].str.contains('6.8nA')))
EBSD_Mode_25mm_6nA_7ms=(sorted_df['Sample Name'].str.contains('EBSDMode'))& (sorted_df['Sample Name'].str.contains('7.65ms'))& (sorted_df['Sample Name'].str.contains('25mm')&(sorted_df['Sample Name'].str.contains('6.8nA')))
EBSD_Mode_25mm_6nA_15ms=(sorted_df['Sample Name'].str.contains('EBSDMode'))& (sorted_df['Sample Name'].str.contains('15.3ms'))& (sorted_df['Sample Name'].str.contains('25mm')&(sorted_df['Sample Name'].str.contains('6.8nA')))

EBSD_Mode_25mm_21nA_2ms=(sorted_df['Sample Name'].str.contains('EBSDMode'))& (sorted_df['Sample Name'].str.contains('2.56ms'))& (sorted_df['Sample Name'].str.contains('25mm')&(sorted_df['Sample Name'].str.contains('20.5nA')))
EBSD_Mode_25mm_21nA_7ms=(sorted_df['Sample Name'].str.contains('EBSDMode'))& (sorted_df['Sample Name'].str.contains('7.65ms'))& (sorted_df['Sample Name'].str.contains('25mm')&(sorted_df['Sample Name'].str.contains('20.5nA')))
EBSD_Mode_25mm_21nA_15ms=(sorted_df['Sample Name'].str.contains('EBSDMode'))& (sorted_df['Sample Name'].str.contains('15.3ms'))& (sorted_df['Sample Name'].str.contains('25mm')&(sorted_df['Sample Name'].str.contains('20.5nA')))

# Tilted and 7ms and 12 mm
Tilted_12mm_6nA=((sorted_df['Sample Name'].str.contains('EBSDMode'))&  (sorted_df['Sample Name'].str.contains('12mm')))|((sorted_df['Sample Name'].str.contains('6.8nA'))& (sorted_df['Sample Name'].str.contains('Tilted')))

# Tilted and 25 mm and 15 mm
Tilted_25mm_20nA=((sorted_df['Sample Name'].str.contains('EBSDMode'))&  (sorted_df['Sample Name'].str.contains('25mm')))|((sorted_df['Sample Name'].str.contains('21nA'))& (sorted_df['Sample Name'].str.contains('Tilted')))

# EBSDMode_15.3ms_12mm_6.8nA
# EBSDMode_2.56ms_25mm_20.5nA
df_tilted_7nA_12mm=sorted_df[Tilted_12mm_6nA]

df_tilted_20nA_25mm=sorted_df[Tilted_25mm_20nA]
# Convert live time into total acquisition time
dt=0.34
df_tilted_7nA_12mm['Total_time'] = df_tilted_7nA_12mm['time']

mask = ~df_tilted_7nA_12mm['Sample Name'].str.contains(
    'EBSDMode', case=False, na=False
)

df_tilted_7nA_12mm.loc[mask, 'Total_time'] = (
    df_tilted_7nA_12mm.loc[mask, 'time'] / (1 - dt)
)

dt=0.52
df_tilted_20nA_25mm['Total_time'] = df_tilted_20nA_25mm['time']

mask = ~df_tilted_20nA_25mm['Sample Name'].str.contains(
    'EBSDMode', case=False, na=False
)
df_tilted_20nA_25mm.loc[mask, 'Total_time'] = (
    df_tilted_20nA_25mm.loc[mask, 'time'] / (1 - dt)
)

# %% 

t_test = np.logspace(-3, 10, 100)  # from 0.001s to 10s

# y = sqrt(A/t + B) - added noise floor cos was a bad fit otherwise. 
def time_model_with_floor(t, A, B):
    t = np.asarray(t, float)
    return np.sqrt(A / t + B)


def fit_time_floor_and_save(
    df,
    oxides=("MgO", "CaO", "SiO2"),
    time_col="time",
    out_path=None,                 # if None, auto-generate
    tag=None,                      # e.g., "12mm" or "25mm"
    overwrite=False,               # safety: don't overwrite by default
    err_col_fmt="{ox}_err",        # default matches dataframe
    save=False,                    # opt-in to writing JSON

):
    """
    Fits y = sqrt(A/t + B) for each oxide's %error column and saves JSON.

    Parameters
    ----------
    out_path : str|Path|None
        If None, uses 'epma_precision_time_floor_{tag}.json' (or no tag).
    tag : str|None
        Appended to filename when out_path is None.
    overwrite : bool
        If False and file exists, raises FileExistsError.
    err_col_fmt : str
        Format string for error column names. Default '{ox}_err'.
    """
    # Decide output path
    if out_path is None:
        suffix = f"_{tag}" if tag else ""
        out_path = Path(f"epma_precision_time_floor{suffix}.json")
    else:
        out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists. Set overwrite=True or change out_path/tag.")

    store = {
        "_meta": {
            "version": "time-floor-1.0",
            "created": datetime.now().isoformat(timespec="seconds"),
            "note": "Single-sample fit with floor: y = sqrt(A/t + B). Uses curve_fit with sigma=1/sqrt(t).",
            "time_col": time_col,
            "tag": tag,
        },
        "models": {}
    }

    t_all = pd.to_numeric(df[time_col], errors="coerce").to_numpy()

    for ox in oxides:
        err_col = err_col_fmt.format(ox=ox)  # e.g., "SiO2_err"
        if err_col not in df.columns:
            continue

        y_all = pd.to_numeric(df[err_col], errors="coerce").to_numpy()

        valid = (
            np.isfinite(t_all) & np.isfinite(y_all) &
            (t_all > 0) & (y_all >= 0) & (y_all <= 100)
        )
        if valid.sum() < 3:
            continue

        t_fit = t_all[valid]
        y_fit = y_all[valid]

        # Initial guesses
        A0 = float(np.nanmedian((y_fit ** 2) * t_fit))
        long_mask = t_fit >= np.nanpercentile(t_fit, 75)
        B0 = float(np.nanmedian(y_fit[long_mask] ** 2)) if np.any(long_mask) else 1e-4
        B0 = max(B0, 1e-6)

        # Weighting (relative): sigma ∝ 1/sqrt(t)
        sigma = 1.0 / np.sqrt(t_fit)

        popt, _ = curve_fit(
            time_model_with_floor,
            t_fit,
            y_fit,
            p0=[A0, B0],
            bounds=([0.0, 0.0], [np.inf, np.inf]),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=20000,
        )

        A, B = map(float, popt)

        store["models"][ox] = {
            "A": A,
            "B": B,
            "n": int(valid.sum()),
            "t_min": float(np.nanmin(t_fit)),
            "t_max": float(np.nanmax(t_fit)),
            "floor_y": float(np.sqrt(B)),
            "err_col": err_col,
        }

    if save:
        if out_path is None:
            suffix = f"_{tag}" if tag else ""
            out_path = Path(f"epma_precision_time_floor{suffix}.json")
        else:
            out_path = Path(out_path)

        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"{out_path} already exists. Set overwrite=True or change out_path/tag."
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
        print("Saved:", str(out_path))
    else:
        print("No file saved. Set save=True to write JSON.")

    return store

def load_time_floor_models(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_percent_error(models, oxide, t):
    params = models["models"][oxide]
    A, B = params["A"], params["B"]
    t = np.asarray(t, float)
    return np.sqrt(A / t + B)

# %% 


model_12mm = fit_time_floor_and_save(df_tilted_7nA_12mm, time_col="Total_time", tag="12mm", overwrite=True,
                                     save=False)
model_25mm = fit_time_floor_and_save(df_tilted_20nA_25mm, time_col="Total_time", tag="25mm", overwrite=True,
                                     save=False)

plot = False 
if plot: 
    fig, (ax2) = plt.subplots(1, 1, figsize = (7,4), sharex=True, sharey=True)
    WD12mm_output_MgO=predict_percent_error(model_12mm, 'MgO', t_test)
    WD12mm_output_SiO2=predict_percent_error(model_12mm, 'SiO2', t_test)
    WD25mm_output_SiO2=predict_percent_error(model_25mm, 'SiO2', t_test)
    ax2.plot(t_test, WD12mm_output_SiO2, '-r')
    ax2.plot(df_tilted_7nA_12mm['Total_time'], df_tilted_7nA_12mm['SiO2_err'], '.r', label='12mm')
    ax2.plot(t_test, WD25mm_output_SiO2, '-k')
    ax2.plot(df_tilted_20nA_25mm['Total_time'], df_tilted_20nA_25mm['SiO2_err'], '.k', label='25mm')
    ax2.set_xlabel('Total acquisition time (s)')
    ax2.set_ylabel('% error (SiO$_2$)')
    ax2.legend()
    ax2.set_xlim([0.001, 10])
    ax2.set_xscale('log')
    fig.tight_layout()


    fig, (ax2) = plt.subplots(1, 1, figsize = (7,4), sharex=True, sharey=True)
    WD12mm_output_MgO=predict_percent_error(model_12mm, 'MgO', t_test)
    WD12mm_output_SiO2=predict_percent_error(model_12mm, 'SiO2', t_test)
    WD25mm_output_SiO2=predict_percent_error(model_25mm, 'SiO2', t_test)
    # ax2.plot(t_test, WD12mm_output_SiO2, '-r')
    # ax2.plot(df_tilted_7nA_12mm['Total_time'], df_tilted_7nA_12mm['SiO2_err'], '.r')
    ax2.plot(t_test, WD25mm_output_SiO2, '-k')
    ax2.plot(df_tilted_20nA_25mm['Total_time'], df_tilted_20nA_25mm['SiO2_err'], '.k')
    ax2.set_xlabel('Total acquisition time (s)')
    ax2.set_ylabel('% error (SiO$_2$)')
    ax2.set_xlim([0.001, 10])
    ax2.set_xscale('log')
    fig.tight_layout()


# %% 
# %% 
# %%

# --- MH0811b Configs ---
mh_base_cols = {
    "Plagioclase": "#66C4C4", "Clinopyroxene": "#E57A7A", 
    "Orthopyroxene": "#931d1d", "Alkali_Feldspar": "#FEF7C2", 
    "Oxide": "#2E2DCE", "Glass": "#F9C300", 
    "Apatite": "#5B6768", #"Rhombohedral_Oxides": "#6FB2E4", 
    "SiO2_Polymorph": "#CEC6CD",
}
mh_cols_ebsd = {**mh_base_cols, "Unindexed": "white"}
mh_cols_eds  = {**mh_base_cols, "Feldspar_Miscibility_Gap": "#003d36", "Amphibole": "#5E2910",
                "Vesicles": "white",  "Olivine": "#6F7608"}

mh_eds_keep = [
    "Plagioclase", "Feldspar_Miscibility_Gap", "Alkali_Feldspar", 
    "Clinopyroxene", "Orthopyroxene", 
    "Oxide", "Glass",
]

map_dirs = [root for root, _, files in os.walk("MH0811b_Maps_25mm_20.5nA") if "Ignore" not in root.split(os.sep) and any(f.lower().endswith(".csv") for f in files)]

# %%
# %%

pred_score_threshold=0.5
mh_1ms = mm.run_map(
    next((s for s in map_dirs if '1ms' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    epoxy_threshold=None,
    remove_islands_flag=False, #phases=mh_eds_keep, 
    fill_holes_flag=False, phase_colors=mh_cols_eds,
)

mh_10ms = mm.run_map(
    next((s for s in map_dirs if '10ms' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    epoxy_threshold=None,
    remove_islands_flag=False, #phases=mh_eds_keep, 
    fill_holes_flag=False, phase_colors=mh_cols_eds,
)

mh_50ms = mm.run_map(
    next((s for s in map_dirs if '50ms' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    epoxy_threshold=None,
    remove_islands_flag=False, #phases=mh_eds_keep, 
    fill_holes_flag=False, phase_colors=mh_cols_eds,
)

mh_eds_keep_lim = ['Plagioclase', 'Feldspar_Miscibility_Gap', 'Alkali_Feldspar',
                   'Clinopyroxene', 'Orthopyroxene', 
                   'Oxide', 'Glass']

maps = [
    ("1 ms",  mh_1ms),
    ("10 ms", mh_10ms),
    ("50 ms", mh_50ms),
]

# %% 

fig, axs = plt.subplots(
    nrows=3, ncols=3,
    figsize=(17, 14),
    # constrained_layout=True,
)

# ── Top row: error plots ──
ax1, ax3, ax2 = axs[0]

frame_times = {
    "t=1 ms":  1e-3,
    "t=10 ms": 10e-3,
    "t=50 ms": 50e-3,
}

frame_style = {
    "t=1 ms":  dict(marker="*", color="k", mfc="yellow", ms=15),
    "t=10 ms": dict(marker="*", color="k", mfc="cyan",   ms=15),
    "t=50 ms": dict(marker="*", color="k", mfc="white",  ms=15),
}

plots = [
    (ax1, "MgO",  "MgO_err",  r"MgO % Error"),
    (ax3, "CaO",  "CaO_err",  r"CaO % Error"),
    (ax2, "SiO2", "SiO2_err", r"$\mathregular{SiO_2}$ % Error"),
]

for ax, oxide, err_col, ylabel in plots:
    y_curve = predict_percent_error(model_25mm, oxide, t_test)
    ax.plot(df_tilted_20nA_25mm["Total_time"],
            df_tilted_20nA_25mm[err_col],
            "ok", alpha=0.1, label='Error Data')
    ax.plot(t_test, y_curve, "-k", label='Error Model')

    for label, t in frame_times.items():
        y = predict_percent_error(model_25mm, oxide, t)
        ax.plot(t, y, **frame_style[label], label=label)

    ax.set_xlabel("Total Acquisition Time (s)")
    ax.set_ylabel(ylabel)

ax1.legend(loc="best", fontsize=10)
ax1.set_ylim([-1, 60])
ax3.set_ylim([-1, 50])
ax2.set_ylim([-1, 32.5])

ax1.set_xscale("log")
ax3.set_xscale("log")
ax2.set_xscale("log")
ax1.set_xlim([0.0005, 10])
ax2.set_xlim([0.0005, 10])
ax3.set_xlim([0.0005, 10])

ax1.annotate("A. 16 wt.% MgO", xy=(0.03, 0.935), xycoords="axes fraction", fontsize=14)
ax3.annotate("B. 19 wt.% CaO", xy=(0.03, 0.935), xycoords="axes fraction", fontsize=14)
ax2.annotate(r"C. 54 wt.% $\mathregular{SiO_2}$", xy=(0.03, 0.935), xycoords="axes fraction", fontsize=14)

ax2.yaxis.set_tick_params(which='both', labelbottom=True)
ax3.yaxis.set_tick_params(which='both', labelbottom=True)


map_axs = axs[1]

# Wrap the zip in enumerate to track the index 'i'
for i, (ax, (lab, res)) in enumerate(zip(map_axs, maps)):
    if res is None:
        ax.set_axis_off()
        ax.set_title(f"{lab} (missing)")
        continue

# Variables to hold our stolen legend data
stolen_handles = []
stolen_labels = []
legend_title = "Phases"

for i, (ax, (lab, res)) in enumerate(zip(map_axs, maps)):
    if res is None:
        ax.set_axis_off()
        ax.set_title(f"{lab} (missing)")
        continue
        
    # Only tell plot_phase_map to draw the legend on the FIRST plot
    is_first_plot = (i == 0)
    
    mm.plot_phase_map(
        res['mineral_map'][:, :-1],
        min_frac=0.0001,
        title=None,
        phase_colors=mh_cols_eds,
        scalebar_um=20, pixel_size_um=5.0,
        scalebar_loc='upper right',
        scalebar_col='black',
        ax=ax,
        legend_on=is_first_plot,
    )
    
    # If this is the first plot, grab the legend and remove it from the plot
    if is_first_plot:
        leg = ax.get_legend()
        if leg is not None:
            # Extract the handles (colors), labels (text), and title
            stolen_handles = leg.legend_handles
            stolen_labels = [text.get_text() for text in leg.texts]
            # legend_title = leg.get_title().get_text()
            leg.remove() 


if stolen_handles:
    map_axs[1].legend(
        handles=stolen_handles, 
        labels=stolen_labels,
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.03),
        frameon=False,
        ncols=8,        
        # title=legend_title, 
        fontsize=10
    )

map_axs[0].annotate("D. Phase Map at t=1ms", xy=(0.03, 0.92), xycoords="axes fraction", fontsize=14, bbox=dict(boxstyle="round,pad=0.1", fc="white", lw=0, alpha=0.75), zorder=10)
map_axs[1].annotate("E. Phase Map at t=10ms", xy=(0.03, 0.92), xycoords="axes fraction", fontsize=14, bbox=dict(boxstyle="round,pad=0.1", fc="white", lw=0, alpha=0.75), zorder=10)
map_axs[2].annotate("F. Phase Map at t=50ms", xy=(0.03, 0.92), xycoords="axes fraction", fontsize=14, bbox=dict(boxstyle="round,pad=0.1", fc="white", lw=0, alpha=0.75), zorder=10)

# Probability Map 
map_axs = axs[2]
cmap='inferno'

for ax, (lab, res) in zip(map_axs, maps):
    if res is None:
        ax.set_axis_off()
        continue
    pred_score_map = res['pred_score_map'][:, :-1]
    ax.imshow(pred_score_map, origin='upper', interpolation='none', cmap=cmap)
    ax.set_axis_off() 

# Titles for the bottom row
map_axs[0].annotate("G. Prediction Score at t=1ms", xy=(0.03, 0.92), xycoords="axes fraction", fontsize=14, bbox=dict(boxstyle="round,pad=0.1", fc="white", lw=0, alpha=0.75))
map_axs[1].annotate("H. Prediction Score at t=10ms", xy=(0.03, 0.92), xycoords="axes fraction", fontsize=14, bbox=dict(boxstyle="round,pad=0.1", fc="white", lw=0, alpha=0.75))
map_axs[2].annotate("I. Prediction Score at t=50ms", xy=(0.03, 0.92), xycoords="axes fraction", fontsize=14, bbox=dict(boxstyle="round,pad=0.1", fc="white", lw=0, alpha=0.75))
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([]) 

cax = fig.add_axes([0.91, 0.1225, 0.0125, 0.205])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_label('Prediction Score', fontsize=14)
# plt.savefig('texfigs/EDS_phase_withuncertainty_map.pdf', pad_inches=0.025, bbox_inches='tight', transparent=False)
plt.show()

# %%
# %%
# %% 
# --- Differential timing maps --- 

# EBSD
# mh_ebsd_phase_map, _, _, _ = mm.plot_ctf_phases(mh_file_path, rename_dict=mh_merge_rules, phase_colors=mh_cols_ebsd, title=None, scalebar_um=100)
# plot_phase_proportions(mh_ebsd_phase_map, mh_cols_ebsd, title="MH0811b EBSD Proportions")

# EDS
pred_score_threshold=0.0
mh_1ms = mm.run_map(
    next((s for s in map_dirs if '1ms' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    epoxy_threshold=None,
    phases=mh_eds_keep, remove_islands_flag=False, 
    fill_holes_flag=False, phase_colors=mh_cols_eds,
)

mh_2ms = mm.run_map(
    next((s for s in map_dirs if '2ms' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    epoxy_threshold=None,
    phases=mh_eds_keep, remove_islands_flag=False, 
    fill_holes_flag=False, phase_colors=mh_cols_eds,
)

mh_5ms = mm.run_map(
    next((s for s in map_dirs if '5ms' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    epoxy_threshold=None,
    phases=mh_eds_keep, remove_islands_flag=False, 
    fill_holes_flag=False, phase_colors=mh_cols_eds,
)

mh_10ms = mm.run_map(
    next((s for s in map_dirs if '10ms' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    epoxy_threshold=None,
    phases=mh_eds_keep, remove_islands_flag=False, 
    fill_holes_flag=False, phase_colors=mh_cols_eds,
)

mh_20ms = mm.run_map(
    next((s for s in map_dirs if '20ms' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    epoxy_threshold=None,
    phases=mh_eds_keep, remove_islands_flag=False, 
    fill_holes_flag=False, phase_colors=mh_cols_eds,
)

mh_50ms = mm.run_map(
    next((s for s in map_dirs if '50ms' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    epoxy_threshold=None,
    phases=mh_eds_keep, remove_islands_flag=False, 
    fill_holes_flag=False, phase_colors=mh_cols_eds,
)

# %%
# %% 

# Plot phase maps 

maps = [
    ("1 ms",  mh_1ms),
    ("2 ms",  mh_2ms),
    ("5 ms",  mh_5ms),
    ("10 ms", mh_10ms),
    ("20 ms", mh_20ms),
    ("50 ms", mh_50ms),
]

fig, axs = plt.subplots(
    nrows=3, ncols=2,
    figsize=(12, 16),
)
axs = axs.ravel()

for ax, (lab, res) in zip(axs, maps):
    if res is None:
        ax.set_axis_off()
        ax.set_title(f"{lab} (missing)")
        continue
    mm.plot_component_composite(
        res,
        phases=mh_eds_keep_lim,
        title=f"EDS Acquisition Time = {lab}",
        phase_colors=mh_cols_eds,
        limits_mode="std",
        min_speck_size=0,
        cbar_vgap=-0.05,
        legend_on=False,
        scalebar_um=10, pixel_size_um=5.0,
        scalebar_col='black',
        scalebar_loc='lower right',
        ax=ax
    )

handles = [mpatches.Patch(color=mh_cols_eds[p], label=p)
           for p in mh_eds_keep_lim if p in mh_cols_eds]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=8, frameon=True, fontsize=10)
plt.tight_layout()
# plt.savefig('texfigs/MH0811_v0030_eds_varytime.pdf', bbox_inches='tight', pad_inches=0.025)
# plt.savefig('texfigs/MH0811_v0030_eds_varytime.png', bbox_inches='tight', pad_inches=0.025, dpi=300)
plt.show()


# %% 

# Plot prediction scores 

maps = [
    ("1 ms",  mh_1ms),
    ("2 ms",  mh_2ms),
    ("5 ms",  mh_5ms),
    ("10 ms", mh_10ms),
    ("20 ms", mh_20ms),
    ("50 ms", mh_50ms),
]

fig, axs = plt.subplots(
    nrows=3, ncols=2,
    figsize=(12, 16),
)
axs = axs.ravel()

for ax, (lab, res) in zip(axs, maps):
    if res is None:
        ax.set_axis_off()
        ax.set_title(f"{lab} (missing)")
        continue
    pred_score_map = res['pred_score_map'][:, :-1]
    ax.imshow(pred_score_map, origin='upper', interpolation='none', cmap=cmap)
    ax.set_axis_off() 
    ax.set_title(f"EDS Acquisition Time = {lab}")

norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([]) 
cax = fig.add_axes([0.99, 0.046, 0.025, 0.26])
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_label('Prediction Score', fontsize=14)

plt.tight_layout()
# plt.savefig('texfigs/MH0811_v0030_eds_errors_varytime.pdf', bbox_inches='tight', pad_inches=0.025)
# plt.savefig('texfigs/MH0811_v0030_eds_errors_varytime.png', bbox_inches='tight', pad_inches=0.025, dpi=300)
plt.show()

# %%
# %% 

# Plot prediction score histograms 

maps = [
    ("1 ms",  mh_1ms),
    ("2 ms",  mh_2ms),
    ("5 ms",  mh_5ms),
    ("10 ms", mh_10ms),
    ("20 ms", mh_20ms),
    ("50 ms", mh_50ms),
]

fig, axs = plt.subplots(3, 2, figsize=(12, 16), constrained_layout=True)
axs = axs.ravel()

bins = 50
xlim = (0.0, 1.0)

for ax, (lab, res) in zip(axs, maps):
    prob = np.asarray(res["pred_score_map"], dtype=float).ravel()
    prob = prob[np.isfinite(prob)]

    ax.hist(prob, bins=bins, color='#3779D5', edgecolor='k') #, histtype='stepfilled')
    ax.set_title(f"EDS Acquisition Time={lab}")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Prediction Score")
    ax.set_ylabel("Pixels")
# plt.savefig('texfigs/MH0811_v0030_prob_histograms_varytime.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()

# %% 
# %% 

# Plot single phase prediction score histograms 

def plot_probability_histograms_on_subspec(
    fig,
    subspec,
    pred_score_map_2d,
    mineral_map_2d,
    pred_score_threshold,
    phases=None,
    bins=50,
    share_y=True,
    title=None,
):
    mineral_map_2d = np.asarray(mineral_map_2d, dtype=object)
    pred_score_map_2d = np.asarray(pred_score_map_2d, dtype=float)

    # Decide phases
    if phases is None:
        vals = mineral_map_2d.ravel()
        vals = vals[np.array([v is not None for v in vals])]
        phases = list(dict.fromkeys([v for v in vals if str(v).lower() not in {"nan", "none", "null", ""}]))
    phases = list(phases)

    if not phases:
        ax = fig.add_subplot(subspec)
        ax.text(0.5, 0.5, "No phases", ha="center", va="center")
        ax.axis("off")
        return [ax]

    per_row = min(5, len(phases))
    nrows = int(np.ceil(len(phases) / per_row))

    cell = subspec.subgridspec(
        nrows=2, ncols=1,
        height_ratios=[0.12, 0.88],   # adjust title height if needed
        hspace=0.05
    )

    # Title axis (separate from plots, no overlap)
    if title:
        ax_title = fig.add_subplot(cell[0, 0])
        ax_title.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
        ax_title.set_axis_off()

    # Inner histogram grid goes in the second row
    inner = cell[1, 0].subgridspec(
        nrows=nrows, ncols=per_row,
        wspace=0.25, hspace=0.4
    )

    finite_prob = np.isfinite(pred_score_map_2d)
    total = float(finite_prob.sum() + 1e-12)

    axes = []
    for i, phase in enumerate(phases):
        rr = i // per_row
        cc = i % per_row
        ax = fig.add_subplot(inner[rr, cc], sharey=axes[0] if (share_y and axes) else None)
        axes.append(ax)

        vals = pred_score_map_2d[mineral_map_2d == phase]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            ax.text(0.5, 0.5, f"{phase}\n(no data)", ha="center", va="center", fontsize=8)
            ax.set_axis_off()
            continue

        ax.hist(vals, bins=bins, orientation="horizontal")
        ax.set_ylim((pred_score_threshold, 1.0))
        ax.set_title(f"{phase}\n{100.0*vals.size/total:.2f} %", fontsize=9, pad=-50)
        ax.set_xlabel("Pixels", fontsize=8)
        if cc == 0:
            ax.set_ylabel("Prediction Probability", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

    # Turn off unused inner cells
    for j in range(len(phases), nrows * per_row):
        rr = j // per_row
        cc = j % per_row
        ax_off = fig.add_subplot(inner[rr, cc])
        ax_off.set_axis_off()

    return axes

maps = [
    ("1 ms",  mh_1ms),
    ("2 ms",  mh_2ms),
    ("5 ms",  mh_5ms),
    ("10 ms", mh_10ms),
    ("20 ms", mh_20ms),
    ("50 ms", mh_50ms),
]

fig = plt.figure(figsize=(20, 20), constrained_layout=True)
outer = fig.add_gridspec(3, 2, wspace=0.1, hspace=0.05)

for idx, (lab, res) in enumerate(maps):
    r, c = divmod(idx, 2)
    subspec = outer[r, c]

    plot_probability_histograms_on_subspec(
        fig=fig,
        subspec=subspec,
        pred_score_map_2d=res["pred_score_map"],
        mineral_map_2d=res["mineral_map"],
        pred_score_threshold=0.,
        phases=mh_eds_keep_lim,
        share_y=True,
        title=f"Prediction Scores for EDS Acquisition Time={lab}",
    )
plt.tight_layout()
# plt.savefig('texfigs/MH0811_v0030_prob_histograms_phases_varytime.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()

# %% 
