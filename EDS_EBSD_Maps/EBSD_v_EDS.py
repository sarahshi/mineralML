# %% 

""" Created on February 1, 2025 // @author: Sarah Shi """

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

sys.path.append('../src')
import mineralML as mm

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.style.use('../style.mplstyle')

# %% 
# Configurations & Dictionaries

# Master Priority List
MASTER_PRIORITY = [
    "Plagioclase", "Feldspar_Miscibility_Gap", "Alkali_Feldspar", 
    "Clinopyroxene", "Orthopyroxene", "Amphibole",
    "SiO2_Polymorph", "Oxide", "Apatite", "Zircon", 
    "Epidote", "Titanite", "Unindexed", "Glass"
]

BII_PRIORITY = [
    "Plagioclase", "Feldspar_Miscibility_Gap", "SiO2_Polymorph", "Titanite", "Amphibole",
    "Clinopyroxene", "Alkali_Feldspar", "Oxide",
    "Epidote", "Zircon", "Apatite", "Unindexed", "Glass", "Vesicles"
]

# --- MH0811b Configs ---
mh_file_path = "EBSD_CTF/MH0811b_EBSD_EDS.ctf"
mh_file_path_bad = "EBSD_CTF/MH0811b_EBSD_EDS_noisy.ctf"


mh_merge_rules = {
    "Andesine": "Plagioclase", "Augite": "Clinopyroxene", "Enstatite": "Orthopyroxene",
    "Orthoclase": "Alkali_Feldspar", "Magnetite": "Oxide", "Ilmenite": "Oxide",
    "Spinel_Group": "Oxide", "Rhombohedral_Oxides": "Oxide", "Oxide": "Oxide",
    "Quartz-new": "SiO2_Polymorph", "Cristobalite": "SiO2_Polymorph",
}

mh_base_cols = {
    "Plagioclase": "#66C4C4", "Clinopyroxene": "#E57A7A", 
    "Orthopyroxene": "#931d1d", "Alkali_Feldspar": "#FEF7C2", 
    "Oxide": "#2E2DCE", "Glass": "#F9C300",
    "Apatite": "#5B6768", "SiO2_Polymorph": "#CEC6CD",
}
mh_cols_ebsd = {**mh_base_cols, "Unindexed": "white"}
mh_cols_eds  = {**mh_base_cols, "Feldspar_Miscibility_Gap": "#003d36", "Amphibole": "#5E2910",
                "Vesicles": "white",}

mh_eds_keep = [
    "Plagioclase", "Feldspar_Miscibility_Gap", "Clinopyroxene", "Orthopyroxene", 
    "Alkali_Feldspar", "Oxide", "Spinel_Group", "Glass", "Apatite", "Rhombohedral_Oxides", "SiO2_Polymorph"
]

# --- Bii Configs ---
bii_file_path = "EBSD_CTF/Bii_EBSD_EDS.ctf"

bii_merge_rules = {
    "Anorthite": "Plagioclase", "Hornblende": "Amphibole", "Orthoclase": "Alkali_Feldspar",
    "Muscovite": "Unindexed", "Biotite": "Unindexed",
    "Magnetite": "Oxide", "Ilmenite": "Oxide",
    "Spinel_Group": "Oxide", "Rhombohedral_Oxides": "Oxide", "Oxide": "Oxide",
    "Quartz-new": "SiO2_Polymorph", 
}

bii_cols  = {**mh_base_cols, "Amphibole": '#5E2910',
             "Zircon": "cyan", "Epidote": "magenta", 
             "Titanite": "#bb730c", "Unindexed": "white",
             "Feldspar_Miscibility_Gap": "#003d36", 
             "Vesicles": "white" # Handled here for EDS plotting
             }

bii_eds_keep = [
    "Plagioclase", "Feldspar_Miscibility_Gap", "SiO2_Polymorph", "Titanite", "Amphibole",
    "Clinopyroxene", "Alkali_Feldspar", "Oxide", "Epidote", "Zircon", "Apatite", 
    "Unindexed", "Glass", "Vesicles"
]

map_dirs = [root for root, _, files in os.walk(".") if "Ignore" not in root.split(os.sep) and any(f.lower().endswith(".csv") for f in files)]

# %%

# --- Helper Functions ---

def get_valid_mask(arr):
    arr = np.asarray(arr)
    return ~((arr != arr) | (arr == 'nan') | (arr == 'NaN') | (arr == 'unindexed') | (arr == 'None'))


def get_proportions(arr, valid_phases=None, replace_unaccepted_with="Vesicles"):
    mask = get_valid_mask(arr)
    valid_pixels = arr[mask].copy()
    if valid_phases is not None:
        valid_pixels[~np.isin(valid_pixels, valid_phases)] = replace_unaccepted_with
    phases, counts = np.unique(valid_pixels, return_counts=True)
    return phases, counts / len(valid_pixels)


def plot_phase_proportions(phase_map, phase_cols, title="Phase Proportions"):
    phases, props = get_proportions(phase_map)
    df = pd.DataFrame({"Phase": phases, "Proportion": props}).sort_values(by="Proportion", ascending=False)
    
    print(f"\n--- {title} ---")
    print(df.round(4).to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(8, 2))
    left = 0
    for _, row in df.iterrows():
        p, prop = row["Phase"], row["Proportion"]
        ax.barh(y=title, width=prop, left=left, color=phase_cols.get(p, "#999999"),
                edgecolor="white", height=0.5, label=f"{p} ({prop*100:.1f}%)")
        left += prop
        
    ax.set_xlim(0, 1)
    ax.set_xlabel("Area Proportion")
    
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
        
    ax.tick_params(axis='y', length=0)
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8, frameon=False, title="Phases")
    
    plt.tight_layout()
    plt.show()


def get_bar_style(p, color_dict, base_plag_col="#66C4C4", is_eds=False):
    """Requires color_dict directly so it works for any sample."""
    c = color_dict.get(p, "#999999")
    
    # Universally style Vesicles and Unindexed
    if p == "Vesicles": 
        return dict(facecolor="white", edgecolor="#F9C300", linewidth=1.0)
    if p == "Unindexed": 
        return dict(facecolor="white", edgecolor="black", linewidth=1.0)
        
    # Phase-specific styles for EDS only
    if is_eds:
        if p == "Feldspar_Miscibility_Gap": 
            return dict(facecolor=base_plag_col, edgecolor="#003d36", linewidth=1.0)
        return dict(facecolor=c, edgecolor=c, linewidth=0.5)
    else:
        return dict(facecolor=c, edgecolor=c, linewidth=0.5)


def sort_phases_priority(phases, props, priority_list=MASTER_PRIORITY):
    prop_dict = dict(zip(phases, props))
    # count_dict = dict(zip(phases, counts))
    ord_phases = [p for p in priority_list if p in prop_dict]
    ord_props = [prop_dict[p] for p in ord_phases]
    # ord_counts = [count_dict[p] for p in ord_phases]
    
    rem_phases = [p for p in phases if p not in priority_list]
    rem_props = [prop_dict[p] for p in rem_phases]
    
    if rem_props:
        sort_idx = np.argsort(rem_props)[::-1]
        ord_phases.extend([rem_phases[i] for i in sort_idx])
        ord_props.extend([rem_props[i] for i in sort_idx])
    return ord_phases, ord_props#, ord_counts


def order_row_major(handles, labels, ncol):
    nrow = int(np.ceil(len(handles) / ncol))
    return [handles[r * ncol + c] for c in range(ncol) for r in range(nrow) if r * ncol + c < len(handles)], \
           [labels[r * ncol + c] for c in range(ncol) for r in range(nrow) if r * ncol + c < len(labels)]


def annotate_stacked_bar_prop_staggered(ax, y, phases, props, fmt="{:.1f}%", fs=10, min_inside=0.03, 
                                        dy_inside=0.0, dy_out=0.35, alternate=True, x_jitter=1.0, 
                                        force_outside=None, force_dx=None, phase_colors=None):
    
    force_outside, force_dx, phase_colors = set(force_outside or []), dict(force_dx or {}), dict(phase_colors or {})
    left, out_i = 0.0, 0
    bbox = dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=1, lw=1.0)

    for p, prop in zip(phases, props):
        prop_pct = prop * 100 
        x = left + (prop_pct / 2.0)
        left += prop_pct
        txt = fmt.format(prop_pct)
        
        # Override the edge color for special phases!
        if p == "Vesicles":
            ec_color = "#F9C300"
        elif p == "Unindexed":
            ec_color = "black"
        else:
            ec_color = phase_colors.get(p, "none")
            
        bb = {**bbox, "ec": ec_color}

        if (prop >= min_inside) and (p not in force_outside):
            ax.text(x, y + dy_inside, txt, ha="center", va="center", fontsize=fs, color="black", bbox=bb, clip_on=False, zorder=10)
            continue

        sgn_y = 1 if (not alternate or out_i % 2 == 0) else -1
        dx = float(np.sign(force_dx[p])) * x_jitter * float(abs(force_dx[p])) if (p in force_outside and x_jitter and p in force_dx) else 0.0

        x_txt = float(np.clip(x + dx, 0.0, 100.0)) 

        ax.annotate(txt, xy=(x, y), xycoords="data", xytext=(x_txt, y + sgn_y * dy_out), textcoords="data",
                    ha="center", va="bottom" if sgn_y > 0 else "top", fontsize=fs, color="black", bbox=bb,
                    arrowprops=dict(arrowstyle="-", lw=0.9, color='k', shrinkA=0, shrinkB=0), clip_on=False, zorder=20)
        out_i += 1


# %% 
# --- Process MH0811b Maps ---

# EBSD
mh_fig, mh_ebsd_phase_map, _, _, _ = mm.plot_ctf_phases(mh_file_path, rename_dict=mh_merge_rules, phase_colors=mh_cols_ebsd, title=None, scalebar_um=100)
plot_phase_proportions(mh_ebsd_phase_map, mh_cols_ebsd, title="MH0811b EBSD Proportions")

pred_score_threshold=0.5
# EDS
mh0811 = mm.run_map(
    next((s for s in map_dirs if 'MH0811b' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold, 
    renormalize=False, epoxy_threshold=None,
    phases=mh_eds_keep, remove_islands_flag=False, 
    fill_holes_flag=True, phase_colors=mh_cols_eds
)

mh_eds_vals_raw = mh0811["df_pred"]['Predict_Mineral'].dropna().values
plot_phase_proportions(mh_eds_vals_raw, mh_cols_eds, title="MH0811b EDS Proportions")

# %% 
# %%

# --- Plot MH0811b Composite ---
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[4, 1], hspace=0.33, wspace=0.025)
ax_ebsd, ax_comp, ax_bar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

# Draw Maps
mh_fig, phase_map, _, _, _ = mm.plot_ctf_phases(mh_file_path, rename_dict=mh_merge_rules, phase_colors=mh_cols_ebsd, ax=ax_ebsd, title="A. EBSD Phase Map", scalebar_um=100, legend_on=False)
_, phase_map_eds, comp_map = mm.plot_component_composite(mh0811, ax=ax_comp, fill_missing=True, 
                                                         title="B. mineralML-Generated EDS Phase Map",
                                                         phases=mh_eds_keep, phase_colors=mh_cols_eds, 
                                                         max_hole_size=5, min_speck_size=5,
                                                         scalebar_um=100, pixel_size_um=2.0, limits_mode='std',
                                                         cbar_hgap=0.02, cbar_vgap=-0.07, cbar_height=0.04, legend_on=False)

# Get Sorted Proportions
ord_phases_ebsd, ord_props_ebsd = sort_phases_priority(*get_proportions(phase_map), priority_list=MASTER_PRIORITY)
ord_phases_eds, ord_props_eds = sort_phases_priority(*get_proportions(mh0811["mineral_map"], valid_phases=mh_eds_keep), priority_list=MASTER_PRIORITY)

# Draw Stacked Bars
y_ebsd, y_eds, bar_h = 1.2, 0.00, 0.4
outlines = []

for is_eds, y_val, phases, props, c_dict in [(False, y_ebsd, ord_phases_ebsd, ord_props_ebsd, mh_cols_ebsd), (True, y_eds, ord_phases_eds, ord_props_eds, mh_cols_eds)]:
    left = 0.0
    for p, prop in zip(phases, props):
        st = get_bar_style(p, color_dict=c_dict, base_plag_col=c_dict.get("Plagioclase", "#66C4C4"), is_eds=is_eds)
        prop_pct = prop * 100
        ax_bar.barh(y=y_val, width=prop_pct, left=left, height=bar_h, color=st["facecolor"], edgecolor=st["edgecolor"], linewidth=st["linewidth"], zorder=2)
        if is_eds and p in ["Feldspar_Miscibility_Gap", "Unindexed"] and prop > 0: 
            outlines.append((left, prop_pct))
        left += prop_pct 

# Redraw thick outlines on EDS row
for left, prop in outlines:
    ax_bar.barh(y=y_eds, width=prop, left=left, height=bar_h, color="none", edgecolor="#003d36", linewidth=1.2, zorder=6)

# Annotations
annotate_stacked_bar_prop_staggered(ax_bar, y_ebsd, ord_phases_ebsd, ord_props_ebsd, dy_out=0.30, 
                                    force_outside={"Rhombohedral_Oxides", "SiO2_Polymorph"}, 
                                    force_dx={"Rhombohedral_Oxides": -3, "SiO2_Polymorph": -2}, phase_colors=mh_cols_ebsd)
annotate_stacked_bar_prop_staggered(ax_bar, y_eds, ord_phases_eds, ord_props_eds, dy_out=0.3, 
                                    force_outside={"Rhombohedral_Oxides", "SiO2_Polymorph"}, 
                                    force_dx={"Rhombohedral_Oxides": -1.75}, phase_colors=mh_cols_eds)

# Unified Legend
all_phases = list(dict.fromkeys(ord_phases_ebsd + ord_phases_eds))
legend_phases = [p for p in MASTER_PRIORITY if p in all_phases] + sorted([p for p in all_phases if p not in MASTER_PRIORITY])

ncol = 8
handles = [mpatches.Patch(**get_bar_style(p, color_dict=mh_cols_eds if p in mh_cols_eds else mh_cols_ebsd, 
                                          is_eds=(p not in mh_cols_ebsd or p in ["Feldspar_Miscibility_Gap", "Unindexed"]))) for p in legend_phases]
h_ord, l_ord = order_row_major(handles, legend_phases, ncol=ncol)

ax_bar.legend(h_ord, l_ord, loc="upper center", bbox_to_anchor=(0.5, -0.4), frameon=True, ncol=ncol, prop={'size': 10})

# Axes Styling
ax_bar.set_title('C. Phase Abundances')
ax_bar.set(xlim=(0, 100), ylim=(-0.75, 1.45), yticks=[y_ebsd, y_eds], yticklabels=["EBSD", "EDS"], xlabel="Modal Phase Proportion (%)")
ax_bar.xaxis.label.set_size(14)
ax_bar.tick_params(axis="both", labelsize=14)
for s in ["top", "right", "left"]: 
    ax_bar.spines[s].set_visible(False)

plt.tight_layout()
fig.canvas.draw()
ax_bar.set_position([ax_ebsd.get_position().x0, ax_bar.get_position().y0, ax_comp.get_position().x1 - ax_ebsd.get_position().x0, ax_bar.get_position().height])
# plt.savefig('texfigs/MH0811_v0030_f.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()

# %%
# %% 

# --- Plot MH0811b low count ---

mh0811_lowct = mm.run_map(
    next((s for s in map_dirs if 'MountHood_MH0811b_lowcount' in s), None),
    units='oxide_wt%', min_frac=0.005, pred_score_threshold=0,
    remove_islands_flag=False, 
    fill_holes_flag=True, phase_colors=mh_cols_eds
)

# %% 

# --- Plot MH0811b low count ---

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[4, 1], hspace=-0.05, wspace=0.025)
ax_ebsd, ax_comp, ax_bar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

# Color dictionaries
mh_merge_rules_new = {**mh_merge_rules, "Forsterite": "Olivine", "Hornblende": "Amphibole", "Tridymite": "SiO2_Polymorph"}
mh_cols_ebsd_new = {**mh_cols_ebsd, "Olivine": "#666633", "Biotite": "tab:brown"}
mh_cols_all = {**mh_cols_ebsd_new, **mh_cols_eds}

bad_phases = set(pd.unique(mh0811_lowct['mineral_map'].ravel()))
bad_phases.discard(None)
bad_phases -= {p for p in bad_phases if str(p) in ('nan', 'NaN', 'None', 'Unknown')}
missing = sorted(bad_phases - set(mh_cols_all.keys()))
print("Phases without colors:", missing)

# Auto-assign tab20 colors to anything missing
tab20 = cm.get_cmap('tab20', max(len(missing), 1))
for i, phase in enumerate(missing):
    mh_cols_all[phase] = tab20(i)


# Draw Maps
fig, phase_map, _, _, _ = mm.plot_ctf_phases(
    mh_file_path_bad, rename_dict=mh_merge_rules_new, phase_colors=mh_cols_ebsd_new,
    ax=ax_ebsd, title="A. EBSD Phase Map", scalebar_um=100, legend_on=False
)

_, _, phase_map_eds = mm.plot_phase_map(
    mh0811_lowct['mineral_map'], ax=ax_comp, min_frac=0.005,
    title="B. mineralML-Generated EDS Phase Map",
    phase_colors=mh_cols_all,
    scalebar_um=100, pixel_size_um=2.0,
    bg_color=(1, 1, 1),
    legend_on=False
)

# Get Sorted Proportions
ord_phases_ebsd, ord_props_ebsd = sort_phases_priority(
    *get_proportions(phase_map), priority_list=MASTER_PRIORITY
)
ord_phases_eds, ord_props_eds = sort_phases_priority(
    *get_proportions(phase_map_eds), priority_list=MASTER_PRIORITY
)

# Draw Stacked Bars
y_ebsd, y_eds, bar_h = 1.2, 0.00, 0.4
outlines = []

for is_eds, y_val, phases, props in [
    (False, y_ebsd, ord_phases_ebsd, ord_props_ebsd),
    (True,  y_eds,  ord_phases_eds,  ord_props_eds),
]:
    left = 0.0
    for p, prop in zip(phases, props):
        st = get_bar_style(
            p, color_dict=mh_cols_all,
            base_plag_col=mh_cols_all.get("Plagioclase", "#66C4C4"),
            is_eds=is_eds
        )
        prop_pct = prop * 100
        ax_bar.barh(
            y=y_val, width=prop_pct, left=left, height=bar_h,
            color=st["facecolor"], edgecolor=st["edgecolor"],
            linewidth=st["linewidth"], zorder=2
        )
        if is_eds and p in ["Feldspar_Miscibility_Gap", "Unindexed"] and prop > 0:
            outlines.append((left, prop_pct))
        left += prop_pct

# Redraw thick outlines on EDS row
for left, prop in outlines:
    ax_bar.barh(
        y=y_eds, width=prop, left=left, height=bar_h,
        color="none", edgecolor="#003d36", linewidth=1.2, zorder=6
    )

# Annotations
annotate_stacked_bar_prop_staggered(
    ax_bar, y_ebsd, ord_phases_ebsd, ord_props_ebsd, dy_out=0.30,
    force_outside={"Rhombohedral_Oxides", "SiO2_Polymorph"},
    force_dx={"Rhombohedral_Oxides": -3, "SiO2_Polymorph": -2},
    phase_colors=mh_cols_all
)
annotate_stacked_bar_prop_staggered(
    ax_bar, y_eds, ord_phases_eds, ord_props_eds, dy_out=0.3,
    force_outside={"Rhombohedral_Oxides", "SiO2_Polymorph"},
    force_dx={"Rhombohedral_Oxides": -1.75},
    phase_colors=mh_cols_all
)

# Unified Legend
all_phases = list(dict.fromkeys(ord_phases_ebsd + ord_phases_eds))
legend_phases = [p for p in MASTER_PRIORITY if p in all_phases] + sorted(
    [p for p in all_phases if p not in MASTER_PRIORITY]
)

ncol = 8
handles = [
    mpatches.Patch(**get_bar_style(
        p, color_dict=mh_cols_all,
        base_plag_col=mh_cols_all.get("Plagioclase", "#66C4C4"),
        is_eds=(p not in mh_cols_ebsd_new or p in ["Feldspar_Miscibility_Gap", "Unindexed"])
    ))
    for p in legend_phases
]
h_ord, l_ord = order_row_major(handles, legend_phases, ncol=ncol)

ax_bar.legend(
    h_ord, l_ord, loc="upper center", bbox_to_anchor=(0.5, -0.4),
    frameon=True, ncol=ncol, prop={"size": 10}
)

# Axes Styling
ax_bar.set_title("C. Phase Abundances")
ax_bar.set(
    xlim=(0, 100), ylim=(-0.75, 1.45),
    yticks=[y_ebsd, y_eds], yticklabels=["EBSD", "EDS"],
    xlabel="Modal Phase Proportion (%)"
)
ax_bar.xaxis.label.set_size(14)
ax_bar.tick_params(axis="both", labelsize=14)
for s in ["top", "right", "left"]:
    ax_bar.spines[s].set_visible(False)

plt.tight_layout()
fig.canvas.draw()
ax_bar.set_position([
    ax_ebsd.get_position().x0, ax_bar.get_position().y0,
    ax_comp.get_position().x1 - ax_ebsd.get_position().x0,
    ax_bar.get_position().height
])
# plt.savefig('texfigs/MH0811_v0030_lowcount.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()

# %% 
# %%
# %% 
# --- Process Bii Maps ---

# EBSD
bii_fig, bii_ebsd_phase_map, _, _, _ = mm.plot_ctf_phases(bii_file_path, rename_dict=bii_merge_rules, phase_colors=bii_cols)
plot_phase_proportions(bii_ebsd_phase_map, bii_cols, title="Bii EBSD Proportions")

# EDS
pred_score_threshold=0.5
bii = mm.run_map(
    next((s for s in map_dirs if 'Bii' in s), None),
    units='element_wt%', pred_score_threshold=pred_score_threshold,
    renormalize=False, epoxy_threshold=None,
    phases=bii_eds_keep, remove_islands_flag=False, 
    fill_holes_flag=True, phase_colors=bii_cols
)

bii_eds_vals_clean = bii["mineral_map"].ravel()[~pd.isna(bii["mineral_map"].ravel())]
plot_phase_proportions(bii_eds_vals_clean, bii_cols, title="Bii EDS Proportions")

# %% 
# --- Plot Bii Composite ---

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[4, 1], hspace=-0.05, wspace=0.05)
ax_ebsd, ax_comp, ax_bar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

# Draw Maps
bii_fig, phase_map_bii, _, _, _ = mm.plot_ctf_phases(bii_file_path, rename_dict=bii_merge_rules, phase_colors=bii_cols, 
                                            ax=ax_ebsd, title="A. EBSD Phase Map", 
                                            scalebar_um=100, scalebar_loc="lower right", legend_on=False)
_, phase_map_eds_bii, comp_map_bii = mm.plot_component_composite(bii, ax=ax_comp, fill_missing=True, 
                                                                 title="B. mineralML-Generated EDS Phase Map", 
                                                                 phases=bii_eds_keep, phase_colors=bii_cols, 
                                                                 max_hole_size=20, min_speck_size=8,
                                                                 scalebar_um=100, pixel_size_um=3.0, 
                                                                 cbar_hgap=0.02, cbar_vgap=-0.09, cbar_height=0.06,
                                                                 scalebar_loc="lower right", limits_mode='std',
                                                                 legend_on=False)

# Get Sorted Proportions
ord_phases_ebsd_bii, ord_props_ebsd_bii = sort_phases_priority(*get_proportions(phase_map_bii), priority_list=BII_PRIORITY)
ord_phases_eds_bii, ord_props_eds_bii = sort_phases_priority(*get_proportions(phase_map_eds_bii, valid_phases=bii_eds_keep), 
                                                             priority_list=BII_PRIORITY)

# Draw Stacked Bars
y_ebsd, y_eds, bar_h = 1.2, 0.00, 0.4
outlines = []

for is_eds, y_val, phases, props in [(False, y_ebsd, ord_phases_ebsd_bii, ord_props_ebsd_bii), (True, y_eds, ord_phases_eds_bii, ord_props_eds_bii)]:
    left = 0.0
    for p, prop in zip(phases, props):
        st = get_bar_style(p, color_dict=bii_cols, base_plag_col=bii_cols.get("Plagioclase", "#66C4C4"), is_eds=is_eds)
        prop_pct = prop * 100
        ax_bar.barh(y=y_val, width=prop_pct, left=left, height=bar_h, color=st["facecolor"], edgecolor=st["edgecolor"], linewidth=st["linewidth"], zorder=2)
        if is_eds and p in ["Feldspar_Miscibility_Gap", "Unindexed"] and prop > 0: 
            outlines.append((left, prop_pct))
        left += prop_pct 

# Redraw thick outlines on EDS row
for left, prop in outlines:
    ax_bar.barh(y=y_eds, width=prop, left=left, height=bar_h, color="none", edgecolor="#003d36", linewidth=1.2, zorder=6)

# Annotations (Empty outside forces by default here, can tweak per sample)
annotate_stacked_bar_prop_staggered(ax_bar, y_ebsd, ord_phases_ebsd_bii, ord_props_ebsd_bii, dy_out=0.25, min_inside=0.03,
                                    force_outside={"Epidote", "Oxide", "Zircon",  "Apatite"},
                                    force_dx={"Epidote": -1.6, "Oxide": -2, "Zircon": -1.6, "Apatite": 0,},
                                    phase_colors=bii_cols)
annotate_stacked_bar_prop_staggered(ax_bar, y_eds, ord_phases_eds_bii, ord_props_eds_bii, dy_out=0.25, min_inside=0.02,
                                    force_outside={"Clinopyroxene", "Epidote", "Zircon", "Oxide",
                                                   "Apatite", "Glass"}, 
                                    force_dx={"Clinopyroxene": -0.5, "Alkali_Feldspar": 0, "Epidote": -3.75,
                                              "Zircon": -2, "Oxide":-2.5, "Apatite": -2.1, "Glass": 0.75},
                                    phase_colors=bii_cols)

# Unified Legend
all_phases = list(dict.fromkeys(ord_phases_ebsd_bii + ord_phases_eds_bii))
legend_phases = [p for p in BII_PRIORITY if p in all_phases] + sorted([p for p in BII_PRIORITY if p not in bii_eds_keep])

ncol = 8
handles = [mpatches.Patch(**get_bar_style(p, color_dict=bii_cols, base_plag_col=bii_cols.get("Plagioclase", "#66C4C4"), 
                                          is_eds=(p in ["Feldspar_Miscibility_Gap", "Unindexed", "Vesicles"]))) for p in legend_phases]
h_ord, l_ord = order_row_major(handles, legend_phases, ncol=ncol)

ax_bar.legend(h_ord, l_ord, loc="upper center", bbox_to_anchor=(0.5, -0.4), frameon=True, ncol=ncol, prop={'size': 10})

# Axes Styling
ax_bar.set_title('C. Phase Abundances')
ax_bar.set(xlim=(0, 100), ylim=(-.5, 1.5), yticks=[y_ebsd, y_eds], yticklabels=["EBSD", "EDS"], xlabel="Modal Phase Proportion (%)")
ax_bar.xaxis.label.set_size(14)
ax_bar.tick_params(axis="both", labelsize=14)
for s in ["top", "right", "left"]:
    ax_bar.spines[s].set_visible(False)

plt.tight_layout()
fig.canvas.draw()
ax_bar.set_position([ax_ebsd.get_position().x0, ax_bar.get_position().y0, ax_comp.get_position().x1 - ax_ebsd.get_position().x0, ax_bar.get_position().height])
# os.makedirs('texfigs', exist_ok=True)
# plt.savefig('texfigs/Bii_v0030.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()

# %%