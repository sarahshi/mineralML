# %% 

""" Created on February 6, 2023 // Modified May 15, 2025 // @author: Sarah Shi """

import numpy as np
import pandas as pd

import sys
sys.path.append('../src/')
import mineralML as mm

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.rcParams.update({
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'pdf.fonttype': 42,
    'font.family': 'Avenir',
    'font.size': 12,
    'xtick.direction': 'in',  # Set x-tick direction to 'in'
    'ytick.direction': 'in',  # Set y-tick direction to 'in'
    'xtick.major.size': 5,    # Set x-tick length
    'ytick.major.size': 5,    # Set y-tick length
    'xtick.major.pad': 6.5,   # Set x-tick padding
    'ytick.major.pad': 6.5    # Set y-tick padding
})

# %% .py for cleaning training dataset, fixing Fe speciation, etc. 

def Fe_Conversion(df):

    """
    Handle inconsistent Fe speciation in PetDB datasets by converting all to FeOt. 

    Parameters
    --------------
    df:class:`pandas.DataFrame`
        Array of oxide compositions.

    Returns
    --------
    df:class:`pandas.DataFrame`
        Array of oxide compositions with corrected Fe.
    """

    fe_conv = 159.688 / (2 * 71.8464)
    conditions = [~np.isnan(df['FeO']) & np.isnan(df['FeOt']) & np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3t']]),
    ~np.isnan(df['FeOt']) & np.isnan(df['FeO']) & np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3t']]), 
    ~np.isnan(df['Fe2O3']) & np.isnan(df['Fe2O3t']) & np.isnan(df['FeO']) & np.isnan([df['FeOt']]), # 2
    ~np.isnan(df['Fe2O3t']) & np.isnan(df['Fe2O3']) & np.isnan(df['FeO']) & np.isnan([df['FeOt']]), # 2
    ~np.isnan(df['FeO']) & ~np.isnan(df['Fe2O3']) & np.isnan(df['FeOt']) & np.isnan([df['Fe2O3t']]), # 3
    ~np.isnan(df['FeO']) & ~np.isnan(df['FeOt']) & ~np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3t']]), # 4
    ~np.isnan(df['FeO']) & ~np.isnan(df['Fe2O3']) & ~np.isnan(df['Fe2O3t']) & np.isnan([df['FeOt']]), # 5
    ~np.isnan(df['FeOt']) & ~np.isnan(df['Fe2O3']) & np.isnan(df['Fe2O3t']) & np.isnan([df['FeO']]), # 6
    ~np.isnan(df['Fe2O3']) & ~np.isnan(df['Fe2O3t']) & np.isnan(df['FeO']) & np.isnan([df['FeOt']]) ] # 7

    choices = [ (df['FeO']), (df['FeOt']),
    (df['Fe2O3']),(df['Fe2O3t']),
    (df['FeO'] + (df['Fe2O3'] / fe_conv)), # 3
    (df['FeOt']), # 4 of interest
    (df['Fe2O3t'] / fe_conv), # 5
    (df['FeOt']), # 6
    (df['Fe2O3t'] / fe_conv) ] # 7

    df.insert(4, 'FeOt_F', np.select(conditions, choices))
    df.drop(['FeOt'], axis=1, inplace=True)
    df.rename(columns={"FeOt_F": "FeOt"}, inplace=True)
    
    return df 


dtypes = {'SiO2': float, 'TiO2': float, 'Al2O3': float, 'FeOt': float, 'Fe2O3t': float, 'FeO': float, 'Fe2O3': float, 
          'MnO': float, 'MgO': float, 'CaO': float, 'Na2O': float, 'K2O': float, 'P2O5': float, 'Cr2O3': float, 'NiO': float, 
          'B2O3': float, 'ZrO2': float}

# %%

# amp_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Amphibole.xlsx', dtype=dtypes)) # 1 
# ap_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Apatite.xlsx', dtype=dtypes)) # 2
# bt_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Biotite.xlsx', dtype=dtypes)) # 3
# cal_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Calcite.xlsx', dtype=dtypes)) # 4
# chl_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Chlorite.xlsx', dtype=dtypes)) # 5
# cpx_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Clinopyroxene.xlsx', dtype=dtypes)) # 6
# ep_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Epidote.xlsx', dtype=dtypes)) # 7
# gt_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Garnet.xlsx', dtype=dtypes)) # 8
# hem_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Hematite.xlsx', dtype=dtypes)) # 9
# ilm_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Ilmenite.xlsx', dtype=dtypes)) # 10
# ks_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Kalsilite.xlsx', dtype=dtypes)) # 11
# ksp_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/KFeldspar.xlsx', dtype=dtypes)) # 12
# lc_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Leucite.xlsx', dtype=dtypes)) # 13
# mt_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Magnetite.xlsx', dtype=dtypes)) # 14
# ml_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Melilite.xlsx', dtype=dtypes)) # 15
# ms_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Muscovite.xlsx', dtype=dtypes)) # 16
# ne_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Nepheline.xlsx', dtype=dtypes)) # 17
# ol_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Olivine.xlsx', dtype=dtypes)) # 18
# opx_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Orthopyroxene.xlsx', dtype=dtypes)) # 19
# pl_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Plagioclase.xlsx', dtype=dtypes)) # 20
# qz_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Quartz.xlsx', dtype=dtypes)) # 21
# rt_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Rutile.xlsx', dtype=dtypes)) # 22
# srp_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Serpentine.xlsx', dtype=dtypes)) # 23
# sp_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Spinel.xlsx', dtype=dtypes)) # 24
# tit_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Titanite.xlsx', dtype=dtypes)) # 25
# trm_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Tourmaline.xlsx', dtype=dtypes)) # 26
# zr_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Zircon.xlsx', dtype=dtypes)) # 27
# gl_df = Fe_Conversion(pd.read_excel('../Training_Data/Mineral/Glass_withMI.xlsx', dtype=dtypes)) # 28

# min_df_all = pd.concat([amp_df, ap_df, bt_df, cal_df, chl_df, cpx_df, ep_df, gt_df, hem_df,                        
#                         ilm_df, ks_df, ksp_df, lc_df, mt_df, ml_df, ms_df, ne_df, ol_df, 
#                         opx_df, pl_df, qz_df, rt_df, srp_df, sp_df, tit_df, trm_df, zr_df, gl_df], axis = 0)

# min_df_work = min_df_all[['Sample Name', 'SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO',
#                           'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Cr2O3', 'NiO', 'B2O3', 'ZrO2', 'Total', 
#                           'Tectonic Setting', 'Mineral', 'Sample Type', 'Volcano', 'Source']]

# min_df = min_df_work.copy()
# min_df.rename(columns={"FeOt_F": "FeOt"}, inplace=True)

# min_df.to_csv('../Training_Data/min_df_v2.csv', index=False)
# constants = ['Sample Name', 'Total', 'Tectonic Setting', 'Mineral', 'Sample Type', 'Volcano', 'Source']

min_df = pd.read_csv('../Training_Data/min_df_v2.csv') 

# %% 

amp_calc = mm.AmphiboleCalculator(min_df[min_df.Mineral=='Amphibole']) # [['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Cr2O3']])
amp_comp = amp_calc.calculate_components()
display(amp_comp)

amp_comp_filt = amp_comp.loc[((amp_comp['Cation_Sum'].between(15, 16)) & 
                              (amp_comp['Ca_B_leake'].between(1.5, 2.1)) & 
                              (amp_comp['Al_T_leake'].between(0.5, 2.25)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
# Blundy and Holland, 1990 plot A_site v. Al_IV and Al_VI v. Al_IV
ax[0].scatter(amp_comp['Al_T_leake'], amp_comp['Na_A_leake'] + amp_comp['K_A_leake'], s = 5, color = 'r')
ax[0].scatter(amp_comp_filt['Al_T_leake'], amp_comp_filt['Na_A_leake'] + amp_comp_filt['K_A_leake'], s = 5, color = 'g')
ax[0].set_xlabel('Al_T_leake (Tetrahedral)')
ax[0].set_ylabel('Na+K_A_leake_Amp (A-site)')
ax[1].scatter(amp_comp['Cation_Sum'], amp_comp['Ca_B_leake'], s = 5, color = 'r')
ax[1].scatter(amp_comp_filt['Cation_Sum'], amp_comp_filt['Ca_B_leake'], s = 5, color = 'g')
ax[1].set_xlabel('Cation_Sum')
ax[1].set_ylabel('Ca_B_leake (B-site)')
plt.tight_layout()

# %% 

ap_calc = mm.ApatiteCalculator(min_df[min_df.Mineral=='Apatite'])
ap_comp = ap_calc.calculate_components()
display(ap_comp)

ap_comp_filt = ap_comp.loc[((ap_comp.Ca_cat_13ox.between(4.9, 5.3)) & (ap_comp.P_cat_13ox.between(3, 3.2)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(ap_comp['CaO'], ap_comp['P2O5'], s = 5, color = 'r')
ax[0].scatter(ap_comp_filt['CaO'], ap_comp_filt['P2O5'], s = 5, color = 'g')
ax[0].set_xlabel('CaO')
ax[0].set_ylabel('P2O5')
ax[1].scatter(ap_comp['Ca_cat_13ox'], ap_comp['P_cat_13ox'], s = 5, color = 'r')
ax[1].scatter(ap_comp_filt['Ca_cat_13ox'], ap_comp_filt['P_cat_13ox'], s = 5, color = 'g')
ax[1].set_xlabel('Ca_cat_13ox')
ax[1].set_ylabel('P_cat_13ox')
plt.tight_layout()


# %%

bt_calc = mm.BiotiteCalculator(min_df[min_df.Mineral=='Biotite'])
bt_comp = bt_calc.calculate_components()
display(bt_comp)

bt_comp_filt = bt_comp.loc[((bt_comp.Cation_Sum.between(7.6, 8.1)) & (bt_comp.M_site.between(2.25, 2.85)) & (bt_comp.T_site.between(3.8, 4.2)) )]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(bt_comp['Cation_Sum'], bt_comp['M_site'], s = 5, color = 'r')
ax[0].scatter(bt_comp_filt['Cation_Sum'], bt_comp_filt['M_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('M_site')
ax[1].scatter(bt_comp['M_site'], bt_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(bt_comp_filt['M_site'], bt_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('Cation_Sum')
ax[1].set_ylabel('T_site')
plt.tight_layout()


# %%

cal_calc = mm.CalciteCalculator(min_df[min_df.Mineral=='Calcite'])
cal_comp = cal_calc.calculate_components()
display(cal_comp)

cal_comp_filt = cal_comp.loc[((cal_comp.Cation_Sum.between(0.98, 1.02)) & (cal_comp.C_site.between(0.99, 1.05)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(cal_comp['Cation_Sum'], cal_comp['C_site'], s = 5, color = 'r')
ax[0].scatter(cal_comp_filt['Cation_Sum'], cal_comp_filt['C_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('C_site')
ax[1].scatter(cal_comp['M_site'], cal_comp['C_site'], s = 5, color = 'r')
ax[1].scatter(cal_comp_filt['M_site'], cal_comp_filt['C_site'], s = 5, color = 'g')
ax[1].set_xlabel('M_site')
ax[1].set_ylabel('C_site')
plt.tight_layout()

# %%

chl_calc = mm.ChloriteCalculator(min_df[min_df.Mineral=='Chlorite'])
chl_comp = chl_calc.calculate_components()
display(chl_comp)

chl_comp_filt = chl_comp.loc[((chl_comp.Cation_Sum.between(9, 10.25)) & (chl_comp.T_site.between(5.3, 6)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(chl_comp['Cation_Sum'], chl_comp['VII_site'], s = 5, color = 'r')
ax[0].scatter(chl_comp_filt['Cation_Sum'], chl_comp_filt['VII_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('VII_site')
ax[1].scatter(chl_comp['M_site'], chl_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(chl_comp_filt['M_site'], chl_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('M_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

cpx_calc = mm.ClinopyroxeneCalculator(min_df[min_df.Mineral=='Clinopyroxene'])
cpx_comp = cpx_calc.calculate_components()
display(cpx_comp)

cpx_comp_filt = cpx_comp.loc[((cpx_comp.Cation_Sum.between(3.95, 4.05)) & (cpx_comp.Wo.between(0.35, 0.525)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(cpx_comp['Cation_Sum'], cpx_comp['Wo'], s = 5, color = 'r')
ax[0].scatter(cpx_comp_filt['Cation_Sum'], cpx_comp_filt['Wo'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('Wo')
ax[1].scatter(cpx_comp['M_site'], cpx_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(cpx_comp_filt['M_site'], cpx_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('M_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

ep_calc = mm.EpidoteCalculator(min_df[min_df.Mineral=='Epidote'])
ep_comp = ep_calc.calculate_components()
display(ep_comp)

ep_comp_filt = ep_comp.loc[((ep_comp.Cation_Sum.between(7.9, 8.1)) & (ep_comp.M_site.between(2.8, 3.3)) & (ep_comp.Z_site.between(2.85, 3.15)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(ep_comp['Cation_Sum'], ep_comp['A_site'], s = 5, color = 'r')
ax[0].scatter(ep_comp_filt['Cation_Sum'], ep_comp_filt['A_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('A_site')
ax[1].scatter(ep_comp['M_site'], ep_comp['Z_site'], s = 5, color = 'r')
ax[1].scatter(ep_comp_filt['M_site'], ep_comp_filt['Z_site'], s = 5, color = 'g')
ax[1].set_xlabel('M_site')
ax[1].set_ylabel('Z_site')
plt.tight_layout()

# %% 

gt_calc = mm.GarnetCalculator(min_df[min_df.Mineral=='Garnet'])
gt_comp = gt_calc.calculate_components()
display(gt_comp)

gt_comp_filt = gt_comp.loc[((gt_comp.Cation_Sum.between(7.96, 8.04)) & (gt_comp.X_site.between(2.9, 3.05)) & (gt_comp.T_site.between(4.3, 5.1)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(gt_comp['Cation_Sum'], gt_comp['X_site'], s = 5, color = 'r')
ax[0].scatter(gt_comp_filt['Cation_Sum'], gt_comp_filt['X_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('X_site')
ax[1].scatter(gt_comp['Y_site'], gt_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(gt_comp_filt['Y_site'], gt_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('Y_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

hem_calc = mm.OxideCalculator(min_df[min_df.Mineral=='Hematite'])
hem_comp = hem_calc.calculate_components()
display(hem_comp)

hem_comp_filt = hem_comp.loc[((hem_comp.Cation_Sum.between(2.0, 2.3)))] # & (hem_comp.Fe_Ti.between(2.05, 2.3)) & (hem_comp.A_site_expanded.between(0.66, 0.76)))

fe_conversion = 159.688 / (2 * 71.8464)
hem_comp_filt["FeOt"] = (hem_comp_filt["FeO"] + hem_comp_filt["Fe2O3"] / fe_conversion)

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(hem_comp['Cation_Sum'], hem_comp['Fe_Ti'], s = 5, color = 'r')
ax[0].scatter(hem_comp_filt['Cation_Sum'], hem_comp_filt['Fe_Ti'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('Fe_Ti')
ax[1].scatter(hem_comp['A_site_expanded'], hem_comp['B_site'], s = 5, color = 'r')
ax[1].scatter(hem_comp_filt['A_site_expanded'], hem_comp_filt['B_site'], s = 5, color = 'g')
ax[1].set_xlabel('A_site_expanded')
ax[1].set_ylabel('B_site')
plt.tight_layout()

# %% 

ilm_calc = mm.OxideCalculator(min_df[min_df.Mineral=='Ilmenite'])
ilm_comp = ilm_calc.calculate_components()
display(ilm_comp)

ilm_comp_filt = ilm_comp.loc[((ilm_comp.Cation_Sum.between(1.99, 2.01)) & (ilm_comp.Fe_Ti.between(1.85, 1.95)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(ilm_comp['Cation_Sum'], ilm_comp['Fe_Ti'], s = 5, color = 'r')
ax[0].scatter(ilm_comp_filt['Cation_Sum'], ilm_comp_filt['Fe_Ti'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('Fe_Ti')
ax[1].scatter(ilm_comp['A_site_expanded'], ilm_comp['B_site'], s = 5, color = 'r')
ax[1].scatter(ilm_comp_filt['A_site_expanded'], ilm_comp_filt['B_site'], s = 5, color = 'g')
ax[1].set_xlabel('A_site_expanded')
ax[1].set_ylabel('B_site')
plt.tight_layout()

# %%

ks_calc = mm.KalsiliteCalculator(min_df[min_df.Mineral=='Kalsilite'])
ks_comp = ks_calc.calculate_components()
display(ks_comp)

ks_comp_filt = ks_comp.loc[((ks_comp.Cation_Sum.between(2.95, 3.05)) & (ks_comp.A_site.between(0.95, 1.05)) & (ks_comp.T_site.between(1.95, 2.05)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(ks_comp['Cation_Sum'], ks_comp['Cation_Sum'], s = 5, color = 'r')
ax[0].scatter(ks_comp_filt['Cation_Sum'], ks_comp_filt['Cation_Sum'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('Cation_Sum')
ax[1].scatter(ks_comp['A_site'], ks_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(ks_comp_filt['A_site'], ks_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('A_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

ksp_calc = mm.FeldsparCalculator(min_df[min_df.Mineral=='KFeldspar'])
ksp_comp = ksp_calc.calculate_components()
display(ksp_comp)

ksp_comp_filt = ksp_comp.loc[((ksp_comp.Cation_Sum.between(4.9, 5.1)) & (ksp_comp.T_site.between(3.96, 4.04)) & (ksp_comp.SiO2.between(40, 80)) & (ksp_comp.K2O.between(5, 17)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(ksp_comp['SiO2'], ksp_comp['K2O'], s = 5, color = 'r')
ax[0].scatter(ksp_comp_filt['SiO2'], ksp_comp_filt['K2O'], s = 5, color = 'g')
ax[0].set_xlabel('SiO2')
ax[0].set_ylabel('K2O')
ax[1].scatter(ksp_comp['Cation_Sum'], ksp_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(ksp_comp_filt['Cation_Sum'], ksp_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('Cation_Sum')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

lc_calc = mm.LeuciteCalculator(min_df[min_df.Mineral=='Leucite'])
lc_comp = lc_calc.calculate_components()
display(lc_comp)

lc_comp_filt = lc_comp.loc[((lc_comp.Cation_Sum.between(3.90, 4.1)) & (lc_comp.T_site.between(2.9, 3.1)) & (lc_comp.Channel_site.between(0.9, 1.05)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(lc_comp['Cation_Sum'], lc_comp['T_site'], s = 5, color = 'r')
ax[0].scatter(lc_comp_filt['Cation_Sum'], lc_comp_filt['T_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('T_site')
ax[1].scatter(lc_comp['Channel_site'], lc_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(lc_comp_filt['Channel_site'], lc_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('Channel_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %%

mt_calc = mm.MagnetiteCalculator(min_df[min_df.Mineral=='Magnetite'])
mt_comp = mt_calc.calculate_components()
display(mt_comp)

mt_comp_filt = mt_comp.loc[((mt_comp.Cation_Sum.between(3.2, 3.8)) & (mt_comp.A_site.between(2.6, 3.6)) & (mt_comp.B_site.between(0.2, 0.8)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(mt_comp['Cation_Sum'], mt_comp['A_site'], s = 5, color = 'r')
ax[0].scatter(mt_comp_filt['Cation_Sum'], mt_comp_filt['A_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('A_site')
ax[1].scatter(mt_comp['A_site'], mt_comp['B_site'], s = 5, color = 'r')
ax[1].scatter(mt_comp_filt['A_site'], mt_comp_filt['B_site'], s = 5, color = 'g')
ax[1].set_xlabel('A_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

ml_calc = mm.MeliliteCalculator(min_df[min_df.Mineral=='Melilite'])
ml_comp = ml_calc.calculate_components()
display(ml_comp)

ml_comp_filt = ml_comp.loc[((ml_comp.Cation_Sum.between(4.95, 5.05)) & (ml_comp.T_site.between(1.9, 2.1)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(ml_comp['Cation_Sum'], ml_comp['T_site'], s = 5, color = 'r')
ax[0].scatter(ml_comp_filt['Cation_Sum'], ml_comp_filt['T_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('T_site')
ax[1].scatter(ml_comp['A_site'], ml_comp['B_site'], s = 5, color = 'r')
ax[1].scatter(ml_comp_filt['A_site'], ml_comp_filt['B_site'], s = 5, color = 'g')
ax[1].set_xlabel('A_site')
ax[1].set_ylabel('B_site')
plt.tight_layout()

# %% 

ms_calc = mm.MuscoviteCalculator(min_df[min_df.Mineral=='Muscovite'])
ms_comp = ms_calc.calculate_components()
display(ms_comp)

ms_comp_filt = ms_comp.loc[((ms_comp.Cation_Sum.between(6.9, 7.1)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(ms_comp['Cation_Sum'], ms_comp['Cation_Sum'], s = 5, color = 'r')
ax[0].scatter(ms_comp_filt['Cation_Sum'], ms_comp_filt['Cation_Sum'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('T_site')
ax[1].scatter(ms_comp['X_site'], ms_comp['M_site'], s = 5, color = 'r')
ax[1].scatter(ms_comp_filt['X_site'], ms_comp_filt['M_site'], s = 5, color = 'g')
ax[1].set_xlabel('A_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

ne_calc = mm.NephelineCalculator(min_df[min_df.Mineral=='Nepheline'])
ne_comp = ne_calc.calculate_components()
display(ne_comp)

ne_comp_filt = ne_comp.loc[((ne_comp.Cation_Sum > 23) & (ne_comp.T_site.between(15.6, 16.4)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(ne_comp['Cation_Sum'], ne_comp['T_site'], s = 5, color = 'r')
ax[0].scatter(ne_comp_filt['Cation_Sum'], ne_comp_filt['T_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('T_site')
ax[1].scatter(ne_comp['A_site'], ne_comp['B_site'], s = 5, color = 'r')
ax[1].scatter(ne_comp_filt['A_site'], ne_comp_filt['B_site'], s = 5, color = 'g')
ax[1].set_xlabel('A_site')
ax[1].set_ylabel('B_site')
plt.tight_layout()

# %% 

ol_calc = mm.OlivineCalculator(min_df[min_df.Mineral=='Olivine'])
ol_comp = ol_calc.calculate_components()
display(ol_comp)

ol_comp_filt = ol_comp.loc[((ol_comp.Cation_Sum.between(2.95, 3.05)) & (ol_comp.M_site_expanded.between(1.95, 2.05) )) ]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(ol_comp['SiO2'], ol_comp['MgO'], s = 5, color = 'r')
ax[0].scatter(ol_comp_filt['SiO2'], ol_comp_filt['MgO'], s = 5, color = 'g')
ax[0].set_xlabel('SiO2')
ax[0].set_ylabel('MgO')
ax[1].scatter(ol_comp['Cation_Sum'], ol_comp['M_site_expanded'], s = 5, color = 'r')
ax[1].scatter(ol_comp_filt['Cation_Sum'], ol_comp_filt['M_site_expanded'], s = 5, color = 'g')
ax[1].set_xlabel('Ol_Cation_Sum')
ax[1].set_ylabel('M_site_expanded')
plt.tight_layout()

# %%

opx_calc = mm.OrthopyroxeneCalculator(min_df[min_df.Mineral=='Orthopyroxene'])
opx_comp = opx_calc.calculate_components()
display(opx_comp)

opx_comp_filt = opx_comp.loc[((opx_comp.Cation_Sum.between(3.95, 4.05)) & (opx_comp.Wo.between(-0.01, 0.05) )) ]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(opx_comp['SiO2'], opx_comp['MgO'], s = 5, color = 'r')
ax[0].scatter(opx_comp_filt['SiO2'], opx_comp_filt['MgO'], s = 5, color = 'g')
ax[0].set_xlabel('SiO2')
ax[0].set_ylabel('MgO')
ax[1].scatter(opx_comp['Cation_Sum'], opx_comp['Wo'], s = 5, color = 'r')
ax[1].scatter(opx_comp_filt['Cation_Sum'], opx_comp_filt['Wo'], s = 5, color = 'g')
ax[1].set_xlabel('Cation_Sum')
ax[1].set_ylabel('Wo')
plt.tight_layout()

# %% 

pl_calc = mm.FeldsparCalculator(min_df[min_df.Mineral=='Plagioclase'])
pl_comp = pl_calc.calculate_components()
display(pl_comp)

pl_comp_filt = pl_comp.loc[((pl_comp.Cation_Sum.between(4.95, 5.05)) & (pl_comp.M_site.between(0.9, 1.05)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(pl_comp['Cation_Sum'], pl_comp['An'], s = 5, color = 'r')
ax[0].scatter(pl_comp_filt['Cation_Sum'], pl_comp_filt['An'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('An')
ax[1].scatter(pl_comp['M_site'], pl_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(pl_comp_filt['M_site'], pl_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('M_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

qz_calc = mm.QuartzCalculator(min_df[min_df.Mineral=='Quartz'])
qz_comp = qz_calc.calculate_components()
display(qz_comp)

qz_comp_filt = qz_comp.loc[((qz_comp.Cation_Sum.between(0.9975, 1.0025)))]

fig, ax = plt.subplots(1, 1, figsize = (5, 5))
ax.scatter(qz_comp['Cation_Sum'], qz_comp['T_site'], s = 5, color = 'r')
ax.scatter(qz_comp_filt['Cation_Sum'], qz_comp_filt['T_site'], s = 5, color = 'g')
ax.set_xlabel('Cation_Sum')
ax.set_ylabel('T_site')
plt.tight_layout()

# %%

rt_calc = mm.RutileCalculator(min_df[min_df.Mineral=='Rutile'])
rt_comp = rt_calc.calculate_components()
display(rt_comp)

rt_comp_filt = rt_comp.loc[((rt_comp.Cation_Sum.between(0.9975, 1.005)) & (rt_comp.M_site.between(0.994, 1.005)))]

fig, ax = plt.subplots(1, 1, figsize = (5, 5))
ax.scatter(rt_comp['Cation_Sum'], rt_comp['M_site'], s = 5, color = 'r')
ax.scatter(rt_comp_filt['Cation_Sum'], rt_comp_filt['M_site'], s = 5, color = 'g')
ax.set_xlabel('Cation_Sum')
ax.set_ylabel('M_site')
plt.tight_layout()

# %% 

srp_calc = mm.SerpentineCalculator(min_df[min_df.Mineral=='Serpentine'])
srp_comp = srp_calc.calculate_components()
display(srp_comp)

srp_comp_filt = srp_comp.loc[((srp_comp.Cation_Sum.between(9.5, 10.5)) & (srp_comp.T_site.between(3.5, 4.5)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(srp_comp['Cation_Sum'], srp_comp['XMg'], s = 5, color = 'r')
ax[0].scatter(srp_comp_filt['Cation_Sum'], srp_comp_filt['XMg'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('XMg')
ax[1].scatter(srp_comp['M_site'], srp_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(srp_comp_filt['M_site'], srp_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('M_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

sp_calc = mm.SpinelCalculator(min_df[min_df.Mineral=='Spinel'])
sp_comp = sp_calc.calculate_components()
display(sp_comp)

sp_comp_filt = sp_comp.loc[((sp_comp.Cation_Sum.between(2.95, 3.45)) & (sp_comp.B_site.between(0.75, 2.05)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(sp_comp['Cation_Sum'], sp_comp['A_B_site'], s = 5, color = 'r')
ax[0].scatter(sp_comp_filt['Cation_Sum'], sp_comp_filt['A_B_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('Fe_Ti')
ax[1].scatter(sp_comp['A_site'], sp_comp['B_site'], s = 5, color = 'r')
ax[1].scatter(sp_comp_filt['A_site'], sp_comp_filt['B_site'], s = 5, color = 'g')
ax[1].set_xlabel('A_site')
ax[1].set_ylabel('B_site')
plt.tight_layout()

# %%

tit_calc = mm.TitaniteCalculator(min_df[min_df.Mineral=='Titanite'])
tit_comp = tit_calc.calculate_components()
display(tit_comp)

tit_comp_filt = tit_comp.loc[((tit_comp.Cation_Sum.between(2.95, 3.1)) & (tit_comp.T_site.between(0.9, 1.1)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(tit_comp['Cation_Sum'], tit_comp['VII_site'], s = 5, color = 'r')
ax[0].scatter(tit_comp_filt['Cation_Sum'], tit_comp_filt['VII_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('VII_site')
ax[1].scatter(tit_comp['M_site'], tit_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(tit_comp_filt['M_site'], tit_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('M_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %%

trm_calc = mm.TourmalineCalculator(min_df[min_df.Mineral=='Tourmaline'])
trm_comp = trm_calc.calculate_components()
display(trm_comp)

trm_comp_filt = trm_comp.loc[((trm_comp.Cation_Sum.between(19.75, 20.25)) & (trm_comp.Y_site.between(9.25, 9.75)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(trm_comp['Cation_Sum'], trm_comp['T_site'], s = 5, color = 'r')
ax[0].scatter(trm_comp_filt['Cation_Sum'], trm_comp_filt['T_site'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('T_site')
ax[1].scatter(trm_comp['X_site'], trm_comp['Y_site'], s = 5, color = 'r')
ax[1].scatter(trm_comp_filt['X_site'], trm_comp_filt['Y_site'], s = 5, color = 'g')
ax[1].set_xlabel('X_site')
ax[1].set_ylabel('Y_site')
plt.tight_layout()

# %% 

zr_calc = mm.ZirconCalculator(min_df[min_df.Mineral=='Zircon'])
zr_comp = zr_calc.calculate_components()
display(zr_comp)

zr_comp_filt = zr_comp.loc[((zr_comp.Cation_Sum.between(1.99, 2.01)) & (zr_comp.T_site.between(0.96, 1.02)))]

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(zr_comp['Cation_Sum'], zr_comp['Zr_cat_4ox'], s = 5, color = 'r')
ax[0].scatter(zr_comp_filt['Cation_Sum'], zr_comp_filt['Zr_cat_4ox'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('Zr_cat_4ox')
ax[1].scatter(zr_comp['M_site'], zr_comp['T_site'], s = 5, color = 'r')
ax[1].scatter(zr_comp_filt['M_site'], zr_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('M_site')
ax[1].set_ylabel('T_site')
plt.tight_layout()

# %% 

from pyrolite.util.classification import TAS

gl_df = min_df[min_df.Mineral=='Glass']
gl_df = gl_df[gl_df.SiO2 > 40]
gl_df = gl_df[['Sample Name', 'SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Cr2O3', 'Mineral']]

cm = TAS()
gl_df['Na2O + K2O'] = gl_df['Na2O'] + gl_df['K2O']
gl_df["TAS"] = cm.predict(gl_df)
gl_df["Rocknames"] = gl_df.TAS.apply(lambda x: cm.fields.get(x, {"name": None})["name"])
gl_df

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, add_labels=True)
plt.scatter(gl_df.SiO2, gl_df.Na2O + gl_df.K2O, marker='o', c='tab:blue', edgecolors='k', linewidth=0.25, label=None)

# First calculate minimum samples needed per group
min_samples = int(2000 / len(gl_df['TAS'].unique()))

# Stratified resampling with replacement for small groups
resampled_df = (gl_df.groupby('TAS', group_keys=False)
                  .apply(lambda x: x.sample(n=max(min_samples, 
                                                  int(2000*len(x)/len(gl_df))),
                                                  replace=True,  # Always use replacement
                                                  random_state=42))
                  .sample(n=2000, random_state=42)
                  .reset_index(drop=True))

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, add_labels=True)
plt.scatter(resampled_df.SiO2, resampled_df['Na2O + K2O'], 
            marker='o', c='tab:blue', edgecolors='k', 
            linewidth=0.25, label=None)
plt.title('Stratified Resampling (n=2000)')
plt.show()

gl_comp_filt = resampled_df
display(gl_comp_filt)

gl_df.to_csv('../Training_Data/glasses_df.csv', index=False)


oxideslab = ['Sample Name', 'SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Cr2O3', 'Mineral']
gl_comp_filt = gl_comp_filt[oxideslab]
gl_comp_filt.to_csv('../Training_Data/glasses_df_sampled.csv', index=False)

# %%

min_df_all = pd.concat([amp_comp_filt, ap_comp_filt, bt_comp_filt, cal_comp_filt, chl_comp_filt, cpx_comp_filt, ep_comp_filt, gt_comp_filt, hem_comp_filt,                        
                        ilm_comp_filt, ks_comp_filt, ksp_comp_filt, lc_comp_filt, mt_comp_filt, ml_comp_filt, ms_comp_filt, ne_comp_filt, ol_comp_filt, 
                        opx_comp_filt, pl_comp_filt, qz_comp_filt, rt_comp_filt, srp_comp_filt, sp_comp_filt, tit_comp_filt, trm_comp_filt, zr_comp_filt, gl_df], axis = 0)
display(min_df_all)

display(min_df_all.Mineral.value_counts())

oxideslab = ['Sample Name', 'SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Cr2O3', 'Mineral']
min_df_lim = min_df_all[oxideslab]
min_df_lim.to_csv('../Training_Data/min_df_v2_clean_natural.csv', index=False)

# %% 

pl_endmembers = {
    # Albite: NaAlSi₃O₈
    'Ab': {'Na': 1, 'Al': 1, 'Si': 3, 'O': 8},
    # Anorthite: CaAl₂Si₂O₈
    'An': {'Ca': 1, 'Al': 2, 'Si': 2, 'O': 8},
}

# define coupled sites
pl_sites = {
    'M': {'elements': ['Na', 'Ca'], 'total': 1,
          'coupling': {'T': {'Na': ['Al', 1], 'Ca': ['Al', 2]}}},
    'T': {'elements': ['Al', 'Si'], 'total': 4}
}

# minor elements
pl_minors = {'K': {'distribution': 'exponential', 'scale': 0.01, 'max_fraction': 0.02}}

# create generator
pl_gen = mm.SolidSolutionGenerator(
    endmembers=pl_endmembers,
    oxygen_basis=8,
    element_noise_scale=0.05,
    min_site_fraction=0.2,
    minor_elements=pl_minors,
    mixing_dist='beta',
    mixing_params={'a': 2, 'b': 2}
)

# generate samples
df_pl = pl_gen.generate(1000)
pl_calc_synth = mm.FeldsparCalculator(df_pl)
pl_comp_synth = pl_calc_synth.calculate_components()
display(pl_comp_synth)

stats_pl = pl_gen.compare_distributions(base_df=pl_comp_filt, synth_df=pl_comp_synth, suptitle="Plagioclase")
display(stats_pl)

# Example scatter‐plot comparing base vs. synthetic oxide proportions
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].scatter(pl_comp_filt["Na2O"],  pl_comp_filt["CaO"],  s=5, c="g")
ax[0].scatter(pl_comp_synth["Na2O"], pl_comp_synth["CaO"], s=5, c="r")
ax[0].set_xlabel("Na2O")
ax[0].set_ylabel("CaO")

ax[1].scatter(pl_comp_filt["Al2O3"],  pl_comp_filt["SiO2"],  s=5, c="g")
ax[1].scatter(pl_comp_synth["Al2O3"], pl_comp_synth["SiO2"], s=5, c="r")
ax[1].set_xlabel("Al2O3")
ax[1].set_ylabel("SiO2")

ax[2].scatter(pl_comp_filt["An"], pl_comp_filt["Ab"], s=5, c="g", label="Base")
ax[2].scatter(pl_comp_synth["An"], pl_comp_synth["Ab"], s=5, c="r", label="Synth")
ax[2].set_xlabel("An (Ca/(Ca+Na))")
ax[2].set_ylabel("Ab (Na/(Ca+Na))")
ax[2].legend()
plt.tight_layout()

# %%

ol_endmembers = {
    # Forsterite: Mg2SiO4
    'Fo': {'Mg': 2, 'Si': 1, 'O': 4},
    # Fayalite: Fe2SiO4
    'Fa': {'Fe2t': 2, 'Si': 1, 'O': 4}
}

# minor elements
ol_minors = {
    'Ca': {'distribution': 'exponential', 'scale': 0.01, 'max_fraction': 0.01},
    'Mn': {'distribution': 'exponential', 'scale': 0.01, 'max_fraction': 0.01}
}

# create generator
ol_gen = mm.SolidSolutionGenerator(
    endmembers=ol_endmembers,
    oxygen_basis=4,
    element_noise_scale=0.025,
    min_site_fraction=0.2,
    minor_elements=ol_minors,
    mixing_dist='beta',
    mixing_params={'a': 1, 'b': 1}
)

# generate samples
df_ol = ol_gen.generate(1000)
ol_calc_synth = mm.OlivineCalculator(df_ol)
ol_comp_synth = ol_calc_synth.calculate_components()
display(ol_comp_synth)

stats_ol = ol_gen.compare_distributions(base_df=ol_comp_filt, synth_df=ol_comp_synth, suptitle="Olivine")
display(stats_ol)

# scatter‐plot comparing base vs. synthetic oxide proportions
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].scatter(ol_comp_filt["FeOt"],  ol_comp_filt["MgO"],  s=5, c="g")
ax[0].scatter(ol_comp_synth["FeOt"], ol_comp_synth["MgO"], s=5, c="r")
ax[0].set_xlabel("FeO")
ax[0].set_ylabel("MgO")

ax[1].scatter(ol_comp_filt["SiO2"],  ol_comp_filt["MgO"],  s=5, c="g")
ax[1].scatter(ol_comp_synth["SiO2"], ol_comp_synth["MgO"], s=5, c="r")
ax[1].set_xlabel("SiO2")
ax[1].set_ylabel("MgO")

ax[2].scatter(ol_comp_filt["XFo"], ol_comp_filt["M_site_expanded"], s=5, c="g", label="Base")
ax[2].scatter(ol_comp_synth["XFo"], ol_comp_synth["M_site_expanded"], s=5, c="r", label="Synth")
ax[2].set_xlabel("XFo (Mg/(Mg+Fe))")
ax[2].set_ylabel("M-site Expanded")
ax[2].legend()
plt.tight_layout()

# %% 

import mineralML.constants as C

C.OXIDE_MASSES.update({"ZrO2":123.222, "HfO2":210.484})
C.CATION_NUMBERS.update({"ZrO2":1, "HfO2":1})
C.OXYGEN_NUMBERS.update({"ZrO2":2, "HfO2":2})
C.OXIDE_TO_CATION_MAP.update({"ZrO2":"Zr", "HfO2":"Hf"})
C.CATION_TO_OXIDE_MAP.update({"Zr":"ZrO2", "Hf":"HfO2"})
C.VALENCES.update({"Zr":4, "Hf":4})

# zircon endmembers
zr_endmembers = {
    'Zr': {'Zr': 1, 'Si': 1, 'O': 4}, # zircon
    'Hf': {'Hf': 1, 'Si': 1, 'O': 4}, # hafnon
}

# no coupled sites for zircon
zr_sites = {}

# optional traces
zr_minors = {
    'Mg': {'distribution': 'exponential', 'scale': 0.01, 'max_fraction': 0.0001},
    'Fe2t': {'distribution': 'exponential', 'scale': 0.01, 'max_fraction': 0.0001}
}  

#beta mix
zr_mix_params = {'a': 2, 'b': 98}

# instantiate generator
zr_gen = mm.SolidSolutionGenerator(
    endmembers = zr_endmembers,
    oxygen_basis = 4,
    element_noise_scale = 0.02,
    min_site_fraction = 0.2,
    minor_elements = zr_minors,
    mixing_dist = 'beta',
    mixing_params = zr_mix_params
)

# generate 1000 synthetic zircons
df_zr = zr_gen.generate(n_samples=(1000-len(min_df_all[min_df_all.Mineral=='Zircon'])))
zr_calc_synth = mm.ZirconCalculator(df_zr)
zr_comp_synth = zr_calc_synth.calculate_components()
display(zr_comp_synth)
zr_comp_synth['Mineral'] = 'Zircon'

stats_zr = zr_gen.compare_distributions(base_df=zr_comp_filt,synth_df=zr_comp_synth, suptitle="Zircon")
display(stats_zr)

zr_comp_synth['Hf_per'] = zr_comp_synth['Hf_cat_4ox'] / (zr_comp_synth['Zr_cat_4ox'] + zr_comp_synth['Hf_cat_4ox'])

# quick histogram of Hf_mol_% to verify HfO2 generation
plt.figure(figsize=(6,4))
plt.hist(zr_comp_synth['Hf_per']*100, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Hf/Hf+Zr")
plt.ylabel("Count")
plt.title("Synthetic Zircon Hf Distribution (1-3 mol % mean)")
plt.tight_layout()
plt.show()

C.OXIDE_MASSES.pop("ZrO2", None)
C.OXIDE_MASSES.pop("HfO2", None)
C.CATION_NUMBERS.pop("ZrO2", None)
C.CATION_NUMBERS.pop("HfO2", None)
C.OXYGEN_NUMBERS.pop("ZrO2", None)
C.OXYGEN_NUMBERS.pop("HfO2", None)
C.OXIDE_TO_CATION_MAP.pop("ZrO2", None)
C.OXIDE_TO_CATION_MAP.pop("HfO2", None)
C.CATION_TO_OXIDE_MAP.pop("Zr", None)
C.CATION_TO_OXIDE_MAP.pop("Hf", None)
C.VALENCES.pop("Zr", None)
C.VALENCES.pop("Hf", None)

# %% 

ks_endmembers = {
    "Ks": {"K":  1, "Al": 1, "Si": 1, "O": 4},
    "Ne": {"Na": 1, "Al": 1, "Si": 1, "O": 4}
}

ks_sites = {}
ks_minors = {} # no minors for pure KAlSiO4–NaAlSiO4

gen_ks = mm.SolidSolutionGenerator(
    endmembers = ks_endmembers,
    oxygen_basis = 4,
    minor_elements = ks_minors,
    element_noise_scale = 0.02,
    min_site_fraction = 0.2,
    mixing_dist = "beta",
    mixing_params = {"a": 1, "b": 200},
)

# generate samples
df_ks = gen_ks.generate(n_samples=(1000-len(min_df_all[min_df_all.Mineral=='Kalsilite'])))
ks_calc_synth = mm.KalsiliteCalculator(df_ks)
ks_comp_synth = ks_calc_synth.calculate_components()
display(ks_comp_synth)
ks_comp_synth['Mineral'] = 'Kalsilite'

stats_ks = gen_ks.compare_distributions(base_df=ks_comp_filt, synth_df=ks_comp_synth, suptitle="Kalsilite")
display(stats_ks)

fig, ax = plt.subplots(1, 4, figsize = (20, 5))
ax = ax.flatten()
ax[0].scatter(ks_comp_synth['Cation_Sum'], ks_comp_synth['Cation_Sum'], s = 5, color = 'r')
ax[0].scatter(ks_comp_filt['Cation_Sum'], ks_comp_filt['Cation_Sum'], s = 5, color = 'g')
ax[0].set_xlabel('Cation_Sum')
ax[0].set_ylabel('Cation_Sum')
ax[1].scatter(ks_comp_synth['A_B_site'], ks_comp_synth['T_site'], s = 5, color = 'r')
ax[1].scatter(ks_comp_filt['A_B_site'], ks_comp_filt['T_site'], s = 5, color = 'g')
ax[1].set_xlabel('A_B_site (K+Na)')
ax[1].set_ylabel('T_site')
ax[2].scatter(ks_comp_synth['K2O'], ks_comp_synth['Na2O'], s = 5, color = 'r')
ax[2].scatter(ks_comp_filt['K2O'], ks_comp_filt['Na2O'], s = 5, color = 'g')
ax[2].set_xlabel('K2O')
ax[2].set_ylabel('Na2O')
ax[3].scatter(ks_comp_synth['SiO2'], ks_comp_synth['Al2O3'], s = 5, color = 'r')
ax[3].scatter(ks_comp_filt['SiO2'], ks_comp_filt['Al2O3'], s = 5, color = 'g')
ax[3].set_xlabel('SiO2')
ax[3].set_ylabel('Al2O3')
plt.tight_layout()
plt.show()


# %% 

quartz_endmembers = {
    "Qz1": {"Si": 0.95, "O": 2},
    "Qz2": {"Si": 1, "O": 2}
}
# define minor elements for quartz
quartz_minors = {
    "Al": {"distribution": "exponential", "scale": 0.001, "max_fraction": 0.0016},
    "Ti": {"distribution": "exponential", "scale": 0.001, "max_fraction": 0.0008},
}

# run SolidSolutionGenerator with oxygen_basis = 2
gen_qz = mm.SolidSolutionGenerator(
    endmembers = quartz_endmembers,
    oxygen_basis = 2,
    minor_elements = quartz_minors,
    site_variation = 0.10,
    element_noise_scale = 0.05,
    min_site_fraction = 0.95,
    mixing_dist = "beta",
    mixing_params = {"a": 1, "b": 1},
    validate_fn = (lambda ox: True)
)

# generate synthetic quartz analyses
df_qz_synth = gen_qz.generate(n_samples=(1000-len(min_df_all[min_df_all.Mineral=='Quartz'])))

# compare synthetic and natural data 
stats_qz = gen_qz.compare_distributions(base_df = qz_comp_filt, synth_df = df_qz_synth, suptitle = "Quartz")
display(stats_qz)

# compute final formulas
qz_calc_synth = mm.QuartzCalculator(df_qz_synth)
qz_comp_synth = qz_calc_synth.calculate_components()
qz_comp_synth["Mineral"] = "Quartz"
display(qz_comp_synth.head())

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax = ax.flatten()
ax[0].scatter(qz_comp_filt["SiO2"], qz_comp_filt["Al2O3"], s=5, c="green", alpha=0.6)
ax[0].scatter(df_qz_synth["SiO2"], df_qz_synth["Al2O3"], s=5, c="crimson", alpha=0.6)
ax[0].set_xlabel("SiO2 (wt%)")
ax[0].set_ylabel("Al2O3 (wt%)")

ax[1].scatter(qz_comp_filt["Si_cat_2ox"], qz_comp_filt["Al_cat_2ox"], s=5, c="green", alpha=0.6)
ax[1].scatter(df_qz_synth["Si_cat_2ox"], df_qz_synth["Al_cat_2ox"], s=5, c="crimson", alpha=0.6)
ax[1].set_xlabel("Si_cat")
ax[1].set_ylabel("Al_cat")
plt.tight_layout()
plt.show()

# %% 

# game by faking two endmembers, mixed with beta distribution
hematite_endmembers = {
    "Hem_fe3": {"Fe3": 2, "O": 3},
    "Hem_fe2": {"Fe2": 2, "O": 3}
}

# define Al as minor element
hematite_minors = {
    "Al": {"distribution": "exponential", "scale": 0.01, "max_fraction": 0.02}
}

# run SolidSolutionGenerator with oxygen_basis = 3
gen_hem = mm.SolidSolutionGenerator(
    endmembers = hematite_endmembers,
    oxygen_basis = 3,
    minor_elements = hematite_minors,
    site_variation = 0.01,
    element_noise_scale = 0.05,
    min_site_fraction = 0.95,
    mixing_dist = "beta",
    mixing_params = {"a": 8, "b": 92}, # mean Fe2‐fraction ≈ 2/(100)=0.02
    validate_fn = (lambda ox: True) # keep all oxide‐wt variants
)

# generate synthetic hematite analyses
df_hem_synth = gen_hem.generate(n_samples=(1000-len(min_df_all[min_df_all.Mineral=='Hematite'])))
fe_conversion = 159.688 / (2 * 71.8464)
df_hem_synth["FeOt"] = (df_hem_synth["FeO"] + df_hem_synth["Fe2O3"] / fe_conversion)

# compare synthetic vs. natural
stats_hem = gen_hem.compare_distributions(base_df = hem_comp_filt, synth_df = df_hem_synth, suptitle = "Hematite")
display(stats_hem)

hem_calc_synth = mm.OxideCalculator(df_hem_synth[["FeOt", "Al2O3"]])
hem_comp_synth = hem_calc_synth.calculate_components()
hem_comp_synth["Mineral"] = "Hematite"
display(hem_comp_synth)
hem_comp_synth["FeOt"] = (hem_comp_synth["FeO"] + hem_comp_synth["Fe2O3"] / fe_conversion)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax = ax.flatten()
ax[0].scatter(hem_comp_filt["FeOt"], hem_comp_filt["Al2O3"], s=5, c="green", alpha=0.6, label="Natural")
ax[0].scatter(hem_comp_synth["FeOt"], hem_comp_synth["Al2O3"], s=5, c="crimson", alpha=0.6, label="Synthetic")
ax[0].set_xlabel("FeOt (wt%)")
ax[0].set_ylabel("Al2O3 (wt%)")

ax[1].scatter(hem_comp_filt["FeOt"], hem_comp_filt["Al2O3"], s=5, c="green", alpha=0.6, label="Natural")
ax[1].scatter(hem_comp_synth["FeOt"], hem_comp_synth["Al2O3"], s=5, c="crimson", alpha=0.6, label="Synthetic")
ax[1].set_xlabel("FeOt (wt%)")
ax[1].set_ylabel("Al2O3 (wt%)")
plt.tight_layout()
plt.show()


# %% 


min_df_all_synth = pd.concat([amp_comp_filt, ap_comp_filt, bt_comp_filt, cal_comp_filt, chl_comp_filt,
                              cpx_comp_filt, ep_comp_filt, gt_comp_filt, hem_comp_filt, hem_comp_synth,
                              ilm_comp_filt, ks_comp_filt, ks_comp_synth, ksp_comp_filt, lc_comp_filt,
                              mt_comp_filt, ml_comp_filt, ms_comp_filt, ne_comp_filt, ol_comp_filt,
                              opx_comp_filt, pl_comp_filt, qz_comp_filt, qz_comp_synth, rt_comp_filt,
                              srp_comp_filt, sp_comp_filt, tit_comp_filt, trm_comp_filt,
                              zr_comp_filt, zr_comp_synth, gl_df], axis=0)
display(min_df_all_synth)
display(min_df_all_synth.Mineral.value_counts())

oxideslab = ['Sample Name', 'SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Cr2O3', 'Mineral']
min_df_lim_synth = min_df_all_synth[oxideslab]
min_df_lim_synth.to_csv('../Training_Data/min_df_v2_clean_synth.csv', index=False)

# %%
