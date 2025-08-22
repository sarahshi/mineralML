# %% 

""" Created on November 9, 2022 // @author: Sarah Shi """

import numpy as np
import pandas as pd
import numpy as np

import os
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 14})
plt.rcParams['pdf.fonttype'] = 42

# %% 

path_parent = os.getcwd()
# output_dir = ["/GEOROC", "/GEOROC_minerals_orig", "/GEOROC_minerals_lim"] 
# for ii in range(len(output_dir)):
#     if not os.path.exists(path_parent + output_dir[ii]):
#         os.makedirs(path_parent + output_dir[ii], exist_ok=True)

# %%

amphiboles = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_AMPHIBOLES.csv', encoding='latin-1')
amphiboles = amphiboles[~amphiboles.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

apatites = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_APATITES.csv', encoding='latin-1')
apatites = apatites[~apatites.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

carbonates = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_CARBONATES.csv', encoding='latin-1')
carbonates = carbonates[~carbonates.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

chalcogenides = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_CHALCOGENIDES.csv', encoding='latin-1')
chalcogenides = chalcogenides[~chalcogenides.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

clays = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_CLAY_MINERALS.csv', encoding='latin-1')
clays = clays[~clays.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

clinopyroxenes = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_CLINOPYROXENES.csv', encoding='latin-1')
clinopyroxenes = clinopyroxenes[~clinopyroxenes.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

feldspars = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_FELDSPARS.csv', encoding='latin-1')
feldspars = feldspars[~feldspars.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

feldspathoids = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_FELDSPATHOIDES.csv', encoding='latin-1')
feldspathoids = feldspathoids[~feldspathoids.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

garnets = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_GARNETS.csv', encoding='latin-1')
garnets = garnets[~garnets.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

ilmenites = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_ILMENITES.csv', encoding='latin-1')
ilmenites = ilmenites[~ilmenites.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

micas = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_MICA.csv', encoding='latin-1')
micas = micas[~micas.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

olivines = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_OLIVINES.csv', encoding='latin-1')
olivines = olivines[~olivines.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

orthopyroxenes = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_ORTHOPYROXENES.csv', encoding='latin-1')
orthopyroxenes = orthopyroxenes[~orthopyroxenes.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

perovskites = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_PEROVSKITES.csv', encoding='latin-1')
perovskites = perovskites[~perovskites.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

pyroxenes = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_PYROXENES.csv', encoding='latin-1')
pyroxenes = pyroxenes[~pyroxenes.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

quartz = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_QUARTZ.csv', encoding='latin-1')
quartz = quartz[~quartz.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

spinels = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_SPINELS.csv', encoding='latin-1')
spinels = spinels[~spinels.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

titanites = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_TITANITES.csv', encoding='latin-1')
titanites = titanites[~titanites.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

zircons = pd.read_csv('../GEOROC_minerals/2024-12-SGFTFN_ZIRCONS.csv', encoding='latin-1')
zircons = zircons[~zircons.applymap(str).apply(lambda x: x.str.contains(r'\\')).any(axis=1)]

# %%

amphiboles = amphiboles.dropna(subset=['SIO2(WT%)'], how='all')
# apatites = apatites.dropna(subset=['SIO2(WT%)'], how='all')
# carbonates = carbonates.dropna(subset=['SIO2(WT%)'], how='all')
# chalcogenides = chalcogenides.dropna(subset=['SIO2(WT%)'], how='all')
# clays = clays.dropna(subset=['SIO2(WT%)'], how='all')
clinopyroxenes = clinopyroxenes.dropna(subset=['SIO2(WT%)'], how='all')
feldspars = feldspars.dropna(subset=['SIO2(WT%)'], how='all')
feldspathoids = feldspathoids.dropna(subset=['SIO2(WT%)'], how='all')
garnets = garnets.dropna(subset=['SIO2(WT%)'], how='all')
# ilmenites = ilmenites.dropna(subset=['SIO2(WT%)'], how='all')
micas = micas.dropna(subset=['SIO2(WT%)'], how='all')
olivines = olivines.dropna(subset=['SIO2(WT%)'], how='all')
orthopyroxenes = orthopyroxenes.dropna(subset=['SIO2(WT%)'], how='all')
perovskites = perovskites.dropna(subset=['SIO2(WT%)'], how='all')
pyroxenes = pyroxenes.dropna(subset=['SIO2(WT%)'], how='all')
quartz = quartz.dropna(subset=['SIO2(WT%)'], how='all')
# spinels = spinels.dropna(subset=['SIO2(WT%)'], how='all')
titanites = titanites.dropna(subset=['SIO2(WT%)'], how='all')
zircons = zircons.dropna(subset=['SIO2(WT%)'], how='all')

# %%

amphiboles.to_csv('../GEOROC_minerals/GEOROC_amphibole.csv')
apatites.to_csv('../GEOROC_minerals/GEOROC_apatite.csv')
carbonates.to_csv('../GEOROC_minerals/GEOROC_carbonate.csv')
chalcogenides.to_csv('../GEOROC_minerals/GEOROC_chalcogenide.csv')
clays.to_csv('../GEOROC_minerals/GEOROC_clay.csv')
clinopyroxenes.to_csv('../GEOROC_minerals/GEOROC_clinopyroxene.csv')
feldspars.to_csv('../GEOROC_minerals/GEOROC_feldspar.csv')
feldspathoids.to_csv('../GEOROC_minerals/GEOROC_feldspathoid.csv')
garnets.to_csv('../GEOROC_minerals/GEOROC_garnet.csv')
ilmenites.to_csv('../GEOROC_minerals/GEOROC_ilmenite.csv')
micas.to_csv('../GEOROC_minerals/GEOROC_mica.csv')
olivines.to_csv('../GEOROC_minerals/GEOROC_olivine.csv')
orthopyroxenes.to_csv('../GEOROC_minerals/GEOROC_orthopyroxene.csv')
perovskites.to_csv('../GEOROC_minerals/GEOROC_perovskite.csv')
pyroxenes.to_csv('../GEOROC_minerals/GEOROC_pyroxene.csv')
quartz.to_csv('../GEOROC_minerals/GEOROC_quartz.csv')
spinels.to_csv('../GEOROC_minerals/GEOROC_spinel.csv')
titanites.to_csv('../GEOROC_minerals/GEOROC_titanite.csv')
zircons.to_csv('../GEOROC_minerals/GEOROC_zircon.csv')

# %%

lim_oxides = ['CITATION', 'SAMPLE NAME', 'TECTONIC SETTING', 'LATITUDE (MIN.)', 'LATITUDE (MAX.)',
              'LONGITUDE (MIN.)', 'LONGITUDE (MAX.)', 'ROCK NAME', 'ROCK TEXTURE', 'ALTERATION', 'MINERAL',
              'SIO2(WT%)', 'TIO2(WT%)', 'AL2O3(WT%)', 'FE2O3T(WT%)', 'FE2O3(WT%)', 'FEOT(WT%)', 'FEO(WT%)', 'MNO(WT%)',
              'MGO(WT%)', 'CAO(WT%)', 'NA2O(WT%)', 'K2O(WT%)', 'P2O5(WT%)', 'CR2O3(WT%)', 'NIO(WT%)']

lim_oxides_nofe3 = ['CITATION', 'SAMPLE NAME', 'TECTONIC SETTING', 'LATITUDE (MIN.)', 'LATITUDE (MAX.)',
                    'LONGITUDE (MIN.)', 'LONGITUDE (MAX.)', 'ROCK NAME', 'ROCK TEXTURE', 'ALTERATION', 'MINERAL',
                    'SIO2(WT%)', 'TIO2(WT%)', 'AL2O3(WT%)', 'FEOT(WT%)', 'FEO(WT%)', 'MNO(WT%)',
                    'MGO(WT%)', 'CAO(WT%)', 'NA2O(WT%)', 'K2O(WT%)', 'P2O5(WT%)', 'CR2O3(WT%)', 'NIO(WT%)', 'ZRO2(WT%)']

amphiboles_lim = amphiboles.loc[:, lim_oxides]
apatites_lim = apatites.loc[:, lim_oxides]
carbonates_lim = carbonates.loc[:, lim_oxides]
chalcogenides_lim = chalcogenides.loc[:, lim_oxides_nofe3]
clays_lim = clays.loc[:, lim_oxides]
clinopyroxenes_lim = clinopyroxenes.loc[:, lim_oxides]
feldspars_lim = feldspars.loc[:, lim_oxides]
feldspathoids_lim = feldspathoids.loc[:, lim_oxides]
garnets_lim = garnets.loc[:, lim_oxides]
ilmenites_lim = ilmenites.loc[:, lim_oxides]
micas_lim = micas.loc[:, lim_oxides]
olivines_lim = olivines.loc[:, lim_oxides]
orthopyroxenes_lim = orthopyroxenes.loc[:, lim_oxides]
perovskites_lim = perovskites.loc[:, lim_oxides]
pyroxenes_lim = pyroxenes.loc[:, lim_oxides]
quartz_lim = quartz.loc[:, lim_oxides]
spinels_lim = spinels.loc[:, lim_oxides]
titanites_lim = titanites.loc[:, lim_oxides]
zircons_lim = zircons.loc[:, lim_oxides_nofe3]

# %% 

subset_ox = ['SIO2(WT%)', 'TIO2(WT%)', 'AL2O3(WT%)', 'FEOT(WT%)', 'FEO(WT%)', 
             'MNO(WT%)', 'MGO(WT%)', 'CAO(WT%)', 'NA2O(WT%)', 'K2O(WT%)', 'P2O5(WT%)',
             'CR2O3(WT%)', 'NIO(WT%)']

subset_ox_zr = ['SIO2(WT%)', 'TIO2(WT%)', 'AL2O3(WT%)', 'FEOT(WT%)', 'FEO(WT%)', 
                'MNO(WT%)', 'MGO(WT%)', 'CAO(WT%)', 'NA2O(WT%)', 'K2O(WT%)', 'P2O5(WT%)',
                'CR2O3(WT%)', 'NIO(WT%)', 'ZRO2(WT%)']

amphiboles_sub = amphiboles_lim.dropna(subset=subset_ox, thresh=4)
amphiboles_sub['MINERAL'] = 'Amphibole'
apatites_sub = apatites_lim.dropna(subset=subset_ox, thresh=4)
apatites_sub['MINERAL'] = 'Apatite'
carbonates_sub = carbonates_lim.dropna(subset=subset_ox, thresh=2)
clays_sub = clays_lim.dropna(subset=subset_ox, thresh=4)
clinopyroxenes_sub = clinopyroxenes_lim.dropna(subset=subset_ox, thresh=4)
clinopyroxenes_sub['MINERAL'] = 'Pyroxene'
feldspars_sub = feldspars_lim.dropna(subset=subset_ox, thresh=4)
feldspars_sub['MINERAL'] = 'Feldspar'
feldspathoids_sub = feldspathoids_lim.dropna(subset=subset_ox, thresh=4)
garnets_sub = garnets_lim.dropna(subset=subset_ox, thresh=4)
garnets_sub['MINERAL'] = 'Garnet'
ilmenites_sub = ilmenites_lim.dropna(subset=subset_ox, thresh=4)
ilmenites_sub['MINERAL'] = 'Ilmenite'
micas_sub = micas_lim.dropna(subset=subset_ox, thresh=4)
olivines_sub = olivines_lim.dropna(subset=subset_ox, thresh=4)
olivines_sub['MINERAL'] = 'Olivine'
orthopyroxenes_sub = orthopyroxenes_lim.dropna(subset=subset_ox, thresh=4)
orthopyroxenes_sub['MINERAL'] = 'Pyroxene'
pyroxenes_sub = pyroxenes_lim.dropna(subset=subset_ox, thresh=4)
pyroxenes_sub['MINERAL'] = 'Pyroxene'
quartz_sub = quartz_lim.dropna(subset=subset_ox, thresh=4)
spinels_sub = spinels_lim.dropna(subset=subset_ox, thresh=4)
spinels_sub['MINERAL'] = 'Spinel'
titanites_sub = titanites_lim.dropna(subset=subset_ox, thresh=4)
titanites_sub['MINERAL'] = 'Titanite'
zircons_sub = zircons_lim.dropna(subset=subset_ox_zr, thresh=4)
zircons_sub['MINERAL'] = 'Zircon'

min_df = pd.concat([amphiboles_sub, apatites_sub, carbonates_sub, clinopyroxenes_sub, feldspars_sub, feldspathoids_sub, garnets_sub, ilmenites_sub, 
                    micas_sub, olivines_sub, orthopyroxenes_sub, pyroxenes_sub, quartz_sub, spinels_sub, titanites_sub, zircons_sub])

min_df.rename(columns = {'MINERAL':'Mineral', 'SIO2(WT%)':'SiO2', 'TIO2(WT%)':'TiO2', 'AL2O3(WT%)':'Al2O3', 'FE2O3T(WT%)':'Fe2O3T', 'FE2O3(WT%)':'Fe2O3', 
                         'FEOT(WT%)':'FeOT', 'FEO(WT%)':'FeO', 'MNO(WT%)':'MnO', 'MGO(WT%)':'MgO', 'CAO(WT%)':'CaO', 'NA2O(WT%)':'Na2O', 'K2O(WT%)':'K2O', 
                         'P2O5(WT%)':'P2O5', 'CR2O3(WT%)':'Cr2O3', 'NIO(WT%)':'NiO', 'ZRO2(WT%)':'ZrO2'}, inplace = True)

def smart_title(s: str) -> str:
    s = str(s)
    # only convert if EVERYTHING is uppercase (e.g., 'CALCITE' -> 'Calcite')
    return s.title() if s.isupper() else s

min_df['Mineral'] = min_df['Mineral'].astype(str).map(smart_title)


# %% 

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
    conditions = [~np.isnan(df['FeO']) & np.isnan(df['FeOt']) & np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3t']]), # 0
    ~np.isnan(df['FeOt']) & np.isnan(df['FeO']) & np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3t']]), # 1
    ~np.isnan(df['Fe2O3']) & np.isnan(df['Fe2O3t']) & np.isnan(df['FeO']) & np.isnan([df['FeOt']]), # 2
    ~np.isnan(df['Fe2O3t']) & np.isnan(df['Fe2O3']) & np.isnan(df['FeO']) & np.isnan([df['FeOt']]), # 3
    ~np.isnan(df['FeO']) & ~np.isnan(df['Fe2O3']) & np.isnan(df['FeOt']) & np.isnan([df['Fe2O3t']]), # 4
    ~np.isnan(df['FeO']) & ~np.isnan(df['FeOt']) & ~np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3t']]), # 5
    ~np.isnan(df['FeO']) & ~np.isnan(df['Fe2O3']) & ~np.isnan(df['Fe2O3t']) & np.isnan([df['FeOt']]), # 6
    ~np.isnan(df['FeOt']) & ~np.isnan(df['Fe2O3']) & np.isnan(df['Fe2O3t']) & np.isnan([df['FeO']]), # 7
    ~np.isnan(df['Fe2O3']) & ~np.isnan(df['Fe2O3t']) & np.isnan(df['FeO']) & np.isnan([df['FeOt']]) ] # 8

    choices = [ (df['FeO']), # 0
    (df['FeOt']), # 1
    (df['Fe2O3'] / fe_conv), # 2
    (df['Fe2O3t'] / fe_conv), # 3
    (df['FeO'] + (df['Fe2O3'] / fe_conv)), # 4
    (df['FeOt']), # 5
    (df['Fe2O3t'] / fe_conv), # 6
    (df['FeOt']), # 7
    (df['Fe2O3t'] / fe_conv) ] # 8

    df.insert(4, 'FeOt_F', np.select(conditions, choices))
    df.drop(['FeOt'], axis=1, inplace=True)
    df.rename(columns={"FeOt_F": "FeOt"}, inplace=True)
    
    return df 

dtypes = {'SiO2': float, 'TiO2': float, 'Al2O3': float, 'FeOT': float, 'Fe2O3T': float, 'FeO': float, 'Fe2O3': float, 
          'MnO': float, 'MgO': float, 'CaO': float, 'Na2O': float, 'K2O': float, 'P2O5': float, 'Cr2O3': float, 'NiO': float, 
          'B2O3': float, 'ZrO2': float}

# %% 

min_df['Fe2O3t'] = min_df['Fe2O3T'].astype(float)
min_df['Fe2O3'] = min_df['Fe2O3'].astype(float)
min_df['FeOt'] = min_df['FeOT'].astype(float)
min_df['FeO'] = min_df['FeO'].astype(float)
min_df_fe = Fe_Conversion(min_df)

# %% 

min_df_fe_lim = min_df_fe[['CITATION', 'SAMPLE NAME', 'Mineral', 'SiO2', 'TiO2', 'Al2O3', 'Cr2O3', 'FeOt', 
                           'MnO', 'MgO', 'NiO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'ZrO2']]

# %% 

min_df_fe_lim.to_csv('../Validation_Data/GEOROC_validationdata_Fe_2025.csv')

# %% 