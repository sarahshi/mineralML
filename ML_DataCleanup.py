# %% 

""" Created on November 9, 2022 // @author: Sarah Shi and Penny Wieser """

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import numpy as np

import os 
import json 
import pickle
import pygmt
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

from sklearn.preprocessing import scale, normalize

import TAS_Functions as tas
import Thermobar as pt

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.path as mpath
import matplotlib.colors as mcolors

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 14})
plt.rcParams['pdf.fonttype'] = 42

pt.__version__

# %%

a=3.95
b=4.05

def SiteCalculator(phase_filt, append, phase_name):

    phase_nosuf = phase_filt.copy()

    amp_sites = pt.get_amp_sites_from_input_not_amp(phase_nosuf, append)
    cpx_sites = pt.calculate_cpx_sites_from_input_not_cpx(phase_nosuf, append)
    opx_phase_nosuf = phase_nosuf.copy()
    opx_phase_nosuf.columns = [col.replace(append, '_Opx') for col in opx_phase_nosuf.columns]
    opx_sites = pt.calculate_orthopyroxene_components(opx_phase_nosuf)
    plag_sites = calculate_plagioclase_components(phase_nosuf, append)
    ol_sites = calculate_olivine_components(phase_nosuf, append)
    sp_sites = calculate_spinel_components(phase_nosuf, append)
    ox_sites = calculate_oxide_components(phase_nosuf, append)
    ap_sites = calculate_apatite_components(phase_nosuf, append)
    bt_sites = calculate_biotite_components(phase_nosuf, append)
    qz_sites = calculate_quartz_components(phase_nosuf, append)
    gt_sites = calculate_garnet_components(phase_nosuf, append)
    kspar_sites = calculate_kspar_components(phase_nosuf, append)

    phase_nosuf['Phase'] = phase_name

    phase_nosuf['Ca_B_Amp'] = amp_sites['Ca_B']
    phase_nosuf['Na_K_A_Amp'] = amp_sites['Na_A'] + amp_sites['K_A']
    phase_nosuf['Al_T_Amp'] = amp_sites['Al_T']
    phase_nosuf['Si_T_Amp'] = amp_sites['Si_T']
    phase_nosuf['Amp_Cation_Sum'] = amp_sites['cation_sum_All']

    phase_nosuf['Ca_CaMgFe_Cpx'] = cpx_sites['Ca_CaMgFe']
    phase_nosuf['Jd_Cpx'] = cpx_sites['Jd']
    phase_nosuf['DiHd_1996_Cpx'] = cpx_sites['DiHd_1996']
    phase_nosuf['Cpx_Cation_Sum'] = cpx_sites['Cation_Sum_Cpx']

    phase_nosuf['Ca_CaMgFe_Opx'] = opx_sites['Ca_CaMgFe']
    phase_nosuf['Opx_Cation_Sum'] = opx_sites['Cation_Sum_Opx']

    phase_nosuf['Na_Ca_M_Plag'] = plag_sites['Na_Ca_M_Plag']
    phase_nosuf['Si_Al_T_Plag'] = plag_sites['Si_Al_T_Plag']
    phase_nosuf['Plag_Cation_Sum'] = plag_sites['Plag_Cation_Sum']

    phase_nosuf['Mg_Fe_M_Ol'] = ol_sites['Mg_Fe_M_Ol']
    phase_nosuf['Ol_Cation_Sum'] = ol_sites['Ol_Cation_Sum']

    # there must be some Fe3+ but don't have a great estimate of speciation
    phase_nosuf['Mg_Fe_M_Sp'] = sp_sites['Mg_Fe_M_Sp']
    phase_nosuf['Al_B_Sp'] = sp_sites['Al_B_Sp']
    phase_nosuf['Sp_Cation_Sum'] = sp_sites['Sp_Cation_Sum']

    phase_nosuf['Fe_Ti_Ox'] = ox_sites['Fe_Ti_Ox']
    phase_nosuf['Ox_Cation_Sum'] = ox_sites['Ox_Cation_Sum']

    phase_nosuf['Ca_P_Ap'] = ap_sites['Ca_P_Ap']
    phase_nosuf['Ap_Cation_Sum'] = ap_sites['Ap_Cation_Sum']

    phase_nosuf['Mg_Fe_Bt'] = bt_sites['Mg_Fe_Bt']
    phase_nosuf['Si_Al_Bt'] = bt_sites['Si_Al_Bt']
    phase_nosuf['Bt_Cation_Sum'] = bt_sites['Bt_Cation_Sum']

    phase_nosuf['Si_Al_Qz'] = qz_sites['Si_Al_Qz']
    phase_nosuf['Qz_Cation_Sum'] = qz_sites['Qz_Cation_Sum']

    phase_nosuf['Mg_MgFeCa_Gt'] = gt_sites['Mg_MgFeCa_Gt']
    phase_nosuf['Fe_MgFeCa_Gt'] = gt_sites['Fe_MgFeCa_Gt']
    phase_nosuf['Ca_MgFeCa_Gt'] = gt_sites['Ca_MgFeCa_Gt']
    phase_nosuf['Gt_Cation_Sum'] = gt_sites['Gt_Cation_Sum']

    phase_nosuf['Na_Ca_M_Kspar'] = kspar_sites['Na_Ca_M_Kspar']
    phase_nosuf['Si_Al_T_Kspar'] = kspar_sites['Si_Al_T_Kspar']
    phase_nosuf['Kspar_Cation_Sum'] = kspar_sites['Kspar_Cation_Sum']

    return phase_nosuf

def calculate_oxygens_plagioclase(plag_comps):
    
    '''Import plagioclase compositions using plag_comps=My_Plags, returns number of oxygens 
    (e.g., mol proportions * number of O in formula unit)
    
    Parameters
    -------
    plag_comps: pandas.DataFrame
        plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Plag_ox
    '''

    oxygen_num_plag = {'SiO2_Plag': 2, 'MgO_Plag': 1, 'FeOt_Plag': 1, 'CaO_Plag': 1, 'Al2O3_Plag': 3, 'Na2O_Plag': 1,
                    'K2O_Plag': 1, 'MnO_Plag': 1, 'TiO2_Plag': 2, 'Cr2O3_Plag': 3, 'P2O5_Plag': 5}
    oxygen_num_plag_df = pd.DataFrame.from_dict(oxygen_num_plag, orient='index').T
    oxygen_num_plag_df['Sample_ID_Plag'] = 'OxNum'
    oxygen_num_plag_df.set_index('Sample_ID_Plag', inplace=True)

    mol_prop = pt.calculate_mol_proportions_plagioclase(plag_comps=plag_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_plag_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd

def calculate_8oxygens_plagioclase(plag_comps):
    
    '''Import plagioclase compositions using plag_comps=My_Plags, returns cations on the basis of 8 oxygens.

    Parameters
    -------
    plag_comps: pandas.DataFrame
        plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 8 oxygens, with column headings of the form... Plag_cat_8ox.
    '''
    cation_num_plag = {'SiO2_Plag': 1, 'MgO_Plag': 1, 'FeOt_Plag': 1, 'CaO_Plag': 1, 'Al2O3_Plag': 2, 'Na2O_Plag': 2,
                   'K2O_Plag': 2, 'MnO_Plag': 1, 'TiO2_Plag': 1, 'Cr2O3_Plag': 2, 'P2O5_Plag': 2}

    cation_num_plag_df = pd.DataFrame.from_dict(cation_num_plag, orient='index').T
    cation_num_plag_df['Sample_ID_Plag'] = 'CatNum'
    cation_num_plag_df.set_index('Sample_ID_Plag', inplace=True)

    oxygens = calculate_oxygens_plagioclase(plag_comps=plag_comps)
    renorm_factor = 8 / (oxygens.sum(axis='columns'))
    mol_prop = pt.calculate_mol_proportions_plagioclase(plag_comps=plag_comps)
    mol_prop['oxy_renorm_factor_plag'] = renorm_factor
    mol_prop_8 = mol_prop.multiply(mol_prop['oxy_renorm_factor_plag'], axis='rows')
    mol_prop_8.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_8.columns]

    ox_num_reindex = cation_num_plag_df.reindex(
        mol_prop_8.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_8])
    cation_8 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_8.columns = [str(col).replace('_mol_prop', '_cat_8ox')
                        for col in mol_prop.columns]

    cation_8_2=cation_8.rename(columns={
                            'SiO2_Plag_cat_8ox': 'Si_Plag_cat_8ox',
                            'TiO2_Plag_cat_8ox': 'Ti_Plag_cat_8ox',
                            'Al2O3_Plag_cat_8ox': 'Al_Plag_cat_8ox',
                            'FeOt_Plag_cat_8ox': 'Fet_Plag_cat_8ox',
                            'MnO_Plag_cat_8ox': 'Mn_Plag_cat_8ox',
                            'MgO_Plag_cat_8ox': 'Mg_Plag_cat_8ox',
                            'CaO_Plag_cat_8ox': 'Ca_Plag_cat_8ox',
                            'Na2O_Plag_cat_8ox': 'Na_Plag_cat_8ox',
                            'K2O_Plag_cat_8ox': 'K_Plag_cat_8ox',
                            'Cr2O3_Plag_cat_8ox': 'Cr_Plag_cat_8ox',
                            'P2O5_Plag_cat_8ox': 'P_Plag_cat_8ox',})

    return cation_8_2

def calculate_plagioclase_components(plag_comps, append):

    '''Import plagioclase compositions using plag_comps=My_Plags, returns components on the basis of 8 oxygens.

    Parameters
    -------
    plag_comps: pandas.DataFrame
        plagioclase compositions with column headings SiO2_Plag, MgO_Plag etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 8 oxygens, with column headings of the form... Plag_cat_8ox.
    '''

    plag_comps_new = plag_comps.copy()
    plag_comps_new.columns = [col.replace(append, '_Plag') for col in plag_comps_new.columns]
    plag_comps = plag_comps_new.copy()

    plag_calc = calculate_8oxygens_plagioclase(plag_comps=plag_comps)
    plag_calc['Plag_Cation_Sum'] = (plag_calc['Si_Plag_cat_8ox']+plag_calc['Ti_Plag_cat_8ox']
    +plag_calc['Al_Plag_cat_8ox']+plag_calc['Fet_Plag_cat_8ox']+plag_calc['Mn_Plag_cat_8ox']
    +plag_calc['Mg_Plag_cat_8ox']+plag_calc['Ca_Plag_cat_8ox']+plag_calc['Na_Plag_cat_8ox']
    +plag_calc['K_Plag_cat_8ox']+plag_calc['Cr_Plag_cat_8ox'])

    plag_calc['Na_Ca_M_Plag'] = plag_calc['Na_Plag_cat_8ox'] + plag_calc['Ca_Plag_cat_8ox']
    plag_calc['Si_Al_T_Plag'] = plag_calc['Si_Plag_cat_8ox'] + plag_calc['Al_Plag_cat_8ox']

    cat_prop = pt.calculate_cat_proportions_plagioclase(plag_comps=plag_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd['An_Plag'] = cat_frac_anhyd['Ca_Plag_cat_frac'] / \
        (cat_frac_anhyd['Ca_Plag_cat_frac'] +
         cat_frac_anhyd['Na_Plag_cat_frac'] + cat_frac_anhyd['K_Plag_cat_frac'])
    cat_frac_anhyd['Ab_Plag'] = cat_frac_anhyd['Na_Plag_cat_frac'] / \
        (cat_frac_anhyd['Ca_Plag_cat_frac'] +
         cat_frac_anhyd['Na_Plag_cat_frac'] + cat_frac_anhyd['K_Plag_cat_frac'])
    cat_frac_anhyd['Or_Plag'] = 1 - \
        cat_frac_anhyd['An_Plag'] - cat_frac_anhyd['Ab_Plag']
    cat_frac_anhyd2 = pd.concat([plag_comps, plag_calc, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

def calculate_oxygens_olivine(ol_comps):

    '''Import olivine compositions using ol_comps=My_Ols, returns number of oxygens 
    (e.g., mol proportions * number of O in formula unit)

    Parameters
    -------
    ol_comps: pandas.DataFrame
        olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Ol_ox
    '''

    oxygen_num_ol = {'SiO2_Ol': 2, 'MgO_Ol': 1, 'FeOt_Ol': 1, 'CaO_Ol': 1, 'Al2O3_Ol': 3, 'Na2O_Ol': 1,
                    'K2O_Ol': 1, 'MnO_Ol': 1, 'TiO2_Ol': 2, 'Cr2O3_Ol': 3, 'P2O5_Ol': 5}
    oxygen_num_ol_df = pd.DataFrame.from_dict(oxygen_num_ol, orient='index').T
    oxygen_num_ol_df['Sample_ID_Ol'] = 'OxNum'
    oxygen_num_ol_df.set_index('Sample_ID_Ol', inplace=True)

    mol_prop = pt.calculate_mol_proportions_olivine(ol_comps=ol_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_ol_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd



def calculate_4oxygens_olivine(ol_comps):

    '''Import olivine compositions using ol_comps=My_Ols, returns cations on the basis of 4 oxygens.

    Parameters
    -------
    ol_comps: pandas.DataFrame
        olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 4 oxygens, with column headings of the form... Ol_cat_4ox.
    '''

    cation_num_ol = {'SiO2_Ol': 1, 'MgO_Ol': 1, 'FeOt_Ol': 1, 'CaO_Ol': 1, 'Al2O3_Ol': 2, 'Na2O_Ol': 2,
                   'K2O_Ol': 2, 'MnO_Ol': 1, 'TiO2_Ol': 1, 'Cr2O3_Ol': 2, 'P2O5_Ol': 2}

    cation_num_ol_df = pd.DataFrame.from_dict(cation_num_ol, orient='index').T
    cation_num_ol_df['Sample_ID_Ol'] = 'CatNum'
    cation_num_ol_df.set_index('Sample_ID_Ol', inplace=True)

    oxygens = calculate_oxygens_olivine(ol_comps=ol_comps)
    renorm_factor = 4 / (oxygens.sum(axis='columns'))
    mol_prop = pt.calculate_mol_proportions_olivine(ol_comps=ol_comps)
    mol_prop['oxy_renorm_factor_ol'] = renorm_factor
    mol_prop_4 = mol_prop.multiply(mol_prop['oxy_renorm_factor_ol'], axis='rows')
    mol_prop_4.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_4.columns]

    ox_num_reindex = cation_num_ol_df.reindex(
        mol_prop_4.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_4])
    cation_4 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_4.columns = [str(col).replace('_mol_prop', '_cat_4ox')
                        for col in mol_prop.columns]

    cation_4_2=cation_4.rename(columns={
                            'SiO2_Ol_cat_4ox': 'Si_Ol_cat_4ox',
                            'TiO2_Ol_cat_4ox': 'Ti_Ol_cat_4ox',
                            'Al2O3_Ol_cat_4ox': 'Al_Ol_cat_4ox',
                            'FeOt_Ol_cat_4ox': 'Fet_Ol_cat_4ox',
                            'MnO_Ol_cat_4ox': 'Mn_Ol_cat_4ox',
                            'MgO_Ol_cat_4ox': 'Mg_Ol_cat_4ox',
                            'CaO_Ol_cat_4ox': 'Ca_Ol_cat_4ox',
                            'Na2O_Ol_cat_4ox': 'Na_Ol_cat_4ox',
                            'K2O_Ol_cat_4ox': 'K_Ol_cat_4ox',
                            'Cr2O3_Ol_cat_4ox': 'Cr_Ol_cat_4ox',
                            'P2O5_Ol_cat_4ox': 'P_Ol_cat_4ox',})

    return cation_4_2

def calculate_olivine_components(ol_comps, append):

    '''Import olivine compositions using ol_comps=My_Ols, returns components on the basis of 4 oxygens.

    Parameters
    -------
    ol_comps: pandas.DataFrame
        olivine compositions with column headings SiO2_Ol, MgO_Ol etc.

    Returns
    -------
    pandas DataFrame
        components on the basis of 4 oxygens, with column headings of the form... Ol_cat_4ox.
    '''

    ol_comps_new = ol_comps.copy()
    ol_comps_new.columns = [col.replace(append, '_Ol') for col in ol_comps_new.columns]
    ol_comps = ol_comps_new.copy()

    ol_calc = calculate_4oxygens_olivine(ol_comps=ol_comps)
    ol_calc['Ol_Cation_Sum'] = (ol_calc['Si_Ol_cat_4ox']+ol_calc['Ti_Ol_cat_4ox']
    +ol_calc['Al_Ol_cat_4ox']+ol_calc['Fet_Ol_cat_4ox']+ol_calc['Mn_Ol_cat_4ox']
    +ol_calc['Mg_Ol_cat_4ox']+ol_calc['Ca_Ol_cat_4ox']+ol_calc['Na_Ol_cat_4ox']
    +ol_calc['K_Ol_cat_4ox'])

    ol_calc['Mg_Fe_M_Ol'] = ol_calc['Mg_Ol_cat_4ox'] + ol_calc['Fet_Ol_cat_4ox']
    ol_calc['Si_T_Ol'] = ol_calc['Si_Ol_cat_4ox'] 

    cat_prop = pt.calculate_cat_proportions_olivine(ol_comps=ol_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd2 = pd.concat([ol_comps, ol_calc, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

def calculate_mol_proportions_spinel(sp_comps):
    
    '''Import spinel compositions using sp_comps=My_Spinels, returns mole proportions

    Parameters
    -------
    sp_comps: pandas.DataFrame
            Panda DataFrame of spinel compositions with column headings SiO2_Sp, MgO_Sp etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for spinels with column headings of the form SiO2_Sp_mol_prop
    '''

    oxide_mass_sp = {'SiO2_Sp': 60.0843, 'MgO_Sp': 40.3044, 'FeOt_Sp': 71.8464,
    'CaO_Sp': 56.0774,'Al2O3_Sp': 101.961, 'Na2O_Sp': 61.9789, 'K2O_Sp': 94.196,
    'MnO_Sp': 70.9375, 'TiO2_Sp': 79.7877, 'Cr2O3_Sp': 151.9982, 'P2O5_Sp': 141.937}
    oxide_mass_sp_df = pd.DataFrame.from_dict(oxide_mass_sp, orient='index').T
    oxide_mass_sp_df['Sample_ID_Sp'] = 'MolWt'
    oxide_mass_sp_df.set_index('Sample_ID_Sp', inplace=True)

    # This makes it match the columns in the oxide mass dataframe
    sp_wt = sp_comps.reindex(oxide_mass_sp_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    sp_wt_combo = pd.concat([oxide_mass_sp_df, sp_wt],)
    # Drop the calculation column
    mol_prop_anhyd = sp_wt_combo.div(
        sp_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd

def calculate_cat_proportions_spinel(*, sp_comps=None, oxide_headers=False):

    '''Import spinel compositions using sp_comps=My_spinels, returns cation proportions

    Parameters
    -------
    sp_comps: pandas.DataFrame
            spinel compositions with column headings SiO2_Sp, MgO_Sp etc.

    oxide_headers: bool
        default=False, returns as Ti_Sp_cat_prop.
        =True returns Ti_Sp_cat_prop.
        This is used for rapid matrix division for
        pre-processing of data for cation fractions etc

    Returns
    -------
    pandas DataFrame
        cation proportions for spinel with column headings of the form... Sp_cat_prop
    '''

    cation_num_sp = {'SiO2_Sp': 1, 'MgO_Sp': 1, 'FeOt_Sp': 1, 'CaO_Sp': 1, 'Al2O3_Sp': 2, 'Na2O_Sp': 2,
                   'K2O_Sp': 2, 'MnO_Sp': 1, 'TiO2_Sp': 1, 'Cr2O3_Sp': 2, 'P2O5_Sp': 2}
    cation_num_sp_df = pd.DataFrame.from_dict(cation_num_sp, orient='index').T
    cation_num_sp_df['Sample_ID_Sp'] = 'CatNum'
    cation_num_sp_df.set_index('Sample_ID_Sp', inplace=True)

    oxide_mass_sp = {'SiO2_Sp': 60.0843, 'MgO_Sp': 40.3044, 'FeOt_Sp': 71.8464,
    'CaO_Sp': 56.0774,'Al2O3_Sp': 101.961, 'Na2O_Sp': 61.9789, 'K2O_Sp': 94.196,
    'MnO_Sp': 70.9375, 'TiO2_Sp': 79.7877, 'Cr2O3_Sp': 151.9982, 'P2O5_Sp': 141.937}
    oxide_mass_sp_df = pd.DataFrame.from_dict(oxide_mass_sp, orient='index').T
    oxide_mass_sp_df['Sample_ID_Sp'] = 'MolWt'
    oxide_mass_sp_df.set_index('Sample_ID_Sp', inplace=True)

    sp_prop_no_cat_num = calculate_mol_proportions_spinel(
        sp_comps=sp_comps)
    sp_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in sp_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_sp_df.reindex(
        oxide_mass_sp_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, sp_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    if oxide_headers is True:
        return cation_prop_anhyd
    if oxide_headers is False:
        cation_prop_anhyd2=cation_prop_anhyd.rename(columns={
                            'SiO2_Sp_cat_prop': 'Si_Sp_cat_prop',
                            'TiO2_Sp_cat_prop': 'Ti_Sp_cat_prop',
                            'Al2O3_Sp_cat_prop': 'Al_Sp_cat_prop',
                            'FeOt_Sp_cat_prop': 'Fet_Sp_cat_prop',
                            'MnO_Sp_cat_prop': 'Mn_Sp_cat_prop',
                            'MgO_Sp_cat_prop': 'Mg_Sp_cat_prop',
                            'CaO_Sp_cat_prop': 'Ca_Sp_cat_prop',
                            'Na2O_Sp_cat_prop': 'Na_Sp_cat_prop',
                            'K2O_Sp_cat_prop': 'K_Sp_cat_prop',
                            'Cr2O3_Sp_cat_prop': 'Cr_Sp_cat_prop',
                            'P2O5_Sp_cat_prop': 'P_Sp_cat_prop',})

        return cation_prop_anhyd2

def calculate_oxygens_spinel(sp_comps):

    '''Import spinel compositions using sp_comps=My_Sps, returns number of oxygens 
    (e.g., mol proportions * number of O in formula unit)

    Parameters
    -------
    sp_comps: pandas.DataFrame
        spinel compositions with column headings SiO2_Sp, MgO_Sp etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Sp_ox
    '''

    oxygen_num_sp = {'SiO2_Sp': 2, 'MgO_Sp': 1, 'FeOt_Sp': 1, 'CaO_Sp': 1, 'Al2O3_Sp': 3, 'Na2O_Sp': 1,
                    'K2O_Sp': 1, 'MnO_Sp': 1, 'TiO2_Sp': 2, 'Cr2O3_Sp': 3, 'P2O5_Sp': 5}
    oxygen_num_sp_df = pd.DataFrame.from_dict(oxygen_num_sp, orient='index').T
    oxygen_num_sp_df['Sample_ID_Sp'] = 'OxNum'
    oxygen_num_sp_df.set_index('Sample_ID_Sp', inplace=True)

    mol_prop = calculate_mol_proportions_spinel(sp_comps=sp_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_sp_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd

def calculate_4oxygens_spinel(sp_comps):

    '''Import spinel compositions using sp_comps=My_Sps, returns cations on the basis of 4 oxygens.

    Parameters
    -------
    sp_comps: pandas.DataFrame
        spinel compositions with column headings SiO2_Sp, MgO_Sp etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 4 oxygens, with column headings of the form... Sp_cat_4ox.
    '''

    cation_num_sp = {'SiO2_Sp': 1, 'MgO_Sp': 1, 'FeOt_Sp': 1, 'CaO_Sp': 1, 'Al2O3_Sp': 2, 'Na2O_Sp': 2,
                   'K2O_Sp': 2, 'MnO_Sp': 1, 'TiO2_Sp': 1, 'Cr2O3_Sp': 2, 'P2O5_Sp': 2}

    cation_num_sp_df = pd.DataFrame.from_dict(cation_num_sp, orient='index').T
    cation_num_sp_df['Sample_ID_Sp'] = 'CatNum'
    cation_num_sp_df.set_index('Sample_ID_Sp', inplace=True)

    oxygens = calculate_oxygens_spinel(sp_comps=sp_comps)
    renorm_factor = 4 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_spinel(sp_comps=sp_comps)
    mol_prop['oxy_renorm_factor_sp'] = renorm_factor
    mol_prop_4 = mol_prop.multiply(mol_prop['oxy_renorm_factor_sp'], axis='rows')
    mol_prop_4.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_4.columns]

    ox_num_reindex = cation_num_sp_df.reindex(
        mol_prop_4.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_4])
    cation_4 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_4.columns = [str(col).replace('_mol_prop', '_cat_4ox')
                        for col in mol_prop.columns]

    cation_4_2=cation_4.rename(columns={
                            'SiO2_Sp_cat_4ox': 'Si_Sp_cat_4ox',
                            'TiO2_Sp_cat_4ox': 'Ti_Sp_cat_4ox',
                            'Al2O3_Sp_cat_4ox': 'Al_Sp_cat_4ox',
                            'FeOt_Sp_cat_4ox': 'Fet_Sp_cat_4ox',
                            'MnO_Sp_cat_4ox': 'Mn_Sp_cat_4ox',
                            'MgO_Sp_cat_4ox': 'Mg_Sp_cat_4ox',
                            'CaO_Sp_cat_4ox': 'Ca_Sp_cat_4ox',
                            'Na2O_Sp_cat_4ox': 'Na_Sp_cat_4ox',
                            'K2O_Sp_cat_4ox': 'K_Sp_cat_4ox',
                            'Cr2O3_Sp_cat_4ox': 'Cr_Sp_cat_4ox',
                            'P2O5_Sp_cat_4ox': 'P_Sp_cat_4ox', })

    return cation_4_2

def calculate_spinel_components(sp_comps, append):

    '''Import spinel compositions using sp_comps=My_Sps, returns components on the basis of 4 oxygens.

    Parameters
    -------
    sp_comps: pandas.DataFrame
        spinel compositions with column headings SiO2_Sp, MgO_Sp etc.

    Returns
    -------
    pandas DataFrame
        components on the basis of 4 oxygens, with column headings of the form... Sp_cat_4ox.
    '''

    sp_comps_new = sp_comps.copy()
    sp_comps_new.columns = [col.replace(append, '_Sp') for col in sp_comps_new.columns]
    sp_comps = sp_comps_new.copy()

    sp_calc = calculate_4oxygens_spinel(sp_comps=sp_comps)
    sp_calc['Sp_Cation_Sum'] = (sp_calc['Si_Sp_cat_4ox']+sp_calc['Ti_Sp_cat_4ox']
    +sp_calc['Al_Sp_cat_4ox']+sp_calc['Fet_Sp_cat_4ox']+sp_calc['Mn_Sp_cat_4ox']
    +sp_calc['Mg_Sp_cat_4ox']+sp_calc['Ca_Sp_cat_4ox']+sp_calc['Na_Sp_cat_4ox']
    +sp_calc['K_Sp_cat_4ox']+sp_calc['Cr_Sp_cat_4ox']+sp_calc['P_Sp_cat_4ox'])

    sp_calc['Mg_Fe_M_Sp'] = sp_calc['Mg_Sp_cat_4ox'] + sp_calc['Fet_Sp_cat_4ox']
    sp_calc['Al_B_Sp'] = sp_calc['Al_Sp_cat_4ox']

    cat_prop = calculate_cat_proportions_spinel(sp_comps=sp_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd2 = pd.concat([sp_comps, sp_calc, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

def calculate_mol_proportions_oxide(ox_comps):

    '''Import Oxide compositions using ox_comps=My_Oxides, returns mole proportions. 
    Retain _Sp appendix

    Parameters
    -------
    ox_comps: pandas.DataFrame
            Panda DataFrame of oxide compositions with column headings SiO2_Sp, MgO_Sp etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for oxides with column headings of the form SiO2_Sp_mol_prop
    '''

    oxide_mass_ox = {'SiO2_Sp': 60.0843, 'MgO_Sp': 40.3044, 'FeOt_Sp': 71.8464,
    'CaO_Sp': 56.0774,'Al2O3_Sp': 101.961, 'Na2O_Sp': 61.9789, 'K2O_Sp': 94.196,
    'MnO_Sp': 70.9375, 'TiO2_Sp': 79.7877, 'Cr2O3_Sp': 151.9982, 'P2O5_Sp': 141.937}

    oxide_mass_ox_df = pd.DataFrame.from_dict(oxide_mass_ox, orient='index').T
    oxide_mass_ox_df['Sample_ID_Sp'] = 'MolWt'
    oxide_mass_ox_df.set_index('Sample_ID_Sp', inplace=True)

    # This makes it match the columns in the oxide mass dataframe
    ox_wt = ox_comps.reindex(oxide_mass_ox_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    ox_wt_combo = pd.concat([oxide_mass_ox_df, ox_wt],)
    # Drop the calculation column
    mol_prop_anhyd = ox_wt_combo.div(
        ox_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_cat_proportions_oxide(*, ox_comps=None, oxide_headers=False):

    '''Import oxide compositions using ox_comps=My_oxides, returns cation proportions
    Retain _Sp appendix

    Parameters
    -------
    ox_comps: pandas.DataFrame
            oxide compositions with column headings SiO2_Sp, MgO_Sp etc.

    oxide_headers: bool
        default=False, returns as Ti_Sp_cat_prop.
        =True returns Ti_Sp_cat_prop.
        This is used for rapid matrix division for
        pre-processing of data for cation fractions etc
    
    Returns
    -------
    pandas DataFrame
        cation proportions for oxide with column headings of the form... Sp_cat_prop
    '''

    cation_num_ox = {'SiO2_Sp': 1, 'MgO_Sp': 1, 'FeOt_Sp': 1, 'CaO_Sp': 1, 'Al2O3_Sp': 2, 'Na2O_Sp': 2,
                   'K2O_Sp': 2, 'MnO_Sp': 1, 'TiO2_Sp': 1, 'Cr2O3_Sp': 2, 'P2O5_Sp': 2}
    cation_num_ox_df = pd.DataFrame.from_dict(cation_num_ox, orient='index').T
    cation_num_ox_df['Sample_ID_Sp'] = 'CatNum'
    cation_num_ox_df.set_index('Sample_ID_Sp', inplace=True)

    oxide_mass_ox = {'SiO2_Sp': 60.0843, 'MgO_Sp': 40.3044, 'FeOt_Sp': 71.8464,
    'CaO_Sp': 56.0774,'Al2O3_Sp': 101.961, 'Na2O_Sp': 61.9789, 'K2O_Sp': 94.196,
    'MnO_Sp': 70.9375, 'TiO2_Sp': 79.7877, 'Cr2O3_Sp': 151.9982, 'P2O5_Sp': 141.937}
    oxide_mass_ox_df = pd.DataFrame.from_dict(oxide_mass_ox, orient='index').T
    oxide_mass_ox_df['Sample_ID_Sp'] = 'MolWt'
    oxide_mass_ox_df.set_index('Sample_ID_Sp', inplace=True)

    ox_prop_no_cat_num = calculate_mol_proportions_oxide(
        ox_comps=ox_comps)
    ox_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in ox_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_ox_df.reindex(
        oxide_mass_ox_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, ox_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    if oxide_headers is True:
        return cation_prop_anhyd
    if oxide_headers is False:
        cation_prop_anhyd2=cation_prop_anhyd.rename(columns={
                            'SiO2_Sp_cat_prop': 'Si_Sp_cat_prop',
                            'TiO2_Sp_cat_prop': 'Ti_Sp_cat_prop',
                            'Al2O3_Sp_cat_prop': 'Al_Sp_cat_prop',
                            'FeOt_Sp_cat_prop': 'Fet_Sp_cat_prop',
                            'MnO_Sp_cat_prop': 'Mn_Sp_cat_prop',
                            'MgO_Sp_cat_prop': 'Mg_Sp_cat_prop',
                            'CaO_Sp_cat_prop': 'Ca_Sp_cat_prop',
                            'Na2O_Sp_cat_prop': 'Na_Sp_cat_prop',
                            'K2O_Sp_cat_prop': 'K_Sp_cat_prop',
                            'Cr2O3_Sp_cat_prop': 'Cr_Sp_cat_prop',
                            'P2O5_Sp_cat_prop': 'P_Sp_cat_prop',})

        return cation_prop_anhyd2

def calculate_oxygens_oxide(ox_comps):

    '''Import oxide compositions using ox_comps=My_Sps, returns number of oxygens 
    (e.g., mol proportions * number of O in formula unit)
    Retain _Sp appendix

   Parameters
    -------
    ox_comps: pandas.DataFrame
        oxide compositions with column headings SiO2_Sp, MgO_Sp etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Sp_ox
    '''

    oxygen_num_ox = {'SiO2_Sp': 2, 'MgO_Sp': 1, 'FeOt_Sp': 1, 'CaO_Sp': 1, 'Al2O3_Sp': 3, 'Na2O_Sp': 1,
                    'K2O_Sp': 1, 'MnO_Sp': 1, 'TiO2_Sp': 2, 'Cr2O3_Sp': 3, 'P2O5_Sp': 5}
    oxygen_num_ox_df = pd.DataFrame.from_dict(oxygen_num_ox, orient='index').T
    oxygen_num_ox_df['Sample_ID_Sp'] = 'OxNum'
    oxygen_num_ox_df.set_index('Sample_ID_Sp', inplace=True)

    mol_prop = calculate_mol_proportions_oxide(ox_comps=ox_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_ox_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd

def calculate_oxygens_oxide(ox_comps):

    '''Import oxide compositions using ox_comps=My_Sps, returns number of oxygens 
    (e.g., mol proportions * number of O in formula unit)
    Retain _Sp appendix

    Parameters
    -------
    ox_comps: pandas.DataFrame
        oxide compositions with column headings SiO2_Sp, MgO_Sp etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Sp_ox
    '''

    oxygen_num_ox = {'SiO2_Sp': 2, 'MgO_Sp': 1, 'FeOt_Sp': 1, 'CaO_Sp': 1, 'Al2O3_Sp': 3, 'Na2O_Sp': 1,
                    'K2O_Sp': 1, 'MnO_Sp': 1, 'TiO2_Sp': 2, 'Cr2O3_Sp': 3, 'P2O5_Sp': 5}
    oxygen_num_ox_df = pd.DataFrame.from_dict(oxygen_num_ox, orient='index').T
    oxygen_num_ox_df['Sample_ID_Sp'] = 'OxNum'
    oxygen_num_ox_df.set_index('Sample_ID_Sp', inplace=True)

    mol_prop = calculate_mol_proportions_oxide(ox_comps=ox_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_ox_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd



def calculate_3oxygens_oxide(ox_comps):

    '''Import oxide compositions using ox_comps=My_Sps, returns cations on the basis of 4 oxygens.
    Retain _Sp appendix

   Parameters
    -------
    ox_comps: pandas.DataFrame
        oxide compositions with column headings SiO2_Sp, MgO_Sp etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 3 oxygens, with column headings of the form... Sp_cat_3ox.
    '''

    cation_num_ox = {'SiO2_Sp': 1, 'MgO_Sp': 1, 'FeOt_Sp': 1, 'CaO_Sp': 1, 'Al2O3_Sp': 2, 'Na2O_Sp': 2,
                   'K2O_Sp': 2, 'MnO_Sp': 1, 'TiO2_Sp': 1, 'Cr2O3_Sp': 2, 'P2O5_Sp': 2}
    cation_num_ox_df = pd.DataFrame.from_dict(cation_num_ox, orient='index').T
    cation_num_ox_df['Sample_ID_Sp'] = 'CatNum'
    cation_num_ox_df.set_index('Sample_ID_Sp', inplace=True)

    oxygens = calculate_oxygens_oxide(ox_comps=ox_comps)
    renorm_factor = 3 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_oxide(ox_comps=ox_comps)
    mol_prop['oxy_renorm_factor_ox'] = renorm_factor
    mol_prop_3 = mol_prop.multiply(mol_prop['oxy_renorm_factor_ox'], axis='rows')
    mol_prop_3.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_3.columns]

    ox_num_reindex = cation_num_ox_df.reindex(
        mol_prop_3.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_3])
    cation_3 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_3.columns = [str(col).replace('_mol_prop', '_cat_3ox')
                        for col in mol_prop.columns]

    cation_3_2=cation_3.rename(columns={
                            'SiO2_Sp_cat_3ox': 'Si_Ox_cat_3ox',
                            'TiO2_Sp_cat_3ox': 'Ti_Ox_cat_3ox',
                            'Al2O3_Sp_cat_3ox': 'Al_Ox_cat_3ox',
                            'FeOt_Sp_cat_3ox': 'Fet_Ox_cat_3ox',
                            'MnO_Sp_cat_3ox': 'Mn_Ox_cat_3ox',
                            'MgO_Sp_cat_3ox': 'Mg_Ox_cat_3ox',
                            'CaO_Sp_cat_3ox': 'Ca_Ox_cat_3ox',
                            'Na2O_Sp_cat_3ox': 'Na_Ox_cat_3ox',
                            'K2O_Sp_cat_3ox': 'K_Ox_cat_3ox',
                            'Cr2O3_Sp_cat_3ox': 'Cr_Ox_cat_3ox',
                            'P2O5_Sp_cat_3ox': 'P_Ox_cat_3ox',})

    return cation_3_2

def calculate_oxide_components(ox_comps, append):

    '''Import oxide compositions using ox_comps=My_Sps, returns cations on the basis of 4 oxygens.
    Retain _Sp appendix

   Parameters
    -------
    ox_comps: pandas.DataFrame
        oxide compositions with column headings SiO2_Sp, MgO_Sp etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 3 oxygens, with column headings of the form... Sp_cat_3ox.
    '''
    
    ox_comps_new = ox_comps.copy()
    ox_comps_new.columns = [col.replace(append, '_Sp') for col in ox_comps_new.columns]
    ox_comps = ox_comps_new.copy()

    ox_calc = calculate_3oxygens_oxide(ox_comps=ox_comps)
    ox_calc['Ox_Cation_Sum'] = (ox_calc['Si_Ox_cat_3ox']+ox_calc['Ti_Ox_cat_3ox']
    +ox_calc['Al_Ox_cat_3ox']+ox_calc['Fet_Ox_cat_3ox']+ox_calc['Mn_Ox_cat_3ox']
    +ox_calc['Mg_Ox_cat_3ox']+ox_calc['Ca_Ox_cat_3ox']+ox_calc['Na_Ox_cat_3ox']
    +ox_calc['K_Ox_cat_3ox']+ox_calc['Cr_Ox_cat_3ox']+ox_calc['P_Ox_cat_3ox'])

    ox_calc['Fe_Ti_Ox'] = ox_calc['Ti_Ox_cat_3ox'] + ox_calc['Fet_Ox_cat_3ox']

    cat_prop = calculate_cat_proportions_oxide(ox_comps=ox_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd2 = pd.concat([ox_comps, ox_calc, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

def calculate_mol_proportions_apatite(ap_comps):

    '''Import Apatite compositions using ap_comps=My_Apatites, returns mole proportions

    Parameters
    -------
    ap_comps: pandas.DataFrame
            Panda DataFrame of apatite compositions with column headings SiO2_Ap, MgO_Ap etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for apatites with column headings of the form SiO2_Ap_mol_prop
    '''

    oxide_mass_ap = {'SiO2_Ap': 60.0843, 'MgO_Ap': 40.3044, 'FeOt_Ap': 71.8464,
    'CaO_Ap': 56.0774,'Al2O3_Ap': 101.961, 'Na2O_Ap': 61.9789, 'K2O_Ap': 94.196,
    'MnO_Ap': 70.9375, 'TiO2_Ap': 79.7877, 'Cr2O3_Ap': 151.9982, 'P2O5_Ap': 141.937}
    oxide_mass_ap_df = pd.DataFrame.from_dict(oxide_mass_ap, orient='index').T
    oxide_mass_ap_df['Sample_ID_Ap'] = 'MolWt'
    oxide_mass_ap_df.set_index('Sample_ID_Ap', inplace=True)


    # This makes it match the columns in the oxide mass dataframe
    ap_wt = ap_comps.reindex(oxide_mass_ap_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    ap_wt_combo = pd.concat([oxide_mass_ap_df, ap_wt],)
    # Drop the calculation column
    mol_prop_anhyd = ap_wt_combo.div(
        ap_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd

def calculate_cat_proportions_apatite(*, ap_comps=None, oxide_headers=False):
    
    '''Import apatite compositions using ap_comps=My_Apatites, returns cation proportions

   Parameters
    -------
    ap_comps: pandas.DataFrame
            apatite compositions with column headings SiO2_Ap, MgO_Ap etc.
    
    oxide_headers: bool
        default=False, returns as Ti_Ap_cat_prop.
        =True returns Ti_Ap_cat_prop.
        This is used for rapid matrix division for
        pre-processing of data for cation fractions etc

    Returns
    -------
    pandas DataFrame
        cation proportions for apatite with column headings of the form... Ap_cat_prop
    '''

    cation_num_ap = {'SiO2_Ap': 1, 'MgO_Ap': 1, 'FeOt_Ap': 1, 'CaO_Ap': 1, 'Al2O3_Ap': 2, 'Na2O_Ap': 2,
                   'K2O_Ap': 2, 'MnO_Ap': 1, 'TiO2_Ap': 1, 'Cr2O3_Ap': 2, 'P2O5_Ap': 2}
    cation_num_ap_df = pd.DataFrame.from_dict(cation_num_ap, orient='index').T
    cation_num_ap_df['Sample_ID_Ap'] = 'CatNum'
    cation_num_ap_df.set_index('Sample_ID_Ap', inplace=True)

    oxide_mass_ap = {'SiO2_Ap': 60.0843, 'MgO_Ap': 40.3044, 'FeOt_Ap': 71.8464,
    'CaO_Ap': 56.0774,'Al2O3_Ap': 101.961, 'Na2O_Ap': 61.9789, 'K2O_Ap': 94.196,
    'MnO_Ap': 70.9375, 'TiO2_Ap': 79.7877, 'Cr2O3_Ap': 151.9982, 'P2O5_Ap': 141.937}

    oxide_mass_ap_df = pd.DataFrame.from_dict(oxide_mass_ap, orient='index').T
    oxide_mass_ap_df['Sample_ID_Ap'] = 'MolWt'
    oxide_mass_ap_df.set_index('Sample_ID_Ap', inplace=True)

    ap_prop_no_cat_num = calculate_mol_proportions_apatite(
        ap_comps=ap_comps)
    ap_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in ap_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_ap_df.reindex(
        oxide_mass_ap_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, ap_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    if oxide_headers is True:
        return cation_prop_anhyd
    if oxide_headers is False:
        cation_prop_anhyd2=cation_prop_anhyd.rename(columns={
                            'SiO2_Ap_cat_prop': 'Si_Ap_cat_prop',
                            'TiO2_Ap_cat_prop': 'Ti_Ap_cat_prop',
                            'Al2O3_Ap_cat_prop': 'Al_Ap_cat_prop',
                            'FeOt_Ap_cat_prop': 'Fet_Ap_cat_prop',
                            'MnO_Ap_cat_prop': 'Mn_Ap_cat_prop',
                            'MgO_Ap_cat_prop': 'Mg_Ap_cat_prop',
                            'CaO_Ap_cat_prop': 'Ca_Ap_cat_prop',
                            'Na2O_Ap_cat_prop': 'Na_Ap_cat_prop',
                            'K2O_Ap_cat_prop': 'K_Ap_cat_prop',
                            'Cr2O3_Ap_cat_prop': 'Cr_Ap_cat_prop',
                            'P2O5_Ap_cat_prop': 'P_Ap_cat_prop',})

        return cation_prop_anhyd2

def calculate_oxygens_apatite(ap_comps):
    
    '''Import apatite compositions using ap_comps=My_Aps, returns number of oxygens 
    (e.g., mol proportions * number of O in formula unit)

   Parameters
    -------
    ap_comps: pandas.DataFrame
        apatite compositions with column headings SiO2_Ap, MgO_Ap etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Ap_ox
    '''

    oxygen_num_ap = {'SiO2_Ap': 2, 'MgO_Ap': 1, 'FeOt_Ap': 1, 'CaO_Ap': 1, 'Al2O3_Ap': 3, 'Na2O_Ap': 1,
                    'K2O_Ap': 1, 'MnO_Ap': 1, 'TiO2_Ap': 2, 'Cr2O3_Ap': 3, 'P2O5_Ap': 5}
    oxygen_num_ap_df = pd.DataFrame.from_dict(oxygen_num_ap, orient='index').T
    oxygen_num_ap_df['Sample_ID_Ap'] = 'OxNum'
    oxygen_num_ap_df.set_index('Sample_ID_Ap', inplace=True)

    mol_prop = calculate_mol_proportions_apatite(ap_comps=ap_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_ap_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd

def calculate_12oxygens_apatite(ap_comps):

    '''Import apatite compositions using ap_comps=My_Aps, returns cations on the basis of 12 oxygens.

   Parameters
    -------
    ap_comps: pandas.DataFrame
        apatite compositions with column headings SiO2_Ap, MgO_Ap etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 12 oxygens, with column headings of the form... Ap_cat_12ox.
    '''

    cation_num_ap = {'SiO2_Ap': 1, 'MgO_Ap': 1, 'FeOt_Ap': 1, 'CaO_Ap': 1, 'Al2O3_Ap': 2, 'Na2O_Ap': 2,
                   'K2O_Ap': 2, 'MnO_Ap': 1, 'TiO2_Ap': 1, 'Cr2O3_Ap': 2, 'P2O5_Ap': 2}

    cation_num_ap_df = pd.DataFrame.from_dict(cation_num_ap, orient='index').T
    cation_num_ap_df['Sample_ID_Ap'] = 'CatNum'
    cation_num_ap_df.set_index('Sample_ID_Ap', inplace=True)

    oxygens = calculate_oxygens_apatite(ap_comps=ap_comps)
    renorm_factor = 12 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_apatite(ap_comps=ap_comps)
    mol_prop['oxy_renorm_factor_ap'] = renorm_factor
    mol_prop_12 = mol_prop.multiply(mol_prop['oxy_renorm_factor_ap'], axis='rows')
    mol_prop_12.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_12.columns]

    ox_num_reindex = cation_num_ap_df.reindex(
        mol_prop_12.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_12])
    cation_12 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_12.columns = [str(col).replace('_mol_prop', '_cat_12ox')
                        for col in mol_prop.columns]

    cation_12_2=cation_12.rename(columns={
                            'SiO2_Ap_cat_12ox': 'Si_Ap_cat_12ox',
                            'TiO2_Ap_cat_12ox': 'Ti_Ap_cat_12ox',
                            'Al2O3_Ap_cat_12ox': 'Al_Ap_cat_12ox',
                            'FeOt_Ap_cat_12ox': 'Fet_Ap_cat_12ox',
                            'MnO_Ap_cat_12ox': 'Mn_Ap_cat_12ox',
                            'MgO_Ap_cat_12ox': 'Mg_Ap_cat_12ox',
                            'CaO_Ap_cat_12ox': 'Ca_Ap_cat_12ox',
                            'Na2O_Ap_cat_12ox': 'Na_Ap_cat_12ox',
                            'K2O_Ap_cat_12ox': 'K_Ap_cat_12ox',
                            'Cr2O3_Ap_cat_12ox': 'Cr_Ap_cat_12ox',
                            'P2O5_Ap_cat_12ox': 'P_Ap_cat_12ox',})

    return cation_12_2

def calculate_apatite_components(ap_comps, append):

    '''Import apatite compositions using ap_comps=My_Aps, returns cations on the basis of 12 oxygens.

   Parameters
    -------
    ap_comps: pandas.DataFrame
        apatite compositions with column headings SiO2_Ap, MgO_Ap etc.

    Returns
    -------
    pandas DataFrame
        components on the basis of 12 oxygens, with column headings of the form... Ap_cat_12ox.
    '''

    ap_comps_new = ap_comps.copy()
    ap_comps_new.columns = [col.replace(append, '_Ap') for col in ap_comps_new.columns]
    ap_comps = ap_comps_new.copy()

    ap_calc = calculate_12oxygens_apatite(ap_comps=ap_comps)
    ap_calc['Ap_Cation_Sum'] = (ap_calc['Si_Ap_cat_12ox']+ap_calc['Ti_Ap_cat_12ox']
    +ap_calc['Al_Ap_cat_12ox']+ap_calc['Fet_Ap_cat_12ox']+ap_calc['Mn_Ap_cat_12ox']
    +ap_calc['Mg_Ap_cat_12ox']+ap_calc['Ca_Ap_cat_12ox']+ap_calc['Na_Ap_cat_12ox']
    +ap_calc['K_Ap_cat_12ox']+ap_calc['Cr_Ap_cat_12ox']+ap_calc['P_Ap_cat_12ox'])

    ap_calc['Ca_P_Ap'] = ap_calc['Ca_Ap_cat_12ox'] + ap_calc['P_Ap_cat_12ox']

    cat_prop = calculate_cat_proportions_apatite(ap_comps=ap_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd2 = pd.concat([ap_comps, ap_calc, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

def calculate_mol_proportions_biotite(bt_comps):

    '''Import Biotite compositions using bt_comps=My_Biotites, returns mole proportions
   Parameters
    -------
    bt_comps: pandas.DataFrame
            Panda DataFrame of biotite compositions with column headings SiO2_Bt, MgO_Bt etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for biotites with column headings of the form SiO2_Bt_mol_prop
    '''

    oxide_mass_bt = {'SiO2_Bt': 60.0843, 'MgO_Bt': 40.3044, 'FeOt_Bt': 71.8464,
    'CaO_Bt': 56.0774,'Al2O3_Bt': 101.961, 'Na2O_Bt': 61.9789, 'K2O_Bt': 94.196,
    'MnO_Bt': 70.9375, 'TiO2_Bt': 79.7877, 'Cr2O3_Bt': 151.9982, 'P2O5_Bt': 141.937}
    oxide_mass_bt_df = pd.DataFrame.from_dict(oxide_mass_bt, orient='index').T
    oxide_mass_bt_df['Sample_ID_Bt'] = 'MolWt'
    oxide_mass_bt_df.set_index('Sample_ID_Bt', inplace=True)

    # This makes it match the columns in the oxide mass dataframe
    bt_wt = bt_comps.reindex(oxide_mass_bt_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    bt_wt_combo = pd.concat([oxide_mass_bt_df, bt_wt],)
    # Drop the calculation column
    mol_prop_anhyd = bt_wt_combo.div(
        bt_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd

def calculate_cat_proportions_biotite(*, bt_comps=None, oxide_headers=False):

    '''Import biotite compositions using bt_comps=My_biotites, returns cation proportions

    Parameters
    -------
    bt_comps: pandas.DataFrame
            biotite compositions with column headings SiO2_Bt, MgO_Bt etc.
    
    oxide_headers: bool
        default=False, returns as Ti_Bt_cat_prop.
        =True returns Ti_Bt_cat_prop.
        This is used for rapid matrix division for
        pre-processing of data for cation fractions etc

    Returns
    -------
    pandas DataFrame
        cation proportions for biotite with column headings of the form... Bt_cat_prop
    '''

    cation_num_bt = {'SiO2_Bt': 1, 'MgO_Bt': 1, 'FeOt_Bt': 1, 'CaO_Bt': 1, 'Al2O3_Bt': 2, 'Na2O_Bt': 2,
                   'K2O_Bt': 2, 'MnO_Bt': 1, 'TiO2_Bt': 1, 'Cr2O3_Bt': 2, 'P2O5_Bt': 2}
    cation_num_bt_df = pd.DataFrame.from_dict(cation_num_bt, orient='index').T
    cation_num_bt_df['Sample_ID_Bt'] = 'CatNum'
    cation_num_bt_df.set_index('Sample_ID_Bt', inplace=True)

    oxide_mass_bt = {'SiO2_Bt': 60.0843, 'MgO_Bt': 40.3044, 'FeOt_Bt': 71.8464,
    'CaO_Bt': 56.0774,'Al2O3_Bt': 101.961, 'Na2O_Bt': 61.9789, 'K2O_Bt': 94.196,
    'MnO_Bt': 70.9375, 'TiO2_Bt': 79.7877, 'Cr2O3_Bt': 151.9982, 'P2O5_Bt': 141.937}

    oxide_mass_bt_df = pd.DataFrame.from_dict(oxide_mass_bt, orient='index').T
    oxide_mass_bt_df['Sample_ID_Bt'] = 'MolWt'
    oxide_mass_bt_df.set_index('Sample_ID_Bt', inplace=True)

    bt_prop_no_cat_num = calculate_mol_proportions_biotite(
        bt_comps=bt_comps)
    bt_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in bt_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_bt_df.reindex(
        oxide_mass_bt_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, bt_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    if oxide_headers is True:
        return cation_prop_anhyd
    if oxide_headers is False:
        cation_prop_anhyd2=cation_prop_anhyd.rename(columns={
                            'SiO2_Bt_cat_prop': 'Si_Bt_cat_prop',
                            'TiO2_Bt_cat_prop': 'Ti_Bt_cat_prop',
                            'Al2O3_Bt_cat_prop': 'Al_Bt_cat_prop',
                            'FeOt_Bt_cat_prop': 'Fet_Bt_cat_prop',
                            'MnO_Bt_cat_prop': 'Mn_Bt_cat_prop',
                            'MgO_Bt_cat_prop': 'Mg_Bt_cat_prop',
                            'CaO_Bt_cat_prop': 'Ca_Bt_cat_prop',
                            'Na2O_Bt_cat_prop': 'Na_Bt_cat_prop',
                            'K2O_Bt_cat_prop': 'K_Bt_cat_prop',
                            'Cr2O3_Bt_cat_prop': 'Cr_Bt_cat_prop',
                            'P2O5_Bt_cat_prop': 'P_Bt_cat_prop',})

        return cation_prop_anhyd2

def calculate_oxygens_biotite(bt_comps):
    
    '''Import biotite compositions using bt_comps=My_Bts, returns number of oxygens (e.g., mol proportions * number of O in formula unit)

    Parameters
    -------
    bt_comps: pandas.DataFrame
        biotite compositions with column headings SiO2_Bt, MgO_Bt etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Bt_ox
    '''

    oxygen_num_bt = {'SiO2_Bt': 2, 'MgO_Bt': 1, 'FeOt_Bt': 1, 'CaO_Bt': 1, 'Al2O3_Bt': 3, 'Na2O_Bt': 1,
                    'K2O_Bt': 1, 'MnO_Bt': 1, 'TiO2_Bt': 2, 'Cr2O3_Bt': 3, 'P2O5_Bt': 5}
    oxygen_num_bt_df = pd.DataFrame.from_dict(oxygen_num_bt, orient='index').T
    oxygen_num_bt_df['Sample_ID_Bt'] = 'OxNum'
    oxygen_num_bt_df.set_index('Sample_ID_Bt', inplace=True)

    mol_prop = calculate_mol_proportions_biotite(bt_comps=bt_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_bt_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd

def calculate_10oxygens_biotite(bt_comps):
    
    '''Import biotite compositions using bt_comps=My_Bts, returns cations on the basis of 10 oxygens.
    
    Parameters
    -------
    bt_comps: pandas.DataFrame
        biotite compositions with column headings SiO2_Bt, MgO_Bt etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 10 oxygens, with column headings of the form... Bt_cat_10ox.
    '''

    cation_num_bt = {'SiO2_Bt': 1, 'MgO_Bt': 1, 'FeOt_Bt': 1, 'CaO_Bt': 1, 'Al2O3_Bt': 2, 'Na2O_Bt': 2,
                   'K2O_Bt': 2, 'MnO_Bt': 1, 'TiO2_Bt': 1, 'Cr2O3_Bt': 2, 'P2O5_Bt': 2}

    cation_num_bt_df = pd.DataFrame.from_dict(cation_num_bt, orient='index').T
    cation_num_bt_df['Sample_ID_Bt'] = 'CatNum'
    cation_num_bt_df.set_index('Sample_ID_Bt', inplace=True)

    oxygens = calculate_oxygens_biotite(bt_comps=bt_comps)
    renorm_factor = 10 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_biotite(bt_comps=bt_comps)
    mol_prop['oxy_renorm_factor_bt'] = renorm_factor
    mol_prop_10 = mol_prop.multiply(mol_prop['oxy_renorm_factor_bt'], axis='rows')
    mol_prop_10.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_10.columns]

    ox_num_reindex = cation_num_bt_df.reindex(
        mol_prop_10.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_10])
    cation_10 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_10.columns = [str(col).replace('_mol_prop', '_cat_10ox')
                        for col in mol_prop.columns]

    cation_10_2=cation_10.rename(columns={
                            'SiO2_Bt_cat_10ox': 'Si_Bt_cat_10ox',
                            'TiO2_Bt_cat_10ox': 'Ti_Bt_cat_10ox',
                            'Al2O3_Bt_cat_10ox': 'Al_Bt_cat_10ox',
                            'FeOt_Bt_cat_10ox': 'Fet_Bt_cat_10ox',
                            'MnO_Bt_cat_10ox': 'Mn_Bt_cat_10ox',
                            'MgO_Bt_cat_10ox': 'Mg_Bt_cat_10ox',
                            'CaO_Bt_cat_10ox': 'Ca_Bt_cat_10ox',
                            'Na2O_Bt_cat_10ox': 'Na_Bt_cat_10ox',
                            'K2O_Bt_cat_10ox': 'K_Bt_cat_10ox',
                            'Cr2O3_Bt_cat_10ox': 'Cr_Bt_cat_10ox',
                            'P2O5_Bt_cat_10ox': 'P_Bt_cat_10ox',})

    return cation_10_2


def calculate_biotite_components(bt_comps, append):
    
    '''Import Biotite compositions using bt_comps=My_Biotites, returns mole proportions
    
    Parameters
    -------
    bt_comps: pandas.DataFrame
            Panda DataFrame of biotite compositions with column headings SiO2_Bt, MgO_Bt etc.
    
    Returns
    -------
    pandas DataFrame
        components for biotites with column headings of the form SiO2_Bt_mol_prop
    '''

    bt_comps_new = bt_comps.copy()
    bt_comps_new.columns = [col.replace(append, '_Bt') for col in bt_comps_new.columns]
    bt_comps = bt_comps_new.copy()

    bt_calc = calculate_10oxygens_biotite(bt_comps=bt_comps)
    bt_calc['Bt_Cation_Sum'] = (bt_calc['Si_Bt_cat_10ox']+bt_calc['Ti_Bt_cat_10ox']
    +bt_calc['Al_Bt_cat_10ox']+bt_calc['Fet_Bt_cat_10ox']+bt_calc['Mn_Bt_cat_10ox']
    +bt_calc['Mg_Bt_cat_10ox']+bt_calc['Ca_Bt_cat_10ox']+bt_calc['Na_Bt_cat_10ox']
    +bt_calc['K_Bt_cat_10ox']+bt_calc['Cr_Bt_cat_10ox']+bt_calc['P_Bt_cat_10ox'])

    bt_calc['Mg_Fe_Bt'] = bt_calc['Mg_Bt_cat_10ox'] + bt_calc['Fet_Bt_cat_10ox']
    bt_calc['Si_Al_Bt'] = bt_calc['Si_Bt_cat_10ox'] + bt_calc['Al_Bt_cat_10ox']

    cat_prop = calculate_cat_proportions_biotite(bt_comps=bt_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd2 = pd.concat([bt_comps, bt_calc, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

def calculate_mol_proportions_quartz(qz_comps):

    '''Import Quartz compositions using qz_comps=My_Quartzs, returns mole proportions

    Parameters
    -------
    qz_comps: pandas.DataFrame
            Panda DataFrame of quartz compositions with column headings SiO2_Qz, MgO_Qz etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for quartzs with column headings of the form SiO2_Qz_mol_prop
    '''

    oxide_mass_qz = {'SiO2_Qz': 60.0843, 'MgO_Qz': 40.3044, 'FeOt_Qz': 71.8464,
    'CaO_Qz': 56.0774,'Al2O3_Qz': 101.961, 'Na2O_Qz': 61.9789, 'K2O_Qz': 94.196,
    'MnO_Qz': 70.9375, 'TiO2_Qz': 79.7877, 'Cr2O3_Qz': 151.9982, 'P2O5_Qz': 141.937}
    oxide_mass_qz_df = pd.DataFrame.from_dict(oxide_mass_qz, orient='index').T
    oxide_mass_qz_df['Sample_ID_Qz'] = 'MolWt'
    oxide_mass_qz_df.set_index('Sample_ID_Qz', inplace=True)

    # This makes it match the columns in the oxide mass dataframe
    qz_wt = qz_comps.reindex(oxide_mass_qz_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    qz_wt_combo = pd.concat([oxide_mass_qz_df, qz_wt],)
    # Drop the calculation column
    mol_prop_anhyd = qz_wt_combo.div(
        qz_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_cat_proportions_quartz(*, qz_comps=None, oxide_headers=False):
    
    '''Import quartz compositions using qz_comps=My_quartzs, returns cation proportions
    
    Parameters
    -------
    qz_comps: pandas.DataFrame
            quartz compositions with column headings SiO2_Plag, MgO_Plag etc.
    oxide_headers: bool
        default=False, returns as Ti_Qz_cat_prop.
        =True returns Ti_Qz_cat_prop.
        This is used for rapid matrix division for
        pre-processing of data for cation fractions etc

    Returns
    -------
    pandas DataFrame
        cation proportions for quartz with column headings of the form... Qz_cat_prop
    '''

    cation_num_qz = {'SiO2_Qz': 1, 'MgO_Qz': 1, 'FeOt_Qz': 1, 'CaO_Qz': 1, 'Al2O3_Qz': 2, 'Na2O_Qz': 2,
                   'K2O_Qz': 2, 'MnO_Qz': 1, 'TiO2_Qz': 1, 'Cr2O3_Qz': 2, 'P2O5_Qz': 2}
    cation_num_qz_df = pd.DataFrame.from_dict(cation_num_qz, orient='index').T
    cation_num_qz_df['Sample_ID_Qz'] = 'CatNum'
    cation_num_qz_df.set_index('Sample_ID_Qz', inplace=True)

    oxide_mass_qz = {'SiO2_Qz': 60.0843, 'MgO_Qz': 40.3044, 'FeOt_Qz': 71.8464,
    'CaO_Qz': 56.0774,'Al2O3_Qz': 101.961, 'Na2O_Qz': 61.9789, 'K2O_Qz': 94.196,
    'MnO_Qz': 70.9375, 'TiO2_Qz': 79.7877, 'Cr2O3_Qz': 151.9982, 'P2O5_Qz': 141.937}

    oxide_mass_qz_df = pd.DataFrame.from_dict(oxide_mass_qz, orient='index').T
    oxide_mass_qz_df['Sample_ID_Qz'] = 'MolWt'
    oxide_mass_qz_df.set_index('Sample_ID_Qz', inplace=True)

    qz_prop_no_cat_num = calculate_mol_proportions_quartz(
        qz_comps=qz_comps)
    qz_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in qz_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_qz_df.reindex(
        oxide_mass_qz_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, qz_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    if oxide_headers is True:
        return cation_prop_anhyd
    if oxide_headers is False:
        cation_prop_anhyd2=cation_prop_anhyd.rename(columns={
                            'SiO2_Qz_cat_prop': 'Si_Qz_cat_prop',
                            'TiO2_Qz_cat_prop': 'Ti_Qz_cat_prop',
                            'Al2O3_Qz_cat_prop': 'Al_Qz_cat_prop',
                            'FeOt_Qz_cat_prop': 'Fet_Qz_cat_prop',
                            'MnO_Qz_cat_prop': 'Mn_Qz_cat_prop',
                            'MgO_Qz_cat_prop': 'Mg_Qz_cat_prop',
                            'CaO_Qz_cat_prop': 'Ca_Qz_cat_prop',
                            'Na2O_Qz_cat_prop': 'Na_Qz_cat_prop',
                            'K2O_Qz_cat_prop': 'K_Qz_cat_prop',
                            'Cr2O3_Qz_cat_prop': 'Cr_Qz_cat_prop',
                            'P2O5_Qz_cat_prop': 'P_Qz_cat_prop',})

        return cation_prop_anhyd2

def calculate_oxygens_quartz(qz_comps):
    
    '''Import quartz compositions using qz_comps=My_Qzs, returns number of oxygens 
    (e.g., mol proportions * number of O in formula unit)

    Parameters
    -------
    qz_comps: pandas.DataFrame
        quartz compositions with column headings SiO2_Qz, MgO_Qz etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Qz_ox
    '''

    oxygen_num_qz = {'SiO2_Qz': 2, 'MgO_Qz': 1, 'FeOt_Qz': 1, 'CaO_Qz': 1, 'Al2O3_Qz': 3, 'Na2O_Qz': 1,
                    'K2O_Qz': 1, 'MnO_Qz': 1, 'TiO2_Qz': 2, 'Cr2O3_Qz': 3, 'P2O5_Qz': 5}
    oxygen_num_qz_df = pd.DataFrame.from_dict(oxygen_num_qz, orient='index').T
    oxygen_num_qz_df['Sample_ID_Qz'] = 'OxNum'
    oxygen_num_qz_df.set_index('Sample_ID_Qz', inplace=True)

    mol_prop = calculate_mol_proportions_quartz(qz_comps=qz_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_qz_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd

def calculate_2oxygens_quartz(qz_comps):
    
    '''Import quartz compositions using qz_comps=My_Qzs, returns cations on the basis of 4 oxygens.

    Parameters
    -------
    qz_comps: pandas.DataFrame
        quartz compositions with column headings SiO2_Qz, MgO_Qz etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 2 oxygens, with column headings of the form... Qz_cat_2ox.
    '''

    cation_num_qz = {'SiO2_Qz': 1, 'MgO_Qz': 1, 'FeOt_Qz': 1, 'CaO_Qz': 1, 'Al2O3_Qz': 2, 'Na2O_Qz': 2,
                   'K2O_Qz': 2, 'MnO_Qz': 1, 'TiO2_Qz': 1, 'Cr2O3_Qz': 2, 'P2O5_Qz': 2}

    cation_num_qz_df = pd.DataFrame.from_dict(cation_num_qz, orient='index').T
    cation_num_qz_df['Sample_ID_Qz'] = 'CatNum'
    cation_num_qz_df.set_index('Sample_ID_Qz', inplace=True)

    oxygens = calculate_oxygens_quartz(qz_comps=qz_comps)
    renorm_factor = 2 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_quartz(qz_comps=qz_comps)
    mol_prop['oxy_renorm_factor_qz'] = renorm_factor
    mol_prop_2 = mol_prop.multiply(mol_prop['oxy_renorm_factor_qz'], axis='rows')
    mol_prop_2.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_2.columns]

    ox_num_reindex = cation_num_qz_df.reindex(
        mol_prop_2.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_2])
    cation_2 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_2.columns = [str(col).replace('_mol_prop', '_cat_2ox')
                        for col in mol_prop.columns]

    cation_2_2=cation_2.rename(columns={
                            'SiO2_Qz_cat_2ox': 'Si_Qz_cat_2ox',
                            'TiO2_Qz_cat_2ox': 'Ti_Qz_cat_2ox',
                            'Al2O3_Qz_cat_2ox': 'Al_Qz_cat_2ox',
                            'FeOt_Qz_cat_2ox': 'Fet_Qz_cat_2ox',
                            'MnO_Qz_cat_2ox': 'Mn_Qz_cat_2ox',
                            'MgO_Qz_cat_2ox': 'Mg_Qz_cat_2ox',
                            'CaO_Qz_cat_2ox': 'Ca_Qz_cat_2ox',
                            'Na2O_Qz_cat_2ox': 'Na_Qz_cat_2ox',
                            'K2O_Qz_cat_2ox': 'K_Qz_cat_2ox',
                            'Cr2O3_Qz_cat_2ox': 'Cr_Qz_cat_2ox',
                            'P2O5_Qz_cat_2ox': 'P_Qz_cat_2ox',})

    return cation_2_2

def calculate_quartz_components(qz_comps, append):

    '''Import quartz compositions using qz_comps=My_Qzs, returns cations on the basis of 4 oxygens.

    Parameters
    -------
    qz_comps: pandas.DataFrame
        quartz compositions with column headings SiO2_Qz, MgO_Qz etc.

    Returns
    -------
    pandas DataFrame
        components on the basis of 2 oxygens, with column headings of the form... Qz_cat_2ox.
    '''

    qz_comps_new = qz_comps.copy()
    qz_comps_new.columns = [col.replace(append, '_Qz') for col in qz_comps_new.columns]
    qz_comps = qz_comps_new.copy()

    qz_calc = calculate_2oxygens_quartz(qz_comps=qz_comps)
    qz_calc['Qz_Cation_Sum'] = (qz_calc['Si_Qz_cat_2ox']+qz_calc['Ti_Qz_cat_2ox']
    +qz_calc['Al_Qz_cat_2ox']+qz_calc['Fet_Qz_cat_2ox']+qz_calc['Mn_Qz_cat_2ox']
    +qz_calc['Mg_Qz_cat_2ox']+qz_calc['Ca_Qz_cat_2ox']+qz_calc['Na_Qz_cat_2ox']
    +qz_calc['K_Qz_cat_2ox']+qz_calc['Cr_Qz_cat_2ox']+qz_calc['P_Qz_cat_2ox'])

    qz_calc['Si_Al_Qz'] = qz_calc['Si_Qz_cat_2ox'] + qz_calc['Al_Qz_cat_2ox']

    cat_prop = calculate_cat_proportions_quartz(qz_comps=qz_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd2 = pd.concat([qz_comps, qz_calc, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2


def calculate_mol_proportions_garnet(gt_comps):

    '''Import Garnet compositions using gt_comps=My_Garnets, returns mole proportions

    Parameters
    -------
    gt_comps: pandas.DataFrame
            Panda DataFrame of garnet compositions with column headings SiO2_Gt, MgO_Gt etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for garnets with column headings of the form SiO2_Gt_mol_prop
    '''

    oxide_mass_gt = {'SiO2_Gt': 60.0843, 'MgO_Gt': 40.3044, 'FeOt_Gt': 71.8464,
    'CaO_Gt': 56.0774,'Al2O3_Gt': 101.961, 'Na2O_Gt': 61.9789, 'K2O_Gt': 94.196,
    'MnO_Gt': 70.9375, 'TiO2_Gt': 79.7877, 'Cr2O3_Gt': 151.9982, 'P2O5_Gt': 141.937,
    'NiO_Gt': 74.6994}
    oxide_mass_gt_df = pd.DataFrame.from_dict(oxide_mass_gt, orient='index').T
    oxide_mass_gt_df['Sample_ID_Gt'] = 'MolWt'
    oxide_mass_gt_df.set_index('Sample_ID_Gt', inplace=True)

    # This makes it match the columns in the oxide mass dataframe
    gt_wt = gt_comps.reindex(oxide_mass_gt_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    gt_wt_combo = pd.concat([oxide_mass_gt_df, gt_wt],)
    # Drop the calculation column
    mol_prop_anhyd = gt_wt_combo.div(
        gt_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd


def calculate_cat_proportions_garnet(*, gt_comps=None, oxide_headers=False):

    '''Import garnet compositions using gt_comps=My_garnets, returns cation proportions

    Parameters
    -------
    gt_comps: pandas.DataFrame
            garnet compositions with column headings SiO2_Plag, MgO_Plag etc.
    
    oxide_headers: bool
        default=False, returns as Ti_Gt_cat_prop.
        =True returns Ti_Gt_cat_prop.
        This is used for rapid matrix division for
        pre-processing of data for cation fractions etc

    Returns
    -------
    pandas DataFrame
        cation proportions for garnet with column headings of the form... Gt_cat_prop
    '''

    cation_num_gt = {'SiO2_Gt': 1, 'MgO_Gt': 1, 'FeOt_Gt': 1, 'CaO_Gt': 1, 'Al2O3_Gt': 2, 'Na2O_Gt': 2,
                   'K2O_Gt': 2, 'MnO_Gt': 1, 'TiO2_Gt': 1, 'Cr2O3_Gt': 2, 'P2O5_Gt': 2, 'NiO_Gt': 1}
    cation_num_gt_df = pd.DataFrame.from_dict(cation_num_gt, orient='index').T
    cation_num_gt_df['Sample_ID_Gt'] = 'CatNum'
    cation_num_gt_df.set_index('Sample_ID_Gt', inplace=True)

    oxide_mass_gt = {'SiO2_Gt': 60.0843, 'MgO_Gt': 40.3044, 'FeOt_Gt': 71.8464,
    'CaO_Gt': 56.0774,'Al2O3_Gt': 101.961, 'Na2O_Gt': 61.9789, 'K2O_Gt': 94.196,
    'MnO_Gt': 70.9375, 'TiO2_Gt': 79.7877, 'Cr2O3_Gt': 151.9982, 'P2O5_Gt': 141.937,
    'NiO_Gt': 74.6994}

    oxide_mass_gt_df = pd.DataFrame.from_dict(oxide_mass_gt, orient='index').T
    oxide_mass_gt_df['Sample_ID_Gt'] = 'MolWt'
    oxide_mass_gt_df.set_index('Sample_ID_Gt', inplace=True)

    gt_prop_no_cat_num = calculate_mol_proportions_garnet(
        gt_comps=gt_comps)
    gt_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in gt_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_gt_df.reindex(
        oxide_mass_gt_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, gt_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    if oxide_headers is True:
        return cation_prop_anhyd
    if oxide_headers is False:
        cation_prop_anhyd2=cation_prop_anhyd.rename(columns={
                            'SiO2_Gt_cat_prop': 'Si_Gt_cat_prop',
                            'TiO2_Gt_cat_prop': 'Ti_Gt_cat_prop',
                            'Al2O3_Gt_cat_prop': 'Al_Gt_cat_prop',
                            'FeOt_Gt_cat_prop': 'Fet_Gt_cat_prop',
                            'MnO_Gt_cat_prop': 'Mn_Gt_cat_prop',
                            'MgO_Gt_cat_prop': 'Mg_Gt_cat_prop',
                            'CaO_Gt_cat_prop': 'Ca_Gt_cat_prop',
                            'Na2O_Gt_cat_prop': 'Na_Gt_cat_prop',
                            'K2O_Gt_cat_prop': 'K_Gt_cat_prop',
                            'Cr2O3_Gt_cat_prop': 'Cr_Gt_cat_prop',
                            'P2O5_Gt_cat_prop': 'P_Gt_cat_prop',
                            'NiO_Gt_cat_prop': 'Ni_Gt_cat_prop',})

        return cation_prop_anhyd2

def calculate_oxygens_garnet(gt_comps):
    
    '''Import garnet compositions using gt_comps=My_Gts, returns number of oxygens 
    (e.g., mol proportions * number of O in formula unit)

    Parameters
    -------
    gt_comps: pandas.DataFrame
        garnet compositions with column headings SiO2_Gt, MgO_Gt etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Gt_ox
    '''

    oxygen_num_gt = {'SiO2_Gt': 2, 'MgO_Gt': 1, 'FeOt_Gt': 1, 'CaO_Gt': 1, 'Al2O3_Gt': 3, 'Na2O_Gt': 1,
                    'K2O_Gt': 1, 'MnO_Gt': 1, 'TiO2_Gt': 2, 'Cr2O3_Gt': 3, 'P2O5_Gt': 5, 'NiO_Gt': 1}
    oxygen_num_gt_df = pd.DataFrame.from_dict(oxygen_num_gt, orient='index').T
    oxygen_num_gt_df['Sample_ID_Gt'] = 'OxNum'
    oxygen_num_gt_df.set_index('Sample_ID_Gt', inplace=True)

    mol_prop = calculate_mol_proportions_garnet(gt_comps=gt_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_gt_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd

def calculate_12oxygens_garnet(gt_comps):

    '''Import garnet compositions using gt_comps=My_Gts, returns cations on the basis of 4 oxygens.

    Parameters
    -------
    gt_comps: pandas.DataFrame
        garnet compositions with column headings SiO2_Gt, MgO_Gt etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 12 oxygens, with column headings of the form... Gt_cat_12ox.
    '''

    cation_num_gt = {'SiO2_Gt': 1, 'MgO_Gt': 1, 'FeOt_Gt': 1, 'CaO_Gt': 1, 'Al2O3_Gt': 2, 'Na2O_Gt': 2,
                   'K2O_Gt': 2, 'MnO_Gt': 1, 'TiO2_Gt': 1, 'Cr2O3_Gt': 2, 'P2O5_Gt': 2, 'NiO_Gt': 1}

    cation_num_gt_df = pd.DataFrame.from_dict(cation_num_gt, orient='index').T
    cation_num_gt_df['Sample_ID_Gt'] = 'CatNum'
    cation_num_gt_df.set_index('Sample_ID_Gt', inplace=True)

    oxygens = calculate_oxygens_garnet(gt_comps=gt_comps)
    renorm_factor = 12 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_garnet(gt_comps=gt_comps)
    mol_prop['oxy_renorm_factor_gt'] = renorm_factor
    mol_prop_12 = mol_prop.multiply(mol_prop['oxy_renorm_factor_gt'], axis='rows')
    mol_prop_12.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_12.columns]

    ox_num_reindex = cation_num_gt_df.reindex(
        mol_prop_12.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_12])
    cation_12 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_12.columns = [str(col).replace('_mol_prop', '_cat_12ox')
                        for col in mol_prop.columns]

    cation_12_2=cation_12.rename(columns={
                            'SiO2_Gt_cat_12ox': 'Si_Gt_cat_12ox',
                            'TiO2_Gt_cat_12ox': 'Ti_Gt_cat_12ox',
                            'Al2O3_Gt_cat_12ox': 'Al_Gt_cat_12ox',
                            'FeOt_Gt_cat_12ox': 'Fet_Gt_cat_12ox',
                            'MnO_Gt_cat_12ox': 'Mn_Gt_cat_12ox',
                            'MgO_Gt_cat_12ox': 'Mg_Gt_cat_12ox',
                            'CaO_Gt_cat_12ox': 'Ca_Gt_cat_12ox',
                            'Na2O_Gt_cat_12ox': 'Na_Gt_cat_12ox',
                            'K2O_Gt_cat_12ox': 'K_Gt_cat_12ox',
                            'Cr2O3_Gt_cat_12ox': 'Cr_Gt_cat_12ox',
                            'P2O5_Gt_cat_12ox': 'P_Gt_cat_12ox',
                            'NiO_Gt_cat_12ox': 'Ni_Gt_cat_12ox',})

    return cation_12_2


def calculate_garnet_components(gt_comps, append):

    '''Import garnet compositions using gt_comps=My_Gts, returns cations on the basis of 12 oxygens.

    Parameters
    -------
    gt_comps: pandas.DataFrame
        garnet compositions with column headings SiO2_Gt, MgO_Gt etc.

    Returns
    -------
    pandas DataFrame
        components on the basis of 12 oxygens, with column headings of the form... Gt_cat_12ox.
    '''

    gt_comps_new = gt_comps.copy()
    gt_comps_new.columns = [col.replace(append, '_Gt') for col in gt_comps_new.columns]
    gt_comps = gt_comps_new.copy()

    gt_calc = calculate_12oxygens_garnet(gt_comps=gt_comps)
    gt_calc['Gt_Cation_Sum'] = (gt_calc['Si_Gt_cat_12ox']+gt_calc['Ti_Gt_cat_12ox']
    +gt_calc['Al_Gt_cat_12ox']+gt_calc['Fet_Gt_cat_12ox']+gt_calc['Mn_Gt_cat_12ox']
    +gt_calc['Mg_Gt_cat_12ox']+gt_calc['Ca_Gt_cat_12ox']+gt_calc['Na_Gt_cat_12ox']
    +gt_calc['K_Gt_cat_12ox']+gt_calc['Cr_Gt_cat_12ox']+gt_calc['P_Gt_cat_12ox']
    +gt_calc['Ni_Gt_cat_12ox'])

    gt_calc['Mg_MgFeCa_Gt'] = gt_calc['Mg_Gt_cat_12ox'] / (gt_calc['Mg_Gt_cat_12ox'] +\
        gt_calc['Fet_Gt_cat_12ox'] + gt_calc['Ca_Gt_cat_12ox'])
    gt_calc['Fe_MgFeCa_Gt'] = gt_calc['Fet_Gt_cat_12ox'] / (gt_calc['Mg_Gt_cat_12ox'] +\
        gt_calc['Fet_Gt_cat_12ox'] + gt_calc['Ca_Gt_cat_12ox'])
    gt_calc['Ca_MgFeCa_Gt'] = gt_calc['Ca_Gt_cat_12ox'] / (gt_calc['Mg_Gt_cat_12ox'] +\
        gt_calc['Fet_Gt_cat_12ox'] + gt_calc['Ca_Gt_cat_12ox'])

    gt_calc['Al_AlCr_Gt'] = gt_calc['Al_Gt_cat_12ox'] / (gt_calc['Al_Gt_cat_12ox'] +\
     gt_calc['Cr_Gt_cat_12ox'])
    gt_calc['Cr_AlCr_Gt'] = gt_calc['Cr_Gt_cat_12ox'] / (gt_calc['Al_Gt_cat_12ox'] +\
     gt_calc['Cr_Gt_cat_12ox'])

    cat_prop = calculate_cat_proportions_garnet(gt_comps=gt_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd2 = pd.concat([gt_comps, gt_calc, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

def calculate_mol_proportions_kspar(kspar_comps):
    
    '''Import Kspar compositions using kspar_comps=My_Kspar, returns mole proportions

    Parameters
    -------
    kspar_comps: pandas.DataFrame
            Panda DataFrame of kspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        mole proportions for kspar with column headings of the form SiO2_Kspar_mol_prop
    '''

    oxide_mass_sp = {'SiO2_Kspar': 60.0843, 'MgO_Kspar': 40.3044, 'FeOt_Kspar': 71.8464,
    'CaO_Kspar': 56.0774,'Al2O3_Kspar': 101.961, 'Na2O_Kspar': 61.9789, 'K2O_Kspar': 94.196,
    'MnO_Kspar': 70.9375, 'TiO2_Kspar': 79.7877, 'Cr2O3_Kspar': 151.9982, 'P2O5_Kspar': 141.937}
    oxide_mass_ksp_df = pd.DataFrame.from_dict(oxide_mass_sp, orient='index').T
    oxide_mass_ksp_df['Sample_ID_Kspar'] = 'MolWt'
    oxide_mass_ksp_df.set_index('Sample_ID_Kspar', inplace=True)

    # This makes it match the columns in the oxide mass dataframe
    ksp_wt = kspar_comps.reindex(oxide_mass_ksp_df.columns, axis=1).fillna(0)
    # Combine the molecular weight and weight percent dataframes
    ksp_wt_combo = pd.concat([oxide_mass_ksp_df, ksp_wt],)
    # Drop the calculation column
    mol_prop_anhyd = ksp_wt_combo.div(
        ksp_wt_combo.loc['MolWt', :], axis='columns').drop(['MolWt'])
    mol_prop_anhyd.columns = [
        str(col) + '_mol_prop' for col in mol_prop_anhyd.columns]
    return mol_prop_anhyd

def calculate_cat_proportions_kspar(*, kspar_comps=None, oxide_headers=False):
    
    '''Import kspar compositions using kspar_comps=My_kspars, returns cation proportions

    Parameters
    -------
    kspar_comps: pandas.DataFrame
            kspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    oxide_headers: bool
        default=False, returns as Ti_Kspar_cat_prop.
        =True returns Ti_Kspar_cat_prop.
        This is used for rapid matrix division for
        pre-processing of data for cation fractions etc

    Returns
    -------
    pandas DataFrame
        cation proportions for kspar with column headings of the form... Ksp_cat_prop
    '''

    cation_num_sp = {'SiO2_Kspar': 1, 'MgO_Kspar': 1, 'FeOt_Kspar': 1, 'CaO_Kspar': 1, 'Al2O3_Kspar': 2, 'Na2O_Kspar': 2,
                   'K2O_Kspar': 2, 'MnO_Kspar': 1, 'TiO2_Kspar': 1, 'Cr2O3_Kspar': 2, 'P2O5_Kspar': 2}
    cation_num_ksp_df = pd.DataFrame.from_dict(cation_num_sp, orient='index').T
    cation_num_ksp_df['Sample_ID_Kspar'] = 'CatNum'
    cation_num_ksp_df.set_index('Sample_ID_Kspar', inplace=True)

    oxide_mass_sp = {'SiO2_Kspar': 60.0843, 'MgO_Kspar': 40.3044, 'FeOt_Kspar': 71.8464,
    'CaO_Kspar': 56.0774,'Al2O3_Kspar': 101.961, 'Na2O_Kspar': 61.9789, 'K2O_Kspar': 94.196,
    'MnO_Kspar': 70.9375, 'TiO2_Kspar': 79.7877, 'Cr2O3_Kspar': 151.9982, 'P2O5_Kspar': 141.937}
    oxide_mass_ksp_df = pd.DataFrame.from_dict(oxide_mass_sp, orient='index').T
    oxide_mass_ksp_df['Sample_ID_Kspar'] = 'MolWt'
    oxide_mass_ksp_df.set_index('Sample_ID_Kspar', inplace=True)

    ksp_prop_no_cat_num = calculate_mol_proportions_kspar(
        kspar_comps=kspar_comps)
    ksp_prop_no_cat_num.columns = [str(col).replace(
        '_mol_prop', '') for col in ksp_prop_no_cat_num.columns]
    ox_num_reindex = cation_num_ksp_df.reindex(
        oxide_mass_ksp_df.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, ksp_prop_no_cat_num])
    cation_prop_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])
    cation_prop_anhyd.columns = [
        str(col) + '_cat_prop' for col in cation_prop_anhyd.columns]
    if oxide_headers is True:
        return cation_prop_anhyd
    if oxide_headers is False:
        cation_prop_anhyd2=cation_prop_anhyd.rename(columns={
                            'SiO2_Kspar_cat_prop': 'Si_Kspar_cat_prop',
                            'TiO2_Kspar_cat_prop': 'Ti_Kspar_cat_prop',
                            'Al2O3_Kspar_cat_prop': 'Al_Kspar_cat_prop',
                            'FeOt_Kspar_cat_prop': 'Fet_Kspar_cat_prop',
                            'MnO_Kspar_cat_prop': 'Mn_Kspar_cat_prop',
                            'MgO_Kspar_cat_prop': 'Mg_Kspar_cat_prop',
                            'CaO_Kspar_cat_prop': 'Ca_Kspar_cat_prop',
                            'Na2O_Kspar_cat_prop': 'Na_Kspar_cat_prop',
                            'K2O_Kspar_cat_prop': 'K_Kspar_cat_prop',
                            'Cr2O3_Kspar_cat_prop': 'Cr_Kspar_cat_prop',
                            'P2O5_Kspar_cat_prop': 'P_Kspar_cat_prop',
                            })

        return cation_prop_anhyd2

def calculate_oxygens_kspar(kspar_comps):

    '''Import kspar compositions using kspar_comps=My_Kspars, returns number of oxygens 
    (e.g., mol proportions * number of O in formula unit)

    Parameters
    -------
    kspar_comps: pandas.DataFrame
        kspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        number of oxygens with column headings of the form SiO2_Kspar_ox
    '''

    oxygen_num_sp = {'SiO2_Kspar': 2, 'MgO_Kspar': 1, 'FeOt_Kspar': 1, 'CaO_Kspar': 1, 'Al2O3_Kspar': 3, 'Na2O_Kspar': 1,
                    'K2O_Kspar': 1, 'MnO_Kspar': 1, 'TiO2_Kspar': 2, 'Cr2O3_Kspar': 3, 'P2O5_Kspar': 5}
    oxygen_num_ksp_df = pd.DataFrame.from_dict(oxygen_num_sp, orient='index').T
    oxygen_num_ksp_df['Sample_ID_Kspar'] = 'OxNum'
    oxygen_num_ksp_df.set_index('Sample_ID_Kspar', inplace=True)

    mol_prop = calculate_mol_proportions_kspar(kspar_comps=kspar_comps)
    mol_prop.columns = [str(col).replace('_mol_prop', '')
                        for col in mol_prop.columns]
    ox_num_reindex = oxygen_num_ksp_df.reindex(
        mol_prop.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop])
    oxygens_anhyd = df_calc_comb.multiply(
        df_calc_comb.loc['OxNum', :], axis='columns').drop(['OxNum'])
    oxygens_anhyd.columns = [str(col) + '_ox' for col in oxygens_anhyd.columns]

    return oxygens_anhyd

def calculate_8oxygens_kspar(kspar_comps):

    '''Import kspar compositions using kspar_comps=My_Kspars, returns cations on the basis of 8 oxygens.

    Parameters
    -------
    kspar_comps: pandas.DataFrame
        kspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 8 oxygens, with column headings of the form... Ksp_cat_8ox.
    '''

    cation_num_sp = {'SiO2_Kspar': 1, 'MgO_Kspar': 1, 'FeOt_Kspar': 1, 'CaO_Kspar': 1, 'Al2O3_Kspar': 2, 'Na2O_Kspar': 2,
                   'K2O_Kspar': 2, 'MnO_Kspar': 1, 'TiO2_Kspar': 1, 'Cr2O3_Kspar': 2, 'P2O5_Kspar': 2}

    cation_num_ksp_df = pd.DataFrame.from_dict(cation_num_sp, orient='index').T
    cation_num_ksp_df['Sample_ID_Kspar'] = 'CatNum'
    cation_num_ksp_df.set_index('Sample_ID_Kspar', inplace=True)

    oxygens = calculate_oxygens_kspar(kspar_comps=kspar_comps)
    renorm_factor = 8 / (oxygens.sum(axis='columns'))
    mol_prop = calculate_mol_proportions_kspar(kspar_comps=kspar_comps)
    mol_prop['oxy_renorm_factor_sp'] = renorm_factor
    mol_prop_8 = mol_prop.multiply(mol_prop['oxy_renorm_factor_sp'], axis='rows')
    mol_prop_8.columns = [str(col).replace('_mol_prop', '')
                          for col in mol_prop_8.columns]

    ox_num_reindex = cation_num_ksp_df.reindex(
        mol_prop_8.columns, axis=1).fillna(0)
    df_calc_comb = pd.concat([ox_num_reindex, mol_prop_8])
    cation_8 = df_calc_comb.multiply(
        df_calc_comb.loc['CatNum', :], axis='columns').drop(['CatNum'])

    cation_8.columns = [str(col).replace('_mol_prop', '_cat_8ox')
                        for col in mol_prop.columns]

    cation_8_2=cation_8.rename(columns={
                            'SiO2_Kspar_cat_8ox': 'Si_Kspar_cat_8ox',
                            'TiO2_Kspar_cat_8ox': 'Ti_Kspar_cat_8ox',
                            'Al2O3_Kspar_cat_8ox': 'Al_Kspar_cat_8ox',
                            'FeOt_Kspar_cat_8ox': 'Fet_Kspar_cat_8ox',
                            'MnO_Kspar_cat_8ox': 'Mn_Kspar_cat_8ox',
                            'MgO_Kspar_cat_8ox': 'Mg_Kspar_cat_8ox',
                            'CaO_Kspar_cat_8ox': 'Ca_Kspar_cat_8ox',
                            'Na2O_Kspar_cat_8ox': 'Na_Kspar_cat_8ox',
                            'K2O_Kspar_cat_8ox': 'K_Kspar_cat_8ox',
                            'Cr2O3_Kspar_cat_8ox': 'Cr_Kspar_cat_8ox',
                            'P2O5_Kspar_cat_8ox': 'P_Kspar_cat_8ox',})

    return cation_8_2


def calculate_kspar_components(kspar_comps, append):

    '''Import kspar compositions using kspar_comps=My_Kspars, returns cations on the basis of 8 oxygens.

    Parameters
    -------
    kspar_comps: pandas.DataFrame
        kspar compositions with column headings SiO2_Kspar, MgO_Kspar etc.

    Returns
    -------
    pandas DataFrame
        cations on the basis of 8 oxygens, with column headings of the form... Ksp_cat_8ox.
    '''

    kspar_comps_new = kspar_comps.copy()
    kspar_comps_new.columns = [col.replace(append, '_Kspar') for col in kspar_comps_new.columns]
    kspar_comps = kspar_comps_new.copy()

    kspar_calc = calculate_8oxygens_kspar(kspar_comps=kspar_comps)
    kspar_calc['Kspar_Cation_Sum'] = (kspar_calc['Si_Kspar_cat_8ox']+kspar_calc['Ti_Kspar_cat_8ox']
    +kspar_calc['Al_Kspar_cat_8ox']+kspar_calc['Fet_Kspar_cat_8ox']+kspar_calc['Mn_Kspar_cat_8ox']
    +kspar_calc['Mg_Kspar_cat_8ox']+kspar_calc['Ca_Kspar_cat_8ox']+kspar_calc['Na_Kspar_cat_8ox']
    +kspar_calc['K_Kspar_cat_8ox']+kspar_calc['Cr_Kspar_cat_8ox']+kspar_calc['P_Kspar_cat_8ox'])

    kspar_calc['Na_Ca_M_Kspar'] = kspar_calc['Na_Kspar_cat_8ox'] + kspar_calc['Ca_Kspar_cat_8ox']
    kspar_calc['Si_Al_T_Kspar'] = kspar_calc['Si_Kspar_cat_8ox'] + kspar_calc['Al_Kspar_cat_8ox']

    cat_prop = pt.calculate_cat_proportions_kspar(kspar_comps=kspar_comps)
    cat_prop['sum'] = cat_prop.sum(axis='columns')
    cat_frac_anhyd = cat_prop.div(cat_prop['sum'], axis='rows')
    cat_frac_anhyd.drop(['sum'], axis='columns', inplace=True)
    cat_frac_anhyd.columns = [str(col).replace('prop', 'frac')
                              for col in cat_frac_anhyd.columns]
    cat_frac_anhyd['An_Kspar'] = cat_frac_anhyd['Ca_Kspar_cat_frac'] / \
        (cat_frac_anhyd['Ca_Kspar_cat_frac'] +
         cat_frac_anhyd['Na_Kspar_cat_frac'] + cat_frac_anhyd['K_Kspar_cat_frac'])
    cat_frac_anhyd['Ab_Kspar'] = cat_frac_anhyd['Na_Kspar_cat_frac'] / \
        (cat_frac_anhyd['Ca_Kspar_cat_frac'] +
         cat_frac_anhyd['Na_Kspar_cat_frac'] + cat_frac_anhyd['K_Kspar_cat_frac'])
    cat_frac_anhyd['Or_Kspar'] = 1 - \
        cat_frac_anhyd['An_Kspar'] - cat_frac_anhyd['Ab_Kspar']
    cat_frac_anhyd2 = pd.concat([kspar_comps, kspar_calc, cat_frac_anhyd], axis=1)

    return cat_frac_anhyd2

# %% 

LEPR_Amp_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Amp")['Amps']
LEPR_Amp_cs = pt.get_amp_sites_from_input_not_amp(LEPR_Amp_PreFilt, "_Amp")
LEPR_Amp_cs = LEPR_Amp_cs.rename(columns = {'Ca_B': 'Ca_B_Amp', 'Al_T': 'Al_T_Amp',
                                            'Si_T': 'Si_T_Amp', 'Na_A': 'Na_A_Amp',
                                            'K_A': 'K_A_Amp', 'cation_sum_All': 'Amp_Cation_Sum'})
LEPR_Amp_cs['Na_K_A_Amp'] = LEPR_Amp_cs['Na_A_Amp'] + LEPR_Amp_cs['K_A_Amp']

LEPR_Amp_Pre = pd.concat([LEPR_Amp_PreFilt, LEPR_Amp_cs], axis = 1)
LEPR_Amp = LEPR_Amp_PreFilt.loc[((LEPR_Amp_cs['Amp_Cation_Sum'].between(15, 16)) & (LEPR_Amp_cs['Ca_B_Amp'].between(1.5, 2.1)))]
LEPR_Amp_nosuf = SiteCalculator(LEPR_Amp, '_Amp', 'Amphibole')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
# Blundy and Holland, 1990 plot A_site v. Al_IV and Al_VI v. Al_IV
ax[0].scatter(LEPR_Amp_Pre['Al_T_Amp'], LEPR_Amp_Pre['Na_K_A_Amp'], s = 5, color = 'r')
ax[0].scatter(LEPR_Amp_nosuf['Al_T_Amp'], LEPR_Amp_nosuf['Na_K_A_Amp'], s = 5, color = 'g')
ax[0].set_xlabel('Al_T_Amp (Tetrahedral)')
ax[0].set_ylabel('Na+K_A_Amp (A-site)')
ax[1].scatter(LEPR_Amp_Pre['Amp_Cation_Sum'], LEPR_Amp_Pre['Ca_B_Amp'], s = 5, color = 'r')
ax[1].scatter(LEPR_Amp_nosuf['Amp_Cation_Sum'], LEPR_Amp_nosuf['Ca_B_Amp'], s = 5, color = 'g')
ax[1].set_xlabel('Amp_Cation_Sum')
ax[1].set_ylabel('Ca_B_Amp (B-site)')
plt.tight_layout()


# %% 

LEPR_Cpx_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Cpx")['Cpxs']
LEPR_Cpx_cs = pt.calculate_clinopyroxene_components(cpx_comps=LEPR_Cpx_PreFilt)
LEPR_Cpx_cs = LEPR_Cpx_cs.rename(columns = {'Cation_Sum_Cpx': 'Cpx_Cation_Sum', 'Ca_CaMgFe': 'Ca_CaMgFe_Cpx'}, )
LEPR_Cpx_Pre = LEPR_Cpx_cs.copy()
LEPR_Cpx = LEPR_Cpx_PreFilt.loc[((LEPR_Cpx_cs.Cpx_Cation_Sum.between(3.95, 4.05)) & (LEPR_Cpx_cs.Ca_CaMgFe_Cpx.between(0.2, 0.5)) )]
LEPR_Cpx_nosuf = SiteCalculator(LEPR_Cpx, '_Cpx', 'Clinopyroxene')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Cpx_cs['SiO2_Cpx'], LEPR_Cpx_cs['CaO_Cpx'], s = 5, color = 'r')
ax[0].scatter(LEPR_Cpx_nosuf['SiO2_Cpx'], LEPR_Cpx_nosuf['CaO_Cpx'], s = 5, color = 'g')
ax[0].set_xlabel('SiO2')
ax[0].set_ylabel('CaO')
ax[1].scatter(LEPR_Cpx_cs['Cpx_Cation_Sum'], LEPR_Cpx_cs['Ca_CaMgFe_Cpx'], s = 5, color = 'r')
ax[1].scatter(LEPR_Cpx_nosuf['Cpx_Cation_Sum'], LEPR_Cpx_nosuf['Ca_CaMgFe_Cpx'], s = 5, color = 'g')
ax[1].set_xlabel('Cpx_Cation_Sum')
ax[1].set_ylabel('Ca_CaMgFe')
plt.tight_layout()


# %%

LEPR_Pig = LEPR_Cpx_PreFilt.loc[((LEPR_Cpx_cs.Cpx_Cation_Sum.between(3.95, 4.05)) & (LEPR_Cpx_cs.Ca_CaMgFe_Cpx.between(0.05, 2)) )]
LEPR_Pig_nosuf = SiteCalculator(LEPR_Pig, '_Cpx', 'Pigeonite')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Cpx_cs['SiO2_Cpx'], LEPR_Cpx_cs['CaO_Cpx'], s = 5, color = 'r')
ax[0].scatter(LEPR_Pig['SiO2_Cpx'], LEPR_Pig['CaO_Cpx'], s = 5, color = 'g')
ax[0].set_xlabel('SiO2')
ax[0].set_ylabel('CaO')
ax[1].scatter(LEPR_Cpx_cs['Cpx_Cation_Sum'], LEPR_Cpx_cs['Ca_CaMgFe_Cpx'], s = 5, color = 'r')
ax[1].scatter(LEPR_Pig_nosuf['Cpx_Cation_Sum'], LEPR_Pig_nosuf['Ca_CaMgFe_Cpx'], s = 5, color = 'g')
ax[1].set_xlabel('Cpx_Cation_Sum')
ax[1].set_ylabel('Ca_CaMgFe')
plt.tight_layout()

# %%

LEPR_Opx_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Opx")['Opxs']
LEPR_Opx_cs = pt.calculate_orthopyroxene_components(opx_comps=LEPR_Opx_PreFilt)
LEPR_Opx_cs = LEPR_Opx_cs.rename(columns = {'Cation_Sum_Opx': 'Opx_Cation_Sum', 'Ca_CaMgFe': 'Ca_CaMgFe_Opx'})
LEPR_Opx_Pre = LEPR_Opx_cs
LEPR_Opx = LEPR_Opx_PreFilt.loc[( (LEPR_Opx_cs.Opx_Cation_Sum.between(3.95, 4.05)) & (LEPR_Opx_cs.Ca_CaMgFe_Opx.between(-0.01, 0.05) )) ]
LEPR_Opx_nosuf = SiteCalculator(LEPR_Opx, '_Opx', 'Orthopyroxene')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Opx_cs['SiO2_Opx'], LEPR_Opx_cs['MgO_Opx'], s = 5, color = 'r')
ax[0].scatter(LEPR_Opx_nosuf['SiO2_Opx'], LEPR_Opx_nosuf['MgO_Opx'], s = 5, color = 'g')
ax[0].set_xlabel('SiO2')
ax[0].set_ylabel('CaO')
ax[1].scatter(LEPR_Opx_cs['Opx_Cation_Sum'], LEPR_Opx_cs['Ca_CaMgFe_Opx'], s = 5, color = 'r')
ax[1].scatter(LEPR_Opx_nosuf['Opx_Cation_Sum'], LEPR_Opx_nosuf['Ca_CaMgFe_Opx'], s = 5, color = 'g')
ax[1].set_xlabel('Cpx_Cation_Sum')
ax[1].set_ylabel('Ca_CaMgFe_Cpx')
plt.tight_layout()

# %%

LEPR_Plag_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Plag")['Plags']
LEPR_Plag_cs = calculate_plagioclase_components(plag_comps=LEPR_Plag_PreFilt, append = '_Plag')
LEPR_Plag_Pre = LEPR_Plag_cs
LEPR_Plag = LEPR_Plag_PreFilt.loc[( (LEPR_Plag_cs.Plag_Cation_Sum.between(4.95, 5.05)) & (LEPR_Plag_cs.Na_Ca_M_Plag.between(0.9, 1.05) )) ]
LEPR_Plag_nosuf = SiteCalculator(LEPR_Plag, '_Plag', 'Plagioclase')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Plag_cs['SiO2_Plag'], LEPR_Plag_cs['CaO_Plag'], s = 5, color = 'r')
ax[0].scatter(LEPR_Plag_nosuf['SiO2_Plag'], LEPR_Plag_nosuf['CaO_Plag'], s = 5, color = 'g')
ax[0].set_xlabel('SiO2')
ax[0].set_ylabel('CaO')
ax[1].scatter(LEPR_Plag_cs['Plag_Cation_Sum'], LEPR_Plag_cs['Na_Ca_M_Plag'], s = 5, color = 'r')
ax[1].scatter(LEPR_Plag_nosuf['Plag_Cation_Sum'], LEPR_Plag_nosuf['Na_Ca_M_Plag'], s = 5, color = 'g')
ax[1].set_xlabel('Plag_Cation_Sum')
ax[1].set_ylabel('Na+Ca')
plt.tight_layout()

# %%

LEPR_Ol_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Ol")['Ols']
LEPR_Ol_cs = calculate_olivine_components(ol_comps=LEPR_Ol_PreFilt, append = '_Ol')
LEPR_Ol_Pre = LEPR_Ol_cs
LEPR_Ol = LEPR_Ol_PreFilt.loc[( (LEPR_Ol_cs.Ol_Cation_Sum.between(2.95, 3.05)) & (LEPR_Ol_cs.Mg_Fe_M_Ol.between(1.95, 2.05) )) ]
LEPR_Ol_nosuf = SiteCalculator(LEPR_Ol, '_Ol', 'Olivine')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Ol_cs['SiO2_Ol'], LEPR_Ol_cs['MgO_Ol'], s = 5, color = 'r')
ax[0].scatter(LEPR_Ol_nosuf['SiO2_Ol'], LEPR_Ol_nosuf['MgO_Ol'], s = 5, color = 'g')
ax[0].set_xlabel('SiO2')
ax[0].set_ylabel('MgO')
ax[1].scatter(LEPR_Ol_cs['Ol_Cation_Sum'], LEPR_Ol_cs['Mg_Fe_M_Ol'], s = 5, color = 'r')
ax[1].scatter(LEPR_Ol_nosuf['Ol_Cation_Sum'], LEPR_Ol_nosuf['Mg_Fe_M_Ol'], s = 5, color = 'g')
ax[1].set_xlabel('Ol_Cation_Sum')
ax[1].set_ylabel('Mg_Fe_M_Ol')
plt.tight_layout()

# %%

LEPR_Sp_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Cr_Sp")['Sps']
LEPR_Sp_cs = calculate_spinel_components(sp_comps=LEPR_Sp_PreFilt, append = '_Sp')
LEPR_Sp_Pre = LEPR_Sp_cs
LEPR_Sp = LEPR_Sp_PreFilt.copy()
LEPR_Sp_nosuf = SiteCalculator(LEPR_Sp, '_Sp', 'Spinel')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Sp_Pre['MgO_Sp'], LEPR_Sp_Pre['FeOt_Sp'], s = 5, color = 'r')
ax[0].scatter(LEPR_Sp_nosuf['MgO_Sp'], LEPR_Sp_nosuf['FeOt_Sp'], s = 5, color = 'g')
ax[0].set_xlabel('Al2O3')
ax[0].set_ylabel('MgO')

ax[1].scatter(LEPR_Sp_Pre['Al_B_Sp'], LEPR_Sp_cs['Mg_Fe_M_Sp'], s = 5, color = 'r')
ax[1].scatter(LEPR_Sp_nosuf['Al_B_Sp'], LEPR_Sp_nosuf['Mg_Fe_M_Sp'], s = 5, color = 'g')
ax[1].set_xlabel('Al_B')
ax[1].set_ylabel('Mg_Fe_M')
plt.tight_layout()

# %%

LEPR_Ox_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Oxide")['Sps']
LEPR_Ox_cs = calculate_oxide_components(ox_comps=LEPR_Ox_PreFilt, append = '_Sp')
LEPR_Ox_Pre = LEPR_Ox_cs
LEPR_Ox = LEPR_Ox_PreFilt.loc[((LEPR_Ox_cs.Ox_Cation_Sum.between(1.95, 3.05)) & (LEPR_Ox_cs.Fe_Ti_Ox.between(1.0, 3.0)) )]

LEPR_Ox_nosuf = SiteCalculator(LEPR_Ox, '_Sp', 'Oxide')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Ox_Pre['FeOt_Sp'], LEPR_Ox_Pre['TiO2_Sp'], s = 5, color = 'r')
ax[0].scatter(LEPR_Ox_nosuf['FeOt_Sp'], LEPR_Ox_nosuf['TiO2_Sp'], s = 5, color = 'g')
ax[0].set_xlabel('FeOT')
ax[0].set_ylabel('TiO2')

ax[1].scatter(LEPR_Ox_Pre['Ox_Cation_Sum'], LEPR_Ox_cs['Fe_Ti_Ox'], s = 5, color = 'r')
ax[1].scatter(LEPR_Ox_nosuf['Ox_Cation_Sum'], LEPR_Ox_nosuf['Fe_Ti_Ox'], s = 5, color = 'g')
ax[1].set_xlabel('Ox_Cation_Sum')
ax[1].set_ylabel('Fe_Ti_Ox')
plt.tight_layout()

# %% Apatite: Ca5(PO4)3.(F,OH,Cl)1 -- Handling OH?

LEPR_Ap_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Apatite")['my_oxides']
LEPR_Ap_PreFilt=LEPR_Ap_PreFilt.add_suffix("_Ap")
LEPR_Ap_cs = calculate_apatite_components(ap_comps=LEPR_Ap_PreFilt, append = '_Ap')
LEPR_Ap_Pre = LEPR_Ap_cs
# LEPR_Ap = LEPR_Ap_PreFilt.loc[((LEPR_Ap_cs.Ap_Cation_Sum.between(1.95, 3.05)) & (LEPR_Ap_cs.Fe_Ti_Ap.between(1.0, 3.0)) )]

LEPR_Ap_nosuf = SiteCalculator(LEPR_Ap_PreFilt, '_Ap', 'Apatite')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Ap_Pre['CaO_Ap'], LEPR_Ap_Pre['P2O5_Ap'], s = 5, color = 'r')
ax[0].scatter(LEPR_Ap_nosuf['CaO_Ap'], LEPR_Ap_nosuf['P2O5_Ap'], s = 5, color = 'g')
ax[0].set_xlabel('CaO')
ax[0].set_ylabel('P2O5')
ax[1].scatter(LEPR_Ap_Pre['Ap_Cation_Sum'], LEPR_Ap_cs['Ca_P_Ap'], s = 5, color = 'r')
ax[1].scatter(LEPR_Ap_nosuf['Ap_Cation_Sum'], LEPR_Ap_nosuf['Ca_P_Ap'], s = 5, color = 'g')
ax[1].set_xlabel('Ap_Cation_Sum')
ax[1].set_ylabel('Ca_P_Ap')
plt.tight_layout()

# %% Biotite: K(Mg,Fe)3AlSi3O10(F,OH)2. -- Handling OH?

LEPR_Bt_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Biotite")['my_oxides']
LEPR_Bt_PreFilt=LEPR_Bt_PreFilt.add_suffix("_Bt")
LEPR_Bt_cs = calculate_biotite_components(bt_comps=LEPR_Bt_PreFilt, append = '_Bt')
LEPR_Bt_Pre = LEPR_Bt_cs
LEPR_Bt = LEPR_Bt_PreFilt.loc[((LEPR_Bt_cs.Bt_Cation_Sum.between(6.8, 7.2)))]

LEPR_Bt_nosuf = SiteCalculator(LEPR_Bt, '_Bt', 'Biotite')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Bt_Pre['Bt_Cation_Sum'], LEPR_Bt_Pre['Mg_Fe_Bt'], s = 5, color = 'r')
ax[0].scatter(LEPR_Bt_nosuf['Bt_Cation_Sum'], LEPR_Bt_nosuf['Mg_Fe_Bt'], s = 5, color = 'g')
ax[0].set_xlabel('Bt_Cation_Sum')
ax[0].set_ylabel('Mg_Fe_Bt')
ax[1].scatter(LEPR_Bt_Pre['Bt_Cation_Sum'], LEPR_Bt_Pre['Si_Al_Bt'], s = 5, color = 'r')
ax[1].scatter(LEPR_Bt_nosuf['Bt_Cation_Sum'], LEPR_Bt_nosuf['Si_Al_Bt'], s = 5, color = 'g')
ax[1].set_xlabel('Bt_Cation_Sum')
ax[1].set_ylabel('Si_Al_Bt')
plt.tight_layout()


# %% Quartz SiO2 

LEPR_Qz_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Quartz")['my_oxides']
LEPR_Qz_PreFilt=LEPR_Qz_PreFilt.add_suffix("_Qz")
LEPR_Qz_cs = calculate_quartz_components(qz_comps=LEPR_Qz_PreFilt, append = '_Qz')
LEPR_Qz_Pre = LEPR_Qz_cs
LEPR_Qz = LEPR_Qz_PreFilt.loc[((LEPR_Qz_cs.Qz_Cation_Sum.between(0.95, 1.1)))]

LEPR_Qz_nosuf = SiteCalculator(LEPR_Qz, '_Qz', 'Quartz')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].hist(LEPR_Qz_Pre['Qz_Cation_Sum'], bins = 10)
ax[0].set_xlabel('Qz_Cation_Sum')
ax[1].hist(LEPR_Qz_Pre['Si_Qz_cat_frac'], bins = 10)
ax[1].set_xlabel('Si_Qz_cat_frac')
plt.tight_layout()

# %% Garnet R3R2(SiO4)3

LEPR_Gt_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Garnet")['my_oxides']
LEPR_Gt_PreFilt = LEPR_Gt_PreFilt.add_suffix("_Gt")
LEPR_Gt_cs = calculate_garnet_components(gt_comps=LEPR_Gt_PreFilt, append = '_Gt')
LEPR_Gt_Pre = pd.concat([LEPR_Gt_PreFilt, LEPR_Gt_cs], axis = 1)
LEPR_Gt = LEPR_Gt_PreFilt.loc[((LEPR_Gt_cs.Gt_Cation_Sum.between(7.9, 8.1)))]

LEPR_Gt_nosuf = SiteCalculator(LEPR_Gt, '_Gt', 'Garnet')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Gt_Pre['Gt_Cation_Sum'], LEPR_Gt_Pre['Mg_MgFeCa_Gt'], s = 5, color = 'r')
ax[0].scatter(LEPR_Gt_nosuf['Gt_Cation_Sum'], LEPR_Gt_nosuf['Mg_MgFeCa_Gt'], s = 5, color = 'g')
ax[0].set_xlabel('Gt_Cation_Sum')
ax[0].set_ylabel('Mg_MgFeCa_Gt')
ax[1].scatter(LEPR_Gt_Pre['Gt_Cation_Sum'], LEPR_Gt_Pre['Fe_MgFeCa_Gt'], s = 5, color = 'r')
ax[1].scatter(LEPR_Gt_nosuf['Gt_Cation_Sum'], LEPR_Gt_nosuf['Fe_MgFeCa_Gt'], s = 5, color = 'g')
ax[1].set_xlabel('Gt_Cation_Sum')
ax[1].set_ylabel('Si_Al_Bt')
plt.tight_layout()

# %%

LEPR_Kspar_PreFilt = pt.import_excel('MachineLearning_MinClass.xlsx', sheet_name="Kspar")['Kspars']
LEPR_Kspar_cs = calculate_kspar_components(kspar_comps=LEPR_Kspar_PreFilt, append = '_Kspar')
LEPR_Kspar_Pre = pd.concat([LEPR_Kspar_PreFilt, LEPR_Kspar_cs], axis = 1)
LEPR_Kspar = LEPR_Kspar_PreFilt.loc[((LEPR_Kspar_cs.Kspar_Cation_Sum.between(4.95, 5.05)) )]
LEPR_Kspar_nosuf = SiteCalculator(LEPR_Kspar, '_Kspar', 'KSpar')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax = ax.flatten()
ax[0].scatter(LEPR_Kspar_Pre['SiO2_Kspar'], LEPR_Kspar_Pre['K2O_Kspar'], s = 5, color = 'r')
ax[0].scatter(LEPR_Kspar_nosuf['SiO2_Kspar'], LEPR_Kspar_nosuf['K2O_Kspar'], s = 5, color = 'g')
ax[0].set_xlabel('SiO2')
ax[0].set_ylabel('K2O')
ax[1].scatter(LEPR_Kspar_Pre['Kspar_Cation_Sum'], LEPR_Kspar_Pre['Na_Ca_M_Kspar'], s = 5, color = 'r')
ax[1].scatter(LEPR_Kspar_nosuf['Kspar_Cation_Sum'], LEPR_Kspar_nosuf['Na_Ca_M_Kspar'], s = 5, color = 'g')
ax[1].set_xlabel('Kspar_Cation_Sum')
ax[1].set_ylabel('Na_Ca_M_Kspar')
plt.tight_layout()

# %% 

# LEPR_Amp_nosuf.loc[:, LEPR_Amp_nosuf.columns!='Amp_Cation_Sum']

def rename_columns(append): 
    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
    oxides_min = [oxide + append for oxide in oxides]
    
    dictionary = dict(zip(oxides_min, oxides))

    return dictionary

LEPR_Amp_nosuf = SiteCalculator(LEPR_Amp, '_Amp', 'Amphibole')
LEPR_Amp_nosuf = LEPR_Amp_nosuf.rename(columns=rename_columns('_Amp'))
LEPR_Amp_nosuf = LEPR_Amp_nosuf.drop(columns=['F_Amp', 'Cl_Amp'], errors='ignore')

LEPR_Cpx_nosuf = SiteCalculator(LEPR_Cpx, '_Cpx', 'Clinopyroxene')
LEPR_Cpx_nosuf = LEPR_Cpx_nosuf.rename(columns=rename_columns('_Cpx'))
LEPR_Cpx_nosuf = LEPR_Cpx_nosuf.drop(columns=['Sample_ID_Cpx'], errors='ignore')

LEPR_Opx_nosuf = SiteCalculator(LEPR_Opx, '_Opx', 'Orthopyroxene')
LEPR_Opx_nosuf = LEPR_Opx_nosuf.rename(columns=rename_columns('_Opx'))
LEPR_Opx_nosuf = LEPR_Opx_nosuf.drop(columns=['Sample_ID_Opx'], errors='ignore')

LEPR_Plag_nosuf = SiteCalculator(LEPR_Plag, '_Plag', 'Plagioclase')
LEPR_Plag_nosuf = LEPR_Plag_nosuf.rename(columns=rename_columns('_Plag'))
LEPR_Plag_nosuf = LEPR_Plag_nosuf.drop(columns=['Sample_ID_Plag'], errors='ignore')

LEPR_Ol_nosuf = SiteCalculator(LEPR_Ol, '_Ol', 'Olivine')
LEPR_Ol_nosuf = LEPR_Ol_nosuf.rename(columns=rename_columns('_Ol'))
LEPR_Ol_nosuf = LEPR_Ol_nosuf.drop(columns=['NiO_Ol'], errors='ignore')
LEPR_Ol_nosuf = LEPR_Ol_nosuf.drop(columns=['Sample_ID_Ol'], errors='ignore')

LEPR_Sp_nosuf = SiteCalculator(LEPR_Sp, '_Sp', 'Spinel')
LEPR_Sp_nosuf = LEPR_Sp_nosuf.rename(columns=rename_columns('_Sp'))
LEPR_Sp_nosuf = LEPR_Sp_nosuf.drop(columns=['NiO_Sp'], errors='ignore')
LEPR_Sp_nosuf = LEPR_Sp_nosuf.drop(columns=['Sample_ID_Sp'], errors='ignore')

LEPR_Ox_nosuf = SiteCalculator(LEPR_Ox, '_Sp', 'Oxide')
LEPR_Ox_nosuf = LEPR_Ox_nosuf.rename(columns=rename_columns('_Sp'))
LEPR_Ox_nosuf = LEPR_Ox_nosuf.drop(columns=['NiO_Sp', 'Sample_ID_Sp'], errors='ignore')

LEPR_Ap_nosuf = SiteCalculator(LEPR_Ap_PreFilt, '_Ap', 'Apatite')
LEPR_Ap_nosuf = LEPR_Ap_nosuf.rename(columns=rename_columns('_Ap'))
LEPR_Ap_nosuf = LEPR_Ap_nosuf.drop(columns=['P2O5_Ap'], errors='ignore')

LEPR_Bt_nosuf = SiteCalculator(LEPR_Bt, '_Bt', 'Biotite')
LEPR_Bt_nosuf = LEPR_Bt_nosuf.rename(columns=rename_columns('_Bt'))
LEPR_Bt_nosuf = LEPR_Bt_nosuf.drop(columns=['P2O5_Bt'], errors='ignore')

LEPR_Qz_nosuf = SiteCalculator(LEPR_Qz, '_Qz', 'Quartz')
LEPR_Qz_nosuf = LEPR_Qz_nosuf.rename(columns=rename_columns('_Qz'))
LEPR_Qz_nosuf = LEPR_Qz_nosuf.drop(columns=['P2O5_Qz'], errors='ignore')

LEPR_Gt_nosuf = SiteCalculator(LEPR_Gt, '_Gt', 'Garnet')
LEPR_Gt_nosuf = LEPR_Gt_nosuf.rename(columns=rename_columns('_Gt'))
LEPR_Gt_nosuf = LEPR_Gt_nosuf.drop(columns=['P2O5_Gt'], errors='ignore')

LEPR_Kspar_nosuf = SiteCalculator(LEPR_Kspar, '_Kspar', 'KSpar')
LEPR_Kspar_nosuf = LEPR_Kspar_nosuf.rename(columns=rename_columns('_Kspar'))
LEPR_Kspar_nosuf = LEPR_Kspar_nosuf.drop(columns=['Sample_ID_Kspar'], errors='ignore')

LEPR_AllPhases = pd.concat([LEPR_Amp_nosuf, LEPR_Cpx_nosuf, LEPR_Opx_nosuf, LEPR_Plag_nosuf, LEPR_Ol_nosuf, 
    LEPR_Sp_nosuf, LEPR_Ox_nosuf, LEPR_Ap_nosuf, LEPR_Bt_nosuf, LEPR_Qz_nosuf, LEPR_Gt_nosuf, LEPR_Kspar_nosuf], 
    axis = 0, ignore_index = True)

# %% 

LEPR_AllPhases.to_csv('./LEPR/LEPR_AllPhases.csv')
LEPR_AllPhases

# %%

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
pd.plotting.scatter_matrix(LEPR_AllPhases[oxides], figsize = (15, 15), hist_kwds={'bins':20})
plt.show()

# %%
# %% 
# %%
# %%