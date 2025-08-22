# %% 

""" Created on February 16, 2023 // @author: Sarah Shi """

import sys
import pickle
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append('src')
import mineralML as mm
from mineralML.constants import OXIDES

from matplotlib import rc, pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

# %%

mm.same_seeds(42)
oxides = OXIDES

# min_df = pd.read_csv('Training_Data/mindf_filt_new.csv', dtype={'Mineral': 'category'})
# min_df = pd.read_csv('Training_Data/mindf_filt_glass_clean.csv', dtype={'Mineral': 'category'})
# min_df = pd.read_csv('Training_Data/min_df_v2_clean.csv', dtype={'Mineral': 'category'}) #, index_col=['Sample Name'])
# min_df = pd.read_csv('Training_Data/min_df_v2_merge_natural.csv', dtype={'Mineral': 'category'})
min_df = pd.read_csv('Training_Data/min_df_v2_clean_synth.csv', dtype={'Mineral': 'category'})
min_df_lim = min_df

counts_df = (
    min_df['Mineral']
      .value_counts()
      .reset_index()
      .rename(columns={'index':'Mineral','Mineral':'Count'})
)
display(counts_df)

# %% 

df_bal = mm.balance(min_df)
min_cat = sorted(df_bal["Mineral"].unique().tolist())
# np.savez("mineral_classes_nn_v008.npz", classes=np.array(min_cat, dtype=object))


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(df_bal[OXIDES])
# mean = pd.Series(scaler.mean_, index=OXIDES)
# std = pd.Series(scaler.scale_, index=OXIDES)
# np.savez("scaler_nn_v008.npz", mean=mean, scale=std)

# %% 

# %% train the neural network 

lr = 5e-3 
wd = 1e-3 
dr = 0.1
n = 0.20

hls_list = [
    # [8],
    # [16],
    # [16,8],
    # [32,16],
    # [64,32],
    # [32,16,8],
    [64,32,16], # large
    # [128,64,32], # large
]
kl_weight_decay_list = [0.75] # [0.0, 0.25, 0.5, 0.75, 1.0]
epochs = 1000 
best_model_state = mm.neuralnetwork(min_df_lim, hls_list, kl_weight_decay_list, lr, wd, dr, epochs, n, balanced=True) 

# %%
# %% 

min_df = pd.read_csv('Training_Data/min_df_v2_clean_natural.csv', dtype={'Mineral': 'category'})
min_df = mm.prep_df_nn(min_df)
min_df_pred, min_probability_matrix = mm.predict_class_prob_nn(min_df)

min_bayes_valid_report = classification_report(
    min_df_pred['Mineral'], min_df_pred['Predict_Mineral'], zero_division=0
)
print("MIN_DF Validation Report:\n", min_bayes_valid_report)

min_cm = mm.confusion_matrix_df(min_df_pred['Mineral'], min_df_pred['Predict_Mineral'])
print("MIN_DF Confusion Matrix:\n", min_cm)

# min_cm[min_cm < len(min_df_pred['Predict_Mineral'])*0.0005] = 0
mm.pp_matrix(min_cm)

# %% 

reykjanes_df_load = mm.load_df('Validation_Data/Reykjanes_Dataset.csv')
reykjanes_df_load['Mineral'] = 'Glass'
reykjanes_df = mm.prep_df_nn(reykjanes_df_load)
reykjanes_df_pred, reykjanes_probability_matrix = mm.predict_class_prob_nn(reykjanes_df)
reykjanes_df_pred = reykjanes_df_pred[reykjanes_df_pred['Predict_Probability']>0.8]

reykjanes_bayes_valid_report = classification_report(
    reykjanes_df_pred['Mineral'], reykjanes_df_pred['Predict_Mineral'], zero_division=0
)
print("Reykjanes Validation Report:\n", reykjanes_bayes_valid_report)

reykjanes_cm = mm.confusion_matrix_df(reykjanes_df_pred['Mineral'], reykjanes_df_pred['Predict_Mineral'])
print("Reykjanes Confusion Matrix:\n", reykjanes_cm)

# min_cm[min_cm < len(min_df_pred['Predict_Mineral'])*0.0005] = 0
mm.pp_matrix(reykjanes_cm)

# reykjanes_df_pred.to_csv('reykjanes_predict.csv')
reykjanes_diff = reykjanes_df_pred[reykjanes_df_pred.Predict_Mineral!=reykjanes_df_pred.Mineral]
# reykjanes_diff.to_csv('reykjanes_diff.csv')

# %% 

mi_df_load = mm.load_df('Training_Data/MI_process.csv')
mi_df_load['Mineral'] = 'Glass'
mi_df = mm.prep_df_nn(mi_df_load)
mi_df_pred, mi_probability_matrix = mm.predict_class_prob_nn(mi_df)
mi_df_pred['Host'] = mi_df_load['Host'].values
mi_df_pred['Citation'] = mi_df_load.index.values
mi_df_pred = mi_df_pred[mi_df_pred['Predict_Probability']>0.8]

mi_bayes_valid_report = classification_report(
    mi_df_pred['Mineral'], mi_df_pred['Predict_Mineral'], zero_division=0
)
print("MI Validation Report:\n", mi_bayes_valid_report)

mi_cm = mm.confusion_matrix_df(mi_df_pred['Mineral'], mi_df_pred['Predict_Mineral'])
print("MI Confusion Matrix:\n", mi_cm)

# min_cm[min_cm < len(min_df_pred['Predict_Mineral'])*0.0005] = 0
mm.pp_matrix(mi_cm)

# mi_df_pred.to_csv('mi_predict.csv')
mi_diff = mi_df_pred[mi_df_pred.Predict_Mineral!=mi_df_pred.Mineral]
# mi_diff.to_csv('mi_diff.csv')

# %% 

MINERALS = ['Amphibole',
            'Apatite',
            'Biotite',
            'Calcite',
            'Chlorite',
            'Epidote',
            'Feldspar',
            'Garnet',
            'Glass',
            'Kalsilite',
            'Leucite',
            'Melilite',
            'Muscovite',
            'Nepheline',
            'Olivine',
            'Oxide', # check for 'oxide' as mineral name
            'Pyroxene',
            'Quartz',
            'Rhombohedral_Oxides',
            'Rutile',
            'Serpentine',
            'Spinel', # check for this in case named this
            'Spinels', # what mineralML encodes as
            'Titanite',
            'Tourmaline',
            'Zircon'
            ]

lepr_alt = pd.read_csv('Validation_Data/LEPR_TraceDs_download_LEPRalt.csv')
lepr_alt["Mineral"] = lepr_alt["Mineral"].astype(str)  # make sure string
lepr_alt_filt = lepr_alt[lepr_alt["Mineral"].str.strip().isin(MINERALS)]

lepr_pmelts = pd.read_csv('Validation_Data/LEPR_TraceDs_download_pMELTS.csv')
lepr_pmelts["Mineral"] = lepr_pmelts["Mineral"].astype(str)  # make sure string
lepr_pmelts_filt = lepr_pmelts[lepr_pmelts["Mineral"].str.strip().isin(MINERALS)]


lepr_alt_df = mm.prep_df_nn(lepr_alt_filt)
lepr_alt_pred, lepr_alt_probability_matrix = mm.predict_class_prob_nn(lepr_alt_df)
lepr_alt_cm = mm.confusion_matrix_df(lepr_alt_pred['Mineral'], lepr_alt_pred['Predict_Mineral'])
mm.pp_matrix(lepr_alt_cm)
plt.show()

lepr_pmelts_df = mm.prep_df_nn(lepr_pmelts_filt)
lepr_pmelts_pred, lepr_pmelts_probability_matrix = mm.predict_class_prob_nn(lepr_pmelts_df)
lepr_pmelts_cm = mm.confusion_matrix_df(lepr_pmelts_pred['Mineral'], lepr_pmelts_pred['Predict_Mineral'])
mm.pp_matrix(lepr_pmelts_cm)
plt.show()


# %% 

# annika_df_load = mm.load_df('Validation_Data/EDS_Annika.csv')
annika_df_load = pd.read_excel('Validation_Data/B2_Oxides.xlsx', sheet_name='Area4')
annika_df = mm.prep_df_nn(annika_df_load)
annika_df_pred, annika_probability_matrix = mm.predict_class_prob_nn(annika_df_load)
annika_df_pred.to_csv('B2_oxides_area4_predict.csv')
annika_df_pred

# %% 

phan_df_load = pd.read_csv('df_xrf_phan.csv')
phan_df_load = phan_df_load.replace(r'^\s*<\s*\d+(\.\d+)?\s*$', np.nan, regex=True)

elem_cols = ['Si','Ti','Al','Fe','Mn','Mg','Ca','Na','K','P','Cr','Ni','S','Zr']

elem_cols = [c for c in elem_cols if c in phan_df_load.columns]

elem_df = phan_df_load[elem_cols].apply(pd.to_numeric, errors='coerce')

# Convert ppm -> wt% (XRF exports are in ppm)
elem_wt = elem_df / 1e4

conv_out, factors = mm.element_to_oxide(elem_wt)

conv_out['Sample ID'] = phan_df_load['Sample ID']
display(conv_out)

# Prep and predict — use the prepped dataframe, not the raw load
phan_df = mm.prep_df_nn(conv_out)
phan_df_pred, phan_probability_matrix = mm.predict_class_prob_nn(phan_df)
display(phan_df_pred)
phan_df_pred.to_csv('df_xrf_phan_pred.csv', index=False)

# %%
# %%

petrelli_df_load = mm.load_df('Validation_Data/Petrelli_cpx.csv')

petrelli_px = mm.PyroxeneClassifier(petrelli_df_load)
petrelli_px_classified = petrelli_px.classify(subclass=True)
petrelli_px_plot, petrelli_px_tax = petrelli_px.plot() 

# %% 

petdb_df_load = mm.load_df('Validation_Data/PetDB_validationdata_Fe.csv')
fspar_ec = petdb_df_load[petdb_df_load['Mineral'] == 'Plagioclase']

classify_fspar = mm.FeldsparClassifier(fspar_ec)
fspar_ec_classified = classify_fspar.classify(subclass=True)
fspar_ec_plot, fspar_ec_tax = classify_fspar.plot()

# %% 

hem_ec = min_df[min_df['Mineral'] == 'Hematite']
il_ec = min_df[min_df['Mineral'] == 'Ilmenite']
sp_ec = min_df[min_df['Mineral'] == 'Spinel']
mt_ec = min_df[min_df['Mineral'] == 'Magnetite']

ox_ec = pd.concat([hem_ec, il_ec, sp_ec, mt_ec], axis=0)
ox_class = mm.OxideClassifier(ox_ec)
ox_comp = ox_class.classify()
ox_class.plot()

# %% 

hem_class = mm.OxideClassifier(hem_ec)
hem_comp = hem_class.classify()
hem_class.plot()

mt_class = mm.OxideClassifier(mt_ec)
mt_comp = mt_class.classify()
mt_class.plot(c='k')

il_class = mm.OxideClassifier(il_ec)
il_comp = il_class.classify()
il_class.plot()

sp_class = mm.OxideClassifier(sp_ec)
sp_comp = sp_class.classify()
sp_class.plot()

# %% 

amph_df = min_df[min_df['Mineral'] == 'Amphibole']
amph_class = mm.AmphiboleClassifier(amph_df)
amph_comp = amph_class.classify()
amph_class.plot()

# %% 


# file_path = "Validation_Data/LEPR_TraceDs_download_LEPRalt.xlsx"

# # Load the Excel workbook
# xls = pd.ExcelFile(file_path)

# # Collect all sheets except "Experiment"
# dfs = []
# for sheet in xls.sheet_names:
#     if sheet != "Experiment":  # exclude the sheet
#         df = pd.read_excel(xls, sheet_name=sheet)
#         df["Mineral"] = sheet  # optional: keep track of origin
#         dfs.append(df)

# # Concatenate into one DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)

# # Save as CSV
# combined_df.to_csv("LEPR_TraceDs_download_LEPRalt.csv", index=False)


# %% 
# %% 

anthony_dec_hem = pd.read_csv('hematiteedsdata/dec_hematite.csv', index_col=0)

# dec_hem = mm.element_to_oxide(anthony_dec_hem)
# dec_hem

# %%
# %%
# %% 

petrelli_df_load = mm.load_df('Validation_Data/Petrelli_cpx.csv')
petrelli_df = mm.prep_df_nn(petrelli_df_load)
petrelli_df_pred, petrelli_probability_matrix = mm.predict_class_prob_nn(petrelli_df)

petrelli_bayes_valid_report = classification_report(
    petrelli_df_pred['Mineral'], petrelli_df_pred['Predict_Mineral'], zero_division=0
)
print("Petrelli Validation Report:\n", petrelli_bayes_valid_report)

petrelli_cm = mm.confusion_matrix_df(petrelli_df_pred['Mineral'], petrelli_df_pred['Predict_Mineral'])
print("Petrelli Confusion Matrix:\n", petrelli_cm)

# petrelli_cm[petrelli_cm < len(petrelli_df_pred['Predict_Mineral'])*0.05] = 0
mm.pp_matrix(petrelli_cm) #, show_null_values=1)

# petrelli_df_pred.to_csv('petrelli_pred.csv')

# %%

lepr_df_load = mm.load_df('Validation_Data/lepr_allphases_lim_sp.csv')
lepr_df = mm.prep_df_nn(lepr_df_load)
lepr_df_pred, lepr_probability_matrix = mm.predict_class_prob_nn(lepr_df)

lepr_bayes_valid_report = classification_report(
    lepr_df_pred['Mineral'], lepr_df_pred['Predict_Mineral'], zero_division=0
)
print("LEPR Validation Report:\n", lepr_bayes_valid_report)

lepr_cm = mm.confusion_matrix_df(lepr_df_pred['Mineral'], lepr_df_pred['Predict_Mineral'])
print("LEPR Confusion Matrix:\n", lepr_cm)

lepr_cm[lepr_cm < len(lepr_df_pred['Predict_Mineral'])*0.0005] = 0
mm.pp_matrix(lepr_cm) 

# lepr_df_pred.to_csv('lepr_pred.csv')

# %% 

petdb_df_load = mm.load_df('Validation_Data/PetDB_validationdata_Fe.csv')
petdb_df = mm.prep_df_nn(petdb_df_load)
petdb_df_pred, petdb_probability_matrix = mm.predict_class_prob_nn(petdb_df)

petdb_bayes_valid_report = classification_report(
    petdb_df_pred['Mineral'], petdb_df_pred['Predict_Mineral'], zero_division=0
)
print("PetDB Validation Report:\n", petdb_bayes_valid_report)

petdb_cm = mm.confusion_matrix_df(petdb_df_pred['Mineral'], petdb_df_pred['Predict_Mineral'])
print("PetDB Confusion Matrix:\n", petdb_cm)

petdb_cm[petdb_cm < len(petdb_df_pred['Predict_Mineral'])*0.0005] = 0
mm.pp_matrix(petdb_cm)

# petdb_df_pred.to_csv('petdb_pred.csv')

# %% 

small_subset = georoc_df.iloc[:1000]

for n_iter in [5, 10, 20, 50, 100, 500]:
    start_time = time.time()
    pred_df, _ = mm.predict_class_prob_nn(small_subset, n_iterations=n_iter)
    accuracy = (pred_df['Mineral'] == pred_df['Predict_Mineral']).mean()
    print(f"Iterations: {n_iter:3d} | Accuracy: {accuracy:.4f} | Time: {time.time() - start_time:.2f}s")

# %%

georoc_df_load = mm.load_df('Validation_Data/GEOROC_validationdata_Fe_2025.csv')
georoc_df = mm.prep_df_nn(georoc_df_load)

# n = len(georoc_df)
# half = n // 2
# georoc_df_half = georoc_df.iloc[:half].reset_index(drop=True)

import time 
start_time = time.time()
georoc_df_pred, georoc_probability_matrix = mm.predict_class_prob_nn(georoc_df)
print("--- %s seconds ---" % (time.time() - start_time))

georoc_bayes_valid_report = classification_report(
    georoc_df_pred['Mineral'], georoc_df_pred['Predict_Mineral'], zero_division=0
)
print("GEOROC Validation Report:\n", georoc_bayes_valid_report)

georoc_cm = mm.confusion_matrix_df(georoc_df_pred['Mineral'], georoc_df_pred['Predict_Mineral'])

mm.pp_matrix(georoc_cm) #, savefig=None) 

# georoc_df_pred.to_csv('georoc_pred.csv')


# %% 


comb_cm = georoc_cm+petdb_cm+lepr_cm

comb_cm[comb_cm < comb_cm['sum_row'].max()*0.0005] = 0


mm.pp_matrix(comb_cm) #, savefig=None) 
# plt.savefig('comb_cm.pdf')

# %% 

georoc_cpx_plag = georoc_df_pred[(georoc_df_pred.Mineral=='Clinopyroxene') & (georoc_df_pred.Predict_Mineral=='Plagioclase')]

import Thermobar as pt 

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']

oxides_plag = [oxide + '_Plag' for oxide in oxides]
georoc_cpx_plag_append = georoc_cpx_plag[oxides].copy()
georoc_cpx_plag_append.columns = oxides_plag

plag_tern_points = pt.tern_points_fspar(fspar_comps=georoc_cpx_plag_append)

fig, tax = pt.plot_fspar_classification(labels=True)
tax.scatter(
    plag_tern_points,
    edgecolor="k",
    marker="^",
    facecolor="red",
    label='Plag',
   s=90
)
fig.tight_layout()

# %% 

cascades_df_load = mm.load_df('Validation_Data/Cascades_CpxAmp_NN.csv')
cascades_df = mm.prep_df_nn(cascades_df_load)

cascades_df_pred, cascades_probability_matrix = mm.predict_class_prob_nn(cascades_df)

cascades_bayes_valid_report = classification_report(
    cascades_df_pred['Mineral'], cascades_df_pred['Predict_Mineral'], zero_division=0
)
print("Cascades Validation Report:\n", cascades_bayes_valid_report)


cascades_cm = mm.confusion_matrix_df(cascades_df_pred['Mineral'], cascades_df_pred['Predict_Mineral'])
print("Cascades Confusion Matrix:\n", cascades_cm)



cascades_df_pred.to_csv('Validation_Data/Cascades_CpxAmp_NN.csv')


# %% 

opx = min_df_lim[min_df_lim.Mineral=='Orthopyroxene']
cpx = min_df_lim[min_df_lim.Mineral=='Clinopyroxene']
constants = ['Sample Name', 'Mineral']

opx_components = mm.calculate_clinopyroxene_components(opx.rename(columns={c: c+'_Cpx' for c in opx.columns if c not in constants}))
cpx_components = mm.calculate_clinopyroxene_components(cpx.rename(columns={c: c+'_Cpx' for c in cpx.columns if c not in constants}))

opx.loc[(opx_components['Ca_CaMgFe_Cpx']<0.05), 'Empirical_Mineral'] = 'Orthopyroxene'
opx.loc[(opx_components['Ca_CaMgFe_Cpx'].between(0.05, 0.2)), 'Empirical_Mineral'] = 'Pigeonite'
opx.loc[(opx_components['Ca_CaMgFe_Cpx']>0.2), 'Empirical_Mineral'] = 'Clinopyroxene'

cpx.loc[(cpx_components['Ca_CaMgFe_Cpx']<0.05), 'Empirical_Mineral'] = 'Orthopyroxene'
cpx.loc[(cpx_components['Ca_CaMgFe_Cpx'].between(0.05, 0.2)), 'Empirical_Mineral'] = 'Pigeonite'
cpx.loc[(cpx_components['Ca_CaMgFe_Cpx']>0.2), 'Empirical_Mineral'] = 'Clinopyroxene'

opx_bad = opx[opx.Mineral != opx.Empirical_Mineral]
cpx_bad = cpx[cpx.Mineral != cpx.Empirical_Mineral]


# %% 


tlepr = lepr_df[lepr_df['Mineral'] == lepr_df['NN_Labels']]
flepr = lepr_df[lepr_df['Mineral'] != lepr_df['NN_Labels']]

import Thermobar as pt 

cpx_corr = tlepr[tlepr.Mineral=='Clinopyroxene']
cpx_incorr = tlepr[tlepr.Mineral=='Clinopyroxene']

opx_corr = tlepr[tlepr.Mineral=='Orthopyroxene']
opx_incorr = tlepr[tlepr.Mineral=='Orthopyroxene']

cpx_tern_corr = pt.tern_points_px(px_comps=cpx_corr.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
cpx_tern_incorr = pt.tern_points_px(px_comps=cpx_incorr.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
opx_tern_corr = pt.tern_points_px(opx_corr.rename(columns={'MgO':'MgO_Opx', 'FeOt':'FeOt_Opx', 'CaO':'CaO_Opx'}))
opx_tern_incorr = pt.tern_points_px(opx_incorr.rename(columns={'MgO':'MgO_Opx', 'FeOt':'FeOt_Opx', 'CaO':'CaO_Opx'}))

opx_comps_corr = pt.calculate_orthopyroxene_components(opx_corr.rename(columns={'MgO':'MgO_Opx', 'FeOt':'FeOt_Opx', 'CaO':'CaO_Opx'}))
opx_comps_incorr = pt.calculate_orthopyroxene_components(opx_incorr.rename(columns={'MgO':'MgO_Opx', 'FeOt':'FeOt_Opx', 'CaO':'CaO_Opx'}))

cpxpred_amplabel = lepr_df[(lepr_df.NN_Labels=='Clinopyroxene') & (lepr_df.Mineral=='Amphibole')]
amppred_cpxlabel = lepr_df[(lepr_df.NN_Labels=='Amphibole') & (lepr_df.Mineral=='Clinopyroxene')]

amp_tern_corr = pt.tern_points_px(px_comps=amppred_cpxlabel.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))

px_points_corr = pt.tern_points(opx_comps_corr["Fs_Simple_MgFeCa_Opx"],  opx_comps_corr["Wo_Simple_MgFeCa_Opx"],  opx_comps_corr["En_Simple_MgFeCa_Opx"])
px_points_incorr = pt.tern_points(opx_comps_incorr["Fs_Simple_MgFeCa_Opx"],  opx_comps_incorr["Wo_Simple_MgFeCa_Opx"],  opx_comps_incorr["En_Simple_MgFeCa_Opx"])

fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
tax.scatter(cpx_tern_corr, edgecolor="k", marker="^", facecolor="tab:blue", label='NN Predicted == LEPR Labeled Cpx', s=75, alpha = 0.25, rasterized=True)
tax.scatter(opx_tern_corr, edgecolor="k", marker="^", facecolor="tab:red", label='NN Predicted == LEPR Labeled Opx', s=75, alpha = 0.25, rasterized=True)

nn_cpx_lepr = lepr_df[lepr_df['NN_Labels']=='Clinopyroxene']
nn_opx_lepr = lepr_df[lepr_df['NN_Labels']=='Orthopyroxene']
nn_cpx_lepr_tern = pt.tern_points_px(px_comps=nn_cpx_lepr.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))
nn_opx_lepr_tern = pt.tern_points_px(px_comps=nn_opx_lepr.rename(columns={'MgO':'MgO_Cpx', 'FeOt':'FeOt_Cpx', 'CaO':'CaO_Cpx'}))

fig, tax = pt.plot_px_classification(figsize=(10, 5), labels=True, fontsize_component_labels=16, fontsize_axes_labels=20)
tax.scatter(nn_cpx_lepr_tern, edgecolor="k", marker="^", facecolor="tab:blue", label='NN Predicted Cpx', s=75, alpha = 0.25, rasterized=True)
tax.scatter(nn_opx_lepr_tern, edgecolor="k", marker="^", facecolor="tab:red", label='NN Predicted Opx', s=75, alpha = 0.25, rasterized=True)
plt.legend()

# %%

with open('src/mineralML/scaler.pkl','rb') as f:
    scaler = pickle.load(f)

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']

georoc = pd.read_csv('Validation_Data/GEOROC_validationdata_Fe.csv', index_col=0)
georoc_df = georoc.dropna(subset=oxides, thresh=6)

# georoc_df = georoc_df[georoc_df.Mineral.isin(['Amphibole', 'Apatite', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide', 'Ilmenite', '(Al)Kalifeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene','Plagioclase', 'Quartz', 'Rutile', 'Spinel', 'Tourmaline', 'Zircon'])]
georoc_df = georoc_df[georoc_df.Mineral.isin(['Amphibole', 'Biotite', 'Clinopyroxene', 'Garnet', 'FeTiOxide', 'Ilmenite', '(Al)Kalifeldspar', 'Magnetite', 'Muscovite', 'Olivine', 'Orthopyroxene','Plagioclase', 'Spinel'])]
georoc_df['Mineral'] = georoc_df['Mineral'].replace('(Al)Kalifeldspar', 'KFeldspar')
georoc_df = georoc_df[~georoc_df['Mineral'].isin(['Tourmaline', 'Quartz', 'Rutile', 'Apatite', 'Zircon'])]

data_idx = np.arange(len(georoc_df))
train_idx, test_idx = train_test_split(data_idx, test_size=0.2, stratify=pd.Categorical(georoc_df['Mineral']).codes, random_state=42, shuffle=True)
georoc_df_lim = georoc_df.iloc[test_idx]

georoc_wt = georoc_df_lim[oxides].fillna(0)
georoc_wt = georoc_wt.to_numpy()
georoc_norm_wt = scaler.transform(georoc_wt)

min_df_lim['Mineral'] = min_df_lim['Mineral'].astype('category')
georoc_df_lim['Mineral'] = georoc_df_lim['Mineral'].astype(pd.CategoricalDtype(categories=min_df_lim['Mineral'].cat.categories))
new_validation_data_y_georoc = (georoc_df_lim['Mineral'].cat.codes).values

# Create a DataLoader for the new validation dataset
new_validation_dataset_georoc = LabelDataset(georoc_norm_wt, new_validation_data_y_georoc)
new_validation_loader_georoc = DataLoader(new_validation_dataset_georoc, batch_size=256, shuffle=False)

input_size = len(new_validation_dataset_georoc.__getitem__(0)[0])

path = 'parametermatrix_neuralnetwork/best_model.pt'

model = MultiClassClassifier(input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=hls).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

load_model(model, optimizer, path)

# Use the trained model to predict the classes for the new validation dataset
model.eval()
new_validation_pred_classes_georoc = []
with torch.no_grad():
    for data, labels in new_validation_loader_georoc:
        x = data.to(device)
        pred_classes = model.predict(x)
        new_validation_pred_classes_georoc.extend(pred_classes.tolist())


new_validation_pred_classes_georoc = np.array(new_validation_pred_classes_georoc)
unique_classes_georoc = np.unique(np.concatenate((new_validation_data_y_georoc[new_validation_data_y_georoc != -1], new_validation_pred_classes_georoc[new_validation_data_y_georoc != -1])))

sort_mapping = {key: value for key, value in sorted(mapping.items(), key=lambda item: item[0]) if key in unique_classes_georoc}

# Calculate classification metrics for the new validation dataset
new_validation_report = classification_report(new_validation_data_y_georoc[new_validation_data_y_georoc != -1], new_validation_pred_classes_georoc[new_validation_data_y_georoc!=-1], labels = unique_classes_georoc, target_names=[sort_mapping[x] for x in unique_classes_georoc], zero_division=0)
print("New validation report:\n", new_validation_report)

cm_valid = confusion_matrix(new_validation_data_y_georoc[new_validation_data_y_georoc!=-1], new_validation_pred_classes_georoc[new_validation_data_y_georoc!=-1])

df_valid_cm = pd.DataFrame(
    cm_valid,
    index=[sort_mapping[x] for x in unique_classes_georoc],
    columns=[sort_mapping[x] for x in unique_classes_georoc],
)

df_valid_cm[df_valid_cm < len(new_validation_pred_classes_georoc)*0.001] = 0

mm.pp_matrix(df_valid_cm, cmap = cmap, savefig='georoc_valid', figsize = (11.5, 11.5)) 
# mm.pp_matrix(df_valid_cm, cmap = cmap, figsize = (11.5, 11.5)) 

# # Convert the predicted integer labels to string labels using the sort_mapping dictionary
new_validation_pred_labels_georoc = np.array([sort_mapping[x] for x in new_validation_pred_classes_georoc])
georoc_df_lim['NN_Labels'] = new_validation_pred_labels_georoc

georoc_df_lim.to_csv('GEOROC_CpxAmp_NN_Variational.csv')

true_georoc = georoc_df_lim[georoc_df_lim['Mineral'] == georoc_df_lim['NN_Labels']]
false_georoc = georoc_df_lim[georoc_df_lim['Mineral'] != georoc_df_lim['NN_Labels']]

false_spinels_georoc = false_georoc[false_georoc['Mineral'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]
false_spinels_georoc = false_spinels_georoc[false_spinels_georoc['NN_Labels'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]

# %% 

oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeOt', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'Cr2O3']
petdb = pd.read_csv('Validation_Data/PetDB_validationdata_Fe.csv', index_col=0)
petdb_df = petdb.dropna(subset=oxides, thresh=6)

petdb_df = petdb_df[petdb_df.Mineral.isin(['Amphibole','Apatite','Biotite','Clinopyroxene','Garnet','Ilmenite','K-Feldspar',
                                             'Magnetite','Muscovite','Olivine','Orthopyroxene','Plagioclase','Quartz','Rutile','Spinel','Zircon'])]
petdb_df['Mineral'] = petdb_df['Mineral'].replace('K-Feldspar', 'KFeldspar')
petdb_df = petdb_df[~petdb_df['Mineral'].isin(['Tourmaline', 'Quartz', 'Rutile', 'Apatite', 'Zircon'])]
petdb_wt = petdb_df[oxides].fillna(0).to_numpy()
petdb_norm_wt = ss.transform(petdb_wt)

min_df_lim['Mineral'] = min_df_lim['Mineral'].astype('category')
petdb_df['Mineral'] = petdb_df['Mineral'].astype(pd.CategoricalDtype(categories=min_df_lim['Mineral'].cat.categories))
new_validation_data_y_petdb = (petdb_df['Mineral'].cat.codes).values

# Create a DataLoader for the new validation dataset
new_validation_dataset_petdb = LabelDataset(petdb_norm_wt, new_validation_data_y_petdb)
new_validation_loader_petdb = DataLoader(new_validation_dataset_petdb, batch_size=256, shuffle=False)

input_size = len(new_validation_dataset_petdb.__getitem__(0)[0])

path = 'parametermatrix_neuralnetwork/best_model.pt'

model = MultiClassClassifier(input_dim=input_size, dropout_rate=dr, hidden_layer_sizes=hls).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

load_model(model, optimizer, path)

# Use the trained model to predict the classes for the new validation dataset

model.eval()
new_validation_pred_classes_petdb = []
with torch.no_grad():
    for data, labels in new_validation_loader_petdb: 
        x = data.to(device)
        pred_classes = model.predict(x)
        new_validation_pred_classes_petdb.extend(pred_classes.tolist())

new_validation_pred_classes_petdb = np.array(new_validation_pred_classes_petdb)
unique_classes_petdb = np.unique(np.concatenate((new_validation_data_y_petdb[new_validation_data_y_petdb != -1], new_validation_pred_classes_petdb[new_validation_data_y_petdb != -1])))
sort_mapping_petdb = {key: value for key, value in sorted(mapping.items(), key=lambda item: item[0]) if key in unique_classes_petdb}

# Calculate classification metrics for the new validation dataset
new_validation_report = classification_report(new_validation_data_y_petdb[new_validation_data_y_petdb!=-1], new_validation_pred_classes_petdb[new_validation_data_y_petdb!=-1], labels = unique_classes_petdb, target_names=[sort_mapping_petdb[x] for x in unique_classes_petdb], zero_division=0)
print("New validation report:\n", new_validation_report)

cm_valid = confusion_matrix(new_validation_data_y_petdb[new_validation_data_y_petdb!=-1], new_validation_pred_classes_petdb[new_validation_data_y_petdb!=-1])

df_valid_cm_petdb = pd.DataFrame(
    cm_valid,
    index=[sort_mapping_petdb[x] for x in unique_classes_petdb],
    columns=[sort_mapping_petdb[x] for x in unique_classes_petdb],
)

df_valid_cm_petdb[df_valid_cm_petdb < len(new_validation_pred_classes_petdb)*0.001] = 0

mm.pp_matrix(df_valid_cm_petdb, cmap = cmap, savefig='petdb_valid', figsize = (11.5, 11.5)) 


new_validation_pred_labels_petdb = np.array([sort_mapping_petdb[x] for x in new_validation_pred_classes_petdb])
petdb_df['NN_Labels'] = new_validation_pred_labels_petdb

petdb_df.to_csv('PetDB_NN_Variational.csv')

true_petdb = petdb_df[petdb_df['Mineral'] == petdb_df['NN_Labels']]
false_petdb = petdb_df[petdb_df['Mineral'] != petdb_df['NN_Labels']]

false_spinels_petdb = false_petdb[false_petdb['Mineral'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]
false_spinels_petdb = false_spinels_petdb[false_spinels_petdb['NN_Labels'].isin(['Magnetite', 'Spinel', 'Ilmenite'])]


# %%
# %%
# %%
# %% 
# %%

import pyrolite 

def spinels(data_oxides): 

    # Set up  mass and charge data
    mr = {'SiO2':60.08, 'TiO2':79.88, 'Al2O3':101.96,'Fe2O3':159.69,
                    'FeO':71.85,'MnO':70.94,'MgO':40.3,'CaO':56.08,'Na2O':61.98,'K2O':94.2,'Cr2O3':151.9904}
    oxygen_numbers = {'SiO2':2,'TiO2':2,'Al2O3':3,'Fe2O3':3,'FeO':1,'MnO':1,'MgO':1,
                    'CaO':1,'Na2O':1,'K2O':1,'Cr2O3':3} # number of Os per mole of oxide
    cation_ratio = {'SiO2':0.5,'TiO2':0.5,'Al2O3':2/3,'Fe2O3':2/3,'FeO':1,'MnO':1,
                    'MgO':1,'CaO':1,'Na2O':2,'K2O':2,'Cr2O3':2/3} # ratio of cations to Os per mole of oxide
    cation = {'SiO2':'Si','TiO2':'Ti','Al2O3':'Al','Fe2O3':'Fe3','FeO':'Fe2', 
            'MnO':'Mn','MgO':'Mg','CaO':'Ca','Na2O':'Na','K2O':'K','Cr2O3':'Cr'}
    # =============================================================================
    # Calculate cations assuming all Fe2, for ferric iron recalculation
    # =============================================================================
    O_prop = pd.DataFrame()

    # Calculate atomic proportion of O from each molecule
    for oxide in data_oxides.columns.to_list():
        O_prop[oxide+'_O_prop'] = data_oxides[oxide]/mr[oxide]*oxygen_numbers[oxide]
            
    # What is the oxygen sum assuming all Fe2?    
    O_prop['O_sum'] = O_prop.sum(axis = 1)

    # What is the scaling factor for the mineral, based on desired numbers of O (4 for spinel)
    O_prop['min_O_sum'] = 4 

    cation_allFe2 = pd.DataFrame()
    for oxide in data_oxides.columns.to_list():
        cation_allFe2[cation[oxide]+'_pfu'] = O_prop[oxide+'_O_prop']*O_prop['min_O_sum']/O_prop['O_sum']*cation_ratio[oxide]

    # =============================================================================
    # Use Droop equation for calculating stoichiometric Fe3: F = 2X(1-T/S) 
    # T = 3 = O_prop['min_cat_sum']
    # X = 4 = O_prop['min_O_sum'] 
    # =============================================================================
    O_prop['min_cat_sum'] = 3

    if cation_allFe2.sum(axis=1).iloc[0] > O_prop['min_cat_sum'].iloc[0]:
        O_prop['Fe3'] = 2*O_prop['min_O_sum']*(1-O_prop['min_cat_sum']/cation_allFe2.sum(axis=1))
    else:
        O_prop['Fe3'] = 0

    cation_pfu = pd.DataFrame()
    cation_pfu['Fe3'] = O_prop['Fe3']

    # Normalise cations to total expected cation number ('Fe2' is really total Fe)
    other_cations = ['Si','Ti','Al','Mn','Mg','Ca','Na','K','Fe2','Cr']
    for cation in other_cations: 
        cation_pfu[cation] = cation_allFe2[cation+'_pfu']*O_prop['min_cat_sum']/cation_allFe2.sum(axis=1)
    # Now replace the Fe2 which is currently actually the total Fe
    cation_pfu['Fe2'] = cation_pfu['Fe2']-O_prop['Fe3']

    return cation_pfu 


def spinels_nocr(data_oxides): 
    # Set up  mass and charge data
    molar_mass = {'SiO2':60.08,'TiO2':79.866,'Al2O3':101.96,'FeO':71.844,'Fe2O3':159.69,
        'MnO':70.9374,'MgO':40.3044,'CaO':56.0774,'Na2O':61.98,'K2O':94.2}
    oxygen_numbers = {'SiO2':2,'TiO2':2,'Al2O3':3,'FeO':1,'Fe2O3':3,'MnO':1,'MgO':1,
                      'Na2O':1,'K2O':1,'CaO':1} # number of Os per mole of oxide
    cation_ratio = {'SiO2':0.5,'TiO2':0.5,'Al2O3':2/3,'FeO':1,'Fe2O3':2/3,'MnO':1,
                    'MgO':1,'Na2O':2,'K2O':2,'CaO':1} # ratio of cations to Os per mole of oxide
    cation = {'SiO2':'Si','TiO2':'Ti','Al2O3':'Al','FeO':'Fe2','Fe2O3':'Fe3',
              'MnO':'Mn','MgO':'Mg','CaO':'Ca','Na2O':'Na','K2O':'K'}
    # =============================================================================
    # Calculate cations assuming all Fe2, for ferric iron recalculation
    # =============================================================================
    O_prop = pd.DataFrame()
    cation_allFe2 = pd.DataFrame()
    cation_pfu = pd.DataFrame()

    # Calculate atomic proportion of O from each molecule
    for oxide in data_oxides.columns.to_list():
        O_prop[oxide+'_O_prop'] = data_oxides[oxide]/molar_mass[oxide]*oxygen_numbers[oxide]

    O_prop['O_sum'] = O_prop.sum(axis = 1)
    
    # What is the scaling factor for the mineral, based on desired numbers of O (4 for spinel)
    # and desired number of cations (3 for spinel)
    X = 4 
    T = 3

    for oxide in data_oxides.columns.to_list():
        # Calculate O and multiply by cation ratio to get to cations. Sum to get 'S' a la Droop
        cation_allFe2[cation[oxide]+'_pfu'] = O_prop[oxide+'_O_prop']/O_prop['O_sum']*X*cation_ratio[oxide]
    # =============================================================================
    # Use Droop equation for calculating stoichiometric Fe3: F=2X(1-T/S) 
    # T = 3 = O_prop['min_cat_sum']
    # X = 4 = O_prop['min_O_sum'] 
    # S = observed cation total, cation_allFe2.sum(axis=1)
    # =============================================================================
    mask = cation_allFe2.sum(axis=1) > T
    O_prop['Fe3'] = np.where(mask, (2*X)*(1-T/cation_allFe2.sum(axis=1)), 0)
    cation_pfu['Fe3'] = O_prop['Fe3']

    # Normalise cations to total expected cation number ('Fe2' is really total Fe)
    other_cations = ['Si','Ti','Al','Fe2','Mn','Mg','Na','K','Ca']
    for cation in other_cations: 
        cation_pfu[cation] = cation_allFe2[cation+'_pfu']/cation_allFe2.sum(axis=1)*T
    
    # Now replace the Fe2 which is currently actually the total Fe
    cation_pfu['Fe2'] = cation_pfu['Fe2'] - O_prop['Fe3'] # np.where(cation_pfu['Fe2'] > 0, cation_pfu['Fe2'] - O_prop['Fe3'], 0)

    return cation_pfu 


def ilmenites(data_oxides): 
    
    # Set up  mass and charge data
    mr = {'SiO2':60.08,'TiO2':79.88,'Al2O3':101.96,'FeO':71.85,'Fe2O3':159.69,
          'MnO':70.94,'MgO':40.3,'CaO':56.08,'Na2O':61.98,'K2O':94.2,'Cr2O3':151.99}
    oxygen_numbers = {'SiO2':2,'TiO2':2,'Al2O3':3,'Fe2O3':3,'FeO':1,'MnO':1,'MgO':1,
                      'CaO':1,'Na2O':1,'K2O':1,'Cr2O3':3} # number of Os per mole of oxide
    cation_ratio = {'SiO2':0.5,'TiO2':0.5,'Al2O3':2/3,'Fe2O3':2/3,'FeO':1,'MnO':1,'MgO':1,
                    'Na2O':2,'K2O':2,'CaO':1,'Cr2O3':2/3} # ratio of cations to Os per mole of oxide
    cation = {'SiO2':'Si','TiO2':'Ti','Al2O3':'Al','Fe2O3':'Fe3','FeO':'Fe2', 
            'MnO':'Mn','MgO':'Mg','CaO':'Ca','Na2O':'Na','K2O':'K','Cr2O3':'Cr'}
    # =============================================================================
    # Calculate cations assuming all Fe2, for ferric iron recalculation
    # =============================================================================
    O_prop = pd.DataFrame()

    # Calculate atomic proportion of O from each molecule
    for oxide in data_oxides.columns.to_list():
        O_prop[oxide+'_O_prop'] = data_oxides[oxide]/mr[oxide]*oxygen_numbers[oxide]
            
    # What is the oxygen sum assuming all Fe2?    
    O_prop['O_sum'] = O_prop.sum(axis = 1)    

    # What is the scaling factor for the mineral, based on desired numbers of O (3 for ilmenite)
    O_prop['min_O_sum'] = 3

    cation_allFe2 = pd.DataFrame()
    for oxide in data_oxides.columns.to_list():
        cation_allFe2[cation[oxide]+'_pfu'] = O_prop[oxide+'_O_prop']*O_prop['min_O_sum']/O_prop['O_sum']*cation_ratio[oxide]

    # =============================================================================
    # Use Droop equation for calculating stoichiometric Fe3: F = 2X(1-T/S)
    # =============================================================================
    O_prop['min_cat_sum'] = 2 # 2 cations for ilmenite, rhombohedral 

    O_prop['Fe3'] = 2*O_prop['min_O_sum']*(1-O_prop['min_cat_sum']/cation_allFe2.sum(axis=1))

    cation_pfu = pd.DataFrame()
    cation_pfu['Fe3'] = O_prop['Fe3']

    # Normalise cations to total expected cation number ('Fe2' is really total Fe)
    other_cations = ['Si','Ti','Al','Fe2','Mn','Mg','Na','K','Ca','Cr']
    for cation in other_cations:
        cation_pfu[cation] = cation_allFe2[cation+'_pfu']*O_prop['min_cat_sum']/cation_allFe2.sum(axis=1)
    # Now replace the Fe2 which is currently actually the total Fe
    cation_pfu['Fe2'] = cation_pfu['Fe2']-O_prop['Fe3']

    return cation_pfu 

# %% 


sp_df = min_df[min_df.Mineral=='Spinel']
il_df = min_df[min_df.Mineral=='Ilmenite']
mt_df = min_df[min_df.Mineral=='Magnetite']

sp_df_calc = sp_df[['SiO2','TiO2','Al2O3','FeOt','MnO','MgO','CaO','Na2O','K2O','Cr2O3']]
sp_df_calc = sp_df_calc.rename(columns={'FeOt':'FeO'})

il_df_calc = il_df[['SiO2','TiO2','Al2O3','FeOt','MnO','MgO','CaO','Na2O','K2O','Cr2O3']]
il_df_calc = il_df_calc.rename(columns={'FeOt':'FeO'})

mt_df_calc = mt_df[['SiO2','TiO2','Al2O3','FeOt','MnO','MgO','CaO','Na2O','K2O','Cr2O3']]
mt_df_calc = mt_df_calc.rename(columns={'FeOt':'FeO'})

sp_cation_pfu = spinels(sp_df_calc) 
il_cation_pfu = ilmenites(il_df_calc)
mt_cation_pfu = spinels(mt_df_calc)
sp_cation_nocr_pfu = spinels_nocr(sp_df_calc[['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O']]) 
mt_cation_nocr_pfu = spinels_nocr(mt_df_calc[['SiO2','TiO2','Al2O3','FeO','MnO','MgO','CaO','Na2O','K2O']]) 

sp_cation_pfu['R3'] = sp_cation_pfu.Fe3 + sp_cation_pfu.Al + sp_cation_pfu.Cr
il_cation_pfu['R3'] = il_cation_pfu.Fe3 + il_cation_pfu.Al + il_cation_pfu.Cr
mt_cation_pfu['R3'] = mt_cation_pfu.Fe3 + mt_cation_pfu.Al + mt_cation_pfu.Cr
sp_cation_nocr_pfu['R3'] = sp_cation_nocr_pfu.Fe3 + sp_cation_nocr_pfu.Al
mt_cation_nocr_pfu['R3'] = mt_cation_nocr_pfu.Fe3 + mt_cation_nocr_pfu.Al

bool = (sp_cation_pfu.Fe3 / (sp_cation_pfu.Al+sp_cation_pfu.Fe3)<0.5) & (sp_cation_pfu.Fe2 / (sp_cation_pfu.Mg+sp_cation_pfu.Fe2)<0.5)
sp_cation_pfu_lim = sp_cation_pfu[bool]

x_sp = sp_cation_pfu_lim.Fe2 / (sp_cation_pfu_lim.Mg+sp_cation_pfu_lim.Fe2)
y_sp = sp_cation_pfu_lim.Fe3 / (sp_cation_pfu_lim.Al+sp_cation_pfu_lim.Fe3)
x1_sp = sp_cation_pfu_lim.Cr / (sp_cation_pfu_lim.Cr+sp_cation_pfu_lim.Al)
y1_sp = sp_cation_pfu_lim.Mg / (sp_cation_pfu_lim.Mg+sp_cation_pfu_lim.Fe2)
y2_sp = sp_cation_pfu_lim.Ti / (sp_cation_pfu_lim.Ti+sp_cation_pfu_lim.Cr)
y3_sp = sp_cation_pfu_lim.Al / (sp_cation_pfu_lim.Al+sp_cation_pfu_lim.Cr)
y4_sp = sp_cation_pfu_lim.Al / (sp_cation_pfu_lim.Al+sp_cation_pfu_lim.Ti)

bool_new = (sp_cation_nocr_pfu.Fe3 / (sp_cation_nocr_pfu.Al+sp_cation_nocr_pfu.Fe3)<0.5) & (sp_cation_nocr_pfu.Fe2 / (sp_cation_nocr_pfu.Mg+sp_cation_nocr_pfu.Fe2)<0.5)
sp_cation_nocr_pfu_lim = sp_cation_nocr_pfu[bool_new]

x_nocr_sp = sp_cation_nocr_pfu_lim.Fe2 / (sp_cation_nocr_pfu_lim.Mg+sp_cation_nocr_pfu_lim.Fe2)
y_nocr_sp = sp_cation_nocr_pfu_lim.Fe3 / (sp_cation_nocr_pfu_lim.Al+sp_cation_nocr_pfu_lim.Fe3)

x_mt = mt_cation_pfu.Fe2 / (mt_cation_pfu.Mg+mt_cation_pfu.Fe2)
y_mt = mt_cation_pfu.Fe3 / (mt_cation_pfu.Al+mt_cation_pfu.Fe3)
x1_mt = mt_cation_pfu.Cr / (mt_cation_pfu.Cr+mt_cation_pfu.Al)
y1_mt = mt_cation_pfu.Mg / (mt_cation_pfu.Mg+mt_cation_pfu.Fe2)
y2_mt = mt_cation_pfu.Ti / (mt_cation_pfu.Ti+mt_cation_pfu.Cr)
y3_mt = mt_cation_pfu.Al / (mt_cation_pfu.Al+mt_cation_pfu.Cr)
y4_mt = mt_cation_pfu.Al / (mt_cation_pfu.Al+mt_cation_pfu.Ti)


# plt.figure(figsize=(8, 6))
# plt.scatter(x_sp, y_sp)
# plt.scatter(x_nocr_sp, y_nocr_sp)
# plt.scatter(x_mt, y_mt)

# plt.figure(figsize=(8, 6))
# plt.scatter(x1_sp, y1_sp)
# plt.scatter(x1_mt, y1_mt)

# plt.figure(figsize=(8, 6))
# plt.scatter(y2_sp, y1_sp)
# plt.scatter(y2_mt, y1_mt)

# plt.figure(figsize=(8, 6))
# plt.scatter(y3_sp, y1_sp)
# plt.scatter(y3_mt, y1_mt)

# plt.figure(figsize=(8, 6))
# plt.scatter(y4_sp, y1_sp)
# plt.scatter(y4_mt, y1_mt)

# %% 

sp_df_ej = pd.read_excel('Training_Data/Mineral/Spinel.xlsx').tail(62)
sp_df_ej_calc = sp_df_ej[['SiO2','TiO2','Al2O3','FeOt','MnO','MgO','CaO','Na2O','K2O','Cr2O3','NiO']]
sp_df_ej_calc = sp_df_ej_calc.rename(columns={'FeOT':'FeO'})


# sp_ej = pd.read_excel('EJ_data.xlsx')

# sp_cation_ej_pfu = spinels(sp_df_ej_calc)
# sp_cation_ej_pfu

# %% 
from pyrolite.plot import pyroplot

fig, ax = plt.subplots(1, 1, figsize=(8,8))
line_df = pd.DataFrame(
    [
        [1/3, 2/3, 0],
        [0, 1/2, 1/2]
    ],
    columns=["Ti", "Fe2", "R3"]
)

line_df1 = pd.DataFrame(
    [
        [1/3, 2/3, 0],
        [0, 1/3, 2/3]
    ],
    columns=["Ti", "Fe2", "R3"]
)
line_df2 = pd.DataFrame(
    [
        [1/2, 1/2, 0],
        [0, 0, 1]
    ],
    columns=["Ti", "Fe2", "R3"]
)
line_df3 = pd.DataFrame(
    [
        [2/3, 1/3, 0],
        [1/3, 0, 2/3]
    ],
    columns=["Ti", "Fe2", "R3"]
)



sp_cation_pfu.loc[:, ["Ti", "Fe2", "R3"]].pyroplot.scatter(c="green", ax=ax, label='spinel with cr')
# sp_cation_nocr_pfu_lim.loc[:, ["Ti", "Fe2", "R3"]].pyroplot.scatter(c="red", ax=ax, label='spinel no cr')
il_cation_pfu.loc[:, ["Ti", "Fe2", "Fe3"]].pyroplot.scatter(c="orange", ax=ax, label='ilmenite')
mt_cation_pfu.loc[:, ["Ti", "Fe2", "Fe3"]].pyroplot.scatter(c="k", ax=ax, label='magnetite soderman')
mt_comp.loc[:, ['Ti_cat_4ox', 'Fe2_cat_4ox', 'Fe3_cat_4ox']].pyroplot.scatter(c="blue", ax=ax, label='magnetite shi')
line_df.pyroplot.scatter(ax=ax, c="k", marker="o")
line_df.pyroplot.plot(ax=ax, color="k", linestyle="--")  # connecting line
line_df1.pyroplot.scatter(ax=ax, c="k", marker="o")
line_df1.pyroplot.plot(ax=ax, color="k", linestyle="--")  # connecting line
line_df2.pyroplot.scatter(ax=ax, c="k", marker="o")
line_df2.pyroplot.plot(ax=ax, color="k", linestyle="--")  # connecting line
line_df3.pyroplot.scatter(ax=ax, c="k", marker="o")
line_df3.pyroplot.plot(ax=ax, color="k", linestyle="--")  # connecting line
plt.legend()
plt.show()

# %%

cat_suffix = f"_cat_{4}ox"
cation_cols = [col for col in mt_comp.columns if col.endswith(cat_suffix)]

mt_comp[cation_cols]


# %% 


# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# sp_cation_pfu.loc[:, ["Cr", "Al", "Fe3"]].pyroplot.scatter(c="green", ax=ax, label='spinel')
# il_cation_pfu.loc[:, ["Cr", "Al", "Fe3"]].pyroplot.scatter(c="orange", ax=ax, label='ilmenite')
# mt_cation_pfu.loc[:, ["Cr", "Al", "Fe3"]].pyroplot.scatter(c="b", ax=ax, label='magnetite')
# plt.legend()
# plt.show()



# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# sp_cation_pfu_lim.loc[:, ["Mg", "Fe2", "Fe3"]].pyroplot.scatter(c="green", ax=ax, label='spinel')
# il_cation_pfu.loc[:, ["Mg", "Fe2", "Fe3"]].pyroplot.scatter(c="orange", ax=ax, label='ilmenite')
# mt_cation_pfu.loc[:, ["Mg", "Fe2", "Fe3"]].pyroplot.scatter(c="b", ax=ax, label='magnetite')
# plt.legend()
# plt.show()

# %%

# %%



data_oxides = sp_df_ej_calc

molar_mass = {'SiO2':60.08,'TiO2':79.866,'Al2O3':101.96,'FeOt':71.844,'Fe2O3':159.69,
    'MnO':70.9374,'MgO':40.3044,'CaO':56.0774,'Cr2O3':151.99,'NiO':74.5928}
cation_numbers = {'SiO2':1,'TiO2':1,'Al2O3':2,'FeOt':1,'Fe2O3':2,'MnO':1,
                'MgO':1,'CaO':1,'Cr2O3':2,'NiO':1} # numbers of cations per mole of oxide
oxygen_numbers = {'SiO2':2,'TiO2':2,'Al2O3':3,'FeOt':1,'Fe2O3':3,'MnO':1,'MgO':1,
                'CaO':1,'Cr2O3':3,'NiO':1} # number of Os per mole of oxide
cation_ratio = {'SiO2':0.5,'TiO2':0.5,'Al2O3':2/3,'FeOt':1,'Fe2O3':2/3,'MnO':1,
                'MgO':1,'CaO':1,'Cr2O3':2/3,'NiO':1} # ratio of cations to Os per mole of oxide
cation = {'SiO2':'Si','TiO2':'Ti','Al2O3':'Al','FeOt':'Fe2','Fe2O3':'Fe3',
            'MnO':'Mn','MgO':'Mg','CaO':'Ca','Cr2O3':'Cr','NiO':'Ni'}
# =============================================================================
# Calculate cations assuming all Fe2, for ferric iron recalculation
# =============================================================================
O_prop = pd.DataFrame()
cat_prop = pd.DataFrame()
cation_pfu = pd.DataFrame()
cation_allFe2 = pd.DataFrame()
# Calculate atomic proportion of O from each molecule
for oxide in data_oxides.columns.to_list():
    O_prop[oxide+'_O_prop'] = data_oxides[oxide]/molar_mass[oxide]*oxygen_numbers[oxide]
    cat_prop[oxide] = data_oxides[oxide]/molar_mass[oxide]*cation_numbers[oxide]

# What is the oxygen sum assuming all Fe2?    
O_prop['O_sum'] = O_prop.sum(axis = 1)
orf = 4 / O_prop['O_sum'] 
cat_prop_norm = cat_prop.mul(orf, axis=0)

cat_prop_norm['cat_sum'] = cat_prop_norm.sum(axis = 1)
cat_prop_norm = cat_prop_norm.fillna(0)
cat_prop_norm['sum_charge'] = (2 * (cat_prop_norm["MgO"] + cat_prop_norm["MnO"] + cat_prop_norm["CaO"] + cat_prop_norm["NiO"])
                            + 3 * (cat_prop_norm["Al2O3"] + cat_prop_norm["Cr2O3"])
                            + 4 * (cat_prop_norm["TiO2"] + cat_prop_norm["SiO2"]))

cat_prop_norm['fe3'] = 0 
cat_prop_norm.loc[(8 * cat_prop_norm["cat_sum"] / 3 - cat_prop_norm["sum_charge"] - 2 * cat_prop_norm["FeO"]) > 0, "fe3",] = (8 * cat_prop_norm["cat_sum"] / 3 - cat_prop_norm["sum_charge"] - 2 * cat_prop_norm["FeO"])
cat_prop_norm["fe2"] = cat_prop_norm["FeO"] - cat_prop_norm["fe3"]


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
cat_prop_norm.loc[:, ["TiO2", "fe2", "fe3"]].pyroplot.scatter(c="red", ax=ax, label='EJ')
sp_cation_ej_pfu.loc[:, ["Ti", "Fe2", "Fe3"]].pyroplot.scatter(c="black", ax=ax, label='EJ')

# %%

