# %% 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('src')
import mineralML as mm

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.rcParams.update({
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.labelsize': 14,
    'pdf.fonttype': 42,
    'font.family': 'Avenir',
    'font.size': 14,
    'xtick.direction': 'in',  # Set x-tick direction to 'in'
    'ytick.direction': 'in',  # Set y-tick direction to 'in'
    'xtick.major.size': 5,    # Set x-tick length
    'ytick.major.size': 5,    # Set y-tick length
    'xtick.major.pad': 6.5,   # Set x-tick padding
    'ytick.major.pad': 6.5    # Set y-tick padding
})


# %% 

df_TC_Px=pd.read_excel('Torres_Orozco.xlsx', sheet_name='3.Pyroxene Gps')
df_TC_Px_YW2=df_TC_Px.loc[df_TC_Px['Sample.1']=='YW2']
df_TC_Px_GP2=df_TC_Px.loc[df_TC_Px['Sample.1']=='GP2']
df_TC_Amp=pd.read_excel('Torres_Orozco.xlsx', sheet_name='4.Amph Gps')
df_TC_Amp_GP2=df_TC_Amp.loc[df_TC_Amp['Sample.1']=='GP2']
df_TC_Amp_YW2=df_TC_Amp.loc[df_TC_Amp['Sample.1']=='YW2']
Act = df_TC_Amp['MineralGroup'] == '(Cr-Actinolite)'

# %% 

df=pd.read_csv('Validation_Data/GEOROC_validationdata_Fe_2025.csv')
min_df = mm.prep_df(df)
GEOROC = mm.predict_class_prob(min_df, n_iterations=25)

# %% 

GEOROC['CITATION'] = df['CITATION']
cit='[26351] TORRES-OROZCO R. (2023)'
cm = pd.crosstab(
    GEOROC['Mineral'].loc[GEOROC['CITATION']==cit],
    GEOROC['Predict_Mineral'].loc[GEOROC['CITATION']==cit]
)

cm_percent = cm.div(cm.sum(axis=1), axis=0) * 100

# %%

GEOROC['Citation']=df['CITATION']
GEOROC['SAMPLE NAME']=df['SAMPLE NAME']
Test = pd.read_csv("Training_Data/min_df_v3_natural.csv")

# %%

mh_base_cols = {
    "Plagioclase": "#66C4C4", "Clinopyroxene": "#E57A7A",
    "Orthopyroxene": "#931d1d", "KFeldspar": "#FEF7C2",
    "Spinels": "#2E2DCE", "Glass": "#F9C300",
    "Apatite": "#5B6768", "Rhombohedral_Oxides": "#56B4E9",
    "SiO2_Polymorph": "#CEC6CD",
}
bii_cols  = {**mh_base_cols, "Amphibole": '#5E2910',
             "Zircon": "cyan", "Epidote": "magenta",
             "Titanite": "tab:brown", "Unindexed": "white",
             "Feldspar_Miscibility_Gap": "#003d36",
             "Vesicles": "white" # Handled here for EDS plotting
             }

# %% 

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10,8))
ax1.plot(Test['SiO2'][Test['Mineral']=='Amphibole'], Test['Al2O3'][Test['Mineral']=='Amphibole'], '.', 
         mfc=bii_cols['Amphibole'], lw=0.5,
         alpha=0.1, label='Calibration Amp', rasterized=True)
ax1.plot(Test['SiO2'][Test['Mineral']=='Clinopyroxene'], Test['Al2O3'][Test['Mineral']=='Clinopyroxene'], '.',
         mfc=bii_cols['Clinopyroxene'], lw=0.5, 
         alpha=0.1, label='Calibration Cpx', rasterized=True)
# Plot their data. 
ax1.plot(df_TC_Amp['SiO2'][~Act], df_TC_Amp['Al2O3'][~Act], 'ok', mfc=bii_cols['Amphibole'], label='Reported Amp')
ax1.plot(df_TC_Px['SiO2'], df_TC_Px['Al2O3'], 'ok', mfc=bii_cols['Clinopyroxene'],  label='Reported Cpx')
ax1.plot(df_TC_Amp['SiO2'][Act], df_TC_Amp['Al2O3'][Act], '^k', mfc='white', label='Reported Cr-Actinolites')
#ax1.legend()
ax1.set_xlabel('SiO$\mathregular{_2}$ (wt.%)')
ax1.set_ylabel('Al$\mathregular{_2}$O$\mathregular{_3}$ (wt.%)')
ax1.legend(fontsize=9)

ax2.plot(Test['Na2O'][Test['Mineral']=='Amphibole'], Test['FeOt'][Test['Mineral']=='Amphibole'], '.',
         mfc=bii_cols['Amphibole'], lw=0.5, alpha=0.1, rasterized=True)
ax2.plot(Test['Na2O'][Test['Mineral']=='Clinopyroxene'], Test['FeOt'][Test['Mineral']=='Clinopyroxene'], '.',
          mfc=bii_cols['Clinopyroxene'], lw=0.5, alpha=0.1, rasterized=True)
# Plot their data. 
ax2.plot(df_TC_Px['Na2O'], df_TC_Px['FeO'], 'ok', mfc=bii_cols['Clinopyroxene'], label='Pyroxene')
ax2.plot(df_TC_Amp['Na2O'][~Act], df_TC_Amp['FeO'][~Act], 'ok', mfc=bii_cols['Amphibole'], label='Amphibole')
ax2.plot(df_TC_Amp['Na2O'][Act], df_TC_Amp['FeO'][Act], '^k', mfc='white', label='Cr-Actinolites')

ax2.set_xlabel('Na$\mathregular{_2}$O (wt.%)')
ax2.set_ylabel('FeO$\mathregular{_t}$ (wt.%)')
ax2.set_xlim([-0.1, 3.5])

## Now plotting up thermobarometry results
ax3.plot(df_TC_Amp['[TP16SiinHbl(oC)a'][~Act], df_TC_Amp['PR10(MPa)a'][~Act], 'ok', mfc=bii_cols['Amphibole'], label='Amphibole')
ax3.plot(df_TC_Amp['[TP16SiinHbl(oC)a'][Act], df_TC_Amp['PR10(MPa)a'][Act], '^k', mfc='white', label='Cr-Actinolites')
ax3.set_xlabel('Temp (°C) \n Si in Amp (P2016)')
ax3.set_ylabel('Pressure (MPa) \n Amp-only Barometer (R2010)')

## Now plotting up thermobarometry results
ax4.plot(df_TC_Amp['H2OmeltR10(%)a'][~Act], df_TC_Amp['SiO2(wt%)a-l]'][~Act], 'ok', mfc=bii_cols['Amphibole'], label='Amphibole')
ax4.plot(df_TC_Amp['H2OmeltR10(%)a'][Act], df_TC_Amp['SiO2(wt%)a-l]'][Act], '^k', mfc='white', label='Cr-Actinolites')
ax4.set_xlabel('Calc Melt $\mathregular{H_2O}$ (wt.%) \n Amp-only Hygrometer (R+2010)')
ax4.set_ylabel('Calc Melt $\mathregular{SiO_2}$ (wt.%) \n Amp-only Chemometer (P2016)')
fig.tight_layout()
fig.savefig('OrozcoIssues.pdf')

# %% 

Amp = df_TC_Amp[~Act]
Amp['Predict_Mineral'] = 'Amphibole'
Amp = mm.prep_df(Amp, convert_fe=True)

Px = df_TC_Px
Px['Predict_Mineral'] = 'Clinopyroxene'
Px = mm.prep_df(Px, convert_fe=True)

CrAct = df_TC_Amp[Act]
CrAct['Predict_Mineral'] = 'Amphibole'
CrAct['Submineral'] = 'Cr-Actinolite'
CrAct = mm.prep_df(CrAct, convert_fe=True)


latent = pd.concat([Amp, CrAct])
latent = mm.prep_df(latent, convert_fe=True)

# %% 

mm.plot_latent_space(CrAct, new_kws={"color": 'white', "marker": "^", "s": 75, "alpha": 1, "rasterized": True}, filename='LS_CrAct.pdf') # filename='LS_CrAct.pdf'

mm.plot_latent_space(Px, new_kws={"color": bii_cols['Clinopyroxene'], "s": 75, "alpha": 1, "rasterized": True}, filename='LS_Px.pdf') # filename='LS_CrAct.pdf'

mm.plot_latent_space(Amp, new_kws={"color": bii_cols['Amphibole'], "s": 75, "alpha": 1, "rasterized": True}, filename='LS_Amp.pdf') # filename='LS_CrAct.pdf'


# %% 