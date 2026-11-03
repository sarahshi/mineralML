# %%

import sys
import tempfile
sys.path.append('../../src')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm
from fontTools.ttLib import TTCollection
import mineralML as mm
from IPython.display import display

# matplotlib only reads Avenir Book from Avenir.ttc; register Avenir Heavy (face 4) too
heavy = tempfile.gettempdir() + '/AvenirHeavy.ttf'
TTCollection('/System/Library/Fonts/Avenir.ttc').fonts[4].save(heavy)
fm.fontManager.addfont(heavy)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.rcParams.update({
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.labelsize': 14,
    'pdf.fonttype': 42,
    'font.family': 'Avenir',
    'font.size': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.major.pad': 6.5,
    'ytick.major.pad': 6.5,
})

# %%

# GEOROC data
df = pd.read_csv('../../Validation_Data/GEOROC_validationdata_Fe_202602.csv')
GEOROC = mm.predict_class_prob(mm.prep_df(df))
GEOROC['Citation'] = df['CITATION']
GEOROC['SAMPLE NAME'] = df['SAMPLE NAME']

# Training Data
Train = pd.read_csv('../../Training_Data/min_df_v3_natural.csv')
T_Amp = Train[Train['Mineral'] == 'Amphibole']
T_Cpx = Train[Train['Mineral'] == 'Clinopyroxene']
T_Plag = Train[Train['Mineral'] == 'Plagioclase']

def citation_cm(cit):
    sub = GEOROC[GEOROC['Citation'] == cit]
    return pd.crosstab(sub['Mineral'], sub['Predict_Mineral'])

# %% Colors

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

# %% Torres-Orozco

df_TC_Px = pd.read_excel('TorresOrozco.xlsx', sheet_name='3.Pyroxene Gps').dropna(subset=['SiO2'])
df_TC_Amp = pd.read_excel('TorresOrozco.xlsx', sheet_name='4.Amph Gps').dropna(subset=['SiO2'])
Act = df_TC_Amp['MineralGroup'] == '(Cr-Actinolite)'
Amp = df_TC_Amp[~Act]
CrAct = df_TC_Amp[Act]

cm = citation_cm('[26351] TORRES-OROZCO R. (2023)')

# %% Figure 3 

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax = ax.flatten()
ax[0].plot(T_Amp['SiO2'], T_Amp['Al2O3'], '.', mfc=bii_cols['Amphibole'], alpha=0.1, label='Amp Cali Dataset')
ax[0].plot(T_Cpx['SiO2'], T_Cpx['Al2O3'], '.', mfc=bii_cols['Clinopyroxene'], alpha=0.1, label='Cpx Cali Dataset')
# Plot their data.
ax[0].plot(df_TC_Px['SiO2'], df_TC_Px['Al2O3'], 'ok', mfc=bii_cols['Clinopyroxene'], label='Reported Cpx')
ax[0].plot(Amp['SiO2'], Amp['Al2O3'], 'ok', mfc=bii_cols['Amphibole'], label='Reported Amp')
ax[0].plot(CrAct['SiO2'], CrAct['Al2O3'], '^k', mfc='white', label='Reported Cr-Actinolites')
ax[0].set_xlabel(r'SiO$\mathregular{_2}$ (wt.%)')
ax[0].set_ylabel(r'Al$\mathregular{_2}$O$\mathregular{_3}$ (wt.%)')
ax[0].legend(fontsize=9)

ax[1].plot(T_Amp['Na2O'], T_Amp['FeOt'], '.', mec=bii_cols['Amphibole'], alpha=0.1)
ax[1].plot(T_Cpx['Na2O'], T_Cpx['FeOt'], '.', mec=bii_cols['Clinopyroxene'], alpha=0.1)
# Plot their data.
ax[1].plot(df_TC_Px['Na2O'], df_TC_Px['FeO'], 'ok', mfc=bii_cols['Clinopyroxene'])
ax[1].plot(Amp['Na2O'], Amp['FeO'], 'ok', mfc=bii_cols['Amphibole'])
ax[1].plot(CrAct['Na2O'], CrAct['FeO'], '^k', mfc='white')
ax[1].set_xlabel(r'Na$\mathregular{_2}$O (wt.%)')
ax[1].set_ylabel(r'FeO$\mathregular{_t}$ (wt.%)')
ax[1].set_xlim([-0.1, 3.5])

## Now plotting up thermobarometry results
ax[2].plot(Amp['[TP16SiinHbl(oC)a'], Amp['PR10(MPa)a'], 'ok', mfc=bii_cols['Amphibole'])
ax[2].plot(CrAct['[TP16SiinHbl(oC)a'], CrAct['PR10(MPa)a'], '^k', mfc='white')
ax[2].set_xlabel('Temp (°C) \n Si in Hbl (P2016)')
ax[2].set_ylabel('Pressure (MPa) \n Amp-only Barometer (R2010)')

ax[3].plot(Amp['H2OmeltR10(%)a'], Amp['SiO2(wt%)a-l]'], 'ok', mfc=bii_cols['Amphibole'])
ax[3].plot(CrAct['H2OmeltR10(%)a'], CrAct['SiO2(wt%)a-l]'], '^k', mfc='white')
ax[3].set_xlabel('Calc Melt $\\mathregular{H_2O}$ (wt.%) \n Amp-only Hygrometer (R+2010)')
ax[3].set_ylabel('Calc Melt $\\mathregular{SiO_2}$ (wt.%) \n Amp-only Chemometer (P2016)')
fig.tight_layout()
fig.savefig('TorresOrozco.pdf', dpi=600)

# %% Torres-Orozco latent space

LS_Amp = mm.prep_df(Amp, convert_fe=True)
LS_Amp['Predict_Mineral'] = 'Amphibole'
fig, ax = mm.plot_latent_space(LS_Amp, title='', ref_kws={'rasterized': True},
                               new_kws={'color': bii_cols['Amphibole'], 's': 75, 'alpha': 1})
wrapper, _, _ = mm.load_hybrid_checkpoint()
mm.same_seeds(88)
for d, m, c in [(df_TC_Px, 's', bii_cols['Clinopyroxene']), (CrAct, '^', 'white')]:
    z, _ = mm.compute_z2_from_df(mm.prep_df(d, convert_fe=True), wrapper)
    ax.scatter(z[:, 0], z[:, 1], marker=m, color=c, s=75, ec='k', lw=0.4)
leg = ax.get_legend()
ref_cols = {t.get_text(): h.get_facecolor()[0] for t, h in zip(leg.texts, leg.legend_handles)}
for x, y, s, c in [(-2.1, 0.1, 'Cpx', bii_cols['Clinopyroxene']),
                   (-1.75, 0.15, 'Cr-Actinolite', 'white'),
                   (-1.45, -0.18, 'Amp', bii_cols['Amphibole']),
                   (-2.1, -0.4, 'Train Px', ref_cols['Pyroxene']),
                   (-1.4, -0.4, 'Train Amp', ref_cols['Amphibole'])]:
    ax.text(x, y, s, color=c, fontsize=20, fontweight='heavy',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='k')])
ax.set_xlabel('Latent Variable 1')
ax.set_ylabel('Latent Variable 2')
fig.savefig('TorresOrozco_LS.pdf', dpi=300, bbox_inches='tight')

# %% Yamasaki

cit = '[26317] YAMASAKI T. (2022)'
display(citation_cm(cit))
Yam_Amp = GEOROC[(GEOROC['Citation'] == cit) & (GEOROC['Mineral'] == 'Amphibole')]

# %% Figure 4 

fig, ax = mm.plot_latent_space(Yam_Amp, title='', ref_kws={'rasterized': True}, new_kws={'s': 75, 'alpha': 1})
leg = ax.get_legend()
ref_cols = {t.get_text(): h.get_facecolor()[0] for t, h in zip(leg.texts, leg.legend_handles)}
for x, y, s in [(0.3, 1.45, 'Feldspar'), (-1.35, 0.2, 'Amphibole'), (0.2, -0.85, 'Biotite')]:
    ax.text(x, y, s, color=ref_cols[s], fontsize=20, fontweight='heavy',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='k')])
ax.set_xlabel('Latent Variable 1')
ax.set_ylabel('Latent Variable 2')
fig.savefig('Yamasaki_LS.pdf', dpi=300, bbox_inches='tight')

# %% Further plotting of Yamasaki Amphiboles in oxide space. 

x = 'CaO'
y = 'MgO'
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(T_Amp[x], T_Amp[y], '.', mfc=bii_cols['Amphibole'], alpha=0.1, label='Amp Cali Dataset')
ax.plot(T_Cpx[x], T_Cpx[y], '.', mfc=bii_cols['Clinopyroxene'], alpha=0.1, label='Cpx Cali Dataset')
ax.plot(T_Plag[x], T_Plag[y], '.', mfc=bii_cols['Plagioclase'], alpha=0.1, label='Plag Cali Dataset')
# Plot their data.
YAmp = Yam_Amp[Yam_Amp['Predict_Mineral'] == 'Amphibole']
YBt = Yam_Amp[Yam_Amp['Predict_Mineral'] == 'Biotite']
YCpx = Yam_Amp[Yam_Amp['Predict_Mineral'] == 'Clinopyroxene']
YPlag = Yam_Amp[Yam_Amp['Predict_Mineral'] == 'Plagioclase']
ax.plot(YAmp[x], YAmp[y], 'ok', mfc=bii_cols['Amphibole'], label='P Amp-MM Amp')
ax.plot(YBt[x], YBt[y], 'ok', mfc='darkred', label='P Amp-MM Bt')
ax.plot(YCpx[x], YCpx[y], 'ok', mfc=bii_cols['Clinopyroxene'], label='P Amp-MM Cpx')
ax.plot(YPlag[x], YPlag[y], 'ok', mfc=bii_cols['Plagioclase'], label='P Amp-MM Plag')
ax.set_xlabel(x)
ax.set_ylabel(y)
ax.legend(fontsize=10)
plt.show()

# %%
