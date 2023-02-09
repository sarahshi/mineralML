# %% 

""" Created on October 31, 2022 // @author: Sarah Shi """

import numpy as np
import pandas as pd
import os 
import json 
import pickle

import Thermobar as pt

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib.path as mpath

# %% 

def DF2Patch(df):
    patches = []
    n = len(df)

    for i in range(n):
        polygon = mpatches.Polygon(df.iloc[i, df.columns.get_loc('Coordinates')], closed = False)
        patches.append(polygon)

    return patches

def Synth_Data(columns = None, mean = None, std = None, n = 500, randseed = None):
    
    oxides = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']

    if columns is None: 
        synth_df = pd.DataFrame(columns = oxides)
    else: 
        synth_df = pd.DataFrame(columsn = columns)

    # Mean and STD from Fuego 2018 MIs. 
    mean_dict = {'SiO2': 0.5153, 'TiO2': 0.0110, 'Al2O3': 0.1745, 'FeO': 0.0988, 'MnO': 0.0020, 'MgO': 0.0407, 'CaO': 0.0768, 'Na2O': 0.0401, 'K2O': 0.0091, 'P2O5': 0.0017}
    std_dict = {'SiO2': 0.0201, 'TiO2': 0.0020, 'Al2O3': 0.0082, 'FeO': 0.0133, 'MnO': 0.0003, 'MgO': 0.0046, 'CaO': 0.0104, 'Na2O': 0.0040, 'K2O': 0.0018, 'P2O5': 0.0004}

    if randseed is not None:
        np.random.seed(randseed)

    if mean is None:
        for oxide in oxides:
            synth_df[oxide] = np.random.normal(loc=mean_dict[oxide], scale = std_dict[oxide], size = n)
    else: 
        for oxide in oxides: 
            synth_df[oxide] = np.random.normal(loc=mean[oxide], scale = std[oxide], size = n)
    
    synth_df *= 100

    return synth_df


def TAS_Plot(edgecolors = 'k', alpha = 0.4):

    import matplotlib.collections as mcollections 

    path_parent = os.getcwd()
    with open(path_parent+'/TAS_Polygon/TASPlottingPolygons.pkl', 'rb') as f:
        polygonplot_df = pickle.load(f)
    plot_patches = DF2Patch(polygonplot_df)

    p = mcollections.PatchCollection(plot_patches, edgecolors = (edgecolors,), facecolors=('None'),  alpha=0.4)
    
    return p

def TAS_Labels(ax, fontsize=8, color=(0.6, 0.6, 0.6)):
    # Adapted from the TAS plotting from https://bitbucket.org/jsteven5/tasplot/src/90ed07ec34fa13405e7d2d5c563341b3e5eef95f/tasplot.py?at=master

    from collections import namedtuple
    FieldName = namedtuple('FieldName', 'name x y rotation')
    names = (FieldName('Picro-\nbasalt', 43, 2, 0),
            FieldName('Alkali\nbasalt', 47, 4.75, 0),
            FieldName('Basalt', 48.5, 3, 0),
            FieldName('Basaltic\nandesite', 54.5, 3, 0),
            FieldName('Andesite', 60, 3, 0),
            FieldName('Dacite', 68.5, 3, 0),
            FieldName('Rhyolite', 72.5, 8, 0),
            FieldName('Trachyte\n(Q<20%)', 63, 12, 0),
            FieldName('Trachydacite\n(Q>20%)', 65, 10, 0),
            FieldName('Basaltic\ntrachy-\nandesite', 53.25, 7.5, 0),
            FieldName('Trachy-\nbasalt', 49, 6.2, 0),
            FieldName('Trachy-\nandesite', 57.5, 9, 0),
            FieldName('Phonotephrite', 49, 9.6, 0),
            FieldName('Tephriphonolite', 53.0, 11.8, 0),
            FieldName('Phonolite', 57.25, 13.5, 0),
            FieldName('Tephrite\n(Ol<10%)', 45, 8, 0),
            FieldName('Foidite', 44, 11.5, 0),
            FieldName('Basanite\n(Ol>10%)', 43.5, 6.6, 0))

    for name in names:
        ax.text(name.x, name.y, name.name, color=color, fontsize = fontsize,
                 horizontalalignment='center', verticalalignment='top',
                 rotation=name.rotation, zorder=0)

def TAS_Class(df):

    tas_label = {0:'NaN', 1:'Basalt', 2:'Foidite', 3:'Basaltic andesite', 4:'Andesite', 5:'Dacite', 6:'Picrite', 7:'Phonolite', 8:'Rhyolite', 9:'Trachybasalt', 10:'Basaltic trachyandesite', 11:'Trachyandesite', 12:'Trachyte-Trachydacite', 13:'Tephrite', 14:'Phonotephrite', 15:'Tephriphonolite'}

    path_parent = os.getcwd()
    with open(path_parent+'/TAS_Polygon/TASPolygons.pkl', 'rb') as f:
        polygons_df = pickle.load(f)

    num_polygons = len(polygons_df)
    class_patches = DF2Patch(polygons_df)

    synth_coord = np.array(list(zip(df['SiO2'],df['Alkalis'])))
    contains_total = np.zeros(len(synth_coord))

    for i in range(num_polygons):
        path_check = mpath.Path(polygons_df.iloc[i, polygons_df.columns.get_loc('Coordinates')])
        contains = path_check.contains_points(synth_coord) * (i+1)
        contains_total += contains

    df.loc[:, 'TAS_No'] = (contains_total).astype(int)

    df.loc[:, 'TAS_Label'] = df['TAS_No'].apply(lambda x: tas_label[x])

    return df

# %%

