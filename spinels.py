

def spinels(data_oxides): 
    # Set up  mass and charge data
    molar_mass = {'SiO2':60.08,'TiO2':79.866,'Al2O3':101.96,'FeO':71.844,'Fe2O3':159.69,
        'MnO':70.9374,'MgO':40.3044,'CaO':56.0774,'Cr2O3':151.99,'NiO':74.5928}
    cation_numbers = {'SiO2':1,'TiO2':1,'Al2O3':2,'FeO':1,'Fe2O3':2,'MnO':1,
                    'MgO':1,'CaO':1,'Cr2O3':2,'NiO':1} # numbers of cations per mole of oxide
    oxygen_numbers = {'SiO2':2,'TiO2':2,'Al2O3':3,'FeO':1,'Fe2O3':3,'MnO':1,'MgO':1,
                    'CaO':1,'Cr2O3':3,'NiO':1} # number of Os per mole of oxide
    cation_ratio = {'SiO2':0.5,'TiO2':0.5,'Al2O3':2/3,'FeO':1,'Fe2O3':2/3,'MnO':1,
                    'MgO':1,'CaO':1,'Cr2O3':2/3,'NiO':1} # ratio of cations to Os per mole of oxide
    cation = {'SiO2':'Si','TiO2':'Ti','Al2O3':'Al','FeO':'Fe2','Fe2O3':'Fe3',
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
        # cat_prop[oxide] = data_oxides[oxide]/molar_mass[oxide]*cation_numbers[oxide]

    O_prop['O_sum'] = O_prop.sum(axis = 1)
    orf = 4 / O_prop['O_sum'] 
    cat_prop_norm = cat_prop.mul(orf, axis=0)

    cat_prop_norm['cat_sum'] = cat_prop_norm.sum(axis = 1)
    cat_prop_norm = cat_prop_norm.fillna(0)
    cat_prop_norm['sum_charge'] = (2 * (cat_prop_norm["MgO"] + cat_prop_norm["MnO"] + cat_prop_norm["CaO"] + cat_prop_norm["NiO"])
                                + 3 * (cat_prop_norm["Al2O3"] + cat_prop_norm["Cr2O3"])
                                + 4 * (cat_prop_norm["TiO2"] + cat_prop_norm["SiO2"]))

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
    O_prop['Fe3'] = np.where(mask, (2*T)*(1-T/cation_allFe2.sum(axis=1)), 0)
    cation_pfu['Fe3'] = O_prop['Fe3']

    # Normalise cations to total expected cation number ('Fe2' is really total Fe)
    other_cations = ['Si','Ti','Al','Fe2','Mn','Mg','Ca','Cr','Ni']
    for cation in other_cations: 
        cation_pfu[cation] = cation_allFe2[cation+'_pfu']/cation_allFe2.sum(axis=1)*T
    
    # Now replace the Fe2 which is currently actually the total Fe
    cation_pfu['Fe2'] = cation_pfu['Fe2'] - O_prop['Fe3'] # np.where(cation_pfu['Fe2'] > 0, cation_pfu['Fe2'] - O_prop['Fe3'], 0)

    return cation_pfu 
