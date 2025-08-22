# %%

# Oxides
OXIDES = ["SiO2", "TiO2", "Al2O3", "FeOt", "MnO", "MgO", "CaO", "Na2O", "K2O", "Cr2O3", "P2O5"]

# Molar masses (g/mol)
OXIDE_MASSES = {
    "SiO2": 60.08,
    "TiO2": 79.866,
    "Al2O3": 101.96,
    "FeOt": 71.844,
    "Fe2O3t": 159.69,
    "FeO": 71.844,
    "Fe2O3": 159.69,
    "MnO": 70.9374,
    "MgO": 40.3044,
    "CaO": 56.0774,
    "Na2O": 61.9789,
    "K2O": 94.196,
    "P2O5": 141.9445,
    "Cr2O3": 151.99,
}

# Number of oxygens per formula unit
OXYGEN_NUMBERS = {
    "SiO2": 2, "TiO2": 2, "Al2O3": 3, "FeOt": 1, "Fe2O3t": 3, "FeO": 1, "Fe2O3": 3,
    "MnO": 1, "MgO": 1, "CaO": 1, "Na2O": 1, "K2O": 1, "P2O5": 5, "Cr2O3": 3,
}

# Number of cations per formula unit
CATION_NUMBERS = {
    "SiO2": 1, "TiO2": 1, "Al2O3": 2, "FeOt": 1, "Fe2O3t": 2, "FeO": 1, "Fe2O3": 2,
    "MnO": 1, "MgO": 1, "CaO": 1, "Na2O": 2, "K2O": 2, "P2O5": 2, "Cr2O3": 2,
}

# Oxide to cation symbol mappings
OXIDE_TO_CATION_MAP = {
    "SiO2": "Si", "TiO2": "Ti", "Al2O3": "Al",
    "FeOt": "Fe2t", "Fe2O3t": "Fe3t",
    "FeO": "Fe2", "Fe2O3": "Fe3",
    "MnO": "Mn", "MgO": "Mg", "CaO": "Ca", "Na2O": "Na",
    "K2O": "K", "P2O5": "P", "Cr2O3": "Cr",
}

# Cation to oxide symbol mappings
CATION_TO_OXIDE_MAP = {
    'Si': 'SiO2', 'Ti': 'TiO2', 'Al': 'Al2O3',
    'Fe2t': 'FeOt', 'Fe3t': 'Fe2O3t',
    'Fe2': 'FeO', 'Fe3': 'Fe2O3',
    'Mn': 'MnO', 'Mg': 'MgO', 'Ca': 'CaO', 'Na': 'Na2O',
    'K': 'K2O', 'P': 'P2O5', 'Cr': 'Cr2O3',
}

VALENCES = {
    "Si": 4, "Ti": 4, "Al": 3, "Fe2t": 2, "Fe3t": 3,
    "Fe2": 2, "Fe3": 3, "Mn": 2, "Mg": 2, "Ca": 2,
    "Na": 1, "K": 1, "P": 5, "Cr": 3, "Zr": 4, "Hf": 4,
}

