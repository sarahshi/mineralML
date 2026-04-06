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
    "Si": "SiO2", "Ti": "TiO2", "Al": "Al2O3",
    "Fe2t": "FeOt", "Fe3t": "Fe2O3t",
    "Fe2": "FeO", "Fe3": "Fe2O3",
    "Mn": "MnO", "Mg": "MgO", "Ca": "CaO", "Na": "Na2O",
    "K": "K2O", "P": "P2O5", "Cr": "Cr2O3",
}

VALENCES = {
    "Si": 4, "Ti": 4, "Al": 3, "Fe2t": 2, "Fe3t": 3,
    "Fe2": 2, "Fe3": 3, "Mn": 2, "Mg": 2, "Ca": 2,
    "Na": 1, "K": 1, "P": 5, "Cr": 3, "Zr": 4, "Hf": 4,
}

TAS_CONFIG = {
    "name": "TAS",
    "axes": {"x": "SiO2", "y": "Na2O + K2O"},
    "fields": {
        "Bs": {
            "name": ["Basalt", "Gabbro"],
            "poly": [[45,0],[45,5],[52,5],[52,0]]
        },
        "F": {
            "name": ["Foidite", "Foidolite"],
            "poly": [[35,9],[37,14],[52.5,18],[52.5,14],[48.4,11.5],[45,9.4],[41,7],[41,3],[37,3]]
        },
        "nan": {"name": "N/A", "poly": []},
        "none": {"name": ["N/A"], "poly": []},
        "O1": {
            "name": ["Basaltic Andesite", "Gabbroic Diorite"],
            "label": ["Basaltic\nAndesite", "Gabbroic\nDiorite"],
            "poly": [[52,0],[52,5],[57,5.9],[57,0]]
        },
        "O2": {
            "name": ["Andesite", "Diorite"],
            "poly": [[57,0],[57,5.9],[63,7],[63,0]]
        },
        "O3": {
            "name": ["Dacite", "Granodiorite"],
            "poly": [[63,0],[63,7],[69,8],[77.3,0]]
        },
        "Pc": {
            "name": ["Picrite", "Peridotgabbro"],
            "poly": [[41,3],[45,3],[45,2],[45,0],[41,0]]
        },
        "Ph": {
            "name": ["Phonolite", "Foid Syenite"],
            "label": ["Phonolite", "Foid\nSyenite"],
            "poly": [[52.5,14],[52.5,18],[57,18],[63,16.2],[61,13.5],[57.6,11.7]]
        },
        "R": {
            "name": ["Rhyolite", "Granite"],
            "poly": [[69,8],[69,13],[85.9,6.8],[87.5,4.7],[77.3,0]]
        },
        "S1": {
            "name": ["Trachybasalt", "Foidolite"],
            "label": ["Trachy-\nbasalt", "Foidolite"],
            "poly": [[45,5],[49.4,7.3],[52,5]]
        },
        "S2": {
            "name": ["Basaltic Trachyandesite", "Foidolite"],
            "label": ["Basaltic\nTrachy-\nandesite", "Foidolite"],
            "poly": [[49.4,7.3],[53,9.3],[57,5.9],[52,5]]
        },
        "S3": {
            "name": ["Trachyandesite", "Foidolite"],
            "label": ["Trachy-\nandesite", "Foidolite"],
            "poly": [[53,9.3],[57.6,11.7],[61,8.6],[63,7],[57,5.9]]
        },
        "T1T2": {
            "name": ["Trachyte-Trachydacite", "Syenite"],
            "label": ["Trachyte/\nTrachy-\ndacite", "Syenite"],
            "poly": [[57.6,11.7],[61,13.5],[63,16.2],[69,13],[69,8],[63,7],[61,8.6]]
        },
        "U1": {
            "name": ["Tephrite", "Foid Gabbro"],
            "label": ["Tephrite", "Foid\nGabbro"],
            "poly": [[41,3],[41,7],[45,9.4],[49.4,7.3],[45,5],[45,3]]
        },
        "U2": {
            "name": ["Phonotephrite", "Foid Monzodiorite"],
            "label": ["Phono-\ntephrite", "Foid\nMonzo-\ndiorite"],
            "poly": [[45,9.4],[48.4,11.5],[53,9.3],[49.4,7.3]]
        },
        "U3": {
            "name": ["Tephriphonolite", "Foid Monzosyenite"],
            "label": ["Tephri-\nphonolite", "Foid\nMonzo-\nsyenite"],
            "poly": [[48.4,11.5],[52.5,14],[57.6,11.7],[53,9.3]]
        }
    }
}