# %%

import numpy as np
import pandas as pd
from scipy import interpolate

from .constants import OXIDES, OXIDE_MASSES, OXYGEN_NUMBERS, CATION_NUMBERS, OXIDE_TO_CATION_MAP

# %%


class BaseMineralCalculator:
    """
    Base class for mineral composition calculations.
    Implement calculate_components() for each mineral.
    """
    
    # Constants for all minerals
    OXIDE_MASSES = OXIDE_MASSES
    OXYGEN_NUMBERS = OXYGEN_NUMBERS
    CATION_NUMBERS = CATION_NUMBERS
    OXIDE_TO_CATION_MAP = OXIDE_TO_CATION_MAP

    # Required subclass definitions
    OXYGEN_BASIS = None  # Oxygen normalization basis
    MINERAL_SUFFIX = None  # Abbreviated mineral suffix
    
    def __init__(self, comps):
        """Initialize with mineral compositions."""
        # Determine oxide columns with suffix
        self.oxide_cols = [
            oxide for oxide in self.OXIDE_MASSES if oxide in comps.columns
        ]
        oxide_cols = self.oxide_cols
        # Keep non-numeric or non-oxide metadata
        self.metadata = comps.drop(columns=self.oxide_cols, errors="ignore")

        self.comps = comps[oxide_cols].clip(lower=0).copy()
        self._validate_subclass()

        _FeOt = 'FeOt' in self.oxide_cols and self.comps['FeOt'].notna().any()
        _Fe2O3t = 'Fe2O3t' in self.oxide_cols and self.comps['Fe2O3t'].notna().any()
        _FeO = 'FeO' in self.oxide_cols and self.comps['FeO'].notna().any()
        _Fe2O3 = 'Fe2O3' in self.oxide_cols and self.comps['Fe2O3'].notna().any()

        if _FeOt and (_FeO or _Fe2O3):
            raise ValueError("Mixing 'FeOt' with 'FeO' and 'Fe2O3'. Provide only 'FeOt', 'Fe2O3t', or both 'FeO' and 'Fe2O3'.")
        if _Fe2O3t and (_FeO or _Fe2O3):
            raise ValueError("Mixing 'Fe2O3t' with 'FeO' and 'Fe2O3'. Provide only 'Fe2O3t', 'FeOt', or both 'FeO' and 'Fe2O3'.")
        if (_FeO != _Fe2O3):
            raise ValueError("If using 'FeO' and 'Fe2O3', both must be provided.")

    def _validate_subclass(self):
        """Check if subclass defined required constants."""
        if self.OXYGEN_BASIS is None:
            raise NotImplementedError("Subclass must define OXYGEN_BASIS")
        if self.MINERAL_SUFFIX is None:
            raise NotImplementedError("Subclass must define MINERAL_SUFFIX")

    def _add_suffix(self, oxide_name):
        """Helper method to add mineral suffix to oxide names, if not already present."""
        if oxide_name.endswith(self.MINERAL_SUFFIX):
            return oxide_name
        return f"{oxide_name}{self.MINERAL_SUFFIX}"

    def _remove_suffix(self, col_name):
        """Helper method to remove mineral suffix from column names, if not already present."""
        if not self.MINERAL_SUFFIX:
            return col_name
        return col_name.replace(self.MINERAL_SUFFIX, "")

    def calculate_moles(self):
        """Calculate moles of each oxide component."""
        valid_cols = [oxide for oxide in self.OXIDE_MASSES if oxide in self.comps.columns]
        if not valid_cols:
            return pd.DataFrame(index=self.comps.index)

        oxide_masses = pd.Series(self.OXIDE_MASSES)
        moles = (
            self.comps[valid_cols]
            .fillna(0)
            .rename(columns=self._remove_suffix)
            .div(oxide_masses)
        )

        return moles.add_suffix('_mols')

    def calculate_oxygens(self):
        """Calculate number of oxygens for each oxide component."""
        moles = self.calculate_moles()
        if moles.empty:
            return pd.DataFrame(index=self.comps.index)

        # Strip "_mols" to match OXYGEN_NUMBERS keys
        moles.columns = [col.replace("_mols", "") for col in moles.columns]
        oxygen_numbers = pd.Series(self.OXYGEN_NUMBERS)

        # Filter only columns present in OXYGEN_NUMBERS
        moles = moles.loc[:, moles.columns.intersection(oxygen_numbers.index)]

        oxygens = moles.multiply(oxygen_numbers, axis="columns")
        return oxygens.add_suffix('_ox')

    def calculate_cations(self, fractions=False):
        """Calculate cations on the basis of the mineral's oxygen basis."""
        moles = self.calculate_moles()
        moles.columns = [col.replace("_mols", "") for col in moles.columns]

        if moles.empty:
            return pd.DataFrame(index=self.comps.index)

        # Renormalize
        oxygens = self.calculate_oxygens()
        renorm_factor = self.OXYGEN_BASIS / oxygens.sum(axis=1)
        mols_renorm = moles.multiply(renorm_factor, axis="rows")

        # Multiply by cation numbers
        cation_numbers = pd.Series(self.CATION_NUMBERS)
        cations = mols_renorm.multiply(cation_numbers, axis="columns")

        # Rename columns
        cation_cols = [f"{col}_cat_{self.OXYGEN_BASIS}ox" for col in cations.columns]
        cations.columns = cation_cols
        cations = cations.rename(columns={
            f"{oxide}_cat_{self.OXYGEN_BASIS}ox": f"{cation}_cat_{self.OXYGEN_BASIS}ox"
            for oxide, cation in self.OXIDE_TO_CATION_MAP.items()
        })

        if fractions:
            # Compute fractions and add them to the same DataFrame
            frac_df = cations.div(cations.sum(axis=1), axis=0)
            frac_df.columns = [col.replace(f"_cat_{self.OXYGEN_BASIS}ox", f"_frac_{self.OXYGEN_BASIS}ox") for col in frac_df.columns]
            return pd.concat([cations, frac_df], axis=1)

        return cations

    def calculate_all(self):
        """Calculate and combine all properties in one DataFrame."""
        idx = self.metadata.index
        first = self.metadata.reindex(
            index=idx,
            columns=['Sample Name'],
            fill_value=np.nan
        )
        last = self.metadata.reindex(
            index=idx,
            columns=['Mineral','Source'],
            fill_value=np.nan
        )

        moles = self.calculate_moles()
        oxygens = self.calculate_oxygens()
        cations = self.calculate_cations()

        # return columns in order of self.OXIDE_MASSES
        oxide_cols = [ox for ox in self.OXIDE_MASSES.keys() if ox in self.comps.columns]

        # corresponding mol‐ and ox‐ columns
        mole_cols   = [f"{ox}_mols" for ox in oxide_cols]
        oxygen_cols = [f"{ox}_ox" for ox in oxide_cols]
        # map each oxide → its cation column name and suffix
        cation_cols = [
            f"{self.OXIDE_TO_CATION_MAP[ox]}_cat_{self.OXYGEN_BASIS}ox"
            for ox in oxide_cols
            if f"{self.OXIDE_TO_CATION_MAP[ox]}_cat_{self.OXYGEN_BASIS}ox" in cations.columns
        ]

        df = pd.concat([first, 
                        self.comps[oxide_cols], 
                        moles[mole_cols], 
                        oxygens[oxygen_cols], 
                        cations[cation_cols], 
                        last],
                       axis=1)

        return df


# %%


def oxide_to_element(df):
    """
    Convert between oxide wt% and elemental wt%.

    Parameters:
        df (pd.DataFrame): DataFrame with oxide or elemental wt% columns.
        direction (str): 'oxide_to_element' or 'element_to_oxide'.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 
            - DataFrame with elemental wt% columns.
            - Series mapping each element to its conversion factor from oxide wt%.
    """
    # Oxide molar masses and corresponding element info
    oxide_data = {
        'SiO2': {'element': 'Si', 'oxide_mass': 60.0843, 'element_mass': 28.0855, 'stoich': 1},
        'TiO2': {'element': 'Ti', 'oxide_mass': 79.866, 'element_mass': 47.867, 'stoich': 1},
        'Al2O3': {'element': 'Al', 'oxide_mass': 101.961, 'element_mass': 26.9815, 'stoich': 2},
        'FeOt': {'element': 'Fe', 'oxide_mass': 71.844, 'element_mass': 55.845, 'stoich': 1},
        'MnO': {'element': 'Mn', 'oxide_mass': 70.9374, 'element_mass': 54.938, 'stoich': 1},
        'MgO': {'element': 'Mg', 'oxide_mass': 40.3044, 'element_mass': 24.305, 'stoich': 1},
        'CaO': {'element': 'Ca', 'oxide_mass': 56.0774, 'element_mass': 40.078, 'stoich': 1},
        'Na2O': {'element': 'Na', 'oxide_mass': 61.9789, 'element_mass': 22.989, 'stoich': 2},
        'K2O': {'element': 'K', 'oxide_mass': 94.196, 'element_mass': 39.0983, 'stoich': 2},
        'P2O5': {'element': 'P', 'oxide_mass': 141.944, 'element_mass': 30.974, 'stoich': 2},
        'Cr2O3': {'element': 'Cr', 'oxide_mass': 151.99, 'element_mass': 51.996, 'stoich': 2},
        'NiO': {'element': 'Ni', 'oxide_mass': 74.6928, 'element_mass': 58.693, 'stoich': 1},
        'SO2': {'element': 'S', 'oxide_mass': 64.066, 'element_mass': 32.065, 'stoich': 1},
        'ZrO2':  {'element': 'Zr', 'oxide_mass': 123.218, 'element_mass': 91.224, 'stoich': 1},
    }

    result = {}
    factors = {}

    for oxide, info in oxide_data.items():
        if oxide in df.columns:
            element = info['element']
            conversion_factor = (info['stoich'] * info['element_mass']) / info['oxide_mass']
            result[element] = df[oxide] * conversion_factor + result.get(element, 0)
            factors[oxide] = conversion_factor

    result_df = pd.DataFrame(result, index=df.index)

    return result_df, pd.Series(factors)

def element_to_oxide(df):
    """
    Convert elemental wt% to oxide wt%, and return the conversion factors used.

    Parameters:
        df (pd.DataFrame): DataFrame with elemental wt% columns.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: 
            - DataFrame with oxide wt% columns.
            - Series mapping each oxide to its conversion factor from element wt%.
    """
    oxide_data = {
        'SiO2': {'element': 'Si', 'oxide_mass': 60.0843, 'element_mass': 28.0855, 'stoich': 1},
        'TiO2': {'element': 'Ti', 'oxide_mass': 79.866,  'element_mass': 47.867,  'stoich': 1},
        'Al2O3': {'element': 'Al', 'oxide_mass': 101.961, 'element_mass': 26.9815, 'stoich': 2},
        'FeOt': {'element': 'Fe', 'oxide_mass': 71.844,   'element_mass': 55.845,  'stoich': 1},
        'MnO':  {'element': 'Mn', 'oxide_mass': 70.9374,  'element_mass': 54.938,  'stoich': 1},
        'MgO':  {'element': 'Mg', 'oxide_mass': 40.3044,  'element_mass': 24.305,  'stoich': 1},
        'CaO':  {'element': 'Ca', 'oxide_mass': 56.0774,  'element_mass': 40.078,  'stoich': 1},
        'Na2O': {'element': 'Na', 'oxide_mass': 61.9789,  'element_mass': 22.989,  'stoich': 2},
        'K2O':  {'element': 'K',  'oxide_mass': 94.196,   'element_mass': 39.0983, 'stoich': 2},
        'P2O5': {'element': 'P',  'oxide_mass': 141.944,  'element_mass': 30.974,  'stoich': 2},
        'Cr2O3':{'element': 'Cr', 'oxide_mass': 151.99,   'element_mass': 51.996,  'stoich': 2},
        'NiO':  {'element': 'Ni', 'oxide_mass': 74.6928,  'element_mass': 58.693,  'stoich': 1},
        'SO2':  {'element': 'S',  'oxide_mass': 64.066,   'element_mass': 32.065,  'stoich': 1},
        'ZrO2':  {'element': 'Zr',  'oxide_mass': 123.218,   'element_mass': 91.224,  'stoich': 1},
    }

    result = {}
    factors = {}

    for oxide, info in oxide_data.items():
        element = info['element']
        if element in df.columns:
            conversion_factor = info['oxide_mass'] / (info['stoich'] * info['element_mass'])
            result[oxide] = df[element] * conversion_factor + result.get(oxide, 0)
            factors[oxide] = conversion_factor

    return pd.DataFrame(result, index=df.index), pd.Series(factors)


# %%


class AmphiboleCalculator(BaseMineralCalculator):
    """Amphibole-specific calculations."""
    OXYGEN_BASIS = 23
    CATION_BASIS = 13
    MINERAL_SUFFIX = "_Amp"

    OXIDE_MASSES = dict(BaseMineralCalculator.OXIDE_MASSES, **{"F": 18.998403, "Cl": 35.453})
    OXYGEN_NUMBERS = dict(BaseMineralCalculator.OXYGEN_NUMBERS, **{"F": 0, "Cl": 0})
    CATION_NUMBERS = dict(BaseMineralCalculator.CATION_NUMBERS, **{"F": 1, "Cl": 1})
    OXIDE_TO_CATION_MAP = dict(BaseMineralCalculator.OXIDE_TO_CATION_MAP, **{"F": "F", "Cl": "Cl"})

    def calculate_components(self):
        """Return complete amphibole composition with site assignments."""
        base = self.calculate_all()
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"
        cat_norm_13_suffix = f"_{self.CATION_BASIS}cat"

        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mg = base[f"Mg{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["Cation_Sum_Si_Mg"] = Si + Ti + Al + Cr + Fe + Mn + Mg
        sites["Cation_Sum_Si_Ca"] = Si + Ca
        sites["Cation_Sum_Amp"] = Si + Ca + Na + K
        sites["XMg"] = Mg / (Mg + Fe)
    
        cat_norm_13, ridolfi_sites = self.calculate_ridolfi_sites(
            base=base,
            sites=sites,
            cat_suffix=cat_suffix,
            cat_norm_13_suffix=cat_norm_13_suffix,
            cation_basis=self.CATION_BASIS
        )

        leake_sites = self.calculate_leake_sites(
            base=base,
            sites=sites,
            cat_suffix=cat_suffix,
            cat_norm_13_suffix=cat_norm_13_suffix,
            cation_basis=self.CATION_BASIS
        )

        return pd.concat([
            base,
            sites,
            ridolfi_sites.add_suffix("_ridolfi"),
            cat_norm_13.add_suffix("_ridolfi_norm"),
            leake_sites.add_suffix("_leake")
        ], axis=1)

    def calculate_ridolfi_sites(self, base, sites, cat_suffix, cat_norm_13_suffix, cation_basis):
        """Compute Ridolfi 13-cation normalized site assignments and perform Ridolfi-style recalc checks."""
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]
        cat_norm_13 = cation_basis * base[cation_cols].div(sites["Cation_Sum_Si_Mg"], axis=0)
        cat_norm_13.columns = [col.replace(cat_suffix, cat_norm_13_suffix) for col in cat_norm_13.columns]

        # H2O and O=F,Cl corrections
        sites['H2O_calc'] = (
            (2 - base.get("F", 0) - base.get("Cl", 0)) *
            sites["Cation_Sum_Si_Mg"] * 17 / self.CATION_BASIS / 2
        )
        sites['O=F,Cl'] = -(
            base.get("F", 0) * 0.421070639014633 +
            base.get("Cl", 0) * 0.225636758525372
        )

        # Fe3+ and Fe2+ recalculated amounts
        charge = (
            cat_norm_13.get(f"Si{cat_norm_13_suffix}", 0) * 4 +
            cat_norm_13.get(f"Ti{cat_norm_13_suffix}", 0) * 4 +
            cat_norm_13.get(f"Al{cat_norm_13_suffix}", 0) * 3 +
            cat_norm_13.get(f"Cr{cat_norm_13_suffix}", 0) * 3 +
            cat_norm_13.get(f"Fe2t{cat_norm_13_suffix}", 0) * 2 +
            cat_norm_13.get(f"Mn{cat_norm_13_suffix}", 0) * 2 +
            cat_norm_13.get(f"Mg{cat_norm_13_suffix}", 0) * 2 +
            cat_norm_13.get(f"Ca{cat_norm_13_suffix}", 0) * 2 +
            cat_norm_13.get(f"Na{cat_norm_13_suffix}", 0) +
            cat_norm_13.get(f"K{cat_norm_13_suffix}", 0)
        )
        sites['Charge'] = charge
        fe3 = (46 - charge).clip(lower=0)
        fe2 = (cat_norm_13.get(f"Fe2t{cat_norm_13_suffix}", 0) - fe3).clip(lower=0)
        sites['Fe3_calc'] = fe3
        sites['Fe2_calc'] = fe2
        sites['Fe2O3_calc'] = fe3 * sites["Cation_Sum_Si_Mg"] * 159.691 / self.CATION_BASIS / 2
        sites['FeO_calc'] = fe2 * sites["Cation_Sum_Si_Mg"] * 71.846 / self.CATION_BASIS

        # Recalculated total
        oxide_cols = [col for col in base.columns if col.endswith("_Amp") and not col.endswith(cat_suffix)]
        sites["Sum_input"] = base[oxide_cols].sum(axis=1)
        sites["Total_recalc"] = (
            sites["Sum_input"]
            - base.get("FeOt_Amp", 0)
            + sites["H2O_calc"]
            + sites['Fe2O3_calc']
            + sites['FeO_calc']
            + sites["O=F,Cl"]
        )

        # Input checks
        sites["Fail Msg"] = ""
        sites["Input_Check"] = True
        sites.loc[sites["Sum_input"] < 90, ["Input_Check", "Fail Msg"]] = [False, "Cation oxide Total<90"]
        sites.loc[sites["Total_recalc"] < 98.5, ["Input_Check", "Fail Msg"]] = [False, "Recalc Total<98.5"]
        sites.loc[sites["Total_recalc"] > 102, ["Input_Check", "Fail Msg"]] = [False, "Recalc Total>102"]
        sites.loc[sites["Charge"] > 46.5, ["Input_Check", "Fail Msg"]] = [False, "unbalanced charge (>46.5)"]
        sites["Fe2_C"] = fe2
        sites.loc[sites["Fe2_C"] < 0, ["Input_Check", "Fail Msg"]] = [False, "unbalanced charge (Fe2<0)"]
        sites["Mgno_Fe2"] = cat_norm_13.get(f"Mg{cat_norm_13_suffix}", 0) / (
            cat_norm_13.get(f"Mg{cat_norm_13_suffix}", 0) + fe2
        )
        sites["Mgno_FeT"] = cat_norm_13.get(f"Mg{cat_norm_13_suffix}", 0) / (
            cat_norm_13.get(f"Mg{cat_norm_13_suffix}", 0) + cat_norm_13.get(f"Fe2t{cat_norm_13_suffix}", 0)
        )
        sites.loc[100 * sites["Mgno_Fe2"] < 54, ["Input_Check", "Fail Msg"]] = [False, "Low Mg# (<54)"]
        sites["Ca_B"] = cat_norm_13.get(f"Ca{cat_norm_13_suffix}", 0)
        sites.loc[sites["Ca_B"] < 1.5, ["Input_Check", "Fail Msg"]] = [False, "Low Ca (<1.5)"]
        sites.loc[sites["Ca_B"] > 2.05, ["Input_Check", "Fail Msg"]] = [False, "High Ca (>2.05)"]
        sites["Na_calc"] = 2 - sites["Ca_B"]
        sites.loc[cat_norm_13.get(f"Na{cat_norm_13_suffix}", 0) < sites["Na_calc"], "Na_calc"] = cat_norm_13.get(f"Na{cat_norm_13_suffix}", 0)
        sites["B_Sum"] = sites["Na_calc"] + sites["Ca_B"]
        sites.loc[sites["B_Sum"] < 1.99, ["Input_Check", "Fail Msg"]] = [False, "Low B Cations"]

        sites["Na_A"] = (cat_norm_13.get(f"Na{cat_norm_13_suffix}", 0) - (2 - sites["Ca_B"]).clip(lower=0))
        sites["K_A"] = cat_norm_13.get(f"K{cat_norm_13_suffix}", 0)
        sites["A_Sum"] = sites["Na_A"] + sites["K_A"]

        # Classification logic
        sites["Classification"] = "N/A"
        lowCa = sites["Ca_B"] < 1.5
        LowMgno = sites["Mgno_Fe2"] < 0.5
        MgHbl = cat_norm_13.get(f"Si{cat_norm_13_suffix}", 0) >= 6.5
        Kaer = (cat_norm_13.get(f"Ti{cat_norm_13_suffix}", 0) - (8 - cat_norm_13.get(f"Si{cat_norm_13_suffix}", 0) - (8 - cat_norm_13.get(f"Si{cat_norm_13_suffix}", 0)))).clip(lower=0) > 0.5
        Tsh = sites["A_Sum"] < 0.5
        MgHast = sites["Fe3_calc"] > (cat_norm_13.get(f"Al{cat_norm_13_suffix}", 0) - (8 - cat_norm_13.get(f"Si{cat_norm_13_suffix}", 0)))

        sites.loc[lowCa, "Classification"] = "low-Ca"
        sites.loc[(~lowCa) & LowMgno, "Classification"] = "low-Mg"
        sites.loc[(~lowCa) & (~LowMgno) & MgHbl, "Classification"] = "Mg-Hornblende"
        sites.loc[(~lowCa) & (~LowMgno) & (~MgHbl) & Kaer, "Classification"] = "kaersutite"
        sites.loc[(~lowCa) & (~LowMgno) & (~MgHbl) & (~Kaer) & Tsh, "Classification"] = "Tschermakitic pargasite"
        sites.loc[(~lowCa) & (~LowMgno) & (~MgHbl) & (~Kaer) & (~Tsh) & MgHast, "Classification"] = "Mg-hastingsite"
        sites.loc[(~lowCa) & (~LowMgno) & (~MgHbl) & (~Kaer) & (~Tsh) & (~MgHast), "Classification"] = "Pargasite"

        return cat_norm_13, sites

    def calculate_leake_sites(self, base, sites, cat_suffix, cat_norm_13_suffix, cation_basis):

        sites = pd.DataFrame(index=base.index, dtype=float)
        columns = ["Si_T", "Al_T", "Al_C", "Ti_C", "Mg_C", "Fe2t_C", 
                   "Mn_C", "Cr_C", "Mg_B", "Fe2t_B", "Mn_B", "Na_B", 
                   "Ca_B", "Na_A", "K_A", "Ca_A"]
        for col in columns:
            sites[col] = 0.0

        sites["Si_T"] = base.get(f"Si{cat_suffix}", 0)
        sites["Ti_C"] = base.get(f"Ti{cat_suffix}", 0)
        sites["Cr_C"] = base.get(f"Cr{cat_suffix}", 0)
        sites["Ca_B"] = base.get(f"Ca{cat_suffix}", 0)
        sites["K_A"] = base.get(f"K{cat_suffix}", 0)

        total_T = sites["Si_T"] + base.get(f"Al{cat_suffix}", 0)
        mask_excess_T = total_T > 8
        sites.loc[mask_excess_T, "Al_T"] = (8 - sites["Si_T"]).clip(lower=0)
        sites.loc[mask_excess_T, "Al_C"] = base.get(f"Al{cat_suffix}", 0) - sites["Al_T"]
        mask_deficient_T = total_T <= 8
        sites.loc[mask_deficient_T, "Al_T"] = base.get(f"Al{cat_suffix}", 0)
        sites.loc[mask_deficient_T, "Al_C"] = 0

        prefilled_C = sites["Al_C"] + sites["Ti_C"] + sites["Cr_C"]
        room_left = 5 - prefilled_C

        for ion in ["Mg", "Fe2t", "Mn"]:
            amt = base.get(f"{ion}_cat_23ox", 0)
            col = f"{ion}_C"
            alloc = room_left.where(amt >= room_left, amt)
            sites[col] = alloc.clip(lower=0)
            room_left -= sites[col]

        for ion in ["Mg", "Fe2t", "Mn"]:
            residual = base.get(f"{ion}_cat_23ox", 0) - sites.get(f"{ion}_C", 0)
            sites[f"{ion}_B"] = residual.clip(lower=0)

        sum_B = sites["Mg_B"] + sites["Fe2t_B"] + sites["Mn_B"] + sites["Ca_B"]
        fill_B = 2 - sum_B
        enough_Na = base.get(f"Na{cat_suffix}", 0) >= fill_B
        sites["Na_B"] = fill_B.where(enough_Na, base.get(f"Na{cat_suffix}", 0))
        sites["Na_A"] = (base.get(f"Na{cat_suffix}", 0) - sites["Na_B"]).clip(lower=0)

        sites["Sum_T"] = sites["Al_T"] + sites["Si_T"]
        sites["Sum_C"] = sites["Al_C"] + sites["Cr_C"] + sites["Mg_C"] + sites["Fe2t_C"] + sites["Mn_C"]
        sites["Sum_B"] = sites["Mg_B"] + sites["Fe2t_B"] + sites["Mn_B"] + sites["Ca_B"] + sites["Na_B"]
        sites["Sum_A"] = sites["K_A"] + sites["Na_A"]

        sites["Cation_Sum"] = sites["Sum_T"] + sites["Sum_C"] + sites["Sum_B"] + sites["Sum_A"]
        mg = base[f"Mg{cat_suffix}"]
        fet = base[f"Fe2t{cat_suffix}"]
        denom = mg + fet
        sites["Mgno"] = mg / denom.replace(0, np.nan)

        return sites


class AmphiboleClassifier(AmphiboleCalculator):
    """General amphibole calculations for classification and plotting."""
    OXYGEN_BASIS = 23
    MINERAL_SUFFIX = "_Amp"

    def _classify_subamphibole(self, x, y, eps=1e-9):
        """
        x = Si (apfu), y = Mg#  (numpy arrays)
        returns: np.array of subtype strings
        """
        sub = np.full(x.shape, "OOD", dtype=object) # out of domain

        # Valid domain (finite, within ranges with slack)
        valid = (
            np.isfinite(x) & np.isfinite(y) &
            (y >= 0.0 - eps) & (y <= 1.0 + eps) &
            (x >= 5.5 - eps) & (x <= 8.0 + eps)
        )
        if not valid.any():
            return sub

        xv = x[valid]; yv = y[valid]
        out = np.full(xv.shape, "Unlabeled", dtype=object)

        # Mg# bands
        top = (yv >= 0.90 - eps)
        mid = (yv >= 0.50 - eps) & (yv < 0.90 - eps)
        bot = (yv < 0.50 - eps)

        # Si bins
        si_hi  = (xv >= 7.5 - eps) & (xv <= 8.0 + eps)   # [7.5, 8.0]
        si_mid = (xv >= 6.5 - eps) & (xv <  7.5 - eps)   # [6.5, 7.5)
        si_lo  = (xv >= 5.5 - eps) & (xv <  6.5 - eps)   # [5.5, 6.5)

        # Assign (order doesn't matter; masks are disjoint)
        out[top & si_hi]  = "Tremolite"
        out[top & si_mid] = "Magnesiohornblende"
        out[top & si_lo]  = "Tschermakite"

        out[mid & si_hi]  = "Actinolite"
        out[mid & si_mid] = "Magnesiohornblende"
        out[mid & si_lo]  = "Tschermakite"

        out[bot & si_hi]  = "Ferroactinolite"
        out[bot & si_mid] = "Ferrohornblende"
        out[bot & si_lo]  = "Ferrotschermakite"

        # Stitch back into full array
        sub[valid] = out
    
        return sub

    def classify(self, subclass=True, eps=1e-9):
        """
        Classify amphibole analyses using Leake-style Si (apfu) and Mg# produced by
        AmphiboleCalculator.calculate_components().

        Returns a DataFrame with `Mineral` and (optionally) `Submineral`.
        """
        comps = super().calculate_components()

        # Require the calculator to have produced these; graceful fallback to NaN if missing
        si = comps.get("Si_T_leake", np.nan).to_numpy()
        mgno = comps.get("Mgno_leake", np.nan).to_numpy()

        if subclass:
            # Any rows with NaN inputs become "Unlabeled"
            mask_valid = np.isfinite(si) & np.isfinite(mgno)
            subs = np.full(si.shape, "Unlabeled", dtype=object)
            if mask_valid.any():
                subs[mask_valid] = self._classify_subamphibole(si[mask_valid], mgno[mask_valid], eps=eps)
        else:
            subs = np.array([None] * len(si), dtype=object)

        df_class = comps.copy()
        df_class["Mineral"] = "Amphibole"
        if subclass:
            df_class["Submineral"] = subs

        return df_class

    def plot(self, df_class=None, subclass=True, figsize=(10, 6), hue=None):

        import matplotlib.pyplot as plt

        if df_class is None:
            df_class = self.classify(subclass=subclass)
        x, y = df_class["Si_T_leake"], df_class["Mgno_leake"]
        # Default hue is Submineral
        if hue is None:
            hue = "Submineral"

        # Ratios for plotting
        fig, ax = plt.subplots(figsize=figsize)

        # Spinel grid definition
        ax.hlines(y=0.5, xmin=5.48, xmax=8.02, color='k', lw=1, zorder=0)
        ax.hlines(y=0.9, xmin=7.5, xmax=8.02, color='k', lw=1, zorder=0)
        ax.vlines(x=7.5, ymin=-0.02, ymax=1.02, color='k', lw=1, zorder=0)
        ax.vlines(x=6.5, ymin=-0.02, ymax=1.02, color='k', lw=1, zorder=0)

        fs = 14
        ax.text(7.75, 0.950, "Tremolite", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(7.75, 0.775, "Actinolite", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(7.75, 0.250, "Ferroactinolite", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(7.00, 0.75, "Magnesiohornblende", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(7.00, 0.25, "Ferrohornblende", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(6.00, 0.75, "Tschermakite", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(6.00, 0.25, "Ferrotschermakite", fontsize=fs, ha="center", va="center", zorder=30)

        # Color by hue using tab10; keep in sorted order
        cmap = plt.get_cmap("tab10")
        if hue in df_class.columns:
            groups = sorted(df_class[hue].astype(str).unique())
            for i, g in enumerate(groups):
                m = (df_class[hue].astype(str) == g).to_numpy()
                ax.scatter(x[m], y[m], label=g, s=30, alpha=0.6,
                           edgecolors="k", linewidth=0.5, color=cmap(i % 10), zorder=20)
            # Put legend OUTSIDE 
            fig.subplots_adjust(right=0.9)  # make room on the right
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=9)
        else:
            ax.scatter(x, y, s=30, alpha=0.6, edgecolors="k", linewidth=0.5,
                       color=cmap(0), zorder=20)

        ax.set_xlim(5.48, 8.02)
        ax.set_ylim(-0.02, 1.02)
        ax.invert_xaxis()
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=5)
        ax.set_xlabel("Si (apfu)")
        ax.set_ylabel("Mg# Amphibole")
        fig.tight_layout()

        return fig, ax


class ApatiteCalculator(BaseMineralCalculator):
    """Apatite-specific calculations. Ca5(PO4)3(F,OH,Cl)."""
    OXYGEN_BASIS = 13
    MINERAL_SUFFIX = "_Ap"

    def calculate_components(self):
        """Return complete apatite composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        P = base[f"P{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Na = base.get(f"Na{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        # https://www.mindat.org/min-274.html
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["M_site"] = Ca + Mn + Na # Pb, Ba, Sr, Ce, La, Y, Bi
        sites["T_site"] = P + Si # As, V, S, B
        # sites["X_site"] = F + Cl + OH
        sites["Ca_P"] = Ca + P

        return pd.concat([base, sites], axis=1)


class BiotiteCalculator(BaseMineralCalculator):
    """Biotite-specific calculations. XM^{2+}3[Si3Al]010(OH)2."""
    OXYGEN_BASIS = 11
    MINERAL_SUFFIX = "_Bt"

    def calculate_components(self):
        """Return complete biotite composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mg = base[f"Mg{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["X_site"] = K + Na + Ca # Ba, Rb, Cs
        sites["M_site"] = Mg + Fe # M2+, octahedral
        sites["M_site_expanded"] = Mg + Fe + Mn + Ti # Fe3+, Li, octahedral
        sites["T_site"] = Si + Al # tetrahedral

        return pd.concat([base, sites], axis=1)


class CalciteCalculator(BaseMineralCalculator):
    """Calcite-specific calculations. CaCO3."""
    OXYGEN_BASIS = 3
    MINERAL_SUFFIX = "_Cal"

    # Extend the parent's dictionaries by merging them with CO2 data
    OXIDE_MASSES = dict(BaseMineralCalculator.OXIDE_MASSES, **{"CO2": 44.009})
    OXYGEN_NUMBERS = dict(BaseMineralCalculator.OXYGEN_NUMBERS, **{"CO2": 2})
    CATION_NUMBERS = dict(BaseMineralCalculator.CATION_NUMBERS, **{"CO2": 1})
    OXIDE_TO_CATION_MAP = dict(BaseMineralCalculator.OXIDE_TO_CATION_MAP, **{"CO2": "C"})

    def calculate_components(self):
        """Return complete calcite composition with site assignments."""
        moles = self.calculate_moles()  
        mol_suffix = f"_mols"
        self.comps["CO2"] = 44.009 * (moles[f"CaO{mol_suffix}"] + moles[f"MgO{mol_suffix}"] + \
                                      moles[f"MnO{mol_suffix}"] + moles[f"FeOt{mol_suffix}"])

        if "CO2" not in self.oxide_cols:
            self.oxide_cols.append("CO2")

        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]
        Ca = base[f"Ca{cat_suffix}"]
        Mg = base.get(f"Mg{cat_suffix}", 0)
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Fe = base.get(f"Fe2t{cat_suffix}", 0)
        C = base.get(f"C{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1) - C
        sites["M_site"] = Ca
        sites["M_site_expanded"] = Ca + Mg + Mn + Fe
        sites['C_site'] = C

        return pd.concat([base, sites], axis=1)


class ChloriteCalculator(BaseMineralCalculator):
    """Chlorite-specific calculations. (Mg,Fe)10Al2[Al2Si6O20](OH)16"""
    OXYGEN_BASIS   = 14
    MINERAL_SUFFIX = "_Chl"

    def calculate_components(self):
        """Return complete chlorite composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base[f"Mg{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["VII_site"] = Ca + Mg + Na + K # Fe2, seven coordinated
        sites["T_site"] = Si + Al # tetrahedral
        sites["Al_IV"] = 4 - Si
        sites["Al_VI"] = Al - sites["Al_IV"]

        sites["M_site"] = sites["Al_VI"] + Ti + Fe + Mn + Mg # Fe3, octahedral 
        sites["M1_vacancy"] = (sites["Al_VI"] - sites["Al_IV"]) / 2
        sites["XMg"] = Mg / (Mg + Fe)

        return pd.concat([base, sites], axis=1)


class ClinopyroxeneCalculator(BaseMineralCalculator):
    """Clinopyroxene-specific calculations. Ca(Mg,Fe)Si2O6."""
    OXYGEN_BASIS = 6
    MINERAL_SUFFIX = "_Cpx"

    def calculate_components(self):
        """Return complete clinopyroxene composition with site assignments and enstatite, ferrosilite, wollastonite, iron assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base[f"Mg{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        # sites = sites.reset_index(drop=True)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["M_site"] = Mg + Fe + Ca + Na + Ti + Cr
        sites["T_site"] = Si + Al
        sites["XMg"] = (Mg / (Mg + Fe))
        sites["En"] = Mg / (Mg + Fe + Ca)
        sites["Fs"] = Fe / (Mg + Fe + Ca)
        sites["Wo"] = Ca / (Mg + Fe + Ca)
        sites["Jd"] = Na

        # This code is modified from Thermobar, with permission from Penny Wieser
        sites["Al_IV"] = 2 - Si
        sites["Al_VI"] = Al - sites["Al_IV"]
        sites["Al_VI"] = sites["Al_VI"].clip(lower=0) # Al_VI can't be negative

        # Fe3+, Fe2+ Calculation
        sites["Fe3_Lindley"] = (
            Na + sites["Al_IV"]
            - sites["Al_VI"]
            - (2 * Ti) - Cr
        ).clip(lower=0, upper=Fe) # Fe3 can't be negative or greater than Fe
        sites.loc[sites["Fe3_Lindley"] < 1e-10, "Fe3_Lindley"] = 0
        sites["Fe2_Lindley"] = Fe - sites["Fe3_Lindley"]
        sites["Fe3_prop_Lindley"] = (sites["Fe3_Lindley"] / Fe).replace(0, np.nan)

        # Independent cpx components
        sites["CrCaTs"] = 0.5 * Cr
        sites['a_cpx_En'] = (
            (1 - Ca - Na - K) * 
            (1 - 0.5 * (Al + Cr + Na + K))
        )

        # If value of AlVI < Na cation fraction
        sites["CaTs"] = sites["Al_VI"] - Na
        CaTs_mask = sites["CaTs"] < 0
        sites["Jd_from 0=Na, 1=Al"] = 0
        sites.loc[CaTs_mask, "Jd_from 0=Na, 1=Al"] = 1
        sites.loc[CaTs_mask, "Jd"] = sites.loc[CaTs_mask, "Al_VI"].to_numpy()
        sites["CaTs"] = sites["CaTs"].clip(lower=0)

        # CaTi component
        sites["CaTi"] = ((sites["Al_IV"] - sites["CaTs"]) / 2).clip(lower=0)

        # DiHd (Diopside-Hedenbergite) component
        sites["DiHd_1996"] = (Ca - sites["CaTs"] - sites["CaTi"] - sites["CrCaTs"]).clip(lower=0)
        sites["EnFs"] = ((Fe + Mg) - sites["DiHd_1996"]) / 2
        sites["DiHd_2003"] = (Ca - sites["CaTs"] - sites["CaTi"] - sites["CrCaTs"]).clip(lower=0)
        sites["Di"] = sites["DiHd_2003"] * (
            Mg / (Mg + Mn + Fe).replace(0, np.nan)
        )

        # Wang 2021 Fe3+
        sites["Fe3_Wang21"] = (Na + sites["Al_IV"] - sites["Al_VI"] - 2 * Ti - Cr)
        sites["Fe2_Wang21"] = Fe - sites["Fe3_Wang21"]

        return pd.concat([base, sites], axis=1)


class EpidoteCalculator(BaseMineralCalculator):
    """Epidote-specific calculations. A2M3Z3(O,OH,F)12."""
    OXYGEN_BASIS = 12.5
    MINERAL_SUFFIX = "_Ep"

    def calculate_components(self):
        """Return complete epidote composition with site assignments."""
        if "FeOt" not in self.comps.columns:
            self.comps["FeOt"] = np.nan
        if "Fe2O3t" not in self.comps.columns:
            self.comps["Fe2O3t"] = np.nan

        mask_convert = self.comps["FeOt"].notna() & self.comps["Fe2O3t"].isna()
        conversion_factor = self.OXIDE_MASSES["Fe2O3t"] / (2 * self.OXIDE_MASSES["FeOt"])
        self.comps.loc[mask_convert, "Fe2O3t"] = self.comps.loc[mask_convert, "FeOt"] * conversion_factor
        self.comps.drop(columns="FeOt", inplace=True)

        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Al = base[f"Al{cat_suffix}"]
        Fe = base.get(f"Fe3t{cat_suffix}", 0)
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base[f"Mg{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["A_site"] = Ca + Mn # Ce, Sr, Pb, La, Y, Th
        sites["M_site"] = Mg + Fe + Mn + Cr + Al
        sites["Z_site"] = Si

        # Allocate Al between M2 and remaining (M1+M3):
        sites["Al_M2"] = Al.clip(upper=1) # Al_M2 cannot exceed 1
        sites["Al_M1M3"] = (Al - 1).clip(lower=0) # Remainder of Al-Al_M2) is never negative

        # Fe-and-Mn in M3: Fe_M3 = min(Fe, 1–Mn)
        sites["Fe_M3"] = np.minimum(Fe, 1 - Mn)

        # Al in M3 is whatever is left over: (1-Fe_M3-Mn), but never negative
        sites["Al_M3"] = (1 - sites["Fe_M3"] - Mn).clip(lower=0)

        # M1 = whatever Fe wasn’t used in M3
        sites["Fe_M1"] = Fe - sites["Fe_M3"]

        # Al in M1 = Al not in M2 or M3
        sites["Al_M1"] = Al - sites["Al_M2"] - sites["Al_M3"]

        # End‐member proportions:
        sites["XMn_Ep"] = Mn.copy() # Mn in M3 is just Mn_M3
        sites["XFe_Ep"] = sites["Fe_M1"]
        sites["XEp"]  = sites["Fe_M3"] - sites["XFe_Ep"]
        sites["XZo"]  = 1 - (sites["XEp"] + sites["XFe_Ep"] + sites["XMn_Ep"])
        sites["XSum"] = sites["XZo"] + sites["XEp"] + sites["XFe_Ep"] + sites["XMn_Ep"]

        return pd.concat([base, sites], axis=1)


class FeldsparCalculator(BaseMineralCalculator):
    """Feldspar-specific calculations."""
    OXYGEN_BASIS = 8
    MINERAL_SUFFIX = "_Feld"

    def calculate_components(self):
        """Return complete feldspar composition with site assignments and anorthite, albite, orthoclase."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Al = base[f"Al{cat_suffix}"]
        Na = base[f"Na{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        K = base[f"K{cat_suffix}"]

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["M_site"] = Na + Ca + K
        sites["T_site"] = Si + Al
        sites["An"] = (Ca / (Ca + Na + K))
        sites["Ab"] = (Na / (Ca + Na + K))
        sites["Or"] = 1 - (sites["An"] + sites["Ab"])

        return pd.concat([base, sites], axis=1)


class FeldsparClassifier(FeldsparCalculator):
    """General feldspar calculations for classification and plotting."""

    def classify(self, subclass=True):
        comps = super().calculate_components()
        An = comps["An"].to_numpy()
        Ab = comps["Ab"].to_numpy()
        Or = comps["Or"].to_numpy()

        # Build a fast lookup for the miscibility gap boundary:
        # The compositional regions are from Thermobar. Plotting has been adapted for mineralML
        # Modified plagioclase line's Or extent from DHZ 0.05 to 0.075. 50% tolerance given
        # dependence on temperature. 
        An_plag = np.array([1.00, 0.90, 0.70, 0.50, 0.30, 0.20, 0.15, 0.10, 0.00])
        Or_plag = np.array([0.075, 0.075, 0.075, 0.075, 0.075, 0.10, 0.15, 0.10, 0.10])
        # Or_plag = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.15, 0.10, 0.10])
        f_plag = interpolate.interp1d(An_plag, Or_plag,
                                      bounds_error=False,
                                      fill_value=(Or_plag[0], Or_plag[-1]))

        # Modified alkali feldspar line's An extent from DHZ 0.05 to 0.075. 50% tolerance given
        # dependence on temperature. 
        Or_kfeld = np.array([0.1, 0.15, 0.37, 0.95])
        An_kfeld = np.array([0.1, 0.15, 0.075, 0.075])
        f_kfeld = interpolate.interp1d(Or_kfeld, An_kfeld,
                                       bounds_error=False,
                                       fill_value=(An_kfeld[0], An_kfeld[-1]))

        def _label(a_n, a_b, o):
            # Plagioclase check first
            if o <= f_plag(a_n):
                main = "Plagioclase"
                if not subclass:
                    return main, None
                # DHZ plagioclase subdivisions are *by* An:
                if a_n >= 0.90: 
                    sub = "Anorthite"
                elif a_n >= 0.70: 
                    sub = "Bytownite"
                elif a_n >= 0.50: 
                    sub = "Labradorite"
                elif a_n >= 0.30: 
                    sub = "Andesine"
                elif a_n >= 0.10: 
                    sub = "Oligoclase"
                else:
                    sub = "Albite"

            # Then K-Feldspar with the interpolated curve
            elif a_n <= f_kfeld(o):
                main = "KFeldspar"
                sub = "Sanidine" if o >= 0.37 else "Anorthoclase"

            # Area in between is miscibility gap
            else:
                return "Feldspar_Miscibility_Gap", "Feldspar_Miscibility_Gap"

            return main, sub

        labels = np.array([_label(a,b,o) for a,b,o in zip(An,Ab,Or)])
        df = comps.copy()
        df["Mineral"] = labels[:,0]
        if subclass:
            df["Submineral"] = labels[:,1]
        return df

    def plot(self, df_class=None, subclass=True, labels="short", figsize=(8, 8), 
             ticks=True, **kwargs):

        import ternary
        import matplotlib.pyplot as plt

        if df_class is None:
            df_class = self.classify(subclass=subclass)

        label_dict = {
            "Sanidine": "San", "Anorthoclase": "AnC", "Albite": "Ab",
            "Oligoclase": "Ol", "Andesine": "Ad", "Labradorite": "La",
            "Bytownite": "By", "Anorthite": "An",
        }

        if labels == "long":
            label_set = {k: k for k in label_dict}
        elif labels == "short" or labels is True:
            label_set = label_dict
        else:
            label_set = None

        # Plot setup
        fig, tax = ternary.figure()
        fig.set_size_inches(figsize)
        tax.boundary(linewidth=1.5, zorder=0)
        tax.right_corner_label("An", fontsize=14)
        tax.top_corner_label("Or", fontsize=14)
        tax.left_corner_label("Ab", fontsize=14)

        tax.gridlines(multiple=0.2, ls=":", lw=0.5, c="k", alpha=0.25, zorder=0)
        tax.gridlines(multiple=0.05, lw=0.25, c="lightgrey", alpha=0.25, zorder=0)

        # Interpolated curves
        An = np.array([0.975, 0.9, 0.7, 0.5, 0.3, 0.20, 0.15])
        Or = np.array([0.075, 0.075, 0.075, 0.075, 0.075, 0.1, 0.15])
        f_plag = interpolate.interp1d(An, Or)
        An_new = np.linspace(An[-1], An[0], 1000)
        Or_new = f_plag(An_new)
        Ab_new = 1 - An_new - Or_new
        plag_curve = np.column_stack([An_new, Or_new, Ab_new])

        An_kp = np.array([0, 0.15])
        Or_kp = np.array([0, 0.15])
        Ab_kp = np.array([1, 0.85])
        f_kp = interpolate.interp1d(An_kp, Ab_kp)
        An_kp_new = np.linspace(0, 0.15, 1000)
        Or_kp_new = An_kp_new
        Ab_kp_new = f_kp(An_kp_new)
        plag_kspar_line = np.column_stack([An_kp_new, Or_kp_new, Ab_kp_new])

        Or_k = np.array([0.975, 0.37, 0.15])
        An_k = np.array([0.075, 0.075, 0.15])
        f_kspar = interpolate.interp1d(Or_k, An_k)
        Or_k_new = np.linspace(Or_k[-1], Or_k[0], 1000)
        An_k_new = f_kspar(Or_k_new)
        Ab_k_new = 1 - Or_k_new
        kspar_curve = np.column_stack([An_k_new, Or_k_new, Ab_k_new])

        # Dividers
        tax.line([0.9, 0, 0], plag_curve[plag_curve[:, 0] >= 0.9][0], color="k", zorder=0)
        tax.line([0.7, 0, 0], plag_curve[plag_curve[:, 0] >= 0.7][0], color="k", zorder=0)
        tax.line([0.5, 0, 0], plag_curve[plag_curve[:, 0] >= 0.5][0], color="k", zorder=0)
        tax.line([0.3, 0, 0], plag_curve[plag_curve[:, 0] >= 0.3][0], color="k", zorder=0)
        tax.line([0.1, 0, 0], plag_kspar_line[plag_kspar_line[:, 0] >= 0.1][0], color="k", zorder=0)
        tax.line([0, 0.37, 0.63], kspar_curve[kspar_curve[:, 1] >= 0.37][0], color="k", zorder=0)
        tax.line([0, 0.1, 0.9], plag_kspar_line[plag_kspar_line[:, 0] >= 0.1][0], color="k", zorder=0)

        # Plot boundaries
        tax.plot(plag_kspar_line[plag_kspar_line[:, 1] > 0.1], color="k", zorder=0)
        tax.plot(plag_curve[:-60], color="k", zorder=0)
        tax.plot(kspar_curve[:-60], color="k", zorder=0)

        if ticks:
            tax.ticks(axis="lbr", linewidth=0.5, multiple=0.2, offset=0.02, tick_formats="%.1f")

        tax.clear_matplotlib_ticks()
        tax.get_axes().axis("off")
        tax._redraw_labels()

        # Labels
        if label_set:
            fs = 12
            lab_z = 500
            ax = tax.get_axes()
            bbox_style = dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.6,
                boxstyle='round,pad=0.05'
            )
            ax.text(0.31, 0.5, label_set["Sanidine"], fontsize=fs, rotation=60, zorder=lab_z, bbox=bbox_style)
            ax.text(0.15, 0.2, label_set["Anorthoclase"], fontsize=fs, zorder=lab_z, bbox=bbox_style)
            ax.text(0.075, 0.03, label_set["Albite"], fontsize=fs, ha='center', zorder=lab_z, bbox=bbox_style)
            ax.text(0.2, 0.03, label_set["Oligoclase"], fontsize=fs, ha='center', zorder=lab_z, bbox=bbox_style)
            ax.text(0.4, 0.02, label_set["Andesine"], fontsize=fs, ha='center', zorder=lab_z, bbox=bbox_style)
            ax.text(0.6, 0.02, label_set["Labradorite"], fontsize=fs, ha='center', zorder=lab_z, bbox=bbox_style)
            ax.text(0.8, 0.02, label_set["Bytownite"], fontsize=fs, ha='center', zorder=lab_z, bbox=bbox_style)
            ax.text(0.95, 0.02, label_set["Anorthite"], fontsize=fs, ha='center', zorder=lab_z, bbox=bbox_style)

        pts = list(zip(df_class["An"], df_class["Or"], df_class["Ab"]))
        cmap = plt.get_cmap("tab10")
        for i, g in enumerate(df_class["Submineral"].unique()):
            if g == "Feldspar_Miscibility_Gap":
                continue
            mask = df_class["Submineral"] == g
            pts_sub = [pts[j] for j in np.where(mask)[0]]
            tax.scatter(pts_sub, marker='o', label=g, color=cmap(i),
                        edgecolor='k', s=20, alpha=0.8, vmin=None, vmax=None)
        # legend for the classified fields
        tax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.02, 1))

        # Plot “Unclassified” as hollow xs:
        mask_unc = df_class["Submineral"] == "Feldspar_Miscibility_Gap"
        if mask_unc.any():
            pts_unc = [pts[j] for j in np.where(mask_unc)[0]]
            tax.scatter(pts_unc, marker='x', label="Feldspar_Miscibility_Gap", color='0.35',
                        s=15, alpha=0.9, zorder=30)
            tax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.02, 1))

        return fig, tax


class GarnetCalculator(BaseMineralCalculator):
    """Garnet-specific calculations."""
    OXYGEN_BASIS = 12
    CATION_BASIS = 8
    MINERAL_SUFFIX = "_Gt"

    def calculate_components(self, Fe_correction="Droop"):
        """Return complete garnet composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols_i = [col for col in base.columns if col.endswith(cat_suffix)]

        Fe = base[f"Fe2t{cat_suffix}"]

        # Optional Fe correction, based on Droop, 1987 Fe assignment
        if Fe_correction == "Droop":
            Fe3 = (2 * self.OXYGEN_BASIS * (1 - self.CATION_BASIS / base[cation_cols_i].sum(axis=1))).clip(lower=0)
            Fe3_prop = (Fe3 / Fe).clip(upper=1)
            Fe2 = Fe - Fe3
        elif Fe_correction == "All_Fe2":
            Fe2 = Fe
            Fe3 = 0
            Fe3_prop = 0
        else:
            raise ValueError("Invalid Fe_correction: choose 'Droop' or 'All_Fe2'")

        base["FeO"] = base["FeOt"] * (1 - Fe3_prop)
        base["Fe2O3"] = base["FeOt"] * Fe3_prop * (1 / 0.89992485)
        update_base = base.drop(columns=["FeOt"])

        update_cation_cols = [ox for ox in self.OXIDE_MASSES if ox in update_base.columns]
        update_comps = update_base[update_cation_cols].copy()
        update_df = pd.concat([
            self.metadata,
            update_comps
            ], axis=1)
        update_calc = type(self)(update_df)
        base_update = update_calc.calculate_all()

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base_update.columns if col.endswith(cat_suffix)]

        # Compute site assignments in sites dataframe
        # X = Na, Ca, Mg, Mn, Fe2, Y, dodecahedral
        # Y = Mn, Mg, Fe2, Fe3, Cr, Ti, viAl, Si, octahedral
        # Z = Fe3, ivAl, Si, tetrahedral
        Si = base_update[f"Si{cat_suffix}"]
        Ti = base_update.get(f"Ti{cat_suffix}", 0)
        Al = base_update[f"Al{cat_suffix}"]
        Fe2 = base_update[f"Fe2{cat_suffix}"]
        Fe3 = base_update[f"Fe3{cat_suffix}"]
        Mn = base_update[f"Mn{cat_suffix}"]
        Mg = base_update[f"Mg{cat_suffix}"]
        Ca = base_update[f"Ca{cat_suffix}"]
        Na = base_update.get(f"Na{cat_suffix}", 0)
        Cr = base_update.get(f"Cr{cat_suffix}", 0)

        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base_update[cation_cols].sum(axis=1)

        sites["X_site"] = Mg + Fe2 + Ca + Mn
        sites["Y_site"] = Al + Cr + Mn
        sites["T_site"] = Si + Al 

        sites['Mg_MgFeCa'] = Mg / (Mg + Fe + Ca)
        sites['Fe_MgFeCa'] = Fe / (Mg + Fe + Ca)
        sites['Ca_MgFeCa'] = Ca / (Mg + Fe + Ca)

        sites['Al_AlCr'] = Al / (Al + Cr)
        sites['Cr_AlCr'] = Cr / (Al + Cr)

        sites['Fe3_prop'] = Fe3 / (Fe2 + Fe3)
        sites["And"] = (Fe3 / (Fe + Al)).clip(lower=0)
        sites["Ca_corr"] = (Ca - (1.5 * Fe3)).clip(lower=0)

        sites["Alm"] = (1 - sites["And"]) * (Fe / (Fe + Mn + Mg + sites["Ca_corr"]))
        sites["Prp"] = (1 - sites["And"]) * (Mg / (Fe + Mn + Mg + sites["Ca_corr"]))
        sites["Sps"] = (1 - sites["And"]) * (Mn / (Fe + Mn + Mg + sites["Ca_corr"]))
        sites["Grs"] = (1 - sites["And"]) * (sites["Ca_corr"] / (Fe + Mn + Mg + sites["Ca_corr"]))
        sites["End_Sum"] = sites["And"] + sites["Alm"] + sites["Prp"] + sites["Sps"] + sites["Grs"]

        sites["XMg"] = Mg / (Mg + Fe2)

        sites["Al_AlCr"] = Al / (Al + Cr)
        sites["Cr_AlCr"] = Cr / (Al + Cr)

        return pd.concat([base_update, sites], axis=1)


class KalsiliteCalculator(BaseMineralCalculator):
    """Kalsilite-specific calculations. K[AlSiO4]."""
    OXYGEN_BASIS = 4
    MINERAL_SUFFIX = "_Kal"

    def calculate_components(self):
        """Return complete kalsilite composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Al = base[f"Al{cat_suffix}"]
        K = base[f"K{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["A_B_site"] = K + Na
        sites["A_site"] = K # mostly K
        sites["B_site"] = Na # mostly Na
        sites["T_site"] = Si + Al # tetrahedral

        return pd.concat([base, sites], axis=1)


class LeuciteCalculator(BaseMineralCalculator):
    """Leucite-specific calculations. K[AlSi2O6]."""
    OXYGEN_BASIS = 6
    MINERAL_SUFFIX = "_Lc"

    def calculate_components(self):
        """Return complete leucite composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Al = base[f"Al{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["Channel_site"] = K + Na
        sites["T_site"] = Si + Al # tetrahedral

        return pd.concat([base, sites], axis=1)


# class MagnetiteCalculator(BaseMineralCalculator):
#     """Magnetite-specific calculations. Fe3O4."""
#     OXYGEN_BASIS = 4
#     MINERAL_SUFFIX = "_Mt"

#     def calculate_components(self):
#         """Return complete magnetite composition with site assignments."""
#         base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
#         cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

#         # Grab just the cation columns from `base`
#         cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

#         Ti = base.get(f"Ti{cat_suffix}", 0)
#         Al = base[f"Al{cat_suffix}"]
#         Fe = base[f"Fe2t{cat_suffix}"]
#         Mn = base.get(f"Mn{cat_suffix}", 0)
#         Mg = base[f"Mg{cat_suffix}"]
#         Cr = base.get(f"Cr{cat_suffix}", 0)

#         # Compute site assignments in sites dataframe
#         sites = pd.DataFrame(index=base.index)
#         sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
#         sites["A_site"] = Mg + Fe # Mn, Zn, Ni, Co, Ni, Cu, Ge
#         sites["A_site_expanded"] = Mg + Fe + Mn # Zn, Ni, Co, Ni, Cu, Ge
#         sites["B_site"] = Al + Ti + Cr # Fe3, V
#         sites["Fe_Ti"] = Fe + Ti

#         return pd.concat([base, sites], axis=1)


class MeliliteCalculator(BaseMineralCalculator):
    """Melilite-specific calculations. (Ca,Na)2[(Mg,Fe2+,Al,Si)3O7]."""
    OXYGEN_BASIS = 7
    MINERAL_SUFFIX = "_Ml"

    def calculate_components(self):
        """Return complete melilite composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mg = base[f"Mg{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["A_site"] = Ca + Na
        sites["B_site"] = Mg + Fe + Al
        sites["T_site"] = Si # tetrahedral

        return pd.concat([base, sites], axis=1)


class MuscoviteCalculator(BaseMineralCalculator):
    """Muscovite-specific calculations. XM^{3+}2[Si3Al]010(OH)2."""
    OXYGEN_BASIS = 11
    MINERAL_SUFFIX = "_Ms"

    def calculate_components(self):
        """Return complete muscovite composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mg = base[f"Mg{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["X_site"] = K + Na + Ca # Ba, Rb, Cs
        sites["Al_IV"] = 4 - Si
        sites["Al_VI"] = Al - sites["Al_IV"]
        sites["M_site"] = sites["Al_VI"] + Mg + Fe + Mn + Cr + Ti # M3+, octahedral
        sites["T_site"] = Si + sites["Al_IV"] # tetrahedral

        return pd.concat([base, sites], axis=1)


class NephelineCalculator(BaseMineralCalculator):
    """Nepheline-specific calculations. Na3(Na,K)[Al4Si4O16]."""
    OXYGEN_BASIS = 32
    MINERAL_SUFFIX = "_Ne"

    def calculate_components(self):
        """Return complete nepheline composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mg = base[f"Mg{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["A_B_site"] = K + Na
        sites["A_site"] = K # mostly K
        sites["B_site"] = Na # mostly Na
        sites["T_site"] = Si + Al # tetrahedral

        return pd.concat([base, sites], axis=1)


class OlivineCalculator(BaseMineralCalculator):
    """Olivine-specific calculations. (Mg,Fe)2SiO4."""
    OXYGEN_BASIS = 4
    MINERAL_SUFFIX = "_Ol"

    def calculate_components(self):
        """Return complete olivine composition with site assignments and forsterite."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations=
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        # G. Cressey, R.A. Howie, in Encyclopedia of Geology, 2005
        # M-site and T-site calculations
        # Mg-Fe olivines - Mg2+ and Fe2+ can occupy M1 and M2 with almost equal preference. 
        # Slight tendency for Fe2+ to occupy the M1 site rather than the M2 site
        # Mg-Fe olivines - Small proportion of Ca and Mn present. Substitution of Mn2+ for Fe2+ in fayalite also occurs.
        # Ca olivines: Ca2+ occupies the (larger) M2 site, while Mg2+ and Fe2+ are randomly distributed on the M1 sites.
        Mg = base[f"Mg{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Si = base[f"Si{cat_suffix}"]
        Ca = base.get(f"Ca{cat_suffix}", 0)
        Mn = base.get(f"Mn{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["M_site"] = Mg + Fe
        sites["T_site"] = Si
        sites["M_site_expanded"] = Mg + Fe + Ca + Mn
        sites["XFo"] = (Mg / (Mg + Fe))

        return pd.concat([base, sites], axis=1)


class OrthopyroxeneCalculator(BaseMineralCalculator):
    """Orthopyroxene-specific calculations."""
    OXYGEN_BASIS = 6
    MINERAL_SUFFIX = "_Opx"

    def calculate_components(self):
        """Return complete orthopyroxene composition with site assignments and enstatite, ferrosilite, wollastonite."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base[f"Mg{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["M_site"] = Mg + Fe + Ca + Na + Ti + Cr
        sites["T_site"] = Si + Al
        sites["XMg"] = (Mg / (Mg + Fe))
        sites["En"] = Mg / (Mg + Fe + Ca)
        sites["Fs"] = Fe / (Mg + Fe + Ca)
        sites["Wo"] = Ca / (Mg + Fe + Ca) # Ca_CaMgFe
        sites["Jd"] = Na

        sites["Al_IV"] = 2 - Si
        sites["Al_VI"] = Al - sites["Al_IV"]
        sites["Al_VI"] = sites["Al_VI"].clip(lower=0) # Al_VI can't be negative

        sites["NaAlSi2O6"] = Na
        sites["FmTiAlSiO6"] = Ti
        sites["CrAl2SiO6"] = Cr
        sites["FmAl2SiO6"] = sites["Al_VI"] - sites["NaAlSi2O6"] - sites["CrAl2SiO6"]
        sites["FmAl2SiO6"] = sites["FmAl2SiO6"].clip(lower=0)
        sites["CaFmSi2O6"] = Ca
        sites["Fm2Si2O6"] = (
            (Fe + Mg + Mn)
            - sites["FmTiAlSiO6"]
            - sites["FmAl2SiO6"]
            - sites["CaFmSi2O6"]
        ) / 2
        sites["En_Opx"] = sites["Fm2Si2O6"] * (
            Mg / (Mg + Fe + Mn).replace(0, np.nan)
        )
        sites["Di_Opx"] = sites["CaFmSi2O6"] * (
            Mg / (Mg + Fe + Mn).replace(0, np.nan)
        )

        return pd.concat([base, sites], axis=1)


class RhombohedralOxideCalculator(BaseMineralCalculator):
    """Rhombohedral oxide-specific calculations. Hematite-Ilmenite, Fe2O3-(FeTi)2O3."""
    OXYGEN_BASIS = 3
    CATION_BASIS = 2
    MINERAL_SUFFIX = "_Ox"

    def calculate_components(self, Fe_correction="Droop"):
        """Return complete Fe-Ti oxide composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]
        total_cat = base[cation_cols].sum(axis=1)
        T_S = self.CATION_BASIS / total_cat
        base["S_init"] = total_cat # initial total cations
        base["T_S"] = T_S
        Fe_init = base[f"Fe2t{cat_suffix}"]

        if Fe_correction == "Droop":
            # Droop (1987) equation
            total_cat = base[cation_cols].sum(axis=1)
            Fe3 = (2 * self.OXYGEN_BASIS * (1 - (self.CATION_BASIS / total_cat))).clip(lower=0)
            Fe3_prop = (Fe3 / Fe_init).clip(upper=1)
            Fe2 = Fe_init - Fe3
        elif Fe_correction == "All_Fe2":
            Fe2 = Fe_init
            Fe3 = pd.Series(0, index=base.index)
            Fe3_prop = pd.Series(0, index=base.index)
        elif Fe_correction == "All_Fe3":
            Fe2 = Fe_init
            Fe3 = pd.Series(0, index=base.index)
            Fe3_prop = pd.Series(1, index=base.index)
            update_base = self.comps.copy()  # only oxide wt% columns
            update_base["FeO"] = update_base["FeOt"] * (1 - Fe3_prop)
            update_base["Fe2O3"] = update_base["FeOt"] * Fe3_prop * (1 / 0.89992485)
            update_base = update_base.drop(columns=["FeOt"])

            update_calc = type(self)(update_base)
            base_update = update_calc.calculate_all()
            cation_cols_update = [col for col in base_update.columns if col.endswith(cat_suffix)]
            base = base_update.copy()
            cation_cols = cation_cols_update
        else:
            raise ValueError("Invalid Fe_correction: choose 'Droop', 'All_Fe2', or 'All_Fe3'.")

        if "FeOt" in base.columns:
            base["FeO"] = base["FeOt"] * (1 - Fe3_prop)
            base["Fe2O3"] = base["FeOt"] * Fe3_prop * (1 / 0.89992485)
            base[f"Fe3{cat_suffix}"] = Fe3
            base[f"Fe2{cat_suffix}"] = Fe2

        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base.get(f"Al{cat_suffix}", 0) 
        Fe2 = base[f"Fe2{cat_suffix}"]
        Fe3 = base[f"Fe3{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base.get(f"Mg{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["A_site"] = Mg + Fe2 # Mn, Zn, Ni, Co, Ni, Cu, Ge
        sites["A_site_expanded"] = Mg + Fe2 + Mn # Zn, Ni, Co, Ni, Cu, Ge
        sites["B_site"] = Al + Ti + Cr + Fe3 # V
        sites["A_B_site"] = Mg + Fe2 + Mn + Al + Ti + Cr + Fe3
        sites["Fe_Ti"] = Fe2 + Fe3 + Ti
        sites['Fe3_prop'] = Fe3 / (Fe2 + Fe3)
        total = (Fe2 + Fe3 + Ti + Al + Cr + Mn + Mg)
        sites["XR2"] = (Fe2 + Mn + Mg) / total # having as true R2+ (Fe+Mg+Mn) pushes all spinels away
        sites["XR3"] = (Fe3 + Al + Cr) / total
        sites["XTi"] = Ti / total

        sites["XHem"] = Fe3 / (Fe3 + Ti) # Hematite
        remainder = 1 - sites["XHem"]
        denominator = Fe2 + Mn + Mg
        sites["XIlm"] = (Fe2 / denominator) * remainder # Ilmenite
        sites["XMnIlm"] = (Mn / denominator) * remainder # Mn-Ilmenite
        sites["XGk"] = (Mg / denominator) * remainder # Geikielite (MgTiO3)
        sites["XSum"] = sites["XHem"] + sites["XIlm"] + sites["XMnIlm"] + sites["XGk"]

        return pd.concat([base, sites], axis=1)


class OxideClassifier:
    """
    General classifier for rhombohedral oxides and spinels. Use either:
    - RhombohedralOxideCalculator for rhombohedral oxides (e.g., Hematite, Ilmenite)
    - SpinelCalculator for spinels (e.g., Spinel, Magnetite)

    Returns:
      Compositions/sites from the specific calculators with XR2, XR3, XTi
      already populated by those calculators. Non-routed rows are passed through.
    """

    def __init__(self, df):
        self.df = df.copy()
        if "Predict_Mineral" in self.df.columns:
            self.mineral_col = "Predict_Mineral"
        elif "Mineral" in self.df.columns:
            self.mineral_col = "Mineral"
        else:
            raise ValueError(
                "Dataframe must contain a 'Predict_Mineral' column"
            )

    def _name_masks(self, frame: pd.DataFrame):
        names = frame[self.mineral_col].astype(str).str.lower()
        rhomb_mask = (
            names.str.contains("rhombohedral_oxides", case=False, regex=False) |
            names.str.contains("ilmenite", case=False, regex=False) |
            names.str.contains("hematite", case=False, regex=False)
        )
        spinel_mask = (
            names.str.contains("spinel", case=False, regex=False) |
            names.str.contains("magnetite", case=False, regex=False)
        )
        return rhomb_mask, spinel_mask

    def calculate_components(self, Fe_correction="Droop"):
        """
        Route rhombohedral oxides and spinels to the appropriate calculator and
        write results back into a copy of the original df.
        """
        out = self.df.copy()
        rhomb_mask, spinel_mask = self._name_masks(out)

        # Rhombohedral oxides
        if rhomb_mask.any():
            ox_df = out.loc[rhomb_mask]

            # Split into hematite-rich (FeOt > 60) and others
            if "FeOt" in ox_df.columns:
                hematite_mask = ox_df["FeOt"] > 60
                hematite_df = ox_df[hematite_mask]
                other_df = ox_df[~hematite_mask]
            else:
                hematite_df = ox_df.iloc[0:0] # empty df
                other_df = ox_df

            # Process hematite-rich samples with All_Fe3
            if not hematite_df.empty:
                hematite_res = RhombohedralOxideCalculator(hematite_df).calculate_components(
                    Fe_correction="All_Fe3"
                )
                out.loc[hematite_df.index, hematite_res.columns] = hematite_res.values

            # Process others with default Fe_correction
            if not other_df.empty:
                other_res = RhombohedralOxideCalculator(other_df).calculate_components(
                    Fe_correction=Fe_correction
                )
                out.loc[other_df.index, other_res.columns] = other_res.values

        # Spinels
        if spinel_mask.any():
            sp_df  = out.loc[spinel_mask]
            sp_res = SpinelCalculator(sp_df).calculate_components(Fe_correction=Fe_correction)
            out.loc[sp_df.index, sp_res.columns] = sp_res.values

        return out

    def _spinel_axes(self, df):
        idx = df.index
        fe2 = df.get("Fe2_cat_4ox", pd.Series(0.0, index=idx)).astype(float)
        mg  = df.get("Mg_cat_4ox",  pd.Series(0.0, index=idx)).astype(float)
        fe3 = df.get("Fe3_cat_4ox", pd.Series(0.0, index=idx)).astype(float)
        al  = df.get("Al_cat_4ox",  pd.Series(0.0, index=idx)).astype(float)
        x = np.divide(fe2, fe2+mg, out=np.zeros_like(fe2, float), where=(fe2+mg)>0)
        y = np.divide(fe3, fe3+al, out=np.zeros_like(fe3, float), where=(fe3+al)>0)
        return np.clip(x,0,1), np.clip(y,0,1)

    def _classify_subspinel(self, x, y, eps=1e-9):
        """
        x = Fe2/(Fe2+Mg), y = Fe3/(Fe3+Al)  (numpy arrays)
        returns: np.array of subtype strings
        """
        sub = np.full(x.shape, "Unlabeled", dtype=object)

        # Top band: y >= 0.75
        top = y >= 0.75 - eps
        sub[top & (x < 0.5 - eps)]  = "Magnesioferrite"
        sub[top & (x >= 0.5 - eps)] = "Magnetite"

        # Middle band: 0.25 <= y < 0.75
        mid = (y >= 0.25 - eps) & (y < 0.75 - eps)
        sub[mid & (x < 0.25 - eps)] = "Ferrian-Spinel"
        mid_mid = mid & (x >= 0.25 - eps) & (x < 0.75 - eps)
        sub[mid_mid] = "Ferrian-Pleonaste"
        right_mid = mid & (x >= 0.75 - eps)
        sub[right_mid & (y >= 0.50 - eps)] = "Al-Magnetite"
        sub[right_mid & (y <  0.50 - eps)] = "Ferrian-Picotite"

        # Bottom band: y < 0.25
        bot = y < 0.25 - eps
        sub[bot & (x < 0.25 - eps)] = "Spinel"
        sub[bot & (x >= 0.25 - eps) & (x < 0.75 - eps)] = "Pleonaste"
        sub[bot & (x >= 0.75 - eps)] = "Hercynite"
        return sub

    def _project_to_line(self, P, A, B, eps=0.1):
        """Helper: Project points P onto line A-B and check distance."""
        v = B - A
        vv = float(v @ v)
        p = P - A
        t = (p @ v) / vv
        t_clip = np.clip(t, 0.0, 1.0)
        # proj = A + t_clip[:, None] * v
        dist = np.linalg.norm(p - t_clip[:, None] * v, axis=1)
        on_line = dist <= eps
        return t_clip, on_line, dist

    def classify(self, eps=0.1, subclass=True):
        comps = self.calculate_components()
        df_class = comps.copy()

        if not subclass:
            return df_class

        df_class["Mineral"] = df_class[self.mineral_col].astype(str)
        df_class["Submineral"] = None
        required_cols = ["XR3", "XTi", "XR2"]
        has_required_cols = all(col in df_class.columns for col in required_cols)
        rhomb_mask, spinel_mask = self._name_masks(df_class)

        if has_required_cols:
            # Rhombohedral oxides: Hematite-Ilmenite line
            if rhomb_mask.any():
                idx = df_class.index[rhomb_mask]
                P = df_class.loc[idx, ["XR3", "XTi", "XR2"]].to_numpy()
                H = np.array([1.0, 0.0, 0.0])  # Hematite
                I = np.array([0.0, 0.5, 0.5])  # Ilmenite
                t_hi, on_hi, dist_hi = self._project_to_line(P, H, I, eps)            
                df_class.loc[idx[on_hi], "Submineral"] = np.where(
                    t_hi[on_hi] <= 0.5, "Hematite", "Ilmenite"
                )
                # df_class.loc[idx[on_hi], "Classification_Confidence"] = 1 - (dist_hi[on_hi]/eps)

            # Spinels: Magnetite-Ulvöspinel line
            if spinel_mask.any():
                idx = df_class.index[spinel_mask]
                P = df_class.loc[idx, ["XR3", "XTi", "XR2"]].to_numpy()
                M = np.array([2/3, 0.0, 1/3]) # Magnetite (Fe3O4)
                U = np.array([0.0, 1/3, 2/3]) # Ulvöspinel (Fe2TiO4)
                t_mu, on_mu, dist_mu = self._project_to_line(P, M, U, eps/4)
                df_class.loc[idx[on_mu], "Submineral"] = np.where(
                    t_mu[on_mu] <= 0.5, "Spinels", "Ulvöspinel"
                )
                # df_class.loc[idx[on_mu], "Classification_Confidence"] = 1 - (dist_mu[on_mu]/eps)

            # This is important if you define Fe2+ rather than R2+ (Fe2+, Mg2+, Mn2+), 
            # as spinel lies between the U-M and I-H line!!! 

            # unl = df_class["Submineral"].isna()
            # if unl.any():
            #     idx = df_class.index[unl]
            #     P = df_class.loc[idx, ["XR3","XTi","XR2"]].to_numpy(dtype=float)
            #     x, y, z = P[:,0], P[:,1], P[:,2]

            #     # # exact tie-lines in barycentric coordinates
            #     z_low  = 0.5 - 0.5 * x
            #     z_high = 2.0/3.0 - 0.5 * x
            #     margin = 0.002

            #     # # keep points strictly between lines and away from the FeO=0 edge
            #     spinel_mask = (z > z_low + margin) & (z < z_high - margin)
            #     df_class.loc[idx[spinel_mask], "Submineral"] = "Spinel Group"

            # FeO-TiO2-Pseudobrookite System
            unl = df_class["Submineral"].isna()
            if unl.any():
                idx = df_class.index[unl]
                P = df_class.loc[idx, ["XR3","XTi","XR2"]].to_numpy()
                A = np.array([0.0, 2/3, 1/3]) # FeO·2TiO2
                B = np.array([1/2, 1/2, 0.0]) # Pseudobrookite
                t_fp, on_fp, dist_fp = self._project_to_line(P, A, B, eps)
                df_class.loc[idx[on_fp], "Submineral"] = "FeO·2TiO2-Pseudobrookite"
                # df_class.loc[idx[on_fp], "Classification_Confidence"] = 1 - (dist_fp[on_fp]/eps)

            end_tol = 0.05
            unl = df_class["Submineral"].isna()
            if unl.any():
                idx = df_class.index[unl]
                near_tio2 = df_class.loc[idx, "XTi"] >= (1.0 - end_tol)
                near_feo = df_class.loc[idx, "XR2"]  >= (1.0 - end_tol)
                near_fe2o3  = df_class.loc[idx, "XR3"]  >= (1.0 - end_tol)
                df_class.loc[idx[near_tio2], "Submineral"] = "Rutile"
                df_class.loc[idx[near_feo], "Submineral"] = "FeO"
                df_class.loc[idx[near_fe2o3], "Submineral"] = "Hematite"


        mineral_col = self.mineral_col
        df_class["Submineral"] = df_class["Submineral"].fillna(df_class[mineral_col].astype(str))

        sp_rows = df_class["Submineral"].astype(str).str.contains("spinel", case=False, na=False)
        if sp_rows.any():
            sp_df = df_class.loc[sp_rows].copy()
            x, y = self._spinel_axes(sp_df)
            df_class.loc[sp_rows, "Subspinel"] = self._classify_subspinel(x, y)
        else:
            df_class["Subspinel"] = np.nan

        return df_class

    def plot(self, figsize=(8, 8), ticks=True, include_unclassified=True, **kw):
        """
        Ternary plot of FeO-Fe2O3-TiO2; colors by the existing 'Mineral' labels.
        """
        import ternary
        import matplotlib.pyplot as plt

        df = self.classify()
        valid = (df[["XR2", "XR3", "XTi"]].sum(axis=1) > 0)
        df = df[valid].copy()

        if "Submineral" not in df.columns:
            df["Submineral"] = "Unclassified"
        if not include_unclassified:
            df = df[df["Submineral"] != "Unclassified"]

        groups = df["Submineral"].astype(str).fillna("(unknown)").unique()
        cmap = plt.get_cmap("tab10")

        fig, tax = ternary.figure()
        fs = 14
        fig.set_size_inches(figsize)
        tax.boundary(linewidth=1.5, zorder=0)
        tax.right_corner_label("R$\\mathregular{^{3+}}$\n$\\mathregular{Fe^{3+}+Cr+Mn}$\nHematite, $\\mathregular{Fe_2O_3}$", fontsize=fs, offset=-0.075)
        tax.top_corner_label("Rutile, anatase, brookite\n$\\mathregular{TiO_2}$", fontsize=fs, offset=0.175)
        tax.left_corner_label("R$\\mathregular{^{2+}}$\n$\\mathregular{Fe^{2+}+Mg+Mn}$\n FeO ", fontsize=fs, offset=-0.075)
        tax.bottom_axis_label("Magnetite\n$\\mathregular{Fe_3O_4}$", fontsize=fs, offset=0)

        ax = tax.get_axes()
        ax.text(0.86, 0.45, "Pseudobrookite\n$\\mathregular{FeTi_2O_5}$", fontsize=fs, ha="center", va="center")
        ax.text(0.24, 0.59, "FeO$\\cdot$${\\mathregular{2TiO_2}}$", fontsize=fs, ha="center", va="center")
        ax.text(0.17, 0.44, "Ilmenite\n$\\mathregular{FeTiO_3}$", fontsize=fs, ha="center", va="center")
        ax.text(0.06, 0.30, "Ulvöspinel\n2FeO$\\cdot$${\\mathregular{TiO_2}}$", fontsize=fs, ha="center", va="center")

        tax.gridlines(multiple=0.2, ls=":", lw=0.5, c="k", alpha=0.25, zorder=0)
        tax.gridlines(multiple=0.05, lw=0.25, c="lightgrey", alpha=0.25, zorder=0)

        # Ilmenite-hematite/maghemite
        tax.line((0, 1/2, 1/2), (1, 0, 0), color="k", **kw)
        # Ulvospinel-magnetite
        tax.line((0, 1/3, 2/3), (2/3, 0, 1/3), color="k", **kw)
        # FeO 2TiO2-pseudobrookite
        tax.line((0, 2/3, 1/3), (1/2, 1/2, 0), color="k", **kw)

        for i, phase in enumerate([g for g in groups if g != "Unclassified"]):
            pts = df[df["Submineral"] == phase][["XR3","XTi","XR2"]].values.tolist()
            if not pts:
                continue
            tax.scatter(pts, marker='o', label=phase, color=cmap(i % 10), edgecolor='k',
                        s=30, alpha=0.85, zorder=50)

        # unclassified last 
        if "Unclassified" in groups:
            pts = df[df["Submineral"] == "Unclassified"][["XR3","XTi","XR2"]].values.tolist()
            if pts:
                tax.scatter(pts, marker='x', label="Unclassified", color='0.35',
                            s=30, alpha=0.9, zorder=30)

        if ticks:
            tax.ticks(axis="lbr", linewidth=0.5, multiple=0.2, offset=0.01, tick_formats="%.1f")

        tax.legend(fontsize=10, bbox_to_anchor=(1.02, 1))
        tax.get_axes().axis("off")

        sp_mask = df["Submineral"].astype(str).str.contains("spinel", case=False, na=False)
        if sp_mask.any():
            self._last_spinel_figax = self.plot_spinel(df=df, figsize=(9, 6), hue="Subspinel")

        return fig, tax
    
    def plot_spinel(self, df=None, figsize=(9, 6), hue=None):

        import matplotlib.pyplot as plt

        if df is None:
            df = self.classify().copy()

        sp_mask = df["Submineral"].astype(str).str.contains("spinel", case=False, na=False)
        sp = df.loc[sp_mask].copy()
        if sp.empty:
            return (None, None)

        # Default hue is Subspinel
        if hue is None:
            hue = "Subspinel"

        # Ratios for plotting
        x, y = self._spinel_axes(sp)
        fig, ax = plt.subplots(figsize=figsize)

        # Spinel grid definition
        ax.hlines(y=0.75, xmin=-0.02, xmax=1.02, color='k', lw=1, zorder=0)
        ax.hlines(y=0.25, xmin=-0.02, xmax=1.02, color='k', lw=1, zorder=0)
        ax.hlines(y=0.50, xmin=0.75, xmax=1.02, color='k', lw=1, zorder=0)
        ax.vlines(x=0.25, ymin=-0.02, ymax=0.75, color='k', lw=1, zorder=0)
        ax.vlines(x=0.75, ymin=-0.02, ymax=0.75, color='k', lw=1, zorder=0)
        ax.vlines(x=0.50, ymin=0.75, ymax=1.02, color='k', lw=1, zorder=0)

        fs = 14
        ax.text(0.225, 0.875, "Magnesioferrite", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(0.775, 0.875, "Magnetite", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(0.125, 0.500, "Ferrian-Spinel", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(0.500, 0.500, "Ferrian-Pleonaste", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(0.875, 0.625, "Al-Magnetite", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(0.875, 0.375, "Ferrian-Picotite", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(0.125, 0.130, "Spinel", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(0.500, 0.130, "Pleonaste", fontsize=fs, ha="center", va="center", zorder=30)
        ax.text(0.875, 0.130, "Hercynite", fontsize=fs, ha="center", va="center", zorder=30)

        # Color by hue using tab10; keep in sorted order
        cmap = plt.get_cmap("tab10")
        if hue in sp.columns:
            groups = sorted(sp[hue].astype(str).unique())
            for i, g in enumerate(groups):
                m = (sp[hue].astype(str) == g).to_numpy()
                ax.scatter(x[m], y[m], label=g, s=30, alpha=0.6,
                        edgecolors="k", linewidth=0.5, color=cmap(i % 10), zorder=20)
            # Put legend OUTSIDE 
            fig.subplots_adjust(right=0.9)  # make room on the right
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True, fontsize=9)
        else:
            ax.scatter(x, y, s=30, alpha=0.6, edgecolors="k", linewidth=0.5,
                    color=cmap(0), zorder=20)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        # ax.margins(x=0.02, y=0.02) # small outer margins to separate tick labels from the frame
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=5)
        ax.set_xlabel("$\\mathregular{Fe^{2+}/(Fe^{2+}+Mg^{2+})}$")
        ax.set_ylabel("$\\mathregular{Fe^{3+}/(Fe^{3+}+Al^{3+})}$")
        fig.tight_layout()

        return fig, ax


class PyroxeneClassifier(BaseMineralCalculator):
    """General pyroxene calculations for classification."""
    OXYGEN_BASIS = 6
    MINERAL_SUFFIX = "_Px"

    def _o_total_from_4cat(self):
        """
        1) compute moles from oxides
        2) convert to cations per oxide (multiply by cation numbers)
        3) normalize so total cations = 4
        4) from those cat-normalized moles, compute O2_total using oxide stoichiometry
        Returns: O_total both as pandas Series
        """
        moles = self.calculate_moles().copy()
        moles.columns = [c.replace("_mols", "") for c in moles.columns]

        # cations per oxide (not yet normalized)
        cat_nums = pd.Series(self.CATION_NUMBERS)
        # keep only oxides defined in cat_nums
        moles2 = moles.loc[:, moles.columns.intersection(cat_nums.index)]
        cations_raw = moles2.multiply(cat_nums, axis="columns")

        # total cations and 4-cation normalization factor
        cat_sum = cations_raw.sum(axis=1).replace(0, np.nan)
        cat_norm = 4.0 / cat_sum

        # cat-normalized moles (apply factor to moles
        moles_4cat = moles.multiply(cat_norm, axis="rows")

        # compute O_total from cat-normalized moles and per-oxide oxygen stoichiometry
        O_per_oxide = pd.Series(self.OXYGEN_NUMBERS)
        moles_4cat = moles_4cat.loc[:, moles_4cat.columns.intersection(O_per_oxide.index)]
        O_total = moles_4cat.multiply(O_per_oxide, axis="columns").sum(axis=1)

        return O_total

    def calculate_components(self):
        """Return complete pyroxene composition with site assignments and enstatite, ferrosilite, wollastonite."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mg = base[f"Mg{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Ca = base[f"Ca{cat_suffix}"]
        K = base.get(f"K{cat_suffix}", 0)
        Na = base.get(f"Na{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["Oxygen_Sum"] = self._o_total_from_4cat()
        sites["M_site"] = Mg + Fe + Ca + Na + Ti + Cr
        sites["T_site"] = Si + Al
        sites["XMg"] = (Mg / (Mg + Fe))
        sites["En"] = Mg / (Mg + Fe + Ca)
        sites["Fs"] = Fe / (Mg + Fe + Ca)
        sites["Wo"] = Ca / (Mg + Fe + Ca)
        sites["Jd"] = Na
        sites["Q"] = Ca + Mg + Fe
        sites["J"] = 2 * Na

        # This code is modified from Thermobar, with permission from Penny Wieser
        sites["Al_IV"] = 2 - Si
        sites["Al_VI"] = (Al - sites["Al_IV"]).clip(lower=0)

        # Fe3+, Fe2+ Calculation
        O_def = (6 - sites["Oxygen_Sum"]).clip(lower=0)
        sites["Fe3_oxdef"] = np.minimum(Fe, 2.0*O_def)
        sites["Fe2_oxdef"] = (Fe - sites["Fe3_oxdef"]).clip(lower=0)

        # Fe3+, Fe2+ Calculation
        sites["Fe3_Lindley"] = (
            Na + sites["Al_IV"]
            - sites["Al_VI"]
            - (2 * Ti) - Cr
        ).clip(lower=0, upper=Fe) # Fe3 can't be negative or greater than Fe
        sites.loc[sites["Fe3_Lindley"] < 1e-10, "Fe3_Lindley"] = 0
        sites["Fe2_Lindley"] = Fe - sites["Fe3_Lindley"]
        sites["Fe3_prop_Lindley"] = (sites["Fe3_Lindley"] / Fe).replace(0, np.nan)

        # Independent cpx components
        sites["CrCaTs"] = 0.5 * Cr
        sites['a_cpx_En'] = (
            (1 - Ca - Na - K) * 
            (1 - 0.5 * (Al + Cr + Na + K))
        )

        # If value of AlVI < Na cation fraction
        sites["CaTs"] = sites["Al_VI"] - Na
        CaTs_mask = sites["CaTs"] < 0
        sites["Jd_from 0=Na, 1=Al"] = 0
        sites.loc[CaTs_mask, "Jd_from 0=Na, 1=Al"] = 1
        sites.loc[CaTs_mask, "Jd"] = sites.loc[CaTs_mask, "Al_VI"].to_numpy()
        sites["CaTs"] = sites["CaTs"].clip(lower=0)

        # CaTi component
        sites["CaTi"] = ((sites["Al_IV"] - sites["CaTs"]) / 2).clip(lower=0)

        # DiHd (Diopside-Hedenbergite) component
        sites["DiHd_1996"] = (Ca - sites["CaTs"] - sites["CaTi"] - sites["CrCaTs"]).clip(lower=0)
        sites["EnFs"] = ((Fe + Mg) - sites["DiHd_1996"]) / 2
        sites["DiHd_2003"] = (Ca - sites["CaTs"] - sites["CaTi"] - sites["CrCaTs"]).clip(lower=0)
        sites["Di"] = sites["DiHd_2003"] * (
            Mg / (Mg + Mn + Fe).replace(0, np.nan)
        )

        # Wang 2021 Fe3+
        sites["Fe3_Wang21"] = (Na + sites["Al_IV"] - sites["Al_VI"] - 2 * Ti - Cr)
        sites["Fe2_Wang21"] = Fe - sites["Fe3_Wang21"]

        # This code was converted from MATLAB to Python, from Jesse Walter's MinPlot
        # Verified for alignment between values. 
        # Sodic pool after balancing Ti
        excess_Si = (Si - 2.0).clip(lower=0)
        A1n = (sites["Al_IV"] - Ti).clip(lower=0) 
        A3n = (sites["Al_VI"] - A1n).clip(lower=0) 
        Fe3 = sites["Fe3_oxdef"].clip(lower=0) # Oxygen deficiency for now, would be different with other implementations
        M3 = Cr + 2.0 * (excess_Si + Ti) + sites["Fe3_oxdef"] + (sites["Al_VI"] - sites["Al_IV"])
        e = (M3 - (Na + K)).clip(lower=0.0) 
        g  = (Ca - 0.5*e - A1n).clip(lower=0.0)

        sites["Di_h"] = np.minimum(g, Mg)
        sites["Hd_h"] = (g - sites["Di_h"]).clip(lower=0.0)
        zp = (Na + K - Ti).clip(lower=0)
        zm = (zp - A3n).clip(lower=0)
        Ko = np.minimum(Cr, zm)
        zk = (zm - Cr).clip(lower=0)

        delta = (sites["Al_VI"]  - (sites["Al_IV"] - Ti))
        F3n = np.where(delta < 0, Fe3 + delta, Fe3) # reduce Fe3 if needed
        F3n = np.clip(F3n, 0.0, None)

        Aeg = np.minimum(Fe3.clip(lower=0.0), zk)
        sites["Aeg_h"] = pd.Series(Aeg, index=sites.index).clip(lower=0)

        Jd  = np.minimum(A3n, zp)
        sites["Jd_h"] = pd.Series(Jd, index=sites.index).clip(lower=0)

        # Harlow's XEn (h): (Mg + Mn + Fe2 - g) / 2, clipped ≥ 0
        En_h = ((Mg + Mn + sites["Fe2_oxdef"]) - g) / 2.0
        En_h = En_h.clip(lower=0.0)
        sites["En_h"] = En_h # store it so classify() can use it

        return pd.concat([base, sites], axis=1)

    def classify(self, subclass=True):
        """
        Classify pyroxene analyses into broad classes (ortho vs. clino) and
        optional DHZ subclasses. Uses Morimoto (1988) scheme to first separate
        sodic vs. non-sodic pyroxenes.

        Parameters:
            subclass (bool): If True, determine `Submineral` classification.

        Returns:
            classified_df (pd.DataFrame): DataFrame with new columns for Mineral, 
                Submineral, En, Fs, and Wo. 
        """

        comps = self.calculate_components()
        en = comps["En"].to_numpy()
        fs = comps["Fs"].to_numpy()
        wo = comps["Wo"].to_numpy()

        en_h = comps["En_h"].to_numpy()
        di_h = comps["Di_h"].to_numpy()
        hd_h = comps["Hd_h"].to_numpy()
        jd_h = comps["Jd_h"].to_numpy()
        aeg_h = comps["Aeg_h"].to_numpy(float)
        q_morimoto = comps["Q"].to_numpy()
        j_morimoto = comps["J"].to_numpy()

        def _classify_non_sodic(en, fs, wo, subclass_flag):
            """Classify non-sodic pyroxenes using Morimoto scheme."""
            main = "Orthopyroxene" if wo < 0.05 else "Clinopyroxene"
            if not subclass_flag:
                return main, None

            # Safe ratio calculation
            en_fs_sum = en + fs
            fs_ratio = fs / en_fs_sum if en_fs_sum > 1e-10 else 0.5

            if wo < 0.05:
                sub = "Enstatite" if fs_ratio < 0.5 else "Ferrosilite"
            elif wo < 0.20:
                sub = "Pigeonite"
            elif wo < 0.45:
                sub = "Augite"
            elif wo < 0.50:
                sub = "Diopside" if fs_ratio < 0.5 else "Hedenbergite"
            else: 
                sub = "Wollastonite"
            return main, sub

        def _classify_sodic(en_h, di_h, hd_h, jd_h, aeg_h, subclass_flag):
            """Classify sodic pyroxenes."""
            main = "Na-Pyroxene"
            if not subclass_flag:
                return main, None
            
            # Safe value conversion
            en_h = max(float(en_h) if np.isfinite(en_h) else 0.0, 0.0)
            jd = max(float(jd_h) if np.isfinite(jd_h) else 0.0, 0.0)
            aeg = max(float(aeg_h) if np.isfinite(aeg_h) else 0.0, 0.0)
            di = max(float(di_h) if np.isfinite(di_h) else 0.0, 0.0)
            hd = max(float(hd_h) if np.isfinite(hd_h) else 0.0, 0.0)

            q_raw = en_h + di + hd
            total = q_raw + jd + aeg
            
            if not np.isfinite(total) or total <= 1e-10:
                return main, None

            q = q_raw / total
            jl = jd / total
            al = aeg / total

            if q <= 0.20:
                sub = "Jadeite" if jl >= al else "Aegirine"
            elif q <= 0.80:
                sub = "Omphacite" if jl >= al else "Aegirine-Augite"
            else:
                sub = "Ca-Mg-Fe Pyroxene"
            return main, sub

        def _label(en, fs, wo, jd_h, aeg_h, di_h, hd_h, en_h, j_morimoto, q_morimoto, subclass_flag):
            """Main classification logic."""
            # Handle NaN/infinite values
            if not all(np.isfinite([en, fs, wo, j_morimoto, q_morimoto])):
                main = "Orthopyroxene" if wo < 0.05 else "Clinopyroxene"
                return main, None
            
            # Morimoto ratio calculation
            denom_na = j_morimoto + q_morimoto
            if denom_na > 1e-10:
                r_morimoto = j_morimoto / denom_na
                r_morimoto = np.clip(r_morimoto, 0.0, 1.0)
            else:
                r_morimoto = 0.0

            # Classification decision
            if r_morimoto < 0.20:
                return _classify_non_sodic(en, fs, wo, subclass_flag)
            else:
                return _classify_sodic(en_h, di_h, hd_h, jd_h, aeg_h, subclass_flag)
        
        # Apply classification to all samples - SINGLE VERSION (remove the duplicate)
        labels = np.array([
            _label(e, f, w, j, a, d, h, e_h, j_m, q_m, subclass) 
            for e, f, w, j, a, d, h, e_h, j_m, q_m in zip(
                en, fs, wo, jd_h, aeg_h, di_h, hd_h, en_h, j_morimoto, q_morimoto
            )
        ])
        df_class = comps.copy()
        df_class["Mineral"] = labels[:,0]

        if subclass:
            df_class["Submineral"] = labels[:,1]
        
        df_class['En'] = comps['En']
        df_class['Fs'] = comps['Fs']
        df_class['Wo'] = comps['Wo']
        df_class['En_h'] = comps['En_h']
        df_class['Jd_h'] = comps['Jd_h']
        df_class['Aeg_h'] = comps['Aeg_h']
        df_class['Di_h'] = comps['Di_h']
        df_class['Hd_h'] = comps['Hd_h']

        return df_class

    def plot(self, df_class=None, subclass=True, labels="short", figsize=(8, 5), 
             **kw):

        """
        Plot pyroxene compositions on the DHZ quadrilateral.

        Parameters:
            df_class: Output of `.classify()`. If None, will call `.classify(subclass)`.
            subclass: Whether to color by Submineral (if False, colors by Mineral).
            figsize: Default (8,5)
            **kw: Passed to the field-boundary `tax.line(…)` calls (e.g. ls=':', lw=0.5).

        Returns:
            fig: matplotlib.figure.Figure
            tax: ternary.TernaryAxesSubplot
        """

        import ternary
        import matplotlib.pyplot as plt

        # get classification if needed
        if df_class is None:
            df_class = self.classify(subclass=subclass)

        non_sodic_px = df_class.loc[(df_class["Mineral"] == 'Orthopyroxene') | (df_class["Mineral"] == 'Clinopyroxene')]
        sodic_px = df_class.loc[(df_class["Mineral"] == "Na-Pyroxene")]

        figs = []

        if len(non_sodic_px) > 0:

            label_dict = {
                "Diopside": "Di", "Hedenbergite": "Hd", "Augite": "Au",
                "Pigeonite": "Pig", "Enstatite": "En", "Ferrosilite": "Fs"
            }

            if labels == "long":
                label_set = {k: k for k in label_dict}
            elif labels == "short" or labels is True:
                label_set = label_dict
            else:
                label_set = None

            # grab En, Fs, Wo for scatter
            pts = list(zip(non_sodic_px["Fs"].to_numpy(float),
                           non_sodic_px["Wo"].to_numpy(float),
                           non_sodic_px["En"].to_numpy(float)))

            # set up ternary
            fig, tax = ternary.figure(scale=1.0)
            fig.set_size_inches(figsize)
            tax.boundary(linewidth=1.5, zorder=0)
            tax.get_axes().set_ylim(-0.035, 0.43375)
            tax.left_corner_label("En\n$(\\mathregular{Mg_2Si_2O_6})$", fontsize=14, offset=-0.2)
            tax.right_corner_label("Fs\n$(\\mathregular{Fe_2Si_2O_6})$", fontsize=14, offset=-0.2)
            tax.top_corner_label("Wo\n$(\\mathregular{Ca_2Si_2O_6})$", fontsize=14)

            tax.gridlines(multiple=0.2, ls=":", lw=0.5, c="k", alpha=0.25, zorder=0)
            tax.gridlines(multiple=0.05, lw=0.25, c="lightgrey", alpha=0.25, zorder=0)

            # DHZ field boundaries
            lines = [
                ([0, 0.5, 0.5],[0.5, 0.5, 0]),
                ([0, 0.45, 0.55],[0.55, 0.45, 0]),
                ([0.25, 0.5, 0.25],[0.275, 0.45, 0.275]),
                ([0, 0.05, 0.95],[0.95, 0.05, 0]),
                ([0, 0.2, 0.8],[0.8, 0.2, 0]),
                ([0.5, 0, 0.5],[0.475, 0.05, 0.475]),
            ]
            for xs, ys in lines:
                tax.line(xs, ys, color="k", **kw, zorder=0)

            # scatter points
            if subclass and "Submineral" in non_sodic_px:
                cmap = plt.get_cmap("tab10")
                for i, g in enumerate(non_sodic_px["Submineral"].unique()):
                    mask = non_sodic_px["Submineral"] == g
                    mask_indices = np.where(mask)[0]
                    if len(mask_indices) > 0:  # Only plot if there are points
                        tax.scatter([pts[j] for j in mask_indices],
                                    marker='o', label=g, color=cmap(i),
                                    edgecolor='k', s=20, alpha=0.8)
                tax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.02,1))
            else:
                tax.scatter(pts, marker='o', color='C0', edgecolor='k', s=20, alpha=0.8)

            # draw & filter ticks so they only appear where the quadrilateral lives
            ax = tax.get_axes()
            def draw_and_filter(axis, offset=0.02, keep_min=None, keep_max=None):
                nL, nT = len(ax.lines), len(ax.texts)
                ticks = [i * 0.1 for i in range(11)]
                tax.ticks(axis=axis, ticks=ticks, linewidth=1, 
                          tick_formats="%.1f", offset=offset, fontsize=10)
                newL, newT = ax.lines[nL:], ax.texts[nT:]
                for L, T in zip(newL, newT):
                    v = float(T.get_text())
                    if (keep_min is not None and v < keep_min) or (keep_max is not None and v > keep_max):
                        L.set_visible(False)
                        T.set_visible(False)

            draw_and_filter('l', keep_min=0.5, keep_max=1.0)  # En axis
            draw_and_filter('r', keep_min=0.0, keep_max=0.5)  # Fs axis
            draw_and_filter('b', offset=0.01, keep_min=0.0, keep_max=1.0)  # Wo axis

            if label_set:
                fs = 12
                lab_z = 120
                ax = tax.get_axes()
                ax.text(0.375, 0.4, label_set["Diopside"], fontsize=fs, ha='center', va='center', zorder=lab_z)
                ax.text(0.625, 0.4, label_set["Hedenbergite"], fontsize=fs, ha='center', va='center', zorder=lab_z)
                ax.text(0.5, 0.275, label_set["Augite"], fontsize=fs, ha='center', va='center', zorder=lab_z)
                ax.text(0.5, 0.1, label_set["Pigeonite"], fontsize=fs, ha='center', va='center', zorder=lab_z)
                ax.text(0.25, 0.02, label_set["Enstatite"], fontsize=fs, ha='center', va='center', zorder=lab_z)
                ax.text(0.75, 0.02, label_set["Ferrosilite"], fontsize=fs, ha='center', va='center', zorder=lab_z)
            tax.clear_matplotlib_ticks()
            ax.axis("off")
            figs.append((fig, tax))

        if len(sodic_px) > 0:
            
            label_dict = {
                "Omphacite": 'Omph', "Aegirine-Augite": "Aeg-Aug",
                "Jadeite": "Jd", "Aegirine": "Aeg"
            }
            if labels == "long":
                label_set = {k: k for k in label_dict}
            elif labels == "short" or labels is True:
                label_set = label_dict
            else:
                label_set = None

            jd_raw  = pd.to_numeric(sodic_px["Jd_h"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
            en_raw = pd.to_numeric(sodic_px["En_h"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
            aeg_raw = pd.to_numeric(sodic_px["Aeg_h"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
            di_raw = pd.to_numeric(sodic_px["Di_h"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
            hd_raw = pd.to_numeric(sodic_px["Hd_h"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
            quad_raw  = en_raw + di_raw + hd_raw 
            total = quad_raw + jd_raw + aeg_raw
            jd = jd_raw/total
            aeg = aeg_raw/total
            quad = quad_raw/total

            pts_sodic = []
            for a, q, j in zip(aeg, quad, jd):
                pts_sodic.append((a, q, j)) # (R, T, L) for python-ternary

            fig_sod, tax_sod = ternary.figure(scale=1.0)
            fig_sod.set_size_inches((8, 8))
            tax_sod.boundary(linewidth=1.5, zorder=0)
            tax_sod.top_corner_label("En+Di-Hd\n(Ca-Mg-Fe Pyroxenes)", fontsize=14)
            tax_sod.left_corner_label("Jd\n$(\\mathregular{NaAlSi_2O_6})$", fontsize=14, offset=0.03)
            tax_sod.right_corner_label("Aeg\n$(\\mathregular{NaFe^{3+}Si_2O_6})$", fontsize=14, offset=0.03)
            tax_sod.gridlines(multiple=0.2, ls=":", lw=0.5, c="k", alpha=0.25, zorder=0)
            tax_sod.gridlines(multiple=0.05, lw=0.25, c="lightgrey", alpha=0.25, zorder=0)

            # Guide lines: top=0.2 and top=0.8; plus center connector
            lines = [
                ([0.0, 0.2, 0.8],[0.8, 0.2, 0.0]),
                ([0.0, 0.8, 0.2],[0.2, 0.8, 0.0]),
                ([0.5, 0.0, 0.5],[0.1, 0.8, 0.1])
            ]
            for xs, ys in lines:
                tax_sod.line(xs, ys, color="k", zorder=0)

            tax_sod.ticks(axis='lr', multiple=0.2, linewidth=1,
                          tick_formats="%.1f", offset=0.02, fontsize=10)
            tax_sod.ticks(axis='b', multiple=0.2, linewidth=1,
                          tick_formats="%.1f", offset=0.01, fontsize=10)

            # Scatter: color by Submineral if present
            if subclass and "Submineral" in sodic_px.columns:
                cmap = plt.get_cmap("tab10")
                subs = sodic_px["Submineral"].fillna(np.nan)
                
                for i, g in enumerate(subs.unique()):
                    # Create mask for this submineral group
                    mask = subs == g
                    # Get the points for this group
                    group_pts = [pt for pt, m in zip(pts_sodic, mask) if m]
                    
                    # plot if there are points
                    if len(group_pts) > 0:
                        tax_sod.scatter(group_pts, marker='o', label=str(g), color=cmap(i),
                                         edgecolor='k', s=20, alpha=0.85)
                # Only add legend if there are any points plotted
                if len(pts_sodic) > 0:
                    tax_sod.legend(loc='upper left', fontsize=10, bbox_to_anchor=(1.02,1))
            else:
                # Fallback: plot all points without subclass coloring
                if len(pts_sodic) > 0:
                    tax_sod.scatter(pts_sodic, marker='o', color='C1',
                                    edgecolor='k', s=22, alpha=0.85)

            if label_set:
                fs = 12
                lab_z = 120
                ax = tax_sod.get_axes()
                ax.text(0.375, 0.4, label_set["Omphacite"], fontsize=fs, ha='center', va='center', zorder=lab_z)
                ax.text(0.625, 0.4, label_set["Aegirine-Augite"], fontsize=fs, ha='center', va='center', zorder=lab_z)
                ax.text(0.3, 0.09, label_set["Jadeite"], fontsize=fs, ha='center', va='center', zorder=lab_z)
                ax.text(0.7, 0.09, label_set["Aegirine"], fontsize=fs, ha='center', va='center', zorder=lab_z)

            tax_sod.clear_matplotlib_ticks()
            tax_sod.get_axes().axis("off")
            figs.append((fig_sod, tax_sod))

        if not figs:
            return None
        if len(figs) == 1:
            return figs[0]
        return figs


class QuartzCalculator(BaseMineralCalculator):
    """Quartz-specific calculations. SiO2."""
    OXYGEN_BASIS = 2
    MINERAL_SUFFIX = "_Qz"

    def calculate_components(self):
        """Return complete quartz composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Al = base.get(f"Al{cat_suffix}", 0)
        Ti = base.get(f"Ti{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["T_site"] = Si + Al + Ti

        return pd.concat([base, sites], axis=1)


class RutileCalculator(BaseMineralCalculator):
    """Rutile-specific calculations. TiO2."""
    OXYGEN_BASIS = 2
    MINERAL_SUFFIX = "_Rt"

    # Extend the parent's dictionaries by merging them with Nb2O5 and Ta2O5 data
    OXIDE_MASSES = dict(BaseMineralCalculator.OXIDE_MASSES, **{"Nb2O5": 265.8098, "Ta2O5": 441.8928})
    OXYGEN_NUMBERS = dict(BaseMineralCalculator.OXYGEN_NUMBERS, **{"Nb2O5": 5, "Ta2O5": 5})
    CATION_NUMBERS = dict(BaseMineralCalculator.CATION_NUMBERS, **{"Nb2O5": 2, "Ta2O5": 2})
    OXIDE_TO_CATION_MAP = dict(BaseMineralCalculator.OXIDE_TO_CATION_MAP, **{"Nb2O5": "Nb", "Ta2O5": "Ta"})

    def calculate_components(self):
        """Return complete rutile composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Ti = base[f"Ti{cat_suffix}"]
        Nb = base.get(f"Nb{cat_suffix}", 0)
        Ta = base.get(f"Ta{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["M_site"] = Ti
        # sites["M_site_expanded"] = Ti + Nb + Ta

        return pd.concat([base, sites], axis=1)


class SerpentineCalculator(BaseMineralCalculator):
    """Serpentine-specific calculations. Mg3[Si2O5](OH)4."""
    OXYGEN_BASIS = 14
    MINERAL_SUFFIX = "_Srp"

    def calculate_components(self):
        """Return complete serpentine composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base[f"Mg{cat_suffix}"]

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["M_site"] = Mg + Fe + Mn
        sites["T_site"] = Si + Al # tetrahedral
        sites["XMg"] = Mg / (Mg + Fe)
        sites["XFe"] = Fe / (Mg + Fe)

        return pd.concat([base, sites], axis=1)


class SodicPyroxeneCalculator(BaseMineralCalculator):
    """Sodic Pyroxene-specific calculations. (Na,Ca)(Mg,Fe3+,Al)Si2O6."""
    OXYGEN_BASIS = 6
    MINERAL_SUFFIX = "_NaPx"

    def _o_total_from_4cat(self):
        """
        1) compute moles from oxides
        2) convert to cations per oxide (multiply by cation numbers)
        3) normalize so total cations = 4
        4) from those cat-normalized moles, compute O2_total using oxide stoichiometry
        Returns: O_total both as pandas Series
        """
        moles = self.calculate_moles().copy()
        moles.columns = [c.replace("_mols", "") for c in moles.columns]

        # cations per oxide (not yet normalized)
        cat_nums = pd.Series(self.CATION_NUMBERS)
        # keep only oxides defined in cat_nums
        moles2 = moles.loc[:, moles.columns.intersection(cat_nums.index)]
        cations_raw = moles2.multiply(cat_nums, axis="columns")

        # total cations and 4-cation normalization factor
        cat_sum = cations_raw.sum(axis=1).replace(0, np.nan)
        cat_norm = 4.0 / cat_sum

        # cat-normalized moles (apply factor to moles
        moles_4cat = moles.multiply(cat_norm, axis="rows")

        # compute O_total from cat-normalized moles and per-oxide oxygen stoichiometry
        O_per_oxide = pd.Series(self.OXYGEN_NUMBERS)
        moles_4cat = moles_4cat.loc[:, moles_4cat.columns.intersection(O_per_oxide.index)]
        O_total = moles_4cat.multiply(O_per_oxide, axis="columns").sum(axis=1)

        return O_total


    def calculate_components(self):
        """Return complete sodic pyroxene composition with site assignments and classic px, jadeite, aegirine, etc. assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base[f"Mg{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        # sites = sites.reset_index(drop=True)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["Oxygen_Sum"] = self._o_total_from_4cat()
        sites["M_site"] = Mg + Fe + Ca + Na + Ti + Cr
        sites["T_site"] = Si + Al
        sites["XMg"] = (Mg / (Mg + Fe))
        sites["En"] = Mg / (Mg + Fe + Ca)
        sites["Fs"] = Fe / (Mg + Fe + Ca)
        sites["Wo"] = Ca / (Mg + Fe + Ca)
        sites["Jd"] = Na
        sites["Q"] = Ca + Mg + Fe
        sites["J"] = 2 * Na

        # This code is modified from Thermobar, with permission from Penny Wieser
        sites["Al_IV"] = 2 - Si
        sites["Al_VI"] = (Al - sites["Al_IV"]).clip(lower=0)

        # Fe3+, Fe2+ Calculation
        O_def = (6 - sites["Oxygen_Sum"]).clip(lower=0)
        sites["Fe3_oxdef"] = np.minimum(Fe, 2.0*O_def)
        sites["Fe2_oxdef"] = (Fe - sites["Fe3_oxdef"]).clip(lower=0)

        # Fe3+, Fe2+ Calculation
        sites["Fe3_Lindley"] = (
            Na + sites["Al_IV"]
            - sites["Al_VI"]
            - (2 * Ti) - Cr
        ).clip(lower=0, upper=Fe) # Fe3 can't be negative or greater than Fe
        sites.loc[sites["Fe3_Lindley"] < 1e-10, "Fe3_Lindley"] = 0
        sites["Fe2_Lindley"] = Fe - sites["Fe3_Lindley"]
        sites["Fe3_prop_Lindley"] = (sites["Fe3_Lindley"] / Fe).replace(0, np.nan)

        # Independent cpx components
        sites["CrCaTs"] = 0.5 * Cr
        sites['a_cpx_En'] = (
            (1 - Ca - Na - K) * 
            (1 - 0.5 * (Al + Cr + Na + K))
        )

        # If value of AlVI < Na cation fraction
        sites["CaTs"] = sites["Al_VI"] - Na
        CaTs_mask = sites["CaTs"] < 0
        sites["Jd_from 0=Na, 1=Al"] = 0
        sites.loc[CaTs_mask, "Jd_from 0=Na, 1=Al"] = 1
        sites.loc[CaTs_mask, "Jd"] = sites.loc[CaTs_mask, "Al_VI"].to_numpy()
        sites["CaTs"] = sites["CaTs"].clip(lower=0)

        # CaTi component
        sites["CaTi"] = ((sites["Al_IV"] - sites["CaTs"]) / 2).clip(lower=0)

        # DiHd (Diopside-Hedenbergite) component
        sites["DiHd_1996"] = (Ca - sites["CaTs"] - sites["CaTi"] - sites["CrCaTs"]).clip(lower=0)
        sites["EnFs"] = ((Fe + Mg) - sites["DiHd_1996"]) / 2
        sites["DiHd_2003"] = (Ca - sites["CaTs"] - sites["CaTi"] - sites["CrCaTs"]).clip(lower=0)
        sites["Di"] = sites["DiHd_2003"] * (
            Mg / (Mg + Mn + Fe).replace(0, np.nan)
        )

        # Wang 2021 Fe3+
        sites["Fe3_Wang21"] = (Na + sites["Al_IV"] - sites["Al_VI"] - 2 * Ti - Cr)
        sites["Fe2_Wang21"] = Fe - sites["Fe3_Wang21"]

        # This code was converted from MATLAB to Python, from Jesse Walter's MinPlot
        # Verified for alignment between values. 
        # Sodic pool after balancing Ti
        excess_Si = (Si - 2.0).clip(lower=0)
        A1n = (sites["Al_IV"] - Ti).clip(lower=0) 
        A3n = (sites["Al_VI"] - A1n).clip(lower=0) 
        Fe3 = sites["Fe3_oxdef"].clip(lower=0) # Oxygen deficiency for now, would be different with other implementations
        M3 = Cr + 2.0 * (excess_Si + Ti) + sites["Fe3_oxdef"] + (sites["Al_VI"] - sites["Al_IV"])
        e = (M3 - (Na + K)).clip(lower=0.0) 
        g  = (Ca - 0.5*e - A1n).clip(lower=0.0)

        sites["Di_h"] = np.minimum(g, Mg)
        sites["Hd_h"] = (g - sites["Di_h"]).clip(lower=0.0)
        zp = (Na + K - Ti).clip(lower=0)
        zm = (zp - A3n).clip(lower=0)
        Ko = np.minimum(Cr, zm)
        zk = (zm - Cr).clip(lower=0)

        delta = (sites["Al_VI"]  - (sites["Al_IV"] - Ti))
        F3n = np.where(delta < 0, Fe3 + delta, Fe3) # reduce Fe3 if needed
        F3n = np.clip(F3n, 0.0, None)

        Aeg = np.minimum(Fe3.clip(lower=0.0), zk)
        sites["Aeg_h"] = pd.Series(Aeg, index=sites.index).clip(lower=0)

        Jd  = np.minimum(A3n, zp)
        sites["Jd_h"] = pd.Series(Jd, index=sites.index).clip(lower=0)

        # Harlow's XEn (h): (Mg + Mn + Fe2 - g) / 2, clipped ≥ 0
        En_h = ((Mg + Mn + sites["Fe2_oxdef"]) - g) / 2.0
        En_h = En_h.clip(lower=0.0)
        sites["En_h"] = En_h # store it so classify() can use it


        return pd.concat([base, sites], axis=1)


class SpinelCalculator(BaseMineralCalculator):
    """Spinel group-specific calculations. MgAl2O4, Fe3O4, AB2X4."""
    OXYGEN_BASIS = 4
    CATION_BASIS = 3
    MINERAL_SUFFIX = "_Sp"

    def calculate_components(self, Fe_correction="Droop"):
        """Return complete spinel composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]
        total_cat = base[cation_cols].sum(axis=1)
        T_S = self.CATION_BASIS / total_cat
        base["S_init"] = total_cat # initial total cations
        base["T_S"] = T_S
        Fe_init = base[f"Fe2t{cat_suffix}"]

        if Fe_correction == "Droop":
            # Droop (1987) equation
            scaled = base.loc[:, cation_cols].mul(T_S.values, axis=0)
            base.loc[:, cation_cols] = scaled.to_numpy()
            Fe_corr = base[f"Fe2t{cat_suffix}"]
            Fe3 = (2 * self.OXYGEN_BASIS * (1 - self.CATION_BASIS / base["S_init"])).clip(lower=0)
            Fe3_prop = (Fe3 / Fe_corr).clip(upper=1)
            Fe2 = Fe_corr - Fe3
        elif Fe_correction == "All_Fe2":
            Fe2 = Fe_init
            Fe3 = pd.Series(0, index=base.index)
            Fe3_prop = pd.Series(0, index=base.index)
        elif Fe_correction == "All_Fe3":
            Fe2 = Fe_init
            Fe3 = pd.Series(0, index=base.index)
            Fe3_prop = pd.Series(1, index=base.index)
        else:
            raise ValueError("Invalid Fe_correction: choose 'Droop', 'All_Fe2', or 'All_Fe3'.")
        base["FeO"] = base["FeOt"] * (1 - Fe3_prop)
        base["Fe2O3"] = base["FeOt"] * Fe3_prop * (1 / 0.89992485)
        base[f"Fe3{cat_suffix}"] = Fe3
        base[f"Fe2{cat_suffix}"] = Fe2

        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base.get(f"Al{cat_suffix}", 0) 
        Fe2 = base[f"Fe2{cat_suffix}"]
        Fe3 = base[f"Fe3{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base.get(f"Mg{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["A_site"] = Mg + Fe2 # Mn, Zn, Ni, Co, Ni, Cu, Ge
        sites["A_site_expanded"] = Mg + Fe2 + Mn # Zn, Ni, Co, Ni, Cu, Ge
        sites["B_site"] = Al + Ti + Cr + Fe3 # V
        sites["A_B_site"] = Mg + Fe2 + Mn + Al + Ti + Cr + Fe3
        sites["Fe_Ti"] = Fe2 + Fe3 + Ti
        sites['Fe3_prop'] = Fe3 / (Fe2 + Fe3)
        total = (Fe2 + Fe3 + Ti + Al + Cr + Mn + Mg)
        sites["XR2"] = (Fe2 + Mn + Mg) / total # having as true R2+ (Fe+Mg+Mn) pushes all spinels away
        sites["XR3"] = (Fe3 + Al + Cr) / total
        sites["XTi"] = Ti / total

        return pd.concat([base, sites], axis=1)
    

class TitaniteCalculator(BaseMineralCalculator):
    """Titanite-specific calculations. CaTiSiO5."""
    OXYGEN_BASIS = 5
    MINERAL_SUFFIX = "_Tit"

    def calculate_components(self):
        """Return complete titanite composition with site assignments."""
        if "FeOt" in self.comps.columns:
            self.comps["Fe2O3t"] = self.comps.pop("FeOt") * (self.OXIDE_MASSES["Fe2O3"] / (2 * self.OXIDE_MASSES["FeO"]))

        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe3t{cat_suffix}"]
        Mg = base[f"Mg{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["VII_site"] = Ca + Mg + Na + K # Fe2, seven coordinated
        sites["M_site"] = Al + Fe + Ti # Fe3, octahedral 
        sites["T_site"] = Si # tetrahedral
        sites["Al_IV"] = 1 - Si
        sites["Al_VI"] = Al - sites["Al_IV"]

        return pd.concat([base, sites], axis=1)


class TourmalineCalculator(BaseMineralCalculator):
    """Tourmaline-specific calculations. XY3Z6[Si6O18](BO3)3(O,OH)3(OH,F,O)."""
    OXYGEN_BASIS = 31
    MINERAL_SUFFIX = "_Trm"

    # Extend the parent's dictionaries by merging them with B2O3 data
    OXIDE_MASSES = dict(BaseMineralCalculator.OXIDE_MASSES, **{"B2O3": 69.6182})
    OXYGEN_NUMBERS = dict(BaseMineralCalculator.OXYGEN_NUMBERS, **{"B2O3": 3})
    CATION_NUMBERS = dict(BaseMineralCalculator.CATION_NUMBERS, **{"B2O3": 2})
    OXIDE_TO_CATION_MAP = dict(BaseMineralCalculator.OXIDE_TO_CATION_MAP, **{"B2O3": "B"})

    def calculate_components(self):
        """Return complete apatite composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base[f"Mg{cat_suffix}"]
        Ca = base[f"Ca{cat_suffix}"]
        Na = base.get(f"Na{cat_suffix}", 0)
        K = base.get(f"K{cat_suffix}", 0)
        Cr = base.get(f"Cr{cat_suffix}", 0)
        B = base.get(f"B{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["X_site"] = Na + Ca + K
        sites["Y_site"] = Mg + Fe + Mn + Al # Fe2, Fe3
        sites["Z_site"] = Mg + Fe + Al + Cr # Fe3
        sites['T_site'] = Si
        # sites["V_site"] = OH, O
        # sites["W_site"] = OH, O, F

        return pd.concat([base, sites], axis=1)


class ZirconCalculator(BaseMineralCalculator):
    """Zircon-specific calculations. ZrSiO4."""
    OXYGEN_BASIS = 4
    MINERAL_SUFFIX = "_Zr"

    # Extend the parent's dictionaries by merging them with ZrO2 and HfO2 data
    OXIDE_MASSES = dict(BaseMineralCalculator.OXIDE_MASSES, **{"ZrO2": 123.222, "HfO2": 210.484})
    OXYGEN_NUMBERS = dict(BaseMineralCalculator.OXYGEN_NUMBERS, **{"ZrO2": 2, "HfO2": 2})
    CATION_NUMBERS = dict(BaseMineralCalculator.CATION_NUMBERS, **{"ZrO2": 1, "HfO2": 1})
    OXIDE_TO_CATION_MAP = dict(BaseMineralCalculator.OXIDE_TO_CATION_MAP, **{"ZrO2": "Zr", "HfO2": "Hf"})

    def calculate_components(self):
        """Return complete zircon composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Si = base[f"Si{cat_suffix}"]
        Zr = base.get(f"Zr{cat_suffix}", 0)
        Hf = base.get(f"Hf{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["M_site"] = Zr
        sites["T_site"] = Si
        # sites["Hf_Zr"] = np.where(Zr > 0, Hf / Zr, np.nan)

        return pd.concat([base, sites], axis=1)


# %%
