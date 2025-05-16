# %%

import numpy as np
import pandas as pd

# %%


class BaseMineralCalculator:
    """
    Base class for mineral composition calculations.
    Implement calculate_components() for each mineral.
    """
    
    # Constants for all minerals
    OXIDE_MASSES = {
        "SiO2": 60.0843,
        "TiO2": 79.7877,
        "Al2O3": 101.961,
        "FeOt": 71.8464,
        "Fe2O3t": 159.688,
        "FeO": 71.8464,
        "Fe2O3": 159.688,
        "MnO": 70.9375,
        "MgO": 40.3044,
        "CaO": 56.0774,
        "Na2O": 61.9789,
        "K2O": 94.196,
        "P2O5": 141.9445,
        "Cr2O3": 151.9902,
    }

    OXYGEN_NUMBERS = {
        "SiO2": 2, "TiO2": 2, "Al2O3": 3, "FeOt": 1, "Fe2O3t": 3, "FeO": 1, "Fe2O3": 3,
        "MnO": 1, "MgO": 1, "CaO": 1, "Na2O": 1, "K2O": 1, "P2O5": 5, "Cr2O3": 3
    }
    
    CATION_NUMBERS = {
        "SiO2": 1, "TiO2": 1, "Al2O3": 2, "FeOt": 1, "Fe2O3t": 2, "FeO": 1, "Fe2O3": 2,
        "MnO": 1, "MgO": 1, "CaO": 1, "Na2O": 2, "K2O": 2, "P2O5": 2, "Cr2O3": 2
    }
    
    OXIDE_TO_CATION_MAP = {
        "SiO2": "Si", "TiO2": "Ti", "Al2O3": "Al", 
        "FeOt": "Fe2t", "Fe2O3t": "Fe3t", 
        "FeO": "Fe2", "Fe2O3": "Fe3", 
        "MnO": "Mn", "MgO": "Mg", "CaO": "Ca", "Na2O": "Na",
        "K2O": "K", "P2O5": "P", "Cr2O3": "Cr"
    }

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
        mole_cols   = [f"{ox}_mols"       for ox in oxide_cols]
        oxygen_cols = [f"{ox}_ox"         for ox in oxide_cols]
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
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["M_site"] = Mg + Fe + Ca + Na + Ti + Cr
        sites["T_site"] = Si + Al
        sites["XMg"] = (Mg / (Mg + Fe))
        sites["En"] = Mg / (Mg + Fe + Ca)
        sites["Fs"] = Fe / (Mg + Fe + Ca)
        sites["Wo"] = Ca / (Mg + Fe + Ca)

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
        sites["Jd"] = Na
        sites["Jd_from 0=Na, 1=Al"] = 0
        sites.loc[sites["CaTs"] < 0, "Jd_from 0=Na, 1=Al"] = 1
        sites.loc[sites["CaTs"] < 0, "Jd"] = sites["Al_VI"]
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
        sites["A_site"] = K + Na
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


class MagnetiteCalculator(BaseMineralCalculator):
    """Magnetite-specific calculations. Fe3O4."""
    OXYGEN_BASIS = 4
    MINERAL_SUFFIX = "_Mt"

    def calculate_components(self):
        """Return complete magnetite composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]

        Ti = base.get(f"Ti{cat_suffix}", 0)
        Al = base[f"Al{cat_suffix}"]
        Fe = base[f"Fe2t{cat_suffix}"]
        Mn = base.get(f"Mn{cat_suffix}", 0)
        Mg = base[f"Mg{cat_suffix}"]
        Cr = base.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["A_site"] = Mg + Fe # Mn, Zn, Ni, Co, Ni, Cu, Ge
        sites["A_site_expanded"] = Mg + Fe + Mn # Zn, Ni, Co, Ni, Cu, Ge
        sites["B_site"] = Al + Ti + Cr # Fe3, V
        sites["Fe_Ti"] = Fe + Ti

        return pd.concat([base, sites], axis=1)


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
        sites["B_site"] = Mg + Fe
        sites["T_site"] = Si + Al # tetrahedral

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


class OxideCalculator(BaseMineralCalculator):
    """Oxide-specific calculations. Fe-Ti Oxides. Hematite-Ilmenite, Fe2O3-(FeTi)2O3."""
    OXYGEN_BASIS = 3
    CATION_BASIS = 2
    MINERAL_SUFFIX = "_Ox"

    def calculate_components(self, Fe_correction="Droop"):
        """Return complete Fe-Ti oxide composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]
        Fe = base[f"Fe2t{cat_suffix}"]

        if Fe_correction == "Droop":
            # Droop (1987) equation
            total_cat = base[cation_cols].sum(axis=1)
            Fe3 = (2 * self.OXYGEN_BASIS * (1 - (self.CATION_BASIS / total_cat))).clip(lower=0)
            Fe3_prop = (Fe3 / Fe).clip(upper=1)
            Fe2 = Fe - Fe3
        elif Fe_correction == "All_Fe2":
            Fe2 = Fe
            Fe3 = pd.Series(0, index=base.index)
            Fe3_prop = pd.Series(0, index=base.index)
        else:
            raise ValueError("Invalid Fe_correction: choose 'Droop' or 'All_Fe2'")

        base["FeO"] = base["FeOt"] * (1 - Fe3_prop)
        base["Fe2O3"] = base["FeOt"] * Fe3_prop * self.OXIDE_MASSES["Fe2O3t"] / (2 * self.OXIDE_MASSES["FeOt"])
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
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base_update[update_cation_cols].sum(axis=1)

        Ti = base_update.get(f"Ti{cat_suffix}", 0)
        Al = base_update[f"Al{cat_suffix}"]
        Fe2 = base_update[f"Fe2{cat_suffix}"]
        Fe3 = base_update[f"Fe3{cat_suffix}"]
        Mn = base_update[f"Mn{cat_suffix}"]
        Mg = base_update[f"Mg{cat_suffix}"]
        Cr = base_update.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base_update.index)
        sites["Cation_Sum"] = base_update[cation_cols].sum(axis=1)
        sites["A_site"] = Mg + Fe2 # Mn, Zn, Ni, Co, Ni, Cu, Ge
        sites["A_site_expanded"] = Mg + Fe2 + Mn # Zn, Ni, Co, Ni, Cu, Ge
        sites["B_site"] = Al + Ti + Cr + Fe3 # V
        sites["A_B_site"] = Mg + Fe2 + Mn + Al + Ti + Cr + Fe3
        sites["Fe_Ti"] = Fe2 + Fe3 + Ti
        sites['Fe3_prop'] = Fe3 / (Fe2 + Fe3)

        sites["XHem"] = Fe3 / (Fe3 + Ti) # Hematite
        remainder = 1 - sites["XHem"]
        denominator = Fe2 + Mn + Mg
        sites["XIlm"] = (Fe2 / denominator) * remainder # Ilmenite
        sites["XMnIlm"] = (Mn / denominator) * remainder # Mn-Ilmenite
        sites["XGk"] = (Mg / denominator) * remainder # Geikielite (MgTiO3)
        sites["XSum"] = sites["XHem"] + sites["XIlm"] + sites["XMnIlm"] + sites["XGk"]

        return pd.concat([base_update, sites], axis=1)


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

    # Extend the parent's dictionaries by merging them with ZrO2 and Ta2O5 data
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


class SpinelCalculator(BaseMineralCalculator):
    """Spinel-specific calculations. MgAl2O4."""
    OXYGEN_BASIS = 4
    CATION_BASIS = 3
    MINERAL_SUFFIX = "_Sp"

    def calculate_components(self, Fe_correction="Droop"):
        """Return complete spinel composition with site assignments."""
        base = self.calculate_all()  # includes self.comps, moles, oxygens, and cations
        cat_suffix = f"_cat_{self.OXYGEN_BASIS}ox"

        # Grab just the cation columns from `base`
        cation_cols = [col for col in base.columns if col.endswith(cat_suffix)]
        Fe = base[f"Fe2t{cat_suffix}"]

        if Fe_correction == "Droop":
            # Droop (1987) equation
            total_cat = base[cation_cols].sum(axis=1)
            Fe3 = (2 * self.OXYGEN_BASIS * (1 - self.CATION_BASIS / total_cat)).clip(lower=0)
            Fe3_prop = (Fe3 / Fe).clip(upper=1)
            Fe2 = Fe - Fe3
        elif Fe_correction == "All_Fe2":
            Fe2 = Fe
            Fe3 = pd.Series(0, index=base.index)
            Fe3_prop = pd.Series(0, index=base.index)
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
        
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base_update[update_cation_cols].sum(axis=1)
        
        Ti = base_update.get(f"Ti{cat_suffix}", 0)
        Al = base_update[f"Al{cat_suffix}"]
        Fe2 = base_update[f"Fe2{cat_suffix}"]
        Fe3 = base_update[f"Fe3{cat_suffix}"]
        Mn = base_update[f"Mn{cat_suffix}"]
        Mg = base_update[f"Mg{cat_suffix}"]
        Cr = base_update.get(f"Cr{cat_suffix}", 0)

        # Compute site assignments in sites dataframe
        sites = pd.DataFrame(index=base.index)
        sites["Cation_Sum"] = base[cation_cols].sum(axis=1)
        sites["A_site"] = Mg + Fe2 # Mn, Zn, Ni, Co, Ni, Cu, Ge
        sites["A_site_expanded"] = Mg + Fe2 + Mn # Zn, Ni, Co, Ni, Cu, Ge
        sites["B_site"] = Al + Ti + Cr + Fe3 # V
        sites["A_B_site"] = Mg + Fe2 + Mn + Al + Ti + Cr + Fe3
        sites["Fe_Ti"] = Fe + Ti
        sites['Fe3_prop'] = Fe3 / (Fe2 + Fe3)

        return pd.concat([base_update, sites], axis=1)


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

    # Extend the parent's dictionaries by merging them with ZrO2 data
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
        sites["Hf_Zr"] = Hf / Zr

        return pd.concat([base, sites], axis=1)


# %%
