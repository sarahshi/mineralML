# %%

__author__ = "Sarah Shi"

import pandas as pd

from .constants import OXIDES

# %% 

def get_oxide_from_elem(elem, found_oxides):
    """
    Helper to match an element name (e.g., 'Ni') to a dynamically found oxide 
    (e.g., 'NiO') in the dataset.
    """
    if elem == 'Fe':
        if 'FeOt' in found_oxides: 
            return 'FeOt'
        if 'FeO' in found_oxides: 
            return 'FeO'
    for ox in found_oxides:
        # e.g., if elem is 'Ni', matches 'NiO'. If elem is 'P', matches 'P2O5'.
        if ox.startswith(elem) and ox.replace(elem, '') in ['O', 'O2', '2O3', 'O3', '2O', '2O5']:
            return ox
    return elem # Fallback to the element name if no oxide match is found


def extract_cameca(file_path, sheet_name=0, oxide_label="Oxide", std_label="StdDev wt%"):
    """
    Extracts, cleans, and formats oxide concentrations and standard deviations 
    from a Cameca EPMA data file (.csv or .xlsx). The function navigates the 
    multi-level headers typical of Cameca exports, identifying the oxide data
    block and their standard deviations. 

    Operations:
      - Identify oxide concentrations up to the 'Total' column. 
      - Dynamically pairs elemental sigmas (e.g. 'Ni') to captured oxides ('NiO')
        and renames columns --> 'NiO_1sigma'. 
      - Converts 'FeO' to 'FeOt'.
      - Sorts columns in an order: [Identifiers] -> [Concentrations] -> 
        [Total] -> [Standard Deviations].
      - Converts the default 3-sigma values to 1-sigma.
      - Renames 'Comment' to 'Sample' for standardization.
    """
    is_excel = file_path.lower().endswith(('.xlsx', '.xls'))
    
    # --- Find Header ---
    def get_header_idx():
        targets = ["Beam curr (nA)", "wt%", "Oxide"]
        if not is_excel:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if any(t in line for t in targets): return i
        else:
            df_temp = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=200)
            for i, row in df_temp.iterrows():
                row_str = " ".join(row.dropna().astype(str))
                if any(t in row_str for t in targets): return i
        return 0

    header_idx = get_header_idx()
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=header_idx, header=[0, 1]) if is_excel else \
         pd.read_csv(file_path, skiprows=header_idx, header=[0, 1])

    # --- Clean MultiIndex ---
    lvl0, lvl1 = df.columns.get_level_values(0).astype(str), df.columns.get_level_values(1)
    new_lvl0, last_cat, stop_filling = [], "", False
    
    for c0, c1 in zip(lvl0, lvl1):
        if str(c1).strip() == "X": stop_filling = True
        if stop_filling: new_lvl0.append("")
        elif "Unnamed" in c0 or c0 == "nan": new_lvl0.append(last_cat)
        else: last_cat = c0.strip(); new_lvl0.append(last_cat)

    df.columns = pd.MultiIndex.from_tuples(
        [(h0, 'FeOt' if str(h1).strip() == 'FeO' else str(h1).strip()) for h0, h1 in zip(new_lvl0, lvl1)]
    )

    id_cols = {"Comment"}

    # --- Dynamically Categorize Columns ---
    id_cols_to_keep, oxide_cols_to_keep, sigma_cols_to_keep = [], [], []

    for col in df.columns:
        if col[1] in id_cols: id_cols_to_keep.append(col)

    # Capture Oxides dynamically up to 'Total'
    for col in df.columns:
        if oxide_label in col[0]:
            oxide_cols_to_keep.append(col)
            if col[1] == 'Total': break

    # Extract the found oxide names so we can map the sigmas accurately
    found_ox = [c[1] for c in oxide_cols_to_keep if c[1] != 'Total']

    # Capture Sigmas dynamically mapping to the found oxides (Fixes the Ni -> NiO bug)
    for col in df.columns:
        if std_label in col[0]:
            mapped_c1 = get_oxide_from_elem(col[1], found_ox)
            sigma_cols_to_keep.append((col[0], mapped_c1, col)) 

    # Extract Data
    raw_keep_cols = id_cols_to_keep + oxide_cols_to_keep + [orig for _, _, orig in sigma_cols_to_keep]
    final_df = df[raw_keep_cols].dropna(how='all').copy()

    # Apply renames
    rename_dict = {orig: (c0, mapped) for c0, mapped, orig in sigma_cols_to_keep}
    final_df.columns = pd.MultiIndex.from_tuples([rename_dict.get(c, c) for c in final_df.columns])

    # --- Reorder Based on OXIDES List ---
    final_ox_order = [ox for ox in OXIDES if ox in found_ox] + \
                     [ox for ox in found_ox if ox not in OXIDES] + \
                     (['Total'] if 'Total' in [c[1] for c in oxide_cols_to_keep] else [])
    
    found_sig = [mapped for _, mapped, _ in sigma_cols_to_keep]
    final_sig_order = [ox for ox in OXIDES if ox in found_sig] + \
                      [ox for ox in found_sig if ox not in OXIDES and ox != 'Total']

    # Build the final ordered multi-index columns
    ordered_multi_cols = id_cols_to_keep.copy()
    ox_tuple_map = {c[1]: c for c in oxide_cols_to_keep}
    
    for ox in final_ox_order: 
        ordered_multi_cols.append(ox_tuple_map[ox])
    for sig in final_sig_order:
        match = [c for c in final_df.columns if c[1] == sig and std_label in c[0]]
        if match: 
            ordered_multi_cols.append(match[0])

    final_df = final_df[ordered_multi_cols]

    # --- Flatten to 1-Level and Calculate 1-Sigma ---
    flat_cols, sigma_targets = [], []
    for c0, c1 in final_df.columns:
        if c1 in id_cols:
            flat_cols.append(c1)
        elif std_label in c0:
            new_name = f"{c1}_1sigma"
            flat_cols.append(new_name)
            sigma_targets.append(new_name)
        else:
            flat_cols.append(c1)
            
    final_df.columns = flat_cols
    
    # Safely convert 3-sigma to 1-sigma on the newly flattened columns
    for col in sigma_targets:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce') / 3.0
        
    # --- Standardize Identifier Name ---
    final_df = final_df.rename(columns={'Comment': 'Sample'})
    
    return final_df


def extract_probe4epma(file_path, sheet_name=0):
    """
    Extracts, cleans, and formats oxide concentrations and standard deviations 
    from a Probe4EPMA data file (.csv or .xlsx). The function identifies the 
    oxide data block and their corresponding relative percent error columns. 

    Operations:
      - Dynamically captures all oxides in the block ending with the TOTAL column.
      - Automatically pairs elemental %ERR columns to their corresponding oxides.
      - Converts Relative % Error to Absolute 1-sigma Error.
      - Sorts columns in an order: [Identifiers] -> [Concentrations] -> 
        [Total] -> [Standard Deviations].
      - Drops 'DESCRIP' and 'NUMBER' and renames 'SAMPLE' to 'Sample'.
    """
    is_excel = file_path.lower().endswith(('.xlsx', '.xls'))
    df = pd.read_excel(file_path, sheet_name=sheet_name) if is_excel else pd.read_csv(file_path)

    # Clean Headers (Probe4EPMA leaves trailing spaces like 'Si %ERR ')
    df.columns = [str(c).strip() for c in df.columns]

    id_cols = ['SAMPLE']
    id_cols_to_keep = [col for col in id_cols if col in df.columns]

    # --- Dynamically Capture Oxides ---
    oxide_cols_to_keep = []
    capturing_oxides = False
    original_total_col = None

    for c in df.columns:
        if not capturing_oxides and c in OXIDES:
            capturing_oxides = True
            
        if capturing_oxides:
            if c == 'O': 
                continue 
            oxide_cols_to_keep.append(c)
            if 'TOTAL' in c:
                original_total_col = c
                break

    # --- Dynamically Capture ALL %ERR Columns ---
    sigma_cols_to_keep = [c for c in df.columns if '%ERR' in c]

    raw_keep_cols = id_cols_to_keep + oxide_cols_to_keep + sigma_cols_to_keep
    final_df = df[raw_keep_cols].dropna(how='all').copy()

    if original_total_col in final_df.columns:
        final_df = final_df.rename(columns={original_total_col: 'Total'})

    # --- Use helper to map Elements to Dynamically Captured Oxides ---
    found_ox = [c for c in oxide_cols_to_keep if c != original_total_col]
    
    # --- Convert Relative % Error to Absolute 1-sigma Error ---
    rename_dict = {}
    for err_col in sigma_cols_to_keep:
        elem = err_col.split(' ')[0] # e.g., 'Ni' from 'Ni %ERR'
        oxide_col = get_oxide_from_elem(elem, found_ox)
        
        if oxide_col and oxide_col in final_df.columns:
            rel_err_decimal = pd.to_numeric(final_df[err_col], errors='coerce') / 100.0
            oxide_wt_pct = pd.to_numeric(final_df[oxide_col], errors='coerce')
            
            final_df[err_col] = oxide_wt_pct * rel_err_decimal
            rename_dict[err_col] = f"{oxide_col}_1sigma"
        else:
            final_df = final_df.drop(columns=[err_col]) 

    # Apply the renaming schema
    final_df = final_df.rename(columns=rename_dict)

    # --- Reorder Columns ---
    final_ox_order = [ox for ox in OXIDES if ox in found_ox] + \
                     [ox for ox in found_ox if ox not in OXIDES] + \
                     (['Total'] if original_total_col else [])
                     
    final_sig_order = []
    for ox in final_ox_order:
        if ox == 'Total': 
            continue
        sig_name = f"{ox}_1sigma"
        if sig_name in final_df.columns:
            final_sig_order.append(sig_name)

    final_ordered_cols = id_cols_to_keep + final_ox_order + final_sig_order
    final_df = final_df[final_ordered_cols]
    
    final_df = final_df.rename(columns={'SAMPLE': 'Sample'})

    return final_df


def extract_aztec(file_path, sheet_name=0):
    """
    Extracts, cleans, and formats oxide concentrations and standard deviations 
    from an AZtec EPMA/EDS data file (.csv or .xlsx). The function parses the 
    multi-block format typical of raw AZtec exports.

    Operations:
      - Identifies data blocks for each sample based on 'Element' and 'Total' rows.
      - Extracts oxide concentrations and their corresponding 'Oxide % Sigma' values.
      - Converts 'FeO' to 'FeOt'.
      - Sorts columns in an order: [Identifiers] -> [Concentrations] -> 
        [Total] -> [Standard Deviations].
    """
    is_excel = file_path.lower().endswith(('.xlsx', '.xls'))
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None) if is_excel else pd.read_csv(file_path, header=None)

    data_blocks_indices = df_raw.index[df_raw[0] == 'Element'].tolist()
    processed_data = []

    for i, start in enumerate(data_blocks_indices):
        label = df_raw.iloc[start - 1, 0]

        total_index_candidates = df_raw.index[(df_raw[0] == 'Total') & (df_raw.index > start)]
        if not total_index_candidates.empty:
            total_index = total_index_candidates.min()
            end = total_index
        else:
            end = data_blocks_indices[i+1] - 1 if i + 1 < len(data_blocks_indices) else len(df_raw)
            total_index = None

        block = df_raw.iloc[start + 1:end].copy()
        
        raw_headers = df_raw.iloc[start, 1:].dropna().values.tolist()
        headers = ['Element'] + [str(h).strip().replace('\n', ' ').replace('\r', '') for h in raw_headers]
        block.columns = headers[:block.shape[1]]

        row_dict = {'SampleID': label}

        # --- Capture Concentrations and Sigmas ---
        for _, row in block.iterrows():
            element = str(row['Element']).strip()
            oxide_formula = str(row.get('Oxide', '')).strip() if 'Oxide' in block.columns and pd.notna(row.get('Oxide')) else None
            
            # Use 'Oxide' formula if available, otherwise fallback to Element
            target_name = oxide_formula if oxide_formula else element
            if target_name == 'FeO':
                target_name = 'FeOt'

            # Extract Oxide Concentration (Prefers Oxide % over Wt%)
            if 'Oxide %' in headers and pd.notna(row.get('Oxide %')):
                row_dict[target_name] = row['Oxide %']
            elif 'Wt%' in headers and pd.notna(row.get('Wt%')):
                row_dict[target_name] = row['Wt%']
                
            # Extract Sigma (AZtec outputs 1-sigma directly)
            if 'Oxide % Sigma' in headers and pd.notna(row.get('Oxide % Sigma')):
                row_dict[f"{target_name}_1sigma"] = row['Oxide % Sigma']
            elif 'Wt% Sigma' in headers and pd.notna(row.get('Wt% Sigma')):
                row_dict[f"{target_name}_1sigma"] = row['Wt% Sigma']

        # --- Capture Total ---
        if total_index is not None:
            total_row = df_raw.iloc[total_index, :].values
            
            if 'Oxide %' in headers:
                col_idx = headers.index('Oxide %')
                if col_idx < len(total_row):
                    row_dict['Total'] = total_row[col_idx]
            elif 'Wt%' in headers:
                col_idx = headers.index('Wt%')
                if col_idx < len(total_row):
                    row_dict['Total'] = total_row[col_idx]

        processed_data.append(row_dict)

    df_out = pd.DataFrame(processed_data)
    if df_out.empty:
        return df_out

    # --- Reorder Columns ---
    id_cols = ['SampleID']
    id_cols_to_keep = [col for col in id_cols if col in df_out.columns]

    found_ox = [col for col in df_out.columns if col in OXIDES or col == 'FeOt']
    extra_ox = [col for col in df_out.columns if col not in id_cols and col != 'Total' and not col.endswith('_1sigma') and col not in found_ox]
    
    final_ox_order = [ox for ox in OXIDES if ox in found_ox] + extra_ox + (['Total'] if 'Total' in df_out.columns else [])
    
    final_sig_order = []
    for ox in final_ox_order:
        if ox == 'Total':
            continue
        sig_name = f"{ox}_1sigma"
        if sig_name in df_out.columns:
            final_sig_order.append(sig_name)

    final_ordered_cols = id_cols_to_keep + final_ox_order + final_sig_order
    df_out = df_out[[c for c in final_ordered_cols if c in df_out.columns]]
    
    df_out = df_out.rename(columns={'SampleID': 'Sample'})

    return df_out


# %%


def format_for_thermobar(df, suffix="_Liq"):
    """
    Converts a mineralML-formatted dataframe into a Thermobar-compatible format.
    Dynamically applies the required phase suffix (e.g., '_Liq', '_Ol', '_Cpx') 
    ONLY to composition/oxide columns.
    
    Parameters:
        df (pd.DataFrame): The output dataframe from mineralML.
        suffix (str): The Thermobar suffix to append (e.g., '_Liq', '_Ol').
        
    Returns:
        pd.DataFrame: A formatted dataframe ready for import into Thermobar.
    """

    df_pt = df.copy()
    rename_dict = {}

    for col in df_pt.columns:
        # Skip identifiers, the Total column, and mineralML metadata
        if col in ['Sample', 'SampleID', 'Total', 'Mineral', 'Submineral'] or 'Predict' in col or 'Score' in col:
            continue
            
        # Skip the 1-sigma standard deviation columns
        if col.endswith('_1sigma'):
            continue
            
        # Everything else (SiO2, FeOt, NiO, Ba, etc.) gets the suffix
        rename_dict[col] = f"{col}{suffix}"

    return df_pt.rename(columns=rename_dict)


# %% 