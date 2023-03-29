#  %% 

import tabula
from tabulate import tabulate
import re
import numpy as np
import pandas as pd
import io, os
os.chdir(os.getcwd())

# %% 

def colTofloat(df_col):
    """
    convert dataframe from string to float and escape TypeError from type like float
    Parameters:
    -------
    df_col: :class: `pandas.Series`: dataframe column data

    Return:
    -------
    col_array: :class: `list`: converted list of data
    std_array: :class: `list`: converted list of stds
    """
    re_before_paran = re.compile("(.*?)\s*\((.*?)\)")  #regex of extracting values before left bracket
    col_array = []  # collect exp data
    std_array = []  # collect exp std
    for idx, ele in enumerate(df_col):
        if "(" in str(ele):  # detect if stds are given in the tabula, if not, assign NaN
            try:
                ele_data = re_before_paran.match(ele).group(1)
                col_array.append(float(ele_data))
                std_str = re.search(r'\((.*?)\)',ele).group(1)  # regex of extracting values between brackets
                std_array.append(float(std_str))
            except:
                std_array.append(np.nan)
                col_array.append(ele)
        else:
            std_array.append(np.nan)
            try: 
                col_array.append(float(ele))
            except:
                col_array.append(ele)
    return col_array, std_array

# %% 

# read your desire page of the table
target_page = 4
pdf_name = 'CarterandDasgupta2015_Supp.pdf'
file = tabula.read_pdf(pdf_name, pages = target_page,
                multiple_tables = True, stream = True)
table = tabulate(file)

# %% 

df_table_data = pd.read_fwf(io.StringIO(table))
df_table_std = df_table_data.copy()  # keep the dataframe for std having same dimension as exp data
# seperate dataframe by multi whitespace, convert string to float for exp data and std
for col in df_table_data.columns:
    split_col = df_table_data[col].str.split(" +", n = 1, expand = True)[1]
    df_table_data[col] = colTofloat(split_col)[0]
    df_table_std[col] = colTofloat(split_col)[1]

# %% 

df_table_data.to_excel("table_data.xlsx")
df_table_std.to_excel("table_std.xlsx")