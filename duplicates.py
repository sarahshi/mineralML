# %% 


# Check the uploaded Excel for duplicate rows or duplicate citation fields
import pandas as pd

# Load file
path = "georoc_citations.xlsx"
df = pd.read_excel(path)

# Basic info
rows, cols = df.shape

# Exact duplicate rows
exact_dupes = df[df.duplicated(keep=False)]

# Column-wise duplicate counts (useful if citations are in one column)
dup_summary = {}
for c in df.columns:
    dup_summary[c] = int(df[c].duplicated().sum())

rows, cols, len(exact_dupes), dup_summary


# %% 

df_unique = df.drop_duplicates()
df_unique.to_csv('georoc_undup.csv')

# %%
