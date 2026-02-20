# %% 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% 
file_path = 'MH0811d_wholeTS_denoised.ctf'

# dynamically find the data start and grid dimensions
data_start_line = 0
x_cells = 0
y_cells = 0

with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        # Extract grid dimensions
        if line.startswith('XCells'):
            x_cells = int(line.split('\t')[1])
        if line.startswith('YCells'):
            y_cells = int(line.split('\t')[1])
            
        # Identify the start of the data table
        if line.startswith('Phase\tX\tY'):
            data_start_line = i
            break

print(f"Grid size: {x_cells} x {y_cells}")
print(f"Data starts at line: {data_start_line + 1}")

# load the data using the dynamic skip
# we use data_start_line because pandas 'skiprows' is 0-indexed
df = pd.read_csv(file_path, sep='\t', skiprows=data_start_line)

# reshape and plot
# CTF data is typically stored Row-by-Row (Y then X)
try:
    # Ensure we only take the number of points expected by the grid
    total_points = x_cells * y_cells
    phase_data = df['Phase'].values[:total_points]
    
    phase_map = phase_data.reshape((y_cells, x_cells))

    plt.figure(figsize=(10, 6))
    plt.imshow(phase_map, cmap='viridis', interpolation='none')
    plt.colorbar(label='Phase ID')
    plt.title(f'Phase Map ({x_cells} x {y_cells})')
    plt.axis('off')
    plt.show()

except ValueError as e:
    print(f"Error reshaping data: {e}")
    print(f"Expected {x_cells * y_cells} points, but found {len(df)} rows.")


# %% 