import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# input in feet
rocket_length = 8.5

# path to CSV tables
TABLE_DIR = os.path.join(os.path.dirname(__file__), "Interpolator Data", "Tuning - Percent Stability", "Tables")

# rocket lengths corresponding to each table (ft)
LENGTHS = list(range(8, 21))  # 8, 9, ..., 20

# load every table into a 3d array [row, column, rocket_length]
tables = [pd.read_csv(os.path.join(TABLE_DIR, f"Spaceshot_{L}ft.CSV")) for L in LENGTHS]
col_names = tables[0].columns.tolist()
S = np.stack([df.to_numpy() for df in tables], axis=2)  # shape: (rows, cols, n_lengths)

# reshape to 2d so each row represents a full flattened table (usable with interp1d)
rows, cols, pages = S.shape
S_flattened = S.transpose(2, 0, 1).reshape(pages, -1)  # shape: (n_lengths, rows*cols)

# interpolate along the rocket_length axis
interpolator = interp1d(LENGTHS, S_flattened, axis=0, kind="linear")
result_flattened = interpolator(rocket_length)

# convert output back to a labelled dataframe
result_array = result_flattened.reshape(rows, cols)
result_table = pd.DataFrame(result_array, columns=col_names)

print(result_table)
