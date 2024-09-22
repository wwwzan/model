import random
import pandas as pd

# Load the uploaded Excel file
file_path = '/mnt/data/vehicle3_data.xlsx'

# Read the content of the Excel file
df = pd.read_excel(file_path)


# Current number of rows in the dataframe
current_row_count = len(df)

# Target row count
target_row_count = 27320

# Number of rows to drop
rows_to_drop = current_row_count - target_row_count

# Randomly sample row indices to drop
drop_indices = random.sample(range(current_row_count), rows_to_drop)

# Drop the selected rows
df_reduced = df.drop(drop_indices).reset_index(drop=True)

# Verify the new number of rows
new_row_count = len(df_reduced)
print(new_row_count)

output_path = '/mnt/data/vehicle3_data_reduced.xlsx'
# df_reduced.to_excel(output_path, index=False)
