import pandas as pd
import os

# Load the main dataset
main_path = 'Data/fulldata_2024_202506.csv'
df = pd.read_csv(main_path, delimiter=';')
df['Datum / Uhrzeit'] = pd.to_datetime(df['Datum / Uhrzeit'], dayfirst=True)
df = df.set_index('Datum / Uhrzeit')

# Example: Add new columns (replace/add your logic here)
df['example_temperature'] = 20.0  # Add a constant column
df['example_wind_speed'] = 5.0    # Add another constant column

# You can add more columns based on your logic, e.g.:
# df['new_col'] = some_function(df)

# Save the new DataFrame with added columns
output_path = 'Data/fulldata_with_weather.csv'
df.to_csv(output_path, sep=';')
print(f'Saved dataframe with new columns to {output_path}')