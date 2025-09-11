# combine full C:\Code\Data_cleanUp_step1\Data\fulldata_2024_202506.csv and C:\Code\Data_cleanUp_step1\Data\fulldata_with_sunrise.csv
import pandas as pd
import os
# Load the main dataset
main_path = 'Data/fulldata_2024_202506.csv'
df_main = pd.read_csv(main_path, delimiter=';')
df_main['Datum / Uhrzeit'] = pd.to_datetime(df_main['Datum / Uhrzeit'], dayfirst=True)
df_main = df_main.set_index('Datum / Uhrzeit')
# Load the sunrise/sunset dataset

sunrise_path = 'Data/fulldata_with_sunrise.csv'
df_sunrise = pd.read_csv(sunrise_path, delimiter=',')
df_sunrise['time'] = pd.to_datetime(df_sunrise['time'])
df_sunrise = df_sunrise.set_index('time')
# Merge datasets on the datetime index
df_combined = df_main.join(df_sunrise, how='inner')
# Save the combined DataFrame
output_path = 'Data/fulldata_combined.csv'
df_combined.to_csv(output_path, sep=';')
print(f'Saved combined dataframe to {output_path}')
print(df_combined.columns.tolist())