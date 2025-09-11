import pandas as pd

# Load both datasets
sunrise_path = 'Data/fulldata_with_sunrise.csv'
main_path = 'Data/fulldata_2024_202506.csv'

df_sunrise = pd.read_csv(sunrise_path)
df_main = pd.read_csv(main_path, delimiter=';')

# Parse datetime columns
# For sunrise: 'time', for main: 'UTC' or 'Datum / Uhrzeit'
df_sunrise['time'] = pd.to_datetime(df_sunrise['time'])
if 'UTC' in df_main.columns:
	df_main['UTC'] = pd.to_datetime(df_main['UTC'])
	main_time_col = 'UTC'
elif 'Datum / Uhrzeit' in df_main.columns:
	df_main['Datum / Uhrzeit'] = pd.to_datetime(df_main['Datum / Uhrzeit'], dayfirst=True)
	main_time_col = 'Datum / Uhrzeit'
else:
	raise ValueError('No UTC or Datum / Uhrzeit column found in main data!')

# Merge on nearest time (tolerance 1 minute, can be adjusted)
df_sunrise = df_sunrise.sort_values('time')
df_main = df_main.sort_values(main_time_col)

# Use merge_asof for nearest match
combined = pd.merge_asof(
	df_main, df_sunrise,
	left_on=main_time_col, right_on='time',
	direction='nearest',
	tolerance=pd.Timedelta('1min')
)

# Save result
combined.to_csv('Data/fulldata_combined.csv', index=False, sep=';')
print('Combined file saved to Data/fulldata_combined.csv')
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