# combine fulldata_combined and swiss_weather_2023_2025_15min_duplicated  and merge on the UTC time column
import pandas as pd
import os
# Load the main dataset
main_path = 'Data/fulldata_combined.csv'
df_main = pd.read_csv(main_path, delimiter=';')
df_main['UTC'] = pd.to_datetime(df_main['UTC'], dayfirst=True)
df_main = df_main.set_index('UTC')
# Ensure index is timezone-naive
if df_main.index.tz is not None:
	df_main.index = df_main.index.tz_convert(None)
# Load the weather dataset
weather_path = 'Data/swiss_weather_2023_2025_15min_duplicated.csv'
df_weather = pd.read_csv(weather_path, delimiter=',')
df_weather['time'] = pd.to_datetime(df_weather['time'])
df_weather = df_weather.set_index('time')
# Ensure index is timezone-naive
if df_weather.index.tz is not None:
	df_weather.index = df_weather.index.tz_convert(None)
# Drop 'station_name' if present
if 'station_name' in df_weather.columns:
    df_weather = df_weather.drop(columns=['station_name'])
# Keep only numeric columns for averaging
df_weather_numeric = df_weather.select_dtypes(include='number')
# Group by time and take mean for each timestamp
df_weather = df_weather_numeric.groupby(df_weather_numeric.index).mean()
# Merge datasets on the datetime index
df_combined = df_main.join(df_weather, how='inner')
# Save the combined DataFrame
output_path = 'Data/fulldata_2024_202506.csv'
df_combined.to_csv(output_path, sep=';')
print(f'Saved combined dataframe to {output_path}')
print(df_combined.columns.tolist())