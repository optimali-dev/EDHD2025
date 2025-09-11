import pandas as pd

# Load your main dataset
df = pd.read_csv('../Data/fulldata_2024_202506.csv', delimiter=';')
df['Datum / Uhrzeit'] = pd.to_datetime(df['Datum / Uhrzeit'], dayfirst=True)
df = df.set_index('Datum / Uhrzeit')

# Load weather data (example format: timestamp, temperature, wind_speed, solar_irradiance, ...)
weather = pd.read_csv('../Data/weather_data.csv')
weather['timestamp'] = pd.to_datetime(weather['timestamp'])
weather = weather.set_index('timestamp')

# Resample or interpolate weather data to 15-min intervals if needed
weather = weather.resample('15T').interpolate()

# Merge weather data into your main DataFrame
df = df.merge(weather, left_index=True, right_index=True, how='left')

# Optional: fill missing weather values (if any)
df = df.fillna(method='ffill')

# Save the merged DataFrame
df.to_csv('../Data/fulldata_with_weather.csv', sep=';')