from datetime import datetime
import pandas as pd
from meteostat import Stations, Hourly

# ------------ Config ------------
country_code = 'CH'  # Switzerland
start = datetime(2023, 1, 1)
end   = datetime(2025, 12, 31)
output_file = 'swiss_weather_2023_2025_15min_duplicated.csv'

# (Optional) restrict stations if you want fewer:
# name_filters = ['Zürich', 'Bern', 'Geneva', 'Basel']  # uncomment to filter by name

# ------------ Fetch station list ------------
stations = Stations().region(country_code).fetch()
# if you want to filter by name, uncomment next 2 lines:
# masks = [stations['name'].str.contains(pat, case=False, na=False) for pat in name_filters]
# stations = stations[pd.concat(masks, axis=1).any(axis=1)]

print(f"Found {len(stations)} Swiss stations")

all_frames = []

# ------------ Loop stations ------------
for station_id, station in stations.iterrows():
    print(f"Fetching {station_id} - {station['name']}")
    try:
        # Fetch HOURLY data in UTC
        data_h = Hourly(station_id, start, end, timezone='UTC').fetch()

        if data_h.empty:
            continue

        # Ensure DateTimeIndex is tz-aware UTC
        if data_h.index.tz is None:
            data_h.index = data_h.index.tz_localize('UTC')

        # ---- Duplicate hourly to 15-min steps ----
        # This creates :00, :15, :30, :45 by forward-filling each hour's value
        data_15 = data_h.resample('15T').ffill()

        # ---- Prepare for saving ----
        data_15 = data_15.reset_index()  # bring 'time' out of index
        # time formatting exactly: YYYY-MM-DD HH:MM:SS UTC
        data_15['time'] = data_15['time'].dt.tz_convert('UTC').dt.strftime('%Y-%m-%d %H:%M:%S') + ' UTC'

        # Add station metadata
        data_15.insert(0, 'station_id', station_id)
        data_15.insert(1, 'station_name', station['name'])
        data_15.insert(2, 'latitude', station['latitude'])
        data_15.insert(3, 'longitude', station['longitude'])
        data_15.insert(4, 'elevation', station['elevation'])

        all_frames.append(data_15)

    except Exception as e:
        print(f"   Skipped {station_id}: {e}")

# ------------ Combine & save ------------
if all_frames:
    final_df = pd.concat(all_frames, ignore_index=True)

    # (Optional) keep only some columns, e.g. time + a few variables:
    # wanted = ['time','station_id','station_name','latitude','longitude','elevation','temp','prcp','wspd']
    # final_df = final_df[[c for c in wanted if c in final_df.columns]]

    final_df.to_csv(output_file, index=False)
    print(f"\nSaved combined CSV: {output_file}")
else:
    print("No data fetched.")
