import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import sys
# read parquet file into dataframe
df = pd.read_parquet('../Data/data_files/df_imbalance_volume_ch.parquet')
scaler = StandardScaler()
df["imbalance_volume_scaled"] = scaler.fit_transform(df[["imbalance_volume_ch"]])
# now df holds the data
df.index = pd.to_datetime(df.index)
df.index = df.index.tz_convert(None)  # remove timezone

# get weekday (0=Mon ... 6=Sun)
weekdays = df.index.weekday
# create binary weekday columns
for i, name in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]):
    df[name] = (weekdays == i).astype(int)

df.index = pd.to_datetime(df.index)
month = df.index.month
df["saison1"] = month.isin([11, 12, 1, 2, 3]).astype(int)  # Nov–März
df["saison2"] = month.isin([6, 7, 8]).astype(int)          # Jun–Aug
df["saison3"] = (~month.isin([11, 12, 1, 2, 3, 6, 7, 8])).astype(int)  # Rest


# simplified monthly sunrise/sunset table (local times)
sun_table = {
    1: ("08:00", "17:00"), 2: ("07:30", "17:45"), 3: ("06:45", "18:30"),
    4: ("06:30", "20:00"), 5: ("05:30", "20:45"), 6: ("05:15", "21:15"),
    7: ("05:45", "21:00"), 8: ("06:15", "20:15"), 9: ("06:45", "19:30"),
    10:("07:30", "18:30"), 11:("07:45", "17:00"), 12:("08:15", "16:45"),
}

# extract month
df["month"] = df.index.month

# convert sunrise/sunset strings to timestamps (for each row, use same day)
def to_datetime_on_day(row, time_str):
    date = row.name.date()  # get the date from index
    hour, minute = map(int, time_str.split(":"))
    return pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=hour, minute=minute)

df["sunrise"] = df.apply(lambda row: to_datetime_on_day(row, sun_table[row["month"]][0]), axis=1)
df["sunset"]  = df.apply(lambda row: to_datetime_on_day(row, sun_table[row["month"]][1]), axis=1)

# add binary flags
df["sunrise_flag"] = ((df.index >= df["sunrise"]) & (df.index <= df["sunrise"] + pd.Timedelta(hours=1))).astype(int)
df["sunset_flag"]  = ((df.index >= df["sunset"] - pd.Timedelta(hours=1)) & (df.index <= df["sunset"])).astype(int)

# optional: drop helper columns
df = df.drop(columns=["month", "sunrise", "sunset"])