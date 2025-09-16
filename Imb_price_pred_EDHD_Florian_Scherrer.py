import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# -------------------------------
# 1. Load & Prepare Data
# -------------------------------
df = pd.read_parquet('../Data/data_files/df_imbalance_penalty_oneprice.parquet')
df_orig = df.copy()

df.index = pd.to_datetime(df.index)
df.index = df.index.tz_convert(None)  # remove timezone


# -------------------------------
# 2. Feature Engineering
# -------------------------------
# Weekday flags
weekdays = df.index.weekday
for i, name in enumerate(["mon", "tue", "wed", "thu", "fri", "sat", "sun"]):
    df[name] = (weekdays == i).astype(int)

# Seasonal flags
month = df.index.month
df["saison1"] = month.isin([11, 12, 1, 2, 3]).astype(int)   # Nov–Mar
df["saison2"] = month.isin([6, 7, 8]).astype(int)           # Jun–Aug
df["saison3"] = (~month.isin([11, 12, 1, 2, 3, 6, 7, 8])).astype(int)  # Rest

# Simplified monthly sunrise/sunset table (local times)
sun_table = {
    1: ("08:00", "17:00"), 2: ("07:30", "17:45"), 3: ("06:45", "18:30"),
    4: ("06:30", "20:00"), 5: ("05:30", "20:45"), 6: ("05:15", "21:15"),
    7: ("05:45", "21:00"), 8: ("06:15", "20:15"), 9: ("06:45", "19:30"),
    10: ("07:30", "18:30"), 11: ("07:45", "17:00"), 12: ("08:15", "16:45"),
}

df["month"] = df.index.month

def to_datetime_on_day(row, time_str):
    date = row.name.date()
    hour, minute = map(int, time_str.split(":"))
    return pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=hour, minute=minute)

df["sunrise"] = df.apply(lambda row: to_datetime_on_day(row, sun_table[row["month"]][0]), axis=1)
df["sunset"]  = df.apply(lambda row: to_datetime_on_day(row, sun_table[row["month"]][1]), axis=1)

# Daytime flags
df["sunrise_flag"] = ((df.index >= df["sunrise"]) & (df.index <= df["sunrise"] + pd.Timedelta(hours=1))).astype(int)
df["sunset_flag"]  = ((df.index >= df["sunset"] - pd.Timedelta(hours=1)) & (df.index <= df["sunset"])).astype(int)
df["night_flag"]   = ((df.index < df["sunrise"]) | (df.index > df["sunset"])).astype(int)

# Drop helper columns
df = df.drop(columns=["month", "sunrise", "sunset"])

# Lag features
for lag in range(1, 25):
    df[f"price_imbalance_penalty_lag{lag}"] = df["price_imbalance_penalty"].shift(lag)


# -------------------------------
# 3. Features & Target
# -------------------------------
target = "price_imbalance_penalty"
df0 = df.iloc[24:, :]   # drop first rows with NaNs from lags

X = df0.drop(columns=[target])
y = df0[target]

y_sign = (y > 0).astype(int)   # sign classification
y_abs = y.abs()


# -------------------------------
# 4. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test, ysign_train, ysign_test, yabs_train, yabs_test = train_test_split(
    X, y, y_sign, y_abs, test_size=0.2, shuffle=False
)


# -------------------------------
# 5. Train Models
# -------------------------------
# Sign classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, ysign_train)
sign_pred = clf.predict(X_test)

# Regressors for abs values
pos_mask = ysign_train == 1
neg_mask = ysign_train == 0

reg_pos = RandomForestRegressor(random_state=0)
reg_pos.fit(X_train[pos_mask], yabs_train[pos_mask])

reg_neg = RandomForestRegressor(random_state=0)
reg_neg.fit(X_train[neg_mask], yabs_train[neg_mask])


# -------------------------------
# 6. Predict & Combine Results
# -------------------------------
yabs_pred = np.zeros_like(y_test, dtype=float)
pos_idx = np.where(sign_pred == 1)[0]
neg_idx = np.where(sign_pred == 0)[0]

yabs_pred[pos_idx] = reg_pos.predict(X_test.iloc[pos_idx])
yabs_pred[neg_idx] = reg_neg.predict(X_test.iloc[neg_idx])

y_pred = np.where(sign_pred == 1, yabs_pred, -yabs_pred)

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred
}, index=y_test.index).sort_index()

print(results.head(10))


# -------------------------------
# 7. Evaluation
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(results["actual"], results["predicted"])

print("Test MSE:", mse)
print("Test RMSE:", mse**0.5)
print("Mean Absolute Error (MAE):", mae)


# -------------------------------
# 8. Plot Results
# -------------------------------
subset = results.iloc[:192, :]

plt.figure(figsize=(10, 5))
plt.plot(subset.index, subset["actual"], label="Actual", color="black", linewidth=1.5)
plt.plot(subset.index, subset["predicted"], label="Predicted", color="red", linestyle="--")
plt.xlabel("Time Index")
plt.ylabel("Penalty")
plt.title("Two-Stage Model: Actual vs Predicted (15–17 Oct 2024)")
plt.legend()
plt.grid(True)
plt.show()
