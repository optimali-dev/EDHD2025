#take fulldata_2024_202506.csv and flatten it so each row also has the previos 48 rows as features labeled as [feature]_lag_[1-48]
import pandas as pd
import os
# Load the dataset
data_path = 'Data/fulldata_2024_202506.csv'
df = pd.read_csv(data_path, delimiter=';')
df['UTC'] = pd.to_datetime(df['UTC'], dayfirst=False)
df = df.sort_values('UTC')
# Fill missing values in the original data (numeric columns only)
numeric_cols = df.select_dtypes(include='number').columns.tolist()
# Columns to exclude from lag feature creation
exclude_cols = [
    'imbalance_volume_ch',
    'Total_System_Imbalance__Positiv_:_long_/_Negativ_:_short_ [MW]',
    'Abgedeckte_Bedarf_der_SA_mFRR- [MW]',
    'Abgedeckte_Bedarf_der_SA_mFRR+ [MW]',
    'AE-Preis long [Euro/MWh]',
    'AE-Preis short [Euro/MWh]',
    'target_15min',
    'longitude',
    'latitude',
    'elevation',
    'imbalance_volume_scaled',
    'AE-Preis Einpreis [Euro/MWh]'
]
numeric_cols_to_lag = [col for col in numeric_cols if col not in exclude_cols]
df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
# Create lag features for the past 48 intervals (15 min each, so 12 hours)
lag_hours = 12
lag_intervals = lag_hours * 4  # 4 intervals per hour
# Build lagged features efficiently
lagged_features = []
for col in numeric_cols_to_lag:
    lagged_cols = {
        f'{col}_lag_{lag}': df[col].shift(lag)
        for lag in range(1, lag_intervals + 1)
    }
    lagged_features.append(pd.DataFrame(lagged_cols))
# Concatenate all lagged features at once
df_lagged = pd.concat(lagged_features, axis=1) if lagged_features else pd.DataFrame(index=df.index)
df_full = pd.concat([df, df_lagged], axis=1)
# Drop rows with NaN values created by lagging
df_full = df_full.dropna().reset_index(drop=True)
# Save the flattened DataFrame
output_path = 'Data/fulldata_2024_202506_flattened.csv'
df_full.to_csv(output_path, sep=';', index=False)
print(f'Saved flattened dataframe to {output_path}')
print(df_full.columns.tolist())
print(f'Number of rows after flattening: {len(df_full)}')
