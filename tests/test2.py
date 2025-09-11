import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Load and preprocess data
df = pd.read_csv('Data/fulldata_2024_202506.csv', delimiter=';')
df['Datum / Uhrzeit'] = pd.to_datetime(df['Datum / Uhrzeit'], dayfirst=True)

# Sort by time
df = df.sort_values('Datum / Uhrzeit')

# Create 15-min ahead target
target_col = 'Total_System_Imbalance__Positiv_:_long_/_Negativ_:_short_ [MW]'
df['target_15min'] = df[target_col].shift(-1)
df = df.dropna(subset=['target_15min'])

X = df.select_dtypes(include='number').drop([
    'Total_System_Imbalance__Positiv_:_long_/_Negativ_:_short_ [MW]',
    'Abgedeckte_Bedarf_der_SA_mFRR- [MW]',
    'Abgedeckte_Bedarf_der_SA_mFRR+ [MW]',
    'target_15min'
], axis=1)


y = df['target_15min']

# TimeSeriesSplit for backtesting
tscv = TimeSeriesSplit(n_splits=5)
last_y_test = None
last_y_pred = None
last_model = None

# plot acf on the time series

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(y, lags=50)
plt.show()

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, y_pred))
    last_y_test = y_test
    last_y_pred = y_pred
    last_model = model

# Print predicted and actual values for the last fold
print('\nPredicted vs Actual values (last fold):')
for pred, actual in zip(last_y_pred, last_y_test):
    print(f'Predicted: {pred:.2f}, Actual: {actual:.2f}')

# Print feature importances at the end
if hasattr(last_model, 'feature_importances_'):
    importances = last_model.feature_importances_
    feature_importance = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    print('\nMost important features:')
    for feature, importance in feature_importance:
        print(f'{feature}: {importance:.4f}')
else:
    print('Model does not provide feature importances.')




