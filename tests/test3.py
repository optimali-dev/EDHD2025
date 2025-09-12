import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# Load and preprocess data
df = pd.read_csv('Data/fulldata_2024_202506_flattened.csv', delimiter=';')
print(df.columns.tolist())
df['UTC'] = pd.to_datetime(df['UTC'], dayfirst=False)
# Sort by time
df = df.sort_values('UTC')


# Remove outliers in the target (z-score > 3)
from scipy.stats import zscore
target_col = 'imbalance_volume_ch'
df['target_z'] = zscore(df[target_col].fillna(0))
df = df[df['target_z'].abs() < 3]
df = df.drop(columns=['target_z'])

# Create 15-min ahead target
df['target_15min'] = df[target_col].shift(-1)
df = df.dropna(subset=['target_15min'])

# Separate positive and negative imbalance for additional analysis
df['imbalance_positive'] = df[target_col].apply(lambda x: x if x > 0 else 0)
df['imbalance_negative'] = df[target_col].apply(lambda x: x if x < 0 else 0)

X = df.select_dtypes(include='number').drop([
    'imbalance_volume_ch',
    'Total_System_Imbalance__Positiv_:_long_/_Negativ_:_short_ [MW]',
    'Abgedeckte_Bedarf_der_SA_mFRR- [MW]',
    'Abgedeckte_Bedarf_der_SA_mFRR+ [MW]',
    'AE-Preis long [Euro/MWh]',
    'AE-Preis short [Euro/MWh]',
    'target_15min',
    'longitude',
    'latitude',
    'elevation'
], axis=1)

y = df['target_15min']


# TimeSeriesSplit for backtesting
tscv = TimeSeriesSplit(n_splits=2)
last_y_test = None
last_y_pred = None
last_model = None
last_y_pred_std = None


# plot acf on the time series
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


# plot_acf(y, lags=50)
# plt.title('Autocorrelation of Target (15-min ahead)')
# plt.show()


# Store metrics for each fold
mae_list = []
mse_list = []
r2_list = []
conf_intervals = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = RandomForestRegressor(n_estimators=10, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Confidence: std of predictions from all trees (convert X_test to numpy array to avoid warning)
    X_test_np = X_test.values
    all_tree_preds = [tree.predict(X_test_np) for tree in model.estimators_]
    y_pred_std = pd.DataFrame(all_tree_preds).std(axis=0)

    # Sign prediction (+/-) and confidence
    sign_labels = (y_train > 0).astype(int)  # 1 for positive, 0 for negative
    sign_model = RandomForestRegressor(n_estimators=10, n_jobs=-1)  # Use regressor for probability, or could use classifier
    sign_model.fit(X_train, sign_labels)
    sign_pred_raw = sign_model.predict(X_test)
    sign_pred = (sign_pred_raw > 0.5).astype(int)
    # Confidence: distance from 0.5 (closer to 0 or 1 = more confident)
    sign_conf = abs(sign_pred_raw - 0.5) * 2  # 0 to 1

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)
    conf_intervals.append(y_pred_std)
    print(f'MAE: {mae:.3f}, MSE: {mse:.3f}, R2: {r2:.3f}')
    last_y_test = y_test
    last_y_pred = y_pred
    last_model = model
    last_y_pred_std = y_pred_std
    last_sign_pred = sign_pred
    last_sign_conf = sign_conf


# Print predicted and actual values for the last fold, with confidence and sign prediction
print('\nPredicted vs Actual values (last fold):')
for pred, actual, conf, sign, sign_conf in zip(last_y_pred, last_y_test, last_y_pred_std, last_sign_pred, last_sign_conf):
    sign_str = '+' if sign == 1 else '-'
    print(f'Predicted: {pred:.2f}, Actual: {actual:.2f}, Conf. Std: {conf:.2f}, Sign: {sign_str}, Sign Conf.: {sign_conf:.2f}')


# Print feature importances at the end
if hasattr(last_model, 'feature_importances_'):
    importances = last_model.feature_importances_
    feature_importance = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    print('\nMost important features:')
    for feature, importance in feature_importance:
        print(f'{feature}: {importance:.4f}')
else:
    print('Model does not provide feature importances.')

# Sensitivity analysis: vary each feature +/- 10% and observe mean prediction change
print('\nSensitivity analysis (last fold):')
X_test = X_test.copy()
base_pred = last_model.predict(X_test)
for col in X_test.columns:
    X_mod = X_test.copy()
    X_mod[col] = X_mod[col] * 1.1
    pred_up = last_model.predict(X_mod)
    X_mod[col] = X_mod[col] * 0.9 / 1.1  # back to 0.9x original
    pred_down = last_model.predict(X_mod)
    mean_change_up = (pred_up - base_pred).mean()
    mean_change_down = (pred_down - base_pred).mean()
    print(f'{col}: +10% -> {mean_change_up:.3f}, -10% -> {mean_change_down:.3f}')

# Positive vs negative imbalance predictions (last fold)
print('\nPositive vs Negative Imbalance Predictions (last fold):')
pos_idx = last_y_test > 0
neg_idx = last_y_test < 0
if pos_idx.any():
    mae_pos = mean_absolute_error(last_y_test[pos_idx], last_y_pred[pos_idx])
    print(f'Positive Imbalance MAE: {mae_pos:.3f}')
else:
    print('No positive imbalance samples in last fold.')
if neg_idx.any():
    mae_neg = mean_absolute_error(last_y_test[neg_idx], last_y_pred[neg_idx])
    print(f'Negative Imbalance MAE: {mae_neg:.3f}')
else:
    print('No negative imbalance samples in last fold.')


# Print average metrics across folds
print('\nAverage metrics across folds:')
avg_mae = sum(mae_list)/len(mae_list)
avg_mse = sum(mse_list)/len(mse_list)
avg_r2 = sum(r2_list)/len(r2_list)
print(f'MAE: {avg_mae:.3f}, MSE: {avg_mse:.3f}, R2: {avg_r2:.3f}')
print(f'accuracy for number prediction (last fold, ±30%): {(abs(last_y_pred - last_y_test) <= 0.3 * abs(last_y_test)).mean():.3f}')
# root mean squared error for the number prediction
print(f'RMSE for number prediction (last fold): {mean_squared_error(last_y_test, last_y_pred, squared=False):.3f}')

# Print accuracy for sign prediction (last fold)
sign_true = (last_y_test > 0).astype(int)
sign_acc = (last_sign_pred == sign_true.values).mean()
print(f"Sign prediction accuracy (last fold): {sign_acc:.3f}")
# root mean squared error for the sign prediction (treating sign as 0/1 regression)
print(f"Sign prediction RMSE (last fold): {mean_squared_error(sign_true, last_sign_pred):.3f}")


