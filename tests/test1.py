# read data from Data\fulldata_2024_202506.csv


import pandas as pd

df = pd.read_csv('Data/plainData.csv', delimiter=';')  # or delimiter='\t'
print(df.columns)
# print headers count
print(len(df.columns))


# run a basic scikitlearn model on the data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


X = df.select_dtypes(include='number').drop([
    'Total_System_Imbalance__Positiv_:_long_/_Negativ_:_short_ [MW]',
    'Abgedeckte_Bedarf_der_SA_mFRR- [MW]',
    'Abgedeckte_Bedarf_der_SA_mFRR+ [MW]',
    'AE-Preis Einpreis [Euro/MWh]'
], axis=1)


y = df['Total_System_Imbalance__Positiv_:_long_/_Negativ_:_short_ [MW]'] > 0


# Print only the features being trained
print('Features used for training:')
print(list(X.columns))


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# After training, print the most important features
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_importance = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    print('Most important features:')
    for feature, importance in feature_importance:
        print(f'{feature}: {importance:.4f}')
else:
    print('Model does not provide feature importances.')


