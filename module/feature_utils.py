import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from typing import List, Tuple

def drop_features(df: pd.DataFrame, features_to_drop: List[str]) -> pd.DataFrame:
    """Drop specified features from DataFrame if they exist."""
    return df.drop([f for f in features_to_drop if f in df.columns], axis=1)

def get_random_features(df: pd.DataFrame, n: int, exclude: List[str]=None) -> List[str]:
    """Return a random subset of n features from df, excluding any in exclude."""
    exclude = exclude or []
    candidates = [col for col in df.columns if col not in exclude]
    return random.sample(candidates, min(n, len(candidates)))

def get_all_models():
    """Return a list of regression model instances to try."""
    return [
        RandomForestRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
        SVR(),
        KNeighborsRegressor()
    ]

def run_feature_model_experiments(X: pd.DataFrame, y: pd.Series, n_trials: int = 10, n_features: int = 5, random_state: int = 42) -> List[Tuple[str, List[str], float]]:
    """Run multiple experiments with random features and all models, return list of (model, features, score)."""
    random.seed(random_state)
    results = []
    models = get_all_models()
    for i in range(n_trials):
        features = get_random_features(X, n_features)
        X_sub = X[features]
        for model in models:
            try:
                model.fit(X_sub, y)
                score = model.score(X_sub, y)
                results.append((type(model).__name__, features, score))
            except Exception as e:
                results.append((type(model).__name__, features, float('nan')))
    return results
