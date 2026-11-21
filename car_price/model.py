"""Model pipeline: preprocessing + Random Forest, trained on log(price)."""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from car_price.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline(n_estimators=300, random_state=42):
    """Preprocessing and model in one object, so prediction applies the same steps as training.

    - numeric columns: median imputation
    - categorical columns: one-hot encoding; unseen categories (e.g. a new brand) are ignored
    - target: log1p-transformed, because prices are heavily right-skewed (2 to 160 lakh)
    """
    preprocess = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL_FEATURES),
        ]
    )
    forest = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1, random_state=random_state
    )
    regressor = TransformedTargetRegressor(regressor=forest, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("preprocess", preprocess), ("model", regressor)])
