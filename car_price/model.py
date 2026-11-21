"""Model pipeline: preprocessing + Random Forest."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from car_price.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_pipeline(n_estimators=300, random_state=42):
    """Preprocessing and model in one object, so prediction applies the same steps as training.

    - numeric columns: median imputation
    - categorical columns: one-hot encoding; unseen categories (e.g. a new brand) are ignored

    The target is modelled as-is. A log1p target was tried, but on the 20% hold-out set it
    scored R² 0.874 / MAE 1.59 lakh against 0.892 / 1.55 lakh for the raw price.
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
    return Pipeline([("preprocess", preprocess), ("model", forest)])
