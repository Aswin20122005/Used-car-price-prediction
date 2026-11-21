"""Train the used-car price model and save it to model.joblib.

    python train.py                 # train on car_prices.csv
    python train.py --data other.csv --out model.joblib
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from car_price.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, load_dataset
from car_price.model import build_pipeline


def evaluate(y_true, y_pred):
    return {
        "mae_lakh": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse_lakh": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train(data_path="car_prices.csv", n_estimators=300, test_size=0.2, random_state=42):
    """Train on a split, report test metrics, then refit on all data. Returns (bundle, metrics)."""
    df = load_dataset(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipeline = build_pipeline(n_estimators=n_estimators, random_state=random_state)
    pipeline.fit(X_train, y_train)
    metrics = evaluate(y_test, pipeline.predict(X_test))
    metrics.update(train_rows=len(X_train), test_rows=len(X_test))

    # Refit on everything for the saved model.
    pipeline.fit(X, y)
    bundle = {
        "pipeline": pipeline,
        "features": FEATURES,
        # Typical values, used by the API when an optional field is left blank.
        "defaults": {c: float(X[c].median()) for c in NUMERIC_FEATURES},
        "categories": {c: sorted(X[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES},
        "metrics": metrics,
    }
    return bundle, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="car_prices.csv")
    parser.add_argument("--out", default="model.joblib")
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args(argv)

    bundle, metrics = train(args.data, n_estimators=args.trees)
    joblib.dump(bundle, args.out, compress=3)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"MAE:  {metrics['mae_lakh']} lakh")
    print(f"RMSE: {metrics['rmse_lakh']} lakh")
    print(f"R²:   {metrics['r2']}")
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
