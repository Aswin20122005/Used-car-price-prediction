"""Load and clean the used-car dataset."""

import pandas as pd

NUMERIC_FEATURES = ["Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
CATEGORICAL_FEATURES = ["Brand", "Fuel_Type", "Transmission", "Owner_Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Price"  # in lakh rupees


def extract_number(series, pattern=r"(\d+\.\d+|\d+)"):
    """Pull the first number out of strings like '19.67 kmpl' or '1582 CC'."""
    return series.astype("string").str.extract(pattern, expand=False).astype(float)


def brand_from_name(name):
    """'Hyundai Creta 1.6 CRDi SX Option' -> 'Hyundai'. Handles 'Land Rover' and 'Mini Cooper'."""
    words = str(name).split()
    if not words:
        return "Unknown"
    if words[0].lower() == "land" and len(words) > 1:
        return "Land Rover"
    brand = words[0]
    # Normalise spelling variants like 'ISUZU'/'Isuzu' but keep acronyms such as 'BMW'.
    if brand.islower() or (brand.isupper() and len(brand) > 3):
        brand = brand.capitalize()
    return brand


def clean(df):
    """Return a cleaned copy with numeric columns parsed and a Brand column added.

    Missing numbers are left as NaN; the model pipeline imputes them so the same
    medians are used at training and prediction time.
    """
    out = df.copy()
    out["Brand"] = out["Name"].map(brand_from_name)
    out["Mileage"] = extract_number(out["Mileage"])
    out.loc[out["Mileage"] == 0, "Mileage"] = float("nan")  # 0 means unknown in this dataset
    out["Engine"] = extract_number(out["Engine"], r"(\d+)")
    out["Power"] = extract_number(out["Power"])
    out["Seats"] = pd.to_numeric(out["Seats"], errors="coerce")
    out.loc[out["Seats"] == 0, "Seats"] = float("nan")
    return out


def load_dataset(path="car_prices.csv"):
    """Read the CSV, drop rows without a price, and clean it."""
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=[TARGET])
    return clean(df)
