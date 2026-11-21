import math

import pandas as pd
import pytest

from car_price.data import FEATURES, brand_from_name, clean, extract_number, load_dataset


@pytest.mark.parametrize(
    "name, brand",
    [
        ("Hyundai Creta 1.6 CRDi SX Option", "Hyundai"),
        ("Land Rover Range Rover 2.2L Pure", "Land Rover"),
        ("ISUZU MUX 4WD", "Isuzu"),
        ("BMW 3 Series 320d", "BMW"),
        ("Mercedes-Benz E-Class", "Mercedes-Benz"),
    ],
)
def test_brand_from_name(name, brand):
    assert brand_from_name(name) == brand


def test_extract_number_handles_units_and_missing():
    values = extract_number(pd.Series(["19.67 kmpl", "1582 CC", None, "null bhp"]))
    assert values[0] == 19.67 and values[1] == 1582
    assert math.isnan(values[2]) and math.isnan(values[3])


def test_clean_treats_zero_mileage_and_seats_as_missing():
    df = pd.DataFrame(
        {"Name": ["Maruti Swift"], "Mileage": ["0.0 kmpl"], "Engine": ["1197 CC"], "Power": ["82 bhp"], "Seats": [0]}
    )
    out = clean(df)
    assert math.isnan(out.loc[0, "Mileage"]) and math.isnan(out.loc[0, "Seats"])
    assert out.loc[0, "Engine"] == 1197 and out.loc[0, "Brand"] == "Maruti"


def test_load_dataset_has_all_features_and_no_index_column():
    df = load_dataset()
    assert set(FEATURES) <= set(df.columns)
    assert "Unnamed: 0" not in df.columns
    assert df["Price"].notna().all()
