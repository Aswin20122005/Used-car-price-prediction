import pytest

VALID = {
    "make": "Hyundai",
    "year": 2015,
    "kilometers_driven": 41000,
    "fuel_type": "Diesel",
    "transmission": "Manual",
    "owner_type": "First",
}


def test_predict_returns_a_positive_price(client):
    res = client.post("/predict", json=VALID)
    assert res.status_code == 200
    assert res.get_json()["predicted_price"] > 0


def test_brand_is_matched_case_insensitively(client):
    upper = client.post("/predict", json={**VALID, "make": "HYUNDAI"}).get_json()
    lower = client.post("/predict", json={**VALID, "make": "hyundai"}).get_json()
    assert upper == lower


def test_luxury_brand_is_priced_higher(client):
    bmw = client.post("/predict", json={**VALID, "make": "BMW", "transmission": "Automatic"}).get_json()
    maruti = client.post("/predict", json={**VALID, "make": "Maruti"}).get_json()
    assert bmw["predicted_price"] > maruti["predicted_price"]


def test_newer_cars_are_worth_more(client):
    old = client.post("/predict", json={**VALID, "year": 2008}).get_json()
    new = client.post("/predict", json={**VALID, "year": 2018}).get_json()
    assert new["predicted_price"] > old["predicted_price"]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"year": "abc"}, "'year' must be a number"),
        ({"year": 1800}, "'year' must be between"),
        ({"fuel_type": "Water"}, "'fuel_type' must be one of"),
        ({"make": ""}, "'make' is required"),
        ({"kilometers_driven": None}, "'kilometers_driven' is required"),
        ({"engine_cc": 99999}, "'engine_cc' must be between"),
    ],
)
def test_invalid_input_returns_400(client, change, message):
    res = client.post("/predict", json={**VALID, **change})
    assert res.status_code == 400
    assert message in res.get_json()["error"]


def test_non_json_body_returns_400(client):
    assert client.post("/predict", data="not json").status_code == 400


def test_missing_model_returns_503(tmp_path, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setattr(app_module, "_bundle", None)
    res = app_module.app.test_client().post("/predict", json=VALID)
    assert res.status_code == 503


def test_options_lists_known_values(client):
    data = client.get("/options").get_json()
    assert "BMW" in data["Brand"] and "Manual" in data["Transmission"]


def test_homepage_is_served(client):
    res = client.get("/")
    assert res.status_code == 200 and b"priceForm" in res.data
