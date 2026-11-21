from datetime import date

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Serve the website from ./static so the page and the API share one origin.
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # Still allow other origins to call the API

MODEL_PATH = "model.joblib"
_bundle = None


def get_bundle():
    """Load the trained model once, on first use."""
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class InvalidInput(ValueError):
    pass


def _number(data, key, low, high, default=None):
    raw = data.get(key)
    if raw in (None, ""):
        if default is None:
            raise InvalidInput(f"'{key}' is required")
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InvalidInput(f"'{key}' must be a number")
    if not low <= value <= high:
        raise InvalidInput(f"'{key}' must be between {low} and {high}")
    return value


def _choice(data, key, allowed):
    value = data.get(key)
    if value not in allowed:
        raise InvalidInput(f"'{key}' must be one of: {', '.join(allowed)}")
    return value


def _brand(make, known):
    """Match the brand case-insensitively ('bmw' -> 'BMW'); unknown brands are still accepted."""
    make = (make or "").strip()
    if not make:
        raise InvalidInput("'make' is required")
    return {b.lower(): b for b in known}.get(make.lower(), make)


def parse_car(data, bundle):
    """Validate the JSON body and turn it into a one-row DataFrame for the pipeline."""
    if not isinstance(data, dict):
        raise InvalidInput("Request body must be a JSON object")
    cats = bundle["categories"]
    brand = _brand(data.get("make"), cats["Brand"])
    defaults = {**bundle["defaults"], **bundle.get("brand_defaults", {}).get(brand, {})}
    row = {
        "Brand": brand,
        "Year": _number(data, "year", 1990, date.today().year),
        "Kilometers_Driven": _number(data, "kilometers_driven", 0, 1_000_000),
        "Fuel_Type": _choice(data, "fuel_type", cats["Fuel_Type"]),
        "Transmission": _choice(data, "transmission", cats["Transmission"]),
        "Owner_Type": _choice(data, "owner_type", cats["Owner_Type"]),
        # Optional technical details: fall back to typical values from the training data.
        "Mileage": _number(data, "mileage_kmpl", 1, 50, defaults["Mileage"]),
        "Engine": _number(data, "engine_cc", 500, 7000, defaults["Engine"]),
        "Power": _number(data, "power_bhp", 20, 700, defaults["Power"]),
        "Seats": _number(data, "seats", 2, 10, defaults["Seats"]),
    }
    return pd.DataFrame([row], columns=bundle["features"])


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/health')
def health():
    try:
        get_bundle()
        return jsonify({'status': 'ok'})
    except FileNotFoundError:
        return jsonify({'status': 'model missing'}), 503


@app.route('/options')
def options():
    """Values the form's dropdowns can offer, taken from the training data."""
    try:
        bundle = get_bundle()
    except FileNotFoundError:
        return jsonify({'error': "Model not found. Run 'python train.py' first."}), 503
    return jsonify({**bundle["categories"], 'metrics': bundle.get("metrics", {})})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        bundle = get_bundle()
    except FileNotFoundError:
        return jsonify({'error': "Model not found. Run 'python train.py' first."}), 503

    try:
        car = parse_car(request.get_json(silent=True), bundle)
    except InvalidInput as e:
        return jsonify({'error': str(e)}), 400

    price = float(bundle["pipeline"].predict(car)[0])
    return jsonify({'predicted_price': round(price, 2), 'unit': 'lakh INR'})


if __name__ == '__main__':
    app.run(debug=True)
