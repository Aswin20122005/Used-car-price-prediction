# ZapValue: Used Car Price Prediction 🚘

ZapValue estimates the resale price of a used car in India from its brand, age, kilometres driven, fuel type, transmission, ownership history and (optionally) its technical specs. A scikit-learn Random Forest is served through a Flask API with a small web front end.

## Results

Trained on 5,847 listings (`car_prices.csv`), evaluated on a 20% hold-out set:

| Metric | Value |
| --- | --- |
| R² | **0.89** |
| Mean absolute error | **₹1.56 lakh** |
| RMSE | ₹3.55 lakh |

`python train.py` prints these and writes them to `metrics.json`.

## Features

- Cleans the raw data: parses "19.67 kmpl", "1582 CC" and "126.2 bhp", treats zero mileage and seats as missing, and extracts the brand from the car name
- One scikit-learn `Pipeline` (median imputation, one-hot encoding, Random Forest), so the API applies exactly the same preprocessing as training
- When engine, power, mileage or seats are left blank, the API fills in **typical values for that brand**
- Flask API with input validation and clear error messages
- Web form served by the same app, with the list of makes loaded from the model

## Getting started

```bash
git clone https://github.com/Aswin20122005/Used-car-price-prediction.git
cd Used-car-price-prediction
pip install -r requirements.txt

python train.py   # trains the model -> model.joblib (about 20 s)
python app.py     # starts the site and API on http://127.0.0.1:5000
```

Open http://127.0.0.1:5000, fill in the form, and click **Get Prediction**.

### Docker

```bash
docker build -t car-price .       # trains the model during the build
docker run -p 8000:8000 car-price # http://127.0.0.1:8000, served by gunicorn
```

For production without Docker: `gunicorn --bind 0.0.0.0:8000 --workers 2 wsgi:app`.

## API

### `POST /predict`

```json
{
  "make": "Hyundai",
  "year": 2015,
  "kilometers_driven": 41000,
  "fuel_type": "Diesel",
  "transmission": "Manual",
  "owner_type": "First",
  "mileage_kmpl": 19.67,
  "engine_cc": 1582,
  "power_bhp": 126.2,
  "seats": 5
}
```

`mileage_kmpl`, `engine_cc`, `power_bhp` and `seats` are optional. Response:

```json
{ "predicted_price": 11.97, "unit": "lakh INR" }
```

Invalid input returns `400` with a message such as `"'fuel_type' must be one of: Diesel, Electric, Petrol"`. If the model hasn't been trained yet, the response is `503`.

### Other endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | Web interface |
| `GET /options` | Brands, fuel types, transmissions and owner types the model knows, plus its metrics |
| `GET /health` | `{"status": "ok"}` when the model is loaded |

## Project structure

```
├── app.py               # Flask API + serves the website
├── train.py             # trains and evaluates the model, saves model.joblib
├── car_price/
│   ├── data.py          # loading and cleaning the dataset
│   └── model.py         # preprocessing + Random Forest pipeline
├── static/              # index.html, about.html, contact.html, style.css, script.js
├── tests/               # pytest suite
├── car_prices.csv       # dataset
└── CarPrediction.ipynb  # exploratory analysis
```

## Development

```bash
pip install -r requirements-dev.txt
pytest          # data, training and API tests
flake8 .
```

GitHub Actions runs flake8 and the tests on every push, then builds the Docker image and smoke-tests a real prediction against it. To run Flask in debug mode locally, set `FLASK_DEBUG=1`.

## Team

Saravanan K, Rithick Praveen G, Thangam Balaji M, Vishnuram V

## License

MIT, see [LICENSE](LICENSE).
