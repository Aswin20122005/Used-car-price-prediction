# Used Car Price Prediction

## Overview
ZapValue is a cutting-edge platform developed to predict the prices of used cars using advanced machine learning techniques. The project includes all necessary code, dataset, and documentation to help you estimate resale values for vehicles accurately and transparently. It features a web interface for users to input car details and instantly receive a price prediction.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)

## Features
- Cleans and preprocesses raw car data.
- Trains a robust machine learning model (RandomForestRegressor) to predict used car prices.
- RESTful API built with Flask for serving predictions.
- Interactive web UI for users to predict prices by entering car details.
- Supports categorical and numerical input features such as manufacturer, model, year, mileage, fuel type, and owner type.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/23f2002620/UsedCarPricePrediction.git
    cd UsedCarPricePrediction
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare the dataset:**
    - Place your dataset as `car_prices.csv` in the project root directory.
    - Ensure columns include: Name, Location, Year, Mileage, Fuel_Type, Transmission, Owner_Type, Engine, Power, Seats, New_Price, Price.

## Usage

### Model Training

To train the model and save it as `model.pkl`:
```sh
python model_creation.py
```
This script:
- Loads and cleans the data.
- Encodes categorical features.
- Trains a RandomForestRegressor.
- Outputs Mean Squared Error and R² metrics.
- Saves the trained model as `model.pkl`.

### Running the Web Application

To start the Flask API and web interface:
```sh
python app.py
```
- The API endpoint `/predict` accepts POST requests with car details in JSON format and returns the predicted price.
- Frontend files (`index.html`, `about.html`, `style.css`, `script.js`) provide a user interface for price prediction.

### Predicting Prices

1. Open `index.html` in a browser.
2. Enter details like make, model, year, mileage, fuel type, and owner type.
3. Click "Get Prediction" to receive the estimated price.

## Model Details

- **Algorithm:** Random Forest Regressor (`sklearn`)
- **Features Used:** Year, Mileage, Fuel Type, Owner Type, Engine, Power, Seats, etc.
- **Metrics:** Mean Squared Error (MSE), R-squared (R²)
- **Output:** Model file `model.pkl` for API use

## Web Application

- Frontend: HTML/CSS/JavaScript (see `index.html`, `style.css`, `script.js`)
- Backend: Flask (`app.py`)
- CORS enabled for API requests.
- Team members: Saravanan K, Rithick Praveen G, Thangam Balaji M, Vishnuram V.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

*Our vision: Making buying and selling used cars easier and more transparent for everyone, with instant and accurate price predictions powered by machine learning.*
