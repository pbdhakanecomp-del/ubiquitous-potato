# House Price Prediction — India

A beginner-friendly machine-learning project using Python, pandas, scikit-learn, and Streamlit.

## Important data note
`housing_data.csv` contains **synthetic illustrative demo data**, not verified property transactions or current market listings. This project demonstrates the full ML workflow. Its metrics and predictions must not be interpreted as real-world accuracy or valuation. Replace the CSV with appropriately licensed, verified local data before relying on predictions.

## Features
- Input city, locality, area, bedrooms, bathrooms, and property age
- Train a Random Forest regression pipeline
- One-hot encode city/locality
- Evaluate with MAE, RMSE, and R²
- Predict an estimated price in Indian rupees using Streamlit

## 1. Install Python
Use Python 3.10 or newer. Check:
```bash
python --version
```

## 2. Open the project folder
Extract the ZIP, then open a terminal in the `house_price_prediction_india` folder.

## 3. Create and activate a virtual environment

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies
```bash
python -m pip install -r requirements.txt
```

## 5. Train the model
```bash
python train_model.py
```
This creates `house_price_model.joblib` and prints evaluation metrics.

## 6. Run the Streamlit app
```bash
streamlit run app.py
```
Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## Dataset columns
- `city`: city name
- `locality`: locality/neighborhood
- `area_sqft`: property area in square feet
- `bedrooms`: number of bedrooms
- `bathrooms`: number of bathrooms
- `property_age_years`: approximate age
- `price_inr`: target price in Indian rupees

## How the model works
1. Loads the CSV and removes rows with missing required values.
2. Splits the data into training and test subsets.
3. One-hot encodes city and locality.
4. Trains a Random Forest Regressor.
5. Reports MAE, RMSE, and R² on the held-out test split.
6. Saves a pipeline that the Streamlit app uses for predictions.

## Improve it for real use
- Replace the synthetic CSV with verified, licensed data from the target market.
- Keep data from the same time period and define whether prices are asking prices or completed-sale prices.
- Clean outliers, duplicate listings, missing values, and inconsistent units.
- Keep enough examples in every city/locality; do not train on tiny or biased samples.
- Use a time-based test split if predicting future prices.
- Compare against a simple baseline and evaluate MAE/RMSE on genuinely unseen records.
- Add features such as property type, furnishing, floor, building age/condition, parking, amenities, and distance to transit where reliable.
- Do not use the model as a substitute for a professional valuation.

## Common problems
- `ModuleNotFoundError`: activate your virtual environment and run `python -m pip install -r requirements.txt`.
- Model not found: run `python train_model.py` before `streamlit run app.py`.
- `streamlit` command not found: use `python -m streamlit run app.py`.
