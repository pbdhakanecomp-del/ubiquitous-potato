from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "housing_data.csv"
MODEL_PATH = BASE_DIR / "house_price_model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    required = ["city", "locality", "area_sqft", "bedrooms",
                "bathrooms", "property_age_years", "price_inr"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=required)

    X = df[["city", "locality", "area_sqft", "bedrooms",
            "bathrooms", "property_age_years"]]
    y = df["price_inr"]

    categorical = ["city", "locality"]
    numerical = ["area_sqft", "bedrooms", "bathrooms", "property_age_years"]
    preprocessor = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numbers", "passthrough", numerical),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=250, random_state=42, min_samples_leaf=2
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)
    print(f"Test MAE:  ₹{mae:,.0f}")
    print(f"Test RMSE: ₹{rmse:,.0f}")
    print(f"Test R²:   {r2:.3f}")
    print()
    print("NOTE: This dataset is synthetic demo data. These scores do not")
    print("measure real-world accuracy. Retrain using verified property data.")

if __name__ == "__main__":
    main()
