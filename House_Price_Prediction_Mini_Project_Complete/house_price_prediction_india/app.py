from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.joblib"
DATA_PATH = BASE_DIR / "housing_data.csv"

st.set_page_config(page_title="House Price Prediction India", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

st.title("🏠 House Price Prediction")
st.caption("Estimate a property's price using a machine-learning model.")

st.warning(
    "Demo project: the included housing dataset is synthetic and illustrative. "
    "Predictions are not market valuations. Use verified local transaction/listing "
    "data before making financial decisions."
)

model = load_model()
if model is None:
    st.error("The trained model was not found. Follow the README instructions to train it first.")
    st.stop()

data = load_data()
cities = sorted(data["city"].dropna().unique().tolist())
city = st.selectbox("City", cities)

localities = sorted(data.loc[data["city"] == city, "locality"].dropna().unique().tolist())
locality = st.selectbox("Locality", localities)

with st.form("prediction_form"):
    st.subheader("Property details")
    area = st.number_input("Area (square feet)", min_value=200, max_value=10000, value=1000, step=50)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=1)
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
    age = st.number_input("Property age (years)", min_value=0, max_value=100, value=5, step=1)
    submitted = st.form_submit_button("Predict house price", type="primary")

if submitted:
    row = pd.DataFrame([{
        "city": city,
        "locality": locality,
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_age_years": age,
    }])
    prediction = max(0, float(model.predict(row)[0]))
    st.subheader("Estimated price")
    st.metric("Predicted house price", f"₹{prediction:,.0f}")
    st.caption(f"Approx. ₹{prediction / 100000:.2f} lakh")
    st.info("This is a model estimate, not a guaranteed sale price. Actual prices depend on building condition, floor, amenities, legal status, and current market conditions.")

with st.expander("About this model"):
    st.write("Algorithm: Random Forest Regression")
    st.write("Categorical features: city and locality (one-hot encoded)")
    st.write("Numerical features: area, bedrooms, bathrooms, property age")
    st.write("To improve this project, replace housing_data.csv with verified and representative Indian housing data, then retrain the model.")
