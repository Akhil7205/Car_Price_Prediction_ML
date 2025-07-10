import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Extract dropdown values
companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

# Streamlit UI
st.title("🚗 Car Price Predictor")
st.markdown("This app predicts the selling price of your car.")

# Inputs
company = st.selectbox("Select Company", companies)
car_model = st.selectbox("Select Car Model", car_models)
year = st.selectbox("Select Year of Purchase", years)
fuel_type = st.selectbox("Select Fuel Type", fuel_types)
kilo_driven = st.number_input("Enter Kilometres Driven", value=10000, step=500)

# Predict Button
if st.button("Predict Price"):
    input_df = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=[[car_model, company, year, kilo_driven, fuel_type]])

    prediction = model.predict(input_df)
    st.success(f"Estimated Car Price: ₹ {np.round(prediction[0], 2)}")
