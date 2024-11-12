import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model and scaler
model = pickle.load(open('Obezite.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Obesity Prediction")

# Input fields
gender = st.selectbox("Gender", ("Male", "Female"))
age = st.number_input("Age", min_value=0)
height = st.number_input("Height (m)", min_value=0.0)
weight = st.number_input("Weight (kg)", min_value=0.0)
cholesterol = st.number_input("Cholesterol Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
smoking = st.selectbox("Smoking Status", ("Non-Smoker", "Smoker"))
alcohol_consumption = st.selectbox("Alcohol Consumption", ("No", "Yes"))
physical_activity = st.selectbox("Physical Activity Level", ("Low", "Moderate", "High"))
diet_quality = st.selectbox("Diet Quality", ("Poor", "Average", "Good"))
family_history_with_overweight = st.selectbox("Family History of Obesity", ("No", "Yes"))
FAVC = st.selectbox("Frequency of Eating Fatty Foods", ("No", "Yes"))
FCVC = st.number_input("Frequency of Vegetables Consumption", min_value=0)
NCP = st.number_input("Number of Main Meals", min_value=1)
CAEC = st.selectbox("Consumption of Food Between Meals", ("No", "Yes"))
CH2O = st.number_input("Water Consumption (L)", min_value=0.0)
SCC = st.number_input("Consumption of Sugar-Sweetened Beverages", min_value=0)
FAF = st.number_input("Physical Activity Level (1-5)", min_value=1, max_value=5)
TUE = st.number_input("Time Spent on Physical Activity", min_value=0)
CALC = st.selectbox("Caloric Intake", ("Low", "Moderate", "High"))
MTRANS = st.selectbox("Transportation Type", ("Walking", "Public Transport", "Private Vehicle"))

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'family_history_with_overweight': [1 if family_history_with_overweight == "Yes" else 0],
    'FAVC': [1 if FAVC == "Yes" else 0],
    'FCVC': [FCVC],
    'NCP': [NCP],
    'CAEC': [1 if CAEC == "Yes" else 0],
    'SMOKE': [1 if smoking == "Smoker" else 0],
    'CH2O': [CH2O],
    'SCC': [SCC],
    'FAF': [FAF],
    'TUE': [TUE],
    'CALC': [CALC],
    'MTRANS': [MTRANS],
})

# Convert categorical variables into dummy/indicator variables
input_data = pd.get_dummies(input_data, drop_first=True)

# Ensure the same features are present as in the scaler
input_data = input_data.reindex(columns=scaler.get_feature_names_out(), fill_value=0)

# Prediction button
if st.button('Predict'):
    input_scaled = scaler.transform(input_data)  # Scale the input data
    prediction = model.predict(input_scaled)  # Get the prediction
    predicted_class = np.argmax(prediction, axis=1)  # Get the predicted class
    st.write(f"Predicted class: {predicted_class[0]}")
