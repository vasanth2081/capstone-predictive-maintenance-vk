import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="v-vasanth2009/capstone-pred-main-vk-12052026", filename="best_cap_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Predictive Maintenance App")
st.write("""
This application predicts the likelihood of a Customer purchasing Wellness Tourism package based on customer and interaction data.
Please enter the Customer and interaction data below to get a prediction.
""")

# User input
Engine_rpm = st.number_input("Engine rpm", min_value=0, max_value=6000, value=1, step=1)
Lub_oil_pressure = st.number_input("Lub oil pressure", min_value=0.0, format="%.9f")
Fuel_pressure = st.number_input("Fuel pressure",  min_value=0.0, format="%.9f")
Coolant_pressure = st.number_input("Coolant pressure",  min_value=0.0, format="%.9f")
lub_oil_temp = st.number_input("lub oil temp",  min_value=0.0, format="%.9f")
Coolant_temp = st.number_input("Coolant temp",  min_value=0.0, format="%.9f")

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Engine rpm': Engine_rpm,
    'Lub oil pressure': Lub_oil_pressure,
    'Fuel pressure': Fuel_pressure,
    'Coolant pressure': Coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': Coolant_temp
}])


if st.button("Predict Engine Failure"):
    prediction = model.predict(input_data)[0]
    result = "Engine Faulty" if prediction == 1 else "Engine Not Faulty"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")


st.write("Input Data")
st.write(input_df)

st.write("Data Types")
st.write(input_df.dtypes)

st.write("Prediction")
st.write(prediction)
