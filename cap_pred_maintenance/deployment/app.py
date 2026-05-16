import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="v-vasanth2009/capstone-pred-main-vk-12052026", filename="best_cap_prediction_model_v1.joblib")
model = joblib.load(model_path)

# FEATURE ENGINEERING
def feature_engineering(df):

    df = df.copy()

    df['thermal_stress'] = (
        df['Engine rpm'] *
        df['lub oil temp']
    )

    df['engine_stress'] = (
        df['Engine rpm'] *
        df['Coolant temp']
    )

    df['pressure_ratio'] = (
        df['Fuel pressure'] /
        (df['Lub oil pressure'] + 1e-5)
    )

    df['rpm_temp_ratio'] = (
        df['Engine rpm'] /
        (df['Coolant temp'] + 1e-5)
    )

    df['pressure_mean'] = (
        df[
            [
                'Fuel pressure',
                'Lub oil pressure',
                'Coolant pressure'
            ]
        ].mean(axis=1)
    )

    return df

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Predictive Maintenance App")
st.write("""
This application predicts whether an engine is:

- Healthy
OR
- Faulty

based on operational sensor readings.

The model uses an optimized ensemble learning pipeline
with engineered predictive maintenance features.
""")

# User input
st.subheader("Enter Engine Parameters")

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

# FEATURE ENGINEERING
input_data = feature_engineering(input_data)

# REMOVE WEAK FEATURES
drop_features = ['Coolant temp']

input_data.drop(
    columns=drop_features,
    inplace=True
)

if st.button("Predict Engine Condition"):

    # Predict probability
    prob = model.predict_proba(input_data)[0][1]

    # Optimized threshold
    threshold = 0.48

    prediction = (

        1 if prob >= threshold

        else 0
    )

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("⚠️ Engine Faulty")
    else:
        st.success("✅ Engine Healthy")

    # PROBABILITY DISPLAY
    st.subheader("Prediction Confidence")

    st.write(

        f"Fault Probability: "
        f"{prob:.2%}"
    )

    st.progress(float(prob))

    # RISK INTERPRETATION
    if prob >= 0.75:

        st.warning(
            "High failure risk detected."
        )

    elif prob >= 0.48:

        st.info(
            "Moderate failure risk detected."
        )

    else:

        st.success(
            "Engine operating normally."
        )

with st.expander("View Input Data"):
  st.write(input_data)

  st.write("Data Types")
  st.write(input_data.dtypes)

  st.write("Prediction")
  st.write(prediction)
