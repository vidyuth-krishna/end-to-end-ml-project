import streamlit as st
import requests
import json

# FastAPI endpoint
api_url = "http://3.94.209.105/predict"

# Streamlit app
st.title("Real-Time Classification with Stacking Model")

st.write("Enter feature values to classify:")

# Input features
num_features = st.number_input("Number of features:", min_value=1, max_value=100, step=1)
features = []
for i in range(num_features):
    feature_value = st.number_input(f"Feature {i+1}:", value=0.0)
    features.append(feature_value)

# Prediction button
if st.button("Predict"):
    try:
        # Send request to FastAPI
        payload = {"features": features}
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            prediction = response.json().get("prediction", "No prediction received")
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
