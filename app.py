from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os 



app = FastAPI()

# Load the model
model_path = '/app/stacking_model.pkl'

if not os.path.exists(model_path):
    raise RuntimeError(f"Model file '{model_path}' not found.")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model file '{model_path}' not found.")

# Define request body for prediction
class PredictionRequest(BaseModel):
    features: list[float]  # List of features for a single prediction

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stacking Model API. Use /predict to make predictions."}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Convert input features to numpy array
        input_features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(input_features)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))