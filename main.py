from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# FastAPI app
app = FastAPI(title="CANCER Prediction API")

# Load trained model
model = joblib.load("svc.pkl")

# Input data structure
class Cancer(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Home API
@app.get("/")
def home():
    return {
        "message": "CANCER Prediction API is running"
    }


# Prediction API
@app.post("/predict")
def predict(data: Cancer):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    return {
        "prediction": str(prediction)
    }