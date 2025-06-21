from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/model.pkl")

class InputData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

import logging
logging.basicConfig(level=logging.INFO)

@app.post("/predict")
def predict(data: InputData):
    logging.info(f"Received input: {data}")
    try:
        features = np.array([[v for v in data.dict().values()]])
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return {"error": str(e)}