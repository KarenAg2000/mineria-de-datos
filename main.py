
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar el modelo entrenado
model = joblib.load('modelo_breast_cancer.pkl')

app = FastAPI()

class PredictionRequest(BaseModel:
    Age: int
    "Tumor Size": int
    "Survival Months": int

@app.post("/predict/")
async def predict(data: PredictionRequest):
    # Crear un DataFrame con los datos de entrada
    input_df = pd.DataFrame([data.dict()])
    
    # Realizar la predicción
    prediction = model.predict(input_df)
    
    # Devolver el resultado de la predicción
    return {"prediction": prediction[0]}
