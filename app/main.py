from fastapi import FastAPI
from app.models import IrisSpecies, IrisModel
import requests

model = IrisModel()  # Load the model at startup

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Iris Classifier API!"}

@app.post("/predict")
def predict_species(iris: IrisSpecies):
    prediction = model.predict(iris)
    return {"species": prediction}

@app.post("/predict_external")  # New route for external requests
def predict_species_external(iris: IrisSpecies):
    url = "http://127.0.0.1:8000/predict"
    data = iris.dict() 
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        prediction = response.json()["species"]
        return {"species": prediction}
    else:
        return {"error": f"Error: {response.status_code}, {response.text}"}
