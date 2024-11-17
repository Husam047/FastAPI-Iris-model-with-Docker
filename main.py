from fastapi import FastAPI
from model import IrisInput, predict_class


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Iris Model API"}

@app.post("predict/")
def predict(iris: IrisInput):
    prediction = predict_class(iris)
    return {"prediction": prediction}