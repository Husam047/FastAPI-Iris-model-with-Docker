from pydantic import BaseModel
import pickle
import numpy as np


with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

class IrisInput (BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def predict_class(iris_input: IrisInput):
    data = np.array([[iris_input.sepal_length, iris_input.sepal_width, iris_input.petal_length, iris_input.petal_width]])
    prediction = model.predict(data)
    return int(prediction[0])