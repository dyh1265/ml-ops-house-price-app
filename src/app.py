from fastapi import FastAPI
from pydantic import BaseModel
from .predict import predict
import pandas as pd

app = FastAPI()

class HouseData(BaseModel):
    area: float
    bedrooms: float
    bathrooms: float
    stories: float
    mainroad: float
    guestroom: float
    basement: float
    hotwaterheating: float
    airconditioning: float
    parking: float
    prefarea: float
    furnishingstatus: float

@app.post("/predict")
def get_prediction(data: HouseData):
    data_dict = data.dict()  # convert Pydantic model to dict
    print("Raw input data:", data_dict)
    df = pd.DataFrame([data_dict])
    print("Converted DataFrame:\n", df)
    result = predict(df)
    return {"prediction": result}
