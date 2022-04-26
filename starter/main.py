# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union
from joblib import dump, load
from starter.ml.data import process_data

# Instantiate the app.
app = FastAPI()


@app.get("/")
async def welcome():
    return {"msg": "Welcome!"}

class Entry(BaseModel):
    age: int = Field(example=38)
    workclass: str = Field(example="Private")
    fnlgt: int = Field(example=215646)
    education: str = Field(example="HS-grad")
    education_num: int = Field(alias="education-num", example=9)
    marital_status: str = Field(alias="marital-status", example="Divorced")
    occupation: str = Field(example="Handlers-cleaners")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="Black")
    sex: str = Field(example="Male")
    capital_gain: int = Field(alias="capital-gain", example=0)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(alias="native-country", example="United-States")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@app.post("/inference")
async def inference(input: Entry):
    model = load('/model/model.joblib') 
    encoder = load('/model/encoder.joblib') 
    lb = load('/model/lb.joblib') 
    
    sample = "asd"
    sample_processed, y_test, encoder, lb = process_data(
        sample, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )
    inference = inference(model, sample_processed)

    return {"pediction": inference}
