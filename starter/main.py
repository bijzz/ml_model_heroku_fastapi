# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from joblib import load
from starter.ml.data import process_data
import pandas as pd

# Instantiate the app.
app = FastAPI()

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

if "DYNO" in os.environ:
    folder_path = "starter/model/"
else:
    folder_path = "model/"


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
    native_country: str = Field(
        alias="native-country", example="United-States")


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


def inputToDataFrame(input):
    updated_dict = {key: [value] for key, value in input.dict().items()}
    df = pd.DataFrame.from_dict(updated_dict)
    hyphen_columns = input.schema()["properties"].keys()
    df.columns = hyphen_columns
    return df


@app.post("/inference")
async def inference(input: Entry):
    # load model
    model = load(folder_path+'model.joblib')
    encoder = load(folder_path+'encoder.joblib')
    lb = load(folder_path+'lb.joblib')
    # adjust input for processing
    input_df = inputToDataFrame(input)
    # prepare input vector
    sample_processed, _, encoder, lb = process_data(
        input_df, categorical_features=cat_features, label=None,
        training=False, encoder=encoder, lb=lb
    )
    inference = model.predict(sample_processed)
    print(inference[0])
    classification = "<=50K" if inference[0] == 0 else ">50K"

    return {"salary prediction class": classification}
