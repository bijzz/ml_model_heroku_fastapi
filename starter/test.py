from fastapi.testclient import TestClient
import json
import pandas as pd
from os.path import exists
from starter.ml.model import train_model, compute_model_metrics
from starter.ml.model import inference
from starter.ml.data import process_data
import sklearn
from joblib import load
from io import StringIO
import numpy as np

# Import our app from main.py.
from main import app
import pytest

# Instantiate the testing client with our app.
client = TestClient(app)

# API Tests
##

# get


def test_api_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"msg": "Welcome!"}

# post


def test_api_post_class_low_salary():
    data = {
        "age": 38, "workclass": "Private", "fnlgt": 215646,
        "education": "HS-grad", "education-num": 9,
        "marital-status": "Divorced", "occupation":
        "Handlers-cleaners", "relationship": "Not-in-family",
        "race": "Black", "sex": "Male", "capital-gain": 0,
        "capital-loss": 0, "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = client.post("/inference", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"salary prediction class": "<=50K"}

# post


def test_api_post_class_high_salary():
    data = {
        "age": 66, "workclass": "Private", "fnlgt": 200818,
        "education": "Doctorate", "education-num": 16,
        "marital-status": "Married-civ-spouse", "occupation":
        "Prof-specialty", "relationship": "Husband",
        "race": "White", "sex": "Male", "capital-gain": 0,
        "capital-loss": 0, "hours-per-week": 60,
        "native-country": "United-States"
    }
    r = client.post("/inference", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"salary prediction class": ">50K"}

# DATA Tests
##


@pytest.fixture(scope='session')
def data(input_data_path="data/census.processed.csv"):
    return pd.read_csv(input_data_path)


def test_check_if_data_files_are_present():
    expected_files = ["data/census.processed.csv"]
    for file in expected_files:
        assert exists(file)


def test_columns_of_input_data(data):
    expected_columns = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary"]
    assert list(data.columns) == expected_columns

# MODEL Tests
##


@pytest.fixture(scope='session')
def simple_df():
    return pd.DataFrame([[1, 1, 1], [1, 0, 0]], columns=["x1", "x2", "y"])


def test_check_if_model_files_are_present():
    expected_files = [
        "model/model.joblib",
        "model/encoder.joblib",
        "model/lb.joblib"]
    for file in expected_files:
        assert exists(file)


def test_train_model_return_type(simple_df):
    X = simple_df[["x1", "x2"]]
    y = simple_df["y"]
    model = train_model(X, y)
    assert isinstance(model, sklearn.tree._classes.DecisionTreeClassifier)


def test_predict_return_type():
    model = load('model/model.joblib')
    encoder = load('model/encoder.joblib')
    lb = load('model/lb.joblib')
    row1_p1 = "age,workclass,fnlgt,education,education-num,marital-status,"
    row1_p2 = "occupation,relationship,race,sex,capital-gain,capital-loss,"
    row1_p3 = "hours-per-week,native-country"
    row2_p1 = "39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,"
    row2_p2 = "Not-in-family,White,Male,2174,0,40,United-States"
    row1 = row1_p1+row1_p2+row1_p3
    row2 = row2_p1+row2_p2
    row = "\n".join([row1, row2])
    X = pd.read_csv(StringIO(row))
    # prepare input vector
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
    sample_processed, _, encoder, lb = process_data(
        X, categorical_features=cat_features, label=None,
        training=False, encoder=encoder, lb=lb
    )
    prediction = inference(model, sample_processed)
    assert len(prediction) == 1
    assert isinstance(prediction, np.ndarray)


def test_model_metrics_return_length(simple_df):
    y = simple_df["x1"]
    ypred = simple_df["x2"]
    result = compute_model_metrics(y, ypred)
    assert len(result) == 3
    assert isinstance(result[0], np.float64)
    assert isinstance(result[1], np.float64)
    assert isinstance(result[2], np.float64)
