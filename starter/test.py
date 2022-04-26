from fastapi.testclient import TestClient
import json
# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# get
def test_api_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"msg": "Welcome!"}

# post 
def test_api_post_class_low_salary():
    data = {
             "age":38
            ,"workclass":"Private"
            ,"fnlgt":215646
            ,"education":"HS-grad"
            ,"education-num":9
            ,"marital-status":"Divorced"
            ,"occupation":"Handlers-cleaners"
            ,"relationship":"Not-in-family"
            ,"race":"Black"
            ,"sex":"Male"
            ,"capital-gain":0
            ,"capital-loss":0
            ,"hours-per-week":40
            ,"native-country":"United-States"
            }
    r = client.post("/inference", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"salary prediction class": "<=50K"}

# post 
def test_api_post_class_high_salary():
    data= {
            "age":66
            ,"workclass":"Private"
            ,"fnlgt":200818
            ,"education":"Doctorate"
            ,"education-num":16
            ,"marital-status":"Married-civ-spouse"
            ,"occupation":"Prof-specialty"
            ,"relationship":"Husband"
            ,"race":"White"
            ,"sex":"Male"
            ,"capital-gain":0
            ,"capital-loss":0
            ,"hours-per-week":60
            ,"native-country":"United-States"
            }
    r = client.post("/inference", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"salary prediction class": ">50K"}
