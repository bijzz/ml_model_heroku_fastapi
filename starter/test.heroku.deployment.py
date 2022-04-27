import requests
import json

data = {
    "age": 38, "workclass": "Private", "fnlgt": 215646,
    "education": "HS-grad", "education-num": 9,
    "marital-status": "Divorced", "occupation":
    "Handlers-cleaners", "relationship": "Not-in-family",
    "race": "Black", "sex": "Male", "capital-gain": 0,
    "capital-loss": 0, "hours-per-week": 40,
    "native-country": "United-States"
}
response = requests.post('https://udacity-project-fastapi.herokuapp.com/inference', data=json.dumps(data))

print(response.status_code)
print(response.json())