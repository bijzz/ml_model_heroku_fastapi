import requests

response = requests.get('https://udacity-project-fastapi.herokuapp.com/')

print(response.status_code)
print(response.json())