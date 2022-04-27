## links

[Github Repo](https://github.com/bijzz/ml_model_heroku_fastapi)

## git & dvc on github actions

Continous Integration (CI) is realized via Github Actions. Screenshot of a succesfull build including flake8 and pytest execution.

![ci](/screenshots/continuous_integration.png)

DVC is used for tracking files.

![dvc](/screenshots/dvcdag.png)

## model building

When testing model performance we can take a look on metrics when the value of a given feature is held fixed. This enables us to judge model performance for each feature at a fixed value. This is referred to as data slicing.

![slice](/screenshots/slice_output.png)

## api creation with fastapi

FastAPI documentation is automatically generated.

![fastapi](/screenshots/example.png)

## api deployment on heroku

At the time of writing the Github Heroku integration is disabled due to security concerns.Quote [...] we will not be reconnecting to GitHub until we are certain that we can do so safely [...] taken from https://status.heroku.com/incidents/2413. Hence instead of Continous Deployment (CD) we deploy to Heroku via Git - see https://devcenter.heroku.com/articles/git.

![cd](/screenshots/continuous_deloyment.png)

Calling the GET endpoint on deployed Heroku instance:

![get](/screenshots/live_get.png)

Testing the POST endpoint via an script:

![post](/screenshots/live_post.png)