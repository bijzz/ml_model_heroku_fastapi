# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from joblib import dump, load
#from sklearn.preprocessing._encoders import OneHotEncoder
#from sklearn.preprocessing._label import LabelBinarizer

# Add code to load in the data.
data = pd.read_csv('../data/census.processed.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)


# Train and save a model.
model = train_model(X_train, y_train)

# Metrics
y_pred = inference(model,X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print("Metrics precision {}, recall {}, fbeta {}".format(precision, recall, fbeta))

dump(model, '../model/model.joblib') 
dump(encoder, '../model/encoder.joblib') 
dump(lb, '../model/lb.joblib') 