from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def performance_on_model_slices(df, y_test, y_pred, columns):
    """ Computes model performance for fixed value of a feature

    Inputs
    ------
    df : input dataframe
    y_test : y test value
    y_pred : y prediction values
    columns : list of columns/features used for slicing

    Returns
    -------
    df : dataframe containing the results
    """
    result_df = pd.DataFrame(columns=['column', 'fixed_value', 'precision',
                                      'recall', 'fbeta'])
    for col in columns:
        col_distinct_value = df[col].unique()
        for fix_value in col_distinct_value:
            y_test_slice = y_test[df[col] == fix_value]
            y_pred_slice = y_pred[df[col] == fix_value]
            precision, recall, fbeta = compute_model_metrics(
                y_test_slice, y_pred_slice)
            result_df = pd.concat([pd.DataFrame([[col,
                                                  fix_value,
                                                  precision,
                                                  recall,
                                                  fbeta]],
                                                columns=result_df.columns),
                                   result_df],
                                  ignore_index=True)
    return result_df
