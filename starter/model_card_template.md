# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model tries to classify whether a person's salary is higher or lower than 50k based on the persons information such as his education, occupation or sex. The data was trained using a logistric regression model.

## Intended Use

The model can be used to estimate whether an individual is a high or low earning individual. This might be for example be interesting for targeting of customers according to their salary or as an classifier to let individuals get an estimate of their own salary allowing a comparision to individuals of similar identity.

## Training Data

The dataset has been split into 80% training data and 20% test data.

## Evaluation Data

Evaluation data corresponds to 20% of the original dataset (see training data).

## Metrics

Precision: 0.71
Recall: 0.26
F-beta 0.38

All metrics listed range from optimal value of 1 to worst value at 0. Details about the metrics can be found at [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).


## Ethical Considerations

Classification of individuals into salary categories can be highly discriminatory if high earning individuals would gain unfair advantages in services provided compared to low earning individuals.

## Caveats and Recommendations

Only a single model has been trained and evaluated. Hence for production use experimentation with different models and hyperparameters is advised to optimize the model performance.
