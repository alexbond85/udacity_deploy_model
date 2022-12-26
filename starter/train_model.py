# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.features import cat_features

# Add the necessary imports for the starter code.
from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

if __name__ == "__main__":
    # Add code to load in the data.
    data = pd.read_csv("../data/census.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    # Write a script that takes in the data, processes it,
    # trains the model, and saves it and the encoder. This script must use
    # the functions you have written.

    # Train and save a model.
    model = train_model(X_train, y_train)

    save_model(model=model, encoder=encoder, path="../model")
    model_loaded, encoder_loaded = load_model("../model")
    y_pred_test = inference(model_loaded, X=X_test)
    test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, y_pred_test)
    print(test_precision, test_recall, test_fbeta)
