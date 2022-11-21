import os
import pickle
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train) -> RandomForestClassifier:
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
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

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


def inference(model: RandomForestClassifier, X: np.array):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def _model_path(path: str, suffix: str) -> str:
    return os.path.join(path, f"model{suffix}.pkl")


def _encoder_path(path: str, suffix: str) -> str:
    return os.path.join(path, f"encoder{suffix}.pkl")


def save_model(
    model: RandomForestClassifier, encoder, path: str, suffix: Optional[str] = None
) -> None:
    if suffix is None:
        suffix = ""
    model_path = _model_path(path, suffix)
    encoder_path = _encoder_path(path, suffix)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)


def load_model(path: str, suffix: Optional[str] = None) -> Tuple:
    if suffix is None:
        suffix = ""
    model_path = _model_path(path, suffix)
    encoder_path = _encoder_path(path, suffix)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder
