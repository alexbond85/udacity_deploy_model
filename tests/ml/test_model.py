import math

import numpy as np

from starter.ml.model import compute_model_metrics, inference, train_model


def test_train_model():
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 0])
    model = train_model(x_train, y_train)
    assert model.predict(np.array([[0, 0]])) == np.array([0])


def test_inference():
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 0])
    model = train_model(x_train, y_train)
    preds = inference(model, x_train)
    assert len(preds) == len(x_train)


def test_compute_model_metrics():
    y = [0]
    y_pred = [0]
    fbeta, precision, recall = compute_model_metrics(y, y_pred)
    assert (precision, recall, fbeta) == (1.0, 1.0, 1.0)
    y = [0, 1]
    y_pred = [1, 1]
    precision, recall, fbeta = compute_model_metrics(y, y_pred)
    assert math.isclose(precision, 0.5)
    assert math.isclose(recall, 1.0)
    assert math.isclose(fbeta, 2 / 3)
