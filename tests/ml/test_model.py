from starter.ml.model import train_model
import numpy as np


def test_train_model():
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 0])
    model = train_model(x_train, y_train)
    assert model.predict(np.array([[0, 0]])) == np.array([0])
