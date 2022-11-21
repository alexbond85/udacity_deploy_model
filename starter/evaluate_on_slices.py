import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import load_model
from starter.features import cat_features


def evaluate_on_slices(df_data: pd.DataFrame, cat_col_name: str, output_filename: str) -> None:
    pass


data = pd.read_csv("../data/census.csv")

model_loaded, encoder_loaded = load_model("../model")


df_input = process_data(
    data, categorical_features=cat_features, training=False, encoder=encoder_loaded
)

print()
