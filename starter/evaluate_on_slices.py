import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import load_model, inference
from starter.features import cat_features
from typing import Iterable


def evaluation_on_slices(
    df_data: pd.DataFrame, cat_col_name: str, model, encoder, lb
) -> Iterable:
    categories = sorted(list(set(df_data[cat_col_name])))
    for c in categories:
        df = df_data[df_data[cat_col_name] == c]
        X, y, _, _ = process_data(
            df,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )
        data_input = process_data(
            df, categorical_features=cat_features, training=False, encoder=encoder
        )[0]
        slice_result = inference(model=model, X=data_input)
        print(slice_result)


data = pd.read_csv("../data/census.csv")

model_loaded, encoder_loaded = load_model("../model")

# df_input = process_data(
#     data, categorical_features=cat_features, training=False,
#     encoder=encoder_loaded
# )
#
# print()
evaluation_on_slices(data, "education", model_loaded, encoder_loaded)
