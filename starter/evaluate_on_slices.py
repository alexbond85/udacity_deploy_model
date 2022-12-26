import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, load_model, inference
from starter.features import cat_features
from typing import Iterable


def evaluation_on_slices(
        df_data: pd.DataFrame, cat_col_name: str, model_data: dict
) -> Iterable[dict]:
    categories = sorted(list(set(df_data[cat_col_name])))
    for c in categories:
        df = df_data[df_data[cat_col_name] == c]
        X, y, _, lb = process_data(
            df,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=model_data["encoder"],
            lb=model_data["lb"],
        )
        slice_result = inference(model=model_data["model"], X=X)
        precision, recall, fbeta = compute_model_metrics(y, preds=slice_result)
        res = dict(category=c, precision=precision, recall=recall, fbeta=fbeta)
        yield res


if __name__ == '__main__':
    data = pd.read_csv("../data/census.csv")
    model_data_loaded = load_model("../model", suffix="_rf")
    slices_res = list(evaluation_on_slices(data, "education", model_data_loaded))
    slices_table = pd.DataFrame(slices_res)
    slices_table.to_csv("../model/slice_output.txt")
