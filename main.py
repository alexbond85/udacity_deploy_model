# Put the code for your API here.
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.features import cat_features
from starter.ml.data import process_data
from starter.ml.model import inference, load_model


class Item(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(example=13)
    marital_status: str = Field(example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=2174)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(example="United-States")

    class Config:
        arbitrary_types_allowed = True


# Instantiate the app.
app = FastAPI()
model_data_loaded = load_model("./model", suffix="_rf")


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello from the census model!"}


@app.post("/infer")
async def infer(body: Item):
    df = pd.DataFrame()
    for col, val in body.dict().items():
        if "_" in col:
            col = col.replace("_", "-")
        df[col] = [val]
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        # label="salary",
        training=False,
        encoder=model_data_loaded["encoder"],
        lb=model_data_loaded["lb"],
    )
    y_pred = int(inference(model_data_loaded["model"], X=X)[0])
    return {"pred": y_pred}
