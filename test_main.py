# test_bar.py

import json
from fastapi.testclient import TestClient

from main import app
import requests


client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["greeting"] == "Hello from the census model!"


def test_post():
    record = dict(
        age=39,
        workclass="State-gov",
        fnlgt=77516,
        education="Bachelors",
        education_num=13,
        marital_status="Never-married",
        occupation="Adm-clerical",
        relationship="Not-in-family",
        race="White",
        sex="Male",
        capital_gain=2174,
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States",
    )
    data = json.dumps(record)
    r = client.post("/infer", data=data)
    assert r.json()["pred"] == 0


def test_get_req():
    record = dict(
        age=39,
        workclass="State-gov",
        fnlgt=77516,
        education="Bachelors",
        education_num=13,
        marital_status="Never-married",
        occupation="Adm-clerical",
        relationship="Not-in-family",
        race="White",
        sex="Male",
        capital_gain=2174111,
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States",
    )
    data = json.dumps(record)
    r = requests.post("http://127.0.0.1:8080/infer", data=data)
    assert r.json() == {"pred": 1}


def test_get_req_heroku():
    record = dict(
        age=39,
        workclass="State-gov",
        fnlgt=77516,
        education="Bachelors",
        education_num=13,
        marital_status="Never-married",
        occupation="Adm-clerical",
        relationship="Not-in-family",
        race="White",
        sex="Male",
        capital_gain=2174111,
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States",
    )
    data = json.dumps(record)
    r = requests.post("https://alexdemotest.herokuapp.com/infer", data=data)
    assert r.status_code == 200
    assert r.json() == {"pred": 1}
