import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import FEATURES
from src.features import build_preprocessor

VALID_PAYLOAD = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
    "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2,
    "ca": 3, "thal": 3,
}


def _make_df(n=150, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 80, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(90, 200, n),
        "chol": rng.integers(120, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(80, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 5, n).round(2),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })
    df.loc[:4, "ca"] = np.nan
    df.loc[5:7, "thal"] = np.nan
    score = df["oldpeak"] + df["ca"].fillna(0) + rng.normal(0, 0.5, n)
    df["target"] = (score > 3.0).astype(int)
    return df


@pytest.fixture(scope="session")
def make_df():
    return _make_df


@pytest.fixture(scope="session")
def sample_df():
    return _make_df()


@pytest.fixture(scope="session")
def trained_pipeline(sample_df):
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(sample_df[FEATURES], sample_df["target"])
    return pipe


@pytest.fixture(scope="session")
def client(trained_pipeline, tmp_path_factory):
    from fastapi.testclient import TestClient

    model_path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(trained_pipeline, model_path)
    os.environ["MODEL_PATH"] = str(model_path)
    from src.app import app
    return TestClient(app)


@pytest.fixture()
def payload():
    return dict(VALID_PAYLOAD)
