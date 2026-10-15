import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import train
from src.data import FEATURES


def test_main_trains_tracks_and_exports_best_model(tmp_path, monkeypatch, make_df):
    (tmp_path / "data").mkdir()
    make_df(n=120).to_csv(tmp_path / "data" / "heart_clean.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "MODELS", {
        "logistic_regression": (LogisticRegression(C=1.0, max_iter=500, random_state=0),
                                {"clf__C": [1]}),
        "random_forest": (RandomForestClassifier(n_estimators=10, min_samples_leaf=1,
                                                 max_features="sqrt", random_state=0),
                          {"clf__n_estimators": [10], "clf__max_features": ["sqrt", "log2"]}),
    })

    train.main()

    model = joblib.load(tmp_path / "models" / "model.joblib")
    meta = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert meta["model"] in ("logistic_regression", "random_forest")
    assert meta["features"] == FEATURES
    assert 0 <= meta["test_metrics"]["test_roc_auc"] <= 1
    assert model.predict(pd.read_csv(tmp_path / "data" / "heart_clean.csv")[FEATURES]).shape
    for plot in ("roc_logistic_regression.png", "confusion_random_forest.png"):
        assert (tmp_path / "report" / "figures" / plot).exists()
    assert (tmp_path / "mlflow.db").exists()