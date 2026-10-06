import joblib
import numpy as np

from src.data import FEATURES


def test_predict_proba_is_valid(trained_pipeline, sample_df):
    proba = trained_pipeline.predict_proba(sample_df[FEATURES])
    assert proba.shape == (len(sample_df), 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_handles_missing_values_at_inference(trained_pipeline, sample_df):
    row = sample_df[FEATURES].iloc[[10]].copy()
    row["ca"] = np.nan
    row["thal"] = np.nan
    assert trained_pipeline.predict(row)[0] in (0, 1)


def test_model_roundtrip_gives_same_predictions(trained_pipeline, sample_df, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(trained_pipeline, path)
    loaded = joblib.load(path)
    X = sample_df[FEATURES]
    assert np.allclose(trained_pipeline.predict_proba(X), loaded.predict_proba(X))


def test_model_beats_chance_on_training_data(trained_pipeline, sample_df):
    acc = (trained_pipeline.predict(sample_df[FEATURES]) == sample_df["target"]).mean()
    assert acc > 0.7
