import json

from src import app as app_module


def test_prediction_updates_ml_metrics(client, payload):
    client.post("/predict", json=payload)
    text = client.get("/metrics").text
    for name in ("ml_prediction_probability_bucket", "ml_prediction_confidence_bucket",
                 "ml_inference_seconds_bucket", "ml_input_age_bucket",
                 "ml_input_chol_bucket"):
        assert name in text


def test_missing_optional_inputs_are_counted(client, payload):
    payload.pop("ca")
    client.post("/predict", json=payload)
    assert 'ml_missing_input_total{feature="ca"}' in client.get("/metrics").text


def test_model_metrics_loaded_from_metadata(tmp_path, client):
    meta = {"model": "demo_model", "test_metrics": {"test_accuracy": 0.9, "test_roc_auc": 0.95}}
    (tmp_path / "model.joblib").write_text("x")
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    app_module.load_model_metrics(tmp_path / "model.joblib")
    text = client.get("/metrics").text
    assert 'ml_model_test_metric{metric="roc_auc",model="demo_model"} 0.95' in text
    assert 'ml_model_info{model="demo_model"} 1.0' in text
