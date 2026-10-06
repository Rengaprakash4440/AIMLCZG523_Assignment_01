def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert 0 <= body["probability_disease"] <= 1
    assert body["label"] in ("heart disease", "no heart disease")


def test_predict_accepts_missing_optional_fields(client, payload):
    payload.pop("ca")
    payload.pop("thal")
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_out_of_range_value(client, payload):
    payload["age"] = 500
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_missing_required_field(client, payload):
    payload.pop("chol")
    assert client.post("/predict", json=payload).status_code == 422


def test_metrics_endpoint_exposes_counters(client, payload):
    client.post("/predict", json=payload)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "predictions_total" in text
