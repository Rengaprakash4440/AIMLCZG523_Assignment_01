from types import SimpleNamespace

import pandas as pd

from src import download_data


def test_main_writes_raw_csv_with_target(tmp_path, monkeypatch):
    features = pd.DataFrame({"age": [50, 60], "chol": [200, 250]})
    targets = pd.DataFrame({"num": [0, 2]})
    fake = SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))
    monkeypatch.setattr(download_data, "fetch_ucirepo", lambda id: fake)
    out = tmp_path / "data" / "heart_raw.csv"
    monkeypatch.setattr(download_data, "RAW", out)

    download_data.main()

    saved = pd.read_csv(out)
    assert list(saved.columns) == ["age", "chol", "target"]
    assert saved["target"].tolist() == [0, 2]