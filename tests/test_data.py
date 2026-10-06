from pathlib import Path

import pandas as pd
import pytest

from src.data import CATEGORICAL, FEATURES, NUMERIC, load_clean

CLEAN = Path(__file__).resolve().parents[1] / "data" / "heart_clean.csv"


def test_feature_lists_are_consistent():
    assert len(FEATURES) == 13
    assert not set(NUMERIC) & set(CATEGORICAL)


def test_load_clean_binarizes_target(tmp_path, make_df):
    df = make_df(50)
    df["target"] = [0, 1, 2, 3, 4] * 10
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    out = load_clean(path)
    assert out["target"].isin([0, 1]).all()
    assert out["target"].sum() == 40


def test_load_clean_drops_duplicates(tmp_path, make_df):
    df = make_df(30)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    path = tmp_path / "raw.csv"
    dup.to_csv(path, index=False)
    assert len(load_clean(path)) == len(df)


@pytest.mark.skipif(not CLEAN.exists(), reason="cleaned dataset not present")
def test_cleaned_dataset_schema():
    df = pd.read_csv(CLEAN)
    assert set(FEATURES + ["target"]) <= set(df.columns)
    assert df["target"].isin([0, 1]).all()
    assert len(df) > 250
