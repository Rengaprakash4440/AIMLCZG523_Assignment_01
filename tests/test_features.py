import numpy as np

from src.data import FEATURES
from src.features import build_preprocessor


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def test_preprocessor_outputs_no_missing_values(sample_df):
    out = _dense(build_preprocessor().fit_transform(sample_df[FEATURES]))
    assert out.shape[0] == len(sample_df)
    assert not np.isnan(out).any()


def test_numeric_columns_are_scaled(sample_df):
    out = _dense(build_preprocessor().fit_transform(sample_df[FEATURES]))
    assert np.allclose(out[:, :5].mean(axis=0), 0, atol=1e-6)


def test_unseen_category_does_not_fail(sample_df):
    prep = build_preprocessor().fit(sample_df[FEATURES])
    row = sample_df[FEATURES].iloc[[0]].copy()
    row["cp"] = 9
    assert _dense(prep.transform(row)).shape[0] == 1
