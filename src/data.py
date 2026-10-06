import pandas as pd

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL

def load_clean(path="data/heart_raw.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target"] = (df["target"] > 0).astype(int)  # 0 = no disease, 1-4 = disease
    return df.drop_duplicates().reset_index(drop=True)
