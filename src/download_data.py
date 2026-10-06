from pathlib import Path
from ucimlrepo import fetch_ucirepo

RAW = Path("data/heart_raw.csv")

def main():
    ds = fetch_ucirepo(id=45)  # Heart Disease (UCI)
    df = ds.data.features.copy()
    df["target"] = ds.data.targets["num"]
    RAW.parent.mkdir(exist_ok=True)
    df.to_csv(RAW, index=False)
    print(f"Saved {RAW} {df.shape}")

if __name__ == "__main__":
    main()
