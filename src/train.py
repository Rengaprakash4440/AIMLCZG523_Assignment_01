import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data import FEATURES, load_clean
from src.features import build_preprocessor

SEED = 42
SCORING = ["accuracy", "precision", "recall", "roc_auc"]
FIG = Path("report/figures")
MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 300],
         "clf__max_depth": [None, 5, 10],
         "clf__min_samples_leaf": [1, 3]},
    ),
}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    df = load_clean("data/heart_clean.csv")
    X, y = df[FEATURES], df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease")

    best = {"auc": -1.0}
    for name, (clf, grid) in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        gs = GridSearchCV(pipe, grid, scoring=SCORING, refit="roc_auc", cv=cv, n_jobs=-1)

        with mlflow.start_run(run_name=name):
            gs.fit(X_tr, y_tr)
            i = gs.best_index_
            cv_m = {f"cv_{m}": float(gs.cv_results_[f"mean_test_{m}"][i]) for m in SCORING}

            model = gs.best_estimator_
            pred = model.predict(X_te)
            proba = model.predict_proba(X_te)[:, 1]
            test_m = {
                "test_accuracy": accuracy_score(y_te, pred),
                "test_precision": precision_score(y_te, pred),
                "test_recall": recall_score(y_te, pred),
                "test_f1": f1_score(y_te, pred),
                "test_roc_auc": roc_auc_score(y_te, proba),
            }

            mlflow.log_params({"model": name, "cv_folds": 5, "seed": SEED,
                               "test_size": 0.2, **gs.best_params_})
            mlflow.log_metrics({**cv_m, **test_m})

            # artifacts: plots, tuning results, data, model
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax)
            ax.set_title(f"ROC - {name}")
            roc_p = FIG / f"roc_{name}.png"
            fig.savefig(roc_p, dpi=150, bbox_inches="tight"); plt.close(fig)

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_te, pred, ax=ax)
            ax.set_title(f"Confusion matrix - {name}")
            cm_p = FIG / f"confusion_{name}.png"
            fig.savefig(cm_p, dpi=150, bbox_inches="tight"); plt.close(fig)

            grid_p = Path(f"models/gridsearch_{name}.csv")
            pd.DataFrame(gs.cv_results_).to_csv(grid_p, index=False)

            for p in (roc_p, cm_p):
                mlflow.log_artifact(str(p), "plots")
            mlflow.log_artifact(str(grid_p), "tuning")
            mlflow.log_artifact("data/heart_clean.csv", "data")
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in {**cv_m, **test_m}.items()))
        if cv_m["cv_roc_auc"] > best["auc"]:
            best = {"auc": cv_m["cv_roc_auc"], "name": name, "model": model,
                    "params": gs.best_params_, "test": test_m}

    joblib.dump(best["model"], "models/model.joblib")
    Path("models/metadata.json").write_text(json.dumps({
        "model": best["name"], "features": FEATURES,
        "best_params": best["params"], "test_metrics": best["test"],
    }, indent=2, default=float))
    print(f"\nBest model (by CV ROC-AUC): {best['name']} -> models/model.joblib")


if __name__ == "__main__":
    main()
