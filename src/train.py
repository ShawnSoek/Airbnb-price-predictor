"""Train a few models, compare them, keep the best one.

Run from the project root:
    uv run python -m src.train
"""
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from src.features import DB_PATH, build_preprocessor, get_train_test

MODEL_PATH = Path("models/price_model.joblib")

# From simplest to most powerful. The baseline always guesses the typical price
MODELS = {
    "baseline (median)": DummyRegressor(strategy="median"),
    "linear regression": Ridge(alpha=1.0),
    "random forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=42),
    "gradient boosting": HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, random_state=42),
}


def evaluate(pipe: Pipeline, X_test, y_test):
    """Score a trained model on the test data, in euros (not log)."""
    pred_eur = np.expm1(pipe.predict(X_test))
    true_eur = np.expm1(y_test).to_numpy()
    scores = {
        "mae_eur": mean_absolute_error(true_eur, pred_eur),           # average miss in €
        "median_pct_error": float(np.median(np.abs(pred_eur - true_eur) / true_eur) * 100),
        "r2": r2_score(true_eur, pred_eur),                          # 1.0 = perfect, 0 = no better than average
    }
    return scores, pred_eur


def main():
    X_train, X_test, y_train, y_test = get_train_test()
    print(f"Training on {len(X_train)} listings, testing on {len(X_test)}\n")

    results, best = [], None
    for name, model in MODELS.items():
        # translator + model glued together -> one object that goes from raw columns to a price
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)

        scores, pred_eur = evaluate(pipe, X_test, y_test)
        results.append({"model": name, **scores})
        print(f"{name:<20} avg miss {scores['mae_eur']:6.1f} €   "
              f"typical miss {scores['median_pct_error']:5.1f} %   R² {scores['r2']:.2f}")

        if best is None or scores["mae_eur"] < best["scores"]["mae_eur"]:
            best = {"name": name, "scores": scores, "pipe": pipe, "pred_eur": pred_eur}

    print(f"\nBest model: {best['name']}")

    # ---------- Save everything ----------
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best["pipe"], MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    run_at = datetime.now().isoformat(timespec="seconds")
    runs = pd.DataFrame(results).assign(run_at=run_at)

    preds = pd.DataFrame({
        "id": X_test.index,
        "actual_eur": np.expm1(y_test).round(2).to_numpy(),
        "predicted_eur": best["pred_eur"].round(2),
    })
    preds["error_eur"] = (preds["predicted_eur"] - preds["actual_eur"]).round(2)

    conn = sqlite3.connect(DB_PATH)
    try:
        runs.to_sql("model_runs", conn, if_exists="append", index=False)     # history of all runs
        preds.to_sql("predictions", conn, if_exists="replace", index=False)  # latest best model only
    finally:
        conn.close()
    print("Saved scores to table model_runs and test predictions to table predictions")


if __name__ == "__main__":
    main()