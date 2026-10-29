from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .preprocessing import TARGET, build_features


def train_models(df: pd.DataFrame, model_path: Path) -> dict:
    prepared = build_features(df)
    if TARGET not in prepared.columns:
        raise ValueError(f"The active CSV must contain the target column '{TARGET}'.")
    X = prepared.drop(columns=[TARGET])
    y = prepared[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {"Linear Regression": LinearRegression(), "Random Forest Regressor": RandomForestRegressor(n_estimators=180, max_depth=18, min_samples_leaf=2, random_state=42, n_jobs=-1)}
    metrics: dict = {}
    fitted: dict = {}
    for name, model in candidates.items():
        started = time.perf_counter(); model.fit(X_train, y_train); train_time = time.perf_counter() - started
        started = time.perf_counter(); pred = model.predict(X_test); prediction_time = time.perf_counter() - started
        metrics[name] = {"mae": float(mean_absolute_error(y_test, pred)), "rmse": float(mean_squared_error(y_test, pred) ** 0.5), "r2": float(r2_score(y_test, pred)), "training_seconds": train_time, "prediction_seconds": prediction_time}
        fitted[name] = model
    best_name = max(metrics, key=lambda n: metrics[n]["r2"])
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump({"model": fitted[best_name], "features": list(X.columns), "target": TARGET, "model_name": best_name}, model_path)
    best_estimator = fitted[best_name]
    if hasattr(best_estimator, "feature_importances_"):
        importance_values = best_estimator.feature_importances_
    else:
        importance_values = [abs(v) for v in best_estimator.coef_]
    importance = pd.DataFrame({"feature": X.columns, "importance": importance_values})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    return {"best_model": best_name, "metrics": metrics, "model": fitted[best_name], "features": list(X.columns), "feature_importance": importance, "baseline": float(y_train.mean())}
