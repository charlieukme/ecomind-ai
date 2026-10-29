from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import TARGET, build_features


def explain_prediction(row: pd.Series, result: dict) -> dict[str, float]:
    prepared = build_features(pd.DataFrame([row]))
    X = prepared.drop(columns=[TARGET], errors="ignore").reindex(columns=result["features"], fill_value=0)
    model = result["model"]
    values = X.iloc[0].astype(float)
    baseline = float(result.get("baseline", np.mean(values)))
    contributions = {}
    importance = result["feature_importance"].set_index("feature")["importance"]
    scale = float(importance.sum()) or 1.0
    for feature in result["features"]:
        contributions[feature] = float((values.get(feature, 0) - baseline) * (importance.get(feature, 0) / scale))
    return dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:10])


def predict_row(row: pd.Series, result: dict, row_index: int) -> dict:
    prepared = build_features(pd.DataFrame([row]))
    X = prepared.drop(columns=[TARGET], errors="ignore").reindex(columns=result["features"], fill_value=0)
    predicted = float(result["model"].predict(X)[0])
    actual = float(row[TARGET])
    error = abs(actual - predicted)
    confidence = max(0.0, min(0.999, 1.0 - error / max(abs(actual), 1.0)))
    return {"row_index": row_index, "actual": actual, "predicted": predicted, "absolute_error": error, "confidence": confidence, "contributions": explain_prediction(row, result)}
