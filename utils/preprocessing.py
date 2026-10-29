from __future__ import annotations

from pathlib import Path
import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
TARGET = "Appliances"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run python scripts/download_dataset.py first.")
    # The bundled file is normalized to commas, while many sensor exports use semicolons.
    return pd.read_csv(path, sep=None, engine="python")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates().reset_index(drop=True)
    date_cols = [c for c in out.columns if c.lower() in {"date", "datetime", "timestamp"}]
    for col in date_cols:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    numeric_cols = out.select_dtypes("number").columns
    out[numeric_cols] = out[numeric_cols].apply(lambda s: s.fillna(s.median()))
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_dataframe(df)
    date_cols = [c for c in out.columns if c.lower() in {"date", "datetime", "timestamp"}]
    if date_cols:
        date_col = date_cols[0]
        parsed = pd.to_datetime(out[date_col], errors="coerce")
        out["hour"] = parsed.dt.hour.fillna(0).astype(int)
        out["day_of_week"] = parsed.dt.dayofweek.fillna(0).astype(int)
        out = out.drop(columns=[date_col])
    out = out.select_dtypes("number")
    return out
