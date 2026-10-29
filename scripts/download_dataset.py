from pathlib import Path
import io
import zipfile
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
DATASET_DIR.mkdir(exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
ZIP_PATH = DATASET_DIR / "uci_energy.zip"
print(f"Downloading trusted UCI dataset from {URL}")
urllib.request.urlretrieve(URL, ZIP_PATH)
with zipfile.ZipFile(ZIP_PATH) as archive:
    csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
    raw = archive.read(csv_name)
# Normalize the source archive into the required project path.
# The UCI archive is comma-separated even though some mirrors publish semicolon exports.
df = pd.read_csv(io.BytesIO(raw))
df.to_csv(DATASET_DIR / "energy_dataset.csv", index=False)
ZIP_PATH.unlink(missing_ok=True)
print(f"Saved {len(df):,} rows and {len(df.columns)} columns to {DATASET_DIR / 'energy_dataset.csv'}")
