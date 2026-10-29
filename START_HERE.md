# EcoMind AI — Start Here

## Fastest start

### Windows
Double-click `run_project.bat`.

### macOS / Linux
Run:

```bash
chmod +x run_project.sh
./run_project.sh
```

The app opens at `http://localhost:8501`.

## Manual start

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python scripts/download_dataset.py
streamlit run app.py
```

## Demonstration order

1. **Overview** — show the lifecycle and live process terminal.
2. **Dataset & Upload** — show `dataset/energy_dataset.csv`, its schema, rows, missing values, duplicates and preview. Upload another CSV here when needed.
3. **EDA Studio** — demonstrate histogram, scatter relationship and correlation heatmap.
4. **Cleaning** — compare source and cleaned data.
5. **Preprocessing** — explain the exact feature preparation steps.
6. **Model Lab** — click **Train / retrain on active dataset**. This trains Linear Regression and Random Forest and saves the selected model under `models/`.
7. **Live Prediction** — select a real dataset row and run a prediction. Explain actual value, predicted value, absolute error, confidence and signed feature contributions.
8. **Feature Importance** — show which features influenced the trained model most.
9. **Report** — download a PDF containing EDA charts, feature importance and the current prediction trace.

## What “connected” means

The sidebar status is intentionally explicit. **Dataset reader: CONNECTED** means the active CSV has been loaded into the Streamlit session. **Model runtime: READY** means a trained in-memory result or Joblib model exists. **Report engine: AVAILABLE** means PDF generation is ready. This is a local live ML session: every upload, retraining action, prediction and report is executed by Python while the app is open.

See `README.md` for the full architecture and viva explanation.
