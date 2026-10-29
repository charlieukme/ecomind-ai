# EcoMind AI
## Smart Building Energy Consumption Prediction Using Machine Learning

EcoMind AI is a Streamlit machine-learning workspace that makes the complete energy prediction lifecycle visible: source data, EDA, cleaning, preprocessing, model training, comparison, live row-level prediction, explainability, and PDF reporting.

## What is real and what is connected?

The Python application reads the active dataset from `dataset/energy_dataset.csv`. The default dataset is the public **Appliances Energy Prediction** dataset from the UCI Machine Learning Repository. It is downloaded by the included script; the project does not fabricate synthetic values.

The ML pipeline is real:

1. Pandas loads the CSV.
2. Duplicate records are removed and numeric missing values are median-imputed.
3. Date/time context is converted into `hour` and `day_of_week` features.
4. The data is split into 80% train and 20% test using `random_state=42`.
5. Exactly two regressors are trained: Linear Regression and Random Forest Regressor.
6. MAE, RMSE, R², training time, and prediction time are calculated.
7. The best model is persisted to `models/energy_prediction_model.pkl` with Joblib.
8. A selected real row is passed through the same preprocessing path and scored live.
9. The app shows the actual value, predicted value, absolute error, confidence estimate, and signed feature contributions.
10. The report engine creates a PDF with EDA charts, feature importance, and prediction trace.

The web dashboard preview is a design shell. The authoritative live ML experience is `app.py`, run through Streamlit, because the supplied brief explicitly requires Python + Streamlit and prohibits React/FastAPI/Flask for the ML application.

## Project structure

```text
EcoMind-AI/
├── app.py
├── dataset/
│   └── energy_dataset.csv              # created by download script
├── models/
│   └── energy_prediction_model.pkl    # created after training
├── reports/
│   └── ecomind_ai_report.pdf           # created from Report page
├── scripts/
│   └── download_dataset.py
├── utils/
│   ├── preprocessing.py
│   ├── training.py
│   └── prediction.py
├── requirements.txt
└── README.md
```

## Run locally

Use Python 3.12+.

```bash
cd EcoMind-AI
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python scripts/download_dataset.py
streamlit run app.py
```

Streamlit opens a local browser URL, normally `http://localhost:8501`.

## How to demonstrate the project

Start on **Overview** and show the live process stream. Then open **Dataset & Upload** to prove the row count, schema, missing values, duplicates, preview, and source URL. Use **EDA Studio** to show the distribution, feature relationship, and heatmap. Use **Cleaning** and **Preprocessing** to explain exactly what changes and why.

Open **Model Lab**, click **Train / retrain on active dataset**, and show the status messages as the two requested regressors are fitted. The selected model is saved to `models/energy_prediction_model.pkl`. Open **Live Prediction**, select a real row, and click **Run live prediction**. Explain that the selected row is cleaned and transformed with the same code used at training time, then passed into the persisted model. The UI shows the actual value, predicted value, absolute error, confidence estimate, and signed feature contributions.

To demonstrate custom data, upload a CSV containing the target column `Appliances` and numeric sensor columns. If it has a date column named `date`, `datetime`, or `timestamp`, EcoMind AI extracts `hour` and `day_of_week`; otherwise it uses numeric columns directly. After uploading, retrain in **Model Lab**. The old session result is cleared so the new dataset cannot silently reuse a stale prediction.

Finally, open **Report**, which generates a PDF containing a cover summary, EDA snapshots, feature importance, and the current prediction trace. Download `ecomind_ai_report.pdf` from the page or find the persisted copy in `reports/`.

## Dataset source

UCI Machine Learning Repository: [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

Download archive used by the script: `https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip`

## Notes on live status

The sidebar connection panel is intentionally explicit:

- **Dataset reader: CONNECTED** means the active CSV has been loaded into memory.
- **Model runtime: READY** means a trained in-memory result or persisted Joblib model is available.
- **Report engine: AVAILABLE** means the PDF generator is installed and can produce a report.

This is a local interactive application, not a hosted telemetry service. “Live” means the app reads the active file and performs the transformation, model call, and report generation during the current Streamlit session. For a production deployment, place the project on a Python-capable host and add authentication, model versioning, and persistent prediction history.

## License

Academic/educational project. Retain the UCI dataset attribution when redistributing.
