EcoMind AI - Windows Quick Start
=================================

1. Extract the ZIP completely.
2. Open the extracted EcoMind-AI folder.
3. Make sure Python 3.12 or newer is installed from https://www.python.org/downloads/
4. Double-click run_project.bat.
5. Wait while dependencies are installed.
6. Your browser should open http://localhost:8501

If Windows SmartScreen appears, choose More info, then Run anyway only if you downloaded this ZIP from your trusted project location.

Manual PowerShell commands:

cd path\to\EcoMind-AI
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\download_dataset.py
streamlit run app.py

If PowerShell blocks activation, you can skip activation and run:

python -m pip install -r requirements.txt
streamlit run app.py

The bundled project already contains the real dataset and trained model, so the dataset download step is only needed if you replace or delete the dataset file.

Do not run app.py by double-clicking it. Use run_project.bat or the streamlit command.
