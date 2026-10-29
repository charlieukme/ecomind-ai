@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if not exist dataset\energy_dataset.csv python scripts\download_dataset.py
streamlit run app.py
