#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
if [ ! -f dataset/energy_dataset.csv ]; then
  python3 scripts/download_dataset.py
fi
streamlit run app.py
