from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocessing import load_dataset
from utils.training import train_models
from utils.prediction import predict_row

root = Path(__file__).resolve().parents[1]
df = load_dataset(root / 'dataset' / 'energy_dataset.csv')
result = train_models(df, root / 'models' / 'energy_prediction_model.pkl')
prediction = predict_row(df.iloc[0], result, 0)
print({
    'rows': len(df),
    'target_present': 'Appliances' in df.columns,
    'best_model': result['best_model'],
    'r2': round(result['metrics'][result['best_model']]['r2'], 4),
    'prediction': round(prediction['predicted'], 2),
    'actual': round(prediction['actual'], 2),
    'importance_features': len(result['feature_importance']),
    'model_file': (root / 'models' / 'energy_prediction_model.pkl').exists(),
})
