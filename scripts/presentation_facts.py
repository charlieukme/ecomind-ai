from pathlib import Path
import sys, json
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.preprocessing import load_dataset
from utils.training import train_models
root=Path(__file__).resolve().parents[1]
df=load_dataset(root/'dataset/energy_dataset.csv')
result=train_models(df, root/'models'/'energy_prediction_model.pkl')
print(json.dumps({
 'rows': len(df), 'columns': len(df.columns), 'target':'Appliances',
 'features':[c for c in df.columns if c!='Appliances'],
 'dtypes':{c:str(df[c].dtype) for c in df.columns},
 'missing':int(df.isna().sum().sum()), 'duplicates':int(df.duplicated().sum()),
 'metrics':result['metrics'], 'importance':result['feature_importance'].to_dict(orient='records'),
 'sample_prediction': float(result['model'].predict(__import__('utils.preprocessing',fromlist=['build_features']).build_features(df).drop(columns=['Appliances']).reindex(columns=result['features'],fill_value=0).iloc[[0]])[0])
}, indent=2))
