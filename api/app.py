# python -m uvicorn api.app:app --reload



from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import json
import joblib
import gdown
from fastapi.middleware.cors import CORSMiddleware

# Add this right after `app = FastAPI()`

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Best model combination
best_bitset = '0010000000'
model_names = ['Linear', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest',
               'GradientBoost', 'AdaBoost', 'SVR', 'KNN', 'MLP']
selected_model_names = [model_names[i] for i, bit in enumerate(best_bitset) if bit == '1']
selected_model_names.append('scaler')  # also include scaler

# Resolve paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.abspath(os.path.join(base_dir, "..", "models"))
json_path = os.path.abspath(os.path.join(base_dir, "..", "model_file_ids.json"))
os.makedirs(models_dir, exist_ok=True)

# Load Google Drive file IDs
with open(json_path, "r") as f:
    file_ids = json.load(f)

# Download missing files
for name in selected_model_names:
    file_id = file_ids.get(name)
    if not file_id:
        raise ValueError(f"No file ID found for {name} in model_file_ids.json")
    
    dest_path = os.path.join(models_dir, f"{name}.pkl")
    if not os.path.isfile(dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"📥 Downloading {name}.pkl from Google Drive...")
        gdown.download(url, dest_path, quiet=False)
    else:
        print(f"✅ {name}.pkl already exists locally.")

# Load scaler
scaler_path = os.path.join(models_dir, "scaler.pkl")
scaler = joblib.load(scaler_path)

# Load selected models
selected_models = [
    joblib.load(os.path.join(models_dir, f"{name}.pkl"))
    for name in selected_model_names if name != 'scaler'
]

print(f"✅ Loaded models: {selected_model_names}")


# Input schema
class SensorData(BaseModel):
    data: list[float]  # Ensure a list of floats

@app.get("/")
def health_check():
    return {"message": "RUL prediction API is up 🚀"}

@app.post("/predict/")
def predict_rul(sensor: SensorData):
    try:
        X = scaler.transform([sensor.data])  # Shape: (1, n_features)
        preds = np.mean([model.predict(X) for model in selected_models], axis=0)
        return {"RUL_prediction": round(float(preds[0]), 2)}
    except Exception as e:
        return {"error": str(e)}
