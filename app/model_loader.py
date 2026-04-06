import joblib
import pandas as pd
from typing import Dict, Any
import os

class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"✅ Модель загружена из {self.model_path}")
        else:
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
    def predict(self, features: Dict[str, Any]) -> float:
        df = pd.DataFrame([features])
        proba = self.model.predict_proba(df)[0, 1]
        return float(proba)

_model_loader = None

def get_model_loader():
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader("models/compliance_model.joblib")
        _model_loader.load()
    return _model_loader