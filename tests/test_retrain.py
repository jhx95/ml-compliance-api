import pytest
import pandas as pd
import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrain import load_new_data, train_model, evaluate_model, FEATURES, TARGET

def test_load_new_data():
    """Тест загрузки данных"""
    df = load_new_data()
    assert df is not None
    assert len(df) > 0
    assert TARGET in df.columns

def test_train_model():
    """Тест обучения модели"""
    df = load_new_data()
    X = df[FEATURES]
    y = df[TARGET]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, _ = train_model(X_train, y_train)  # train_model возвращает (model, params)
    assert model is not None
    
    # Проверяем, что модель умеет предсказывать
    preds = model.predict_proba(X_test)
    assert preds.shape[0] == len(y_test)

def test_evaluate_model():
    """Тест оценки модели"""
    df = load_new_data()
    X = df[FEATURES]
    y = df[TARGET]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    auc = evaluate_model(model, X_test, y_test)
    assert 0 <= auc <= 1