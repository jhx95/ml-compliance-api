import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from datetime import datetime

# ========== НАСТРОЙКИ ==========
MODEL_PATH = "models/compliance_model.joblib"
NEW_DATA_PATH = "data/new_data.csv"
FEATURES = ['complaint_status', 'num_reassignments', 'has_photo_evidence',
            'is_monsoon_season', 'resolution_days', 'has_gps_location',
            'repeat_complainant', 'severity', 'ward_code', 'complaint_channel']
TARGET = 'citizen_satisfied'

# MLflow настройки (локально, можно заменить на http://localhost:5000 если сервер запущен)
MLFLOW_TRACKING_URI = None  # None = локальная файловая база

def load_new_data():
    """Загружает новые данные"""
    if os.path.exists(NEW_DATA_PATH):
        df = pd.read_csv(NEW_DATA_PATH)
        print(f"✅ Загружено {len(df)} новых записей")
        return df
    else:
        print(f"❌ Файл {NEW_DATA_PATH} не найден")
        return None

def train_model(X_train, y_train, params=None):
    """Обучает новую модель"""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.05,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model, params

def evaluate_model(model, X_test, y_test):
    """Оценивает модель"""
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)

def get_current_model_auc():
    """Загружает текущую модель и считает её AUC"""
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        df = load_new_data()
        if df is not None:
            X = df[FEATURES]
            y = df[TARGET]
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return evaluate_model(model, X_test, y_test)
    return 0

def save_to_mlflow(model, params, auc, X_train, y_train, X_test, y_test):
    """Сохраняет модель в MLflow"""
    # Устанавливаем tracking URI
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Создаём эксперимент
    mlflow.set_experiment("compliance_model_retraining")
    
    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_params(params)
        
        # Логируем метрики
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        
        # Логируем модель
        mlflow.xgboost.log_model(model, "model")
        
        # Логируем артефакты (признаки)
        mlflow.log_text(str(FEATURES), "features.txt")
        
        # Регистрируем модель
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        try:
            # Пытаемся зарегистрировать
            mlflow.register_model(model_uri, "compliance_model")
            print(f"📦 Модель зарегистрирована в MLflow Registry")
        except:
            print(f"📦 Модель сохранена в run {run_id}")
        
        print(f"🔗 Смотреть эксперимент: http://localhost:5000 (если MLflow server запущен)")

def main():
    print("🚀 Запуск автоматического переобучения с MLflow...")
    print(f"⏰ Время запуска: {datetime.now()}")
    
    # 1. Загружаем новые данные
    df = load_new_data()
    if df is None:
        return
    
    X = df[FEATURES]
    y = df[TARGET]
    
    # 2. Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 3. Обучаем новую модель
    print("📚 Обучение новой модели...")
    new_model, params = train_model(X_train, y_train)
    new_auc = evaluate_model(new_model, X_test, y_test)
    print(f"📊 Новая модель AUC: {new_auc:.4f}")
    
    # 4. Получаем AUC текущей модели
    current_auc = get_current_model_auc()
    print(f"📊 Текущая модель AUC: {current_auc:.4f}")
    
    # 5. Сравниваем
    if new_auc > current_auc:
        print(f"✅ Новая модель лучше (+{new_auc - current_auc:.4f})")
        
        # 6. Сохраняем локально
        joblib.dump(new_model, MODEL_PATH)
        print(f"💾 Модель сохранена в {MODEL_PATH}")
        
        # 7. Сохраняем в MLflow
        save_to_mlflow(new_model, params, new_auc, X_train, y_train, X_test, y_test)
        
        print("🎉 Переобучение завершено! Модель обновлена.")
    else:
        print(f"⏸️ Новая модель не лучше (хуже на {current_auc - new_auc:.4f}). Обновление отменено.")
        print("📝 Логируем эксперимент в MLflow (без обновления)...")
        
        # Всё равно логируем эксперимент для истории
        mlflow.set_experiment("compliance_model_retraining")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("roc_auc", new_auc)
            mlflow.log_metric("better_than_current", 0)
            mlflow.log_text("Model was not better than current", "result.txt")
        print("📊 Эксперимент залогирован")

if __name__ == "__main__":
    main()