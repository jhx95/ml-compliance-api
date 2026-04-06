from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
import logging
import time

from .schemas import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from .model_loader import get_model_loader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск сервера...")
    get_model_loader()
    logger.info("✅ Сервер готов")
    yield
    logger.info("🛑 Остановка сервера")

app = FastAPI(
    title="ML Prediction API",
    description="API для предсказаний ML модели",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "ML Prediction API", "status": "running", "docs": "/docs"}

@app.get("/health")
async def health():
    try:
        model_loader = get_model_loader()
        return {"status": "healthy", "model_loaded": model_loader.model is not None}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    try:
        logger.info(f"Запрос: {request.request_id}")
        model_loader = get_model_loader()
        prediction = model_loader.predict(request.features)
        latency = time.time() - start_time
        logger.info(f"Ответ: {prediction}, время: {latency:.3f}s")
        return PredictionResponse(prediction=prediction, request_id=request.request_id)
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    predictions = []
    for req in request.requests:
        model_loader = get_model_loader()
        pred = model_loader.predict(req.features)
        predictions.append(pred)
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))