import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ModelInfoResponse,
)
from api.dependencies import get_cached_predictor, get_model_metadata
from src.predict import SpamPredictor


ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")


app = FastAPI(
    title="Spam Classifier API",
    description="Email/SMS Spam Classification Service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check(predictor: SpamPredictor = Depends(get_cached_predictor)):
    return HealthResponse(
        status="healthy",
        model_loaded=predictor._pipeline is not None,
        model_version=predictor.model_path.stem,
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check(predictor: SpamPredictor = Depends(get_cached_predictor)):
    if predictor._pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_version=predictor.model_path.stem,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info(metadata: dict = Depends(get_model_metadata)):
    return ModelInfoResponse(**metadata)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    predictor: SpamPredictor = Depends(get_cached_predictor),
):
    try:
        result = predictor.predict(request.text)
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)