from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import predict_all

app = FastAPI(title="Solar Sync Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolarWindow(BaseModel):
    soft: List[float] = Field(..., min_length=60, max_length=60)
    hard: List[float] = Field(..., min_length=60, max_length=60)
    mask: List[float] = Field(..., min_length=60, max_length=60)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(payload: SolarWindow):
    return predict_all(payload.soft, payload.hard, payload.mask)