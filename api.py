"""Local FastAPI interface for the shadow release."""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictStr

from .core import load_release


class NewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    complaint_type: StrictStr
    borough: StrictStr
    created_date: StrictStr


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = load_release(os.environ.get("NYC311_RELEASE_DIR", "."))
    yield


app = FastAPI(title="NYC 311 Shadow Prediction Demo", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    release = request.app.state.service.release
    return {
        "status": "ready",
        "model_id": release.get("model_id", release.get("final_evaluation_id")),
        "operational_status": release["operational_status"],
        "mode": "shadow",
    }


@app.post("/predict")
def predict(item: NewRequest, request: Request):
    try:
        return request.app.state.service.predict(item.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
