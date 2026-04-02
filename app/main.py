# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.ml.model import load_model

# Global dictionary to hold the model
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the Detectron2 model on startup
    print("Loading Detectron2 model...")
    ml_models["hold_segmenter"] = load_model()
    yield
    # Clean up resources on shutdown
    ml_models.clear()

app = FastAPI(title="just climb segmentation API", lifespan=lifespan)

app.include_router(router)