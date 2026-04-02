# app/api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ml.inference import segment_image
import app.main as main_app # To access the loaded model

router = APIRouter()

@router.post("/api/v1/segment-holds")
async def segment_holds(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    try:
        image_bytes = await file.read()
        model = main_app.ml_models["hold_segmenter"]
        
        # Run inference
        detections = segment_image(model, image_bytes)
        
        return {
            "filename": file.filename,
            "holds_detected": len(detections),
            "data": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))