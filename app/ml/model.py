# app/ml/model.py
from ultralytics import FastSAM

def load_model():
    print("Loading FastSAM Zero-Shot Segmentation model...")
    # This will automatically download the FastSAM weights
    model = FastSAM('FastSAM-s.pt') 
    return model