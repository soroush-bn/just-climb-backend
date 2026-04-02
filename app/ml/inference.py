# app/ml/inference.py
import numpy as np
import cv2

def segment_image(model, image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run FastSAM inference
    # conf=0.2 lowers the threshold to catch smaller holds
    # retina_masks=True gives higher resolution polygons
    results = model(img, conf=0.2, retina_masks=True)
    
    output = []
    for result in results:
        if result.masks is not None:
            # FastSAM doesn't use standard "classes", so everything is just an "object"
            for mask_points in result.masks.xy:
                # Filter out tiny artifacts (polygons with too few points)
                if len(mask_points) > 10: 
                    output.append({
                        "polygon": mask_points.tolist(),
                        "class_id": 0, # Generic "object"
                        "confidence": 1.0 # SAM doesn't output traditional confidence scores
                    })
                
    return output