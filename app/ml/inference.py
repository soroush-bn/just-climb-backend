import numpy as np
import cv2

def segment_image(model, image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # --- CRITICAL MEMORY FIX ---
    # Shrink the image to 640px maximum before feeding it to FastSAM
    orig_h, orig_w = original_img.shape[:2]
    max_dim = 640
    scale = 1.0
    
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        img = cv2.resize(original_img, (int(orig_w * scale), int(orig_h * scale)))
    else:
        img = original_img

    # Run inference on the smaller image (and turn OFF retina_masks)
    results = model(img, conf=0.2, imgsz=640, retina_masks=False)
    
    output = []
    for result in results:
        if result.masks is not None:
            for mask_points in result.masks.xy:
                # Filter out tiny artifacts
                if len(mask_points) > 10:
                    # Scale the coordinates BACK UP to match the original image size
                    # so your React frontend draws the polygons in the right place!
                    original_size_points = mask_points / scale
                    
                    output.append({
                        "polygon": original_size_points.tolist(),
                        "class_id": 0, 
                        "confidence": 1.0 
                    })
                
    return output