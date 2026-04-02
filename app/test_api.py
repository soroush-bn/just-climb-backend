# test_api.py
import requests
import json
import time
import cv2
import numpy as np

url = "http://127.0.0.1:8000/api/v1/segment-holds"
image_path = "sample_wall.jpg" 

print(f"Sending {image_path} to {url}...")
start_time = time.time()

with open(image_path, "rb") as f:
    files = {"file": (image_path, f, "image/jpeg")}
    response = requests.post(url, files=files)

end_time = time.time()
print(f"Request took {end_time - start_time:.2f} seconds\n")

if response.status_code == 200:
    data = response.json()
    print(f"Success! Detected {data['holds_detected']} objects.")
    
    # --- VISUALIZATION LOGIC ---
    
    # 1. Load the original image using OpenCV
    img = cv2.imread(image_path)
    
    # Create a copy for a transparent overlay effect
    overlay = img.copy()
    
    # 2. Loop through every detected hold in the JSON data
    for hold in data['data']:
        # Convert the Python list of coordinates back to a Numpy array for OpenCV
        pts = np.array(hold['polygon'], np.int32)
        
        # Reshape the array to the format OpenCV expects for drawing polygons
        pts = pts.reshape((-1, 1, 2))
        
        # 3. Draw a solid green outline (thickness=2)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 4. Fill the polygon on the overlay to create a translucent tint
        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
    
    # 5. Blend the overlay with the original image (30% opacity for the fill)
    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # 6. Save the final image to your directory
    output_filename = "segmented_wall_output.jpg"
    cv2.imwrite(output_filename, img)
    print(f"Saved visualization to '{output_filename}'")

else:
    print(f"Failed with status code: {response.status_code}")
    print(response.text)