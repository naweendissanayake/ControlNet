import cv2
import os
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# Load YOLO model
yolo_model = YOLO("best.pt")

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"  # or "cpu" if you don't have a GPU
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

input_folder = ''
output_folder = ""
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Function to generate random points inside a polygon
def random_points_in_polygon(polygon, num_points):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append((int(pnt.x), int(pnt.y)))
    return points

# Process each image and save predictions
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    
    # Run YOLO inference
    yolo_results = yolo_model(image, imgsz=640, conf=0.6)
    
    # Set image for SAM
    predictor.set_image(image)
    
    for result in yolo_results:
        if result.masks is not None:
            for mask in result.masks.xy:
                polygon_np = np.array(mask)
                polygon = Polygon(polygon_np)
                
                # Generate random points inside the polygon
                input_points = np.array(random_points_in_polygon(polygon, num_points=10))
                
                # Generate SAM mask
                input_labels = np.ones(input_points.shape[0])  # Assign label 1 to all points
                sam_masks, _, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=False,
                )
                
                # Create a mask from the YOLO polygon
                yolo_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(yolo_mask, [polygon_np.astype(np.int32)], 1)
                
                # Apply YOLO mask to SAM mask
                sam_mask = sam_masks[0] & yolo_mask
                
                # Overlay masked SAM result on the image
                color_mask = np.array([0, 255, 0], dtype=np.uint8)  # Green color
                colored_mask = np.expand_dims(sam_mask, axis=2) * color_mask
                image = cv2.addWeighted(image, 1, colored_mask.astype(np.uint8), 0.5, 0)

    # Save the output image with overlays
    output_image_path = os.path.join(output_folder, f"predicted_{image_file}")
    cv2.imwrite(output_image_path, image)

print(f"✅ Processing complete. Results saved in '{output_folder}'")
