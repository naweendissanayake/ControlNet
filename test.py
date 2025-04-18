import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

torch.cuda.empty_cache()


def load_models(yolo_path: str, sam_checkpoint: str) -> tuple[YOLO, SamPredictor]:
    model = YOLO(yolo_path)
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, SamPredictor(sam)

def process_image(image_path: str, model: YOLO, predictor: SamPredictor, conf_thresholds: dict[str, float]) -> np.ndarray:
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    black_background = np.zeros((height, width), dtype=np.uint8)
    results = model(image, imgsz=640, conf=0.1)
    predictor.set_image(image)

    for result in results:
        if result.masks is None:
            continue

        for i, mask in enumerate(result.masks.xy):
            class_id = int(result.boxes.cls[i])
            class_name = model.names[class_id]
            conf = result.boxes.conf[i]

            if class_name in conf_thresholds and conf >= conf_thresholds[class_name]:
                mask = np.array(mask, dtype=np.int32)
                polygon_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(polygon_mask, [mask], 255)

                y_indices, x_indices = np.where(polygon_mask > 0)
                input_points = np.column_stack((x_indices, y_indices))

                sam_masks, _, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=np.ones(len(input_points)),
                    multimask_output=False
                )

                refined_mask = sam_masks[0].astype(np.uint8) * 255
                final_mask = cv2.bitwise_and(refined_mask, polygon_mask)
                
                #update part from last code :  to have better binary mask for the contronet
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated_mask = cv2.dilate(final_mask, kernel, iterations=1)

                black_background[dilated_mask > 0] = 255  

    return black_background

def main(input_folder: str, output_folder: str, yolo_path: str, sam_checkpoint: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    model, predictor = load_models(yolo_path, sam_checkpoint)
    conf_thresholds = {"hand": 0.6, "fingers": 0.4}

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        binary_mask = process_image(image_path, model, predictor, conf_thresholds)
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, binary_mask)

    print(f"✅ Processing complete. Results saved in '{output_folder}'")

if __name__ == "__main__":
    main(
        input_folder='C:/Users/n.dissanayake.GEOSATING/upwork/images',
        output_folder='controlNet',
        yolo_path="C:/Users/n.dissanayake.GEOSATING/upwork/runs/segment/yoloseg6/weights/best.pt",
        sam_checkpoint="sam_vit_h_4b8939.pth"
    )
