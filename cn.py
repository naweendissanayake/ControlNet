import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import mediapipe as mp
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from sklearn.cluster import KMeans
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel
from typing import Tuple, List, Union


def load_models(yolo_path: str, sam_checkpoint: str) -> Tuple[YOLO, SamPredictor]:
    model = YOLO(yolo_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    return model, SamPredictor(sam)


def extract_hand_keypoints(image: np.ndarray) -> np.ndarray:
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
    results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    canvas = np.zeros_like(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x = int(lm.x * image.shape[1])
                y = int(lm.y * image.shape[0])
                cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)
    return canvas


def generate_depth_map(image: np.ndarray) -> np.ndarray:
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").eval().to("cuda")
    inputs = feature_extractor(images=pil_image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted = outputs.predicted_depth
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    depth_min, depth_max = predicted.min(), predicted.max()
    normalized = 255 * (predicted - depth_min) / (depth_max - depth_min)
    return normalized.astype(np.uint8)


def nearest_colour(palette: np.ndarray, colour: np.ndarray) -> np.ndarray:
    return palette[np.argmin(np.linalg.norm(palette - colour, axis=1))]


def apply_colour_palette_to_masked_area(
    image: Image.Image,
    mask: Image.Image,
    palette: List[Tuple[int, int, int]],
    n_colors: int = 5,
) -> Image.Image:
    img_np = np.array(image)
    mask_resized = mask.resize(image.size)
    mask_np = np.array(mask_resized.convert("L")) > 128
    if img_np.shape[:2] != mask_np.shape:
        raise ValueError("Image and mask dimensions do not match")
    masked_pixels = img_np[mask_np]
    if masked_pixels.size == 0:
        return image
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(masked_pixels)
    new_colors = np.array(palette)
    cluster_centers = kmeans.cluster_centers_
    mapped_centers = np.array([nearest_colour(new_colors, colour) for colour in cluster_centers])
    labels = kmeans.predict(masked_pixels)
    img_remap = img_np.copy()
    img_remap[mask_np] = mapped_centers[labels]
    return Image.fromarray(img_remap)


def apply_mask_to_image(
    generated_img: Image.Image,
    mask_img: Union[np.ndarray, Image.Image],
    bg_colour: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Applies the given mask to the generated image, setting the non-masked areas to the given background colour.

    Args:
        generated_img: The image to which the mask will be applied.
        mask_img: The mask to be applied, where non-zero values represent the masked area.
        bg_colour: The background colour to apply where the mask is not applied.

    Returns:
        The final image with the mask applied.
    """
    if isinstance(mask_img, np.ndarray):
        mask_img = Image.fromarray(mask_img)
    mask_img = mask_img.convert("RGB")
    mask_resized = mask_img.resize(generated_img.size, resample=Image.BICUBIC)
    gen_np = np.array(generated_img)
    mask_np = np.array(mask_resized.convert("L"))
    mask_bool = mask_np > 128
    gen_np[~mask_bool] = bg_colour
    return Image.fromarray(gen_np)


def inpaint_with_controlnet(
    hand_mask_path: str,
    depth_map_path: str,
    palette: List[Tuple[int, int, int]],
    output_folder: str,
    image_name: str,
) -> None:
    hand_mask = Image.open(hand_mask_path).convert("RGB")
    hand_mask_gray = hand_mask.convert("L")
    mask_np = np.array(hand_mask_gray)
    binary_mask_np = (mask_np > 128).astype(np.uint8) * 255
    hand_mask_gray = Image.fromarray(binary_mask_np)
    canny_np = cv2.Canny(binary_mask_np, 100, 200)
    control_image_canny = Image.fromarray(canny_np).convert("RGB")
    control_image_depth = Image.open(depth_map_path).convert("RGB")
    controlnets = [
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16),
    ]
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnets,
        torch_dtype=torch.float16,
    ).to("cuda")
    prompt = "realistic dark-skinned human hand, photorealistic, soft light, detailed dark red skin texture"
    negative_prompt = "blurry, deformed fingers, extra fingers, missing fingers, cartoon, sketch, painted, no black colour"
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=hand_mask,
        mask_image=hand_mask_gray,
        control_image=[control_image_canny, control_image_depth],
        controlnet_conditioning_scale=[1.0, 0.8],
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
        guidance_scale=8.0,
    )
    generated_image = result.images[0]
    final_image = apply_mask_to_image(generated_image, hand_mask, bg_colour=(255, 255, 255))
    final_palette_image = apply_colour_palette_to_masked_area(final_image, hand_mask_gray, palette)
    final_image.save(os.path.join(output_folder, f"{image_name}_controlnet_final.png"))
    final_palette_image.save(os.path.join(output_folder, f"{image_name}_controlnet_final_palette.png"))


def main(input_folder: str, output_folder: str, yolo_path: str, sam_checkpoint: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    model, predictor = load_models(yolo_path, sam_checkpoint)
    conf_thresholds = {"hand": 0.6, "fingers": 0.4}
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
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
                    mask_np = np.array(mask, dtype=np.int32)
                    polygon_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(polygon_mask, [mask_np], 255)
                    y_indices, x_indices = np.where(polygon_mask > 0)
                    input_points = np.column_stack((x_indices, y_indices))
                    sam_masks, _, _ = predictor.predict(
                        point_coords=input_points,
                        point_labels=np.ones(len(input_points)),
                        multimask_output=False,
                    )
                    refined_mask = sam_masks[0].astype(np.uint8) * 255
                    final_mask = cv2.bitwise_and(refined_mask, polygon_mask)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    dilated_mask = cv2.dilate(final_mask, kernel, iterations=1)
                    black_background[dilated_mask > 0] = 255
        masked_image = cv2.bitwise_and(image, image, mask=black_background)
        openpose_map = extract_hand_keypoints(image)
        depth_map = generate_depth_map(masked_image)
        base_name = os.path.splitext(image_file)[0]
        mask_path = os.path.join(output_folder, f"{base_name}_mask.png")
        depth_path = os.path.join(output_folder, f"{base_name}_depth.png")
        final_image = apply_mask_to_image(Image.fromarray(masked_image), black_background, bg_colour=(255, 255, 255))
        final_palette_image = apply_colour_palette_to_masked_area(final_image, Image.fromarray(black_background), [(139, 0, 0)])
        final_image.save(os.path.join(output_folder, f"{base_name}_final.png"))
        final_palette_image.save(os.path.join(output_folder, f"{base_name}_final_palette.png"))
        cv2.imwrite(mask_path, black_background)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_openpose.png"), openpose_map)
        cv2.imwrite(depth_path, depth_map)
        palette = [(139, 0, 0)]
        inpaint_with_controlnet(mask_path, depth_path, palette, output_folder, base_name)


if __name__ == "__main__":
    main(
        input_folder "images",
        output_folder="controlNet2",
        yolo_path "best.pt",
        sam_checkpoint="sam_vit_h_4b8939.pth",
    )
