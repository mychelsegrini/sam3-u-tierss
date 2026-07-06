'''
Runs inference on a folder of images, producing masks and jsons.
'''

import os
from dotenv import load_dotenv
import glob
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
import json

# Import models
from sam3.model_builder import build_sam3_image_model
from segmenter import SAM3Segmenter

def load_inference_model(sam3_checkpoint_path: str, custom_weights_path: str, device: torch.device) -> SAM3Segmenter:
    """
    Loads the base SAM 3 model and injects the custom trained weights for the neck and decoder.

    Args:
        sam3_checkpoint_path (str): The file path to the original Meta SAM 3 backbone weights.
        custom_weights_path (str): The file path to your custom fine-tuned weights (.pt file).
        device (torch.device): The hardware device (CPU or CUDA) to load the model onto.

    Returns:
        SAM3Segmenter: The fully initialized model set to evaluation mode.
    """
    print("Loading base SAM 3 backbone...")
    base_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint_path)
    
    print("Attaching custom semantic head...")
    model = SAM3Segmenter(base_model, num_classes=3)
    
    print(f"Injecting trained weights from {custom_weights_path}...")
    checkpoint = torch.load(custom_weights_path, map_location=device)
    model.neck.load_state_dict(checkpoint['neck_state'])
    model.head.load_state_dict(checkpoint['head_state'])
    
    model.to(device)
    model.eval() # Lock batch norm layers
    return model

def process_images(
    model: SAM3Segmenter, 
    input_dir: str, 
    colored_mask_output_dir: str, 
    ind_mask_output_dir: str, 
    blended_mask_output_dir: str, 
    json_output_dir: str, 
    device: torch.device
) -> None:
    """
    Runs semantic segmentation inference on an entire folder of images and exports 
    multiple mask formats alongside JSON metadata for downstream processing.

    Args:
        model (SAM3Segmenter): The fully loaded PyTorch segmentation model.
        input_dir (str): The directory containing the raw input images.
        colored_mask_output_dir (str): The destination directory for human-readable RGB masks.
        ind_mask_output_dir (str): The destination directory for machine-readable indexed (0, 1, 2) masks.
        blended_mask_output_dir (str): The destination directory for visual overlays (50% opacity blend).
        json_output_dir (str): The destination directory for the structural metadata.
        device (torch.device): The hardware device executing the tensors.
    """
    os.makedirs(colored_mask_output_dir, exist_ok=True)
    os.makedirs(ind_mask_output_dir, exist_ok=True)
    os.makedirs(blended_mask_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Grab a few images from your input folder
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images. Starting export...")
    
    # Define our colors (BGR format for OpenCV)
    # Class 0 (Path) -> Green
    # Class 1 (Obstacle) -> Red
    # Class 2 (Background) -> Dark Gray
    colors: dict[int, list[int]] = {
        0: [0, 255, 0],
        1: [0, 0, 255], 
        2: [50, 50, 50]
    }

    with torch.no_grad():
        for path in image_paths:
            filename: str = os.path.basename(path)
            name_no_ext: str = os.path.splitext(filename)[0]
            print(f"Processing {filename}...")

            # 1. Load Image
            raw_image = Image.open(path).convert("RGB")
            original_width, original_height = raw_image.size
            
            # 2. Preprocess
            img_resized = TF.resize(raw_image, (1008, 1008), interpolation=TF.InterpolationMode.BILINEAR)
            img_tensor: torch.Tensor = TF.to_tensor(img_resized)
            img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = img_tensor.unsqueeze(0).to(device) 
            
            # 3. Inference
            with torch.autocast(device_type=device.type):
                logits: torch.Tensor = model(img_tensor)
                predicted_mask: torch.Tensor = torch.argmax(logits, dim=1).unsqueeze(1).float()
                
            # 4. Upscale back to original resolution
            restored_mask: torch.Tensor = F.interpolate(predicted_mask, size=(original_height, original_width), mode='nearest')
            
            # Convert to numpy (values will be exactly 0, 1, or 2)
            indexed_mask: np.ndarray = restored_mask.squeeze().cpu().numpy().astype(np.uint8)
            
            # 5. Generate Colored Mask
            color_map: np.ndarray = np.zeros((original_height, original_width, 3), dtype=np.uint8)
            for cls_idx, color in colors.items():
                color_map[indexed_mask == cls_idx] = color
                
            # 6. Generate Blended Image
            original_bgr: np.ndarray = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
            blended: np.ndarray = cv2.addWeighted(original_bgr, 0.5, color_map, 0.5, 0)
            
            # ========================================================
            # 7. EXPORT FILES
            # ========================================================
            # Save Colored Mask
            cv2.imwrite(os.path.join(colored_mask_output_dir, f"{name_no_ext}_colored.png"), color_map)
            
            # Save Blended Visualization
            cv2.imwrite(os.path.join(blended_mask_output_dir, f"{name_no_ext}_blended.jpg"), blended)
            
            # Save Indexed Mask (WARNING: Will look totally black in image viewers, but data is safe!)
            cv2.imwrite(os.path.join(ind_mask_output_dir, f"{name_no_ext}_indexed.png"), indexed_mask)

            metadata: dict = {
                "name": filename,
                "attributes": {
                    "width": original_width,
                    "height": original_height
                },
                "segmentation": {
                    "indexed_mask": f"{name_no_ext}_indexed.png",
                    "colored_mask": f"{name_no_ext}_colored.png",
                    "class_mapping": {
                        "0": "Path",
                        "1": "Obstacle",
                        "2": "Background"
                    }
                }
            }
            
            json_path: str = os.path.join(json_output_dir, f"{name_no_ext}_meta.json")
            with open(json_path, 'w') as json_file:
                json.dump(metadata, json_file, indent=4)

    print(f"Done! Check the '{colored_mask_output_dir}' and '{json_output_dir}' folders.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CONFIGURATION ---
    # Put the best epoch number you found from your evaluate script here
    load_dotenv()
    ROOT_PATH = os.getenv("ROOT_PATH")
    WEIGHTS_RELATIVE_PATH = os.getenv("WEIGHTS_RELATIVE_PATH")
    WEIGHT_NUM = 31
    BEST_WEIGHTS = os.path.join(ROOT_PATH, WEIGHTS_RELATIVE_PATH, f'advanced_semantic_graft_epoch_{WEIGHT_NUM}.pt') 
    
    # Create a folder and drop the images for inference inside it
    TEST_RELATIVE_PATH = os.getenv("TEST_RELATIVE_PATH")

    IMAGES_FOLDER = os.path.join(ROOT_PATH, TEST_RELATIVE_PATH, "images") 
    COLORED_MASKS_FOLDER = os.path.join(ROOT_PATH, TEST_RELATIVE_PATH, "masks/colored")
    INDEXED_MASKS_FOLDER = os.path.join(ROOT_PATH, TEST_RELATIVE_PATH, "masks/indexed")
    BLENDED_MASKS_FOLDER = os.path.join(ROOT_PATH, TEST_RELATIVE_PATH, "masks/blended")
    JSON_FOLDER = os.path.join(ROOT_PATH, TEST_RELATIVE_PATH, "jsons")
    
    # Run the pipeline
    SAM3_RELATIVE_PATH = os.getenv("SAM3_RELATIVE_PATH")
    SAM3_PATH = os.path.join(ROOT_PATH, SAM3_RELATIVE_PATH)
    model = load_inference_model(SAM3_PATH, BEST_WEIGHTS, device)
    process_images(model, IMAGES_FOLDER, COLORED_MASKS_FOLDER, INDEXED_MASKS_FOLDER, BLENDED_MASKS_FOLDER, JSON_FOLDER, device)