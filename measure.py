"""
Evaluates the custom SAM 3 Semantic Segmenter on the validation dataset.
Uses Stratified Pixel Sampling to extract raw class probabilities, while simultaneously 
calculating the global Intersection over Union (IoU) across the entire dataset.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from torch.utils.data import ConcatDataset, DataLoader

# Import custom modules
from sam3.model_builder import build_sam3_image_model
from segmenter import SAM3Segmenter
from dataset import (
    BDD100KSemanticDataset, 
    CityscapesSemanticDataset, 
    MapillarySemanticDataset
)

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Any, **_kwargs: Any) -> Any:
        return iterable

def update_iou_metrics(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the raw Intersection and Union pixel counts for a single batch.
    Operating directly on the GPU keeps this calculation fast.
    """
    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        label_mask = (labels == cls)
        
        intersection[cls] = (pred_mask & label_mask).sum().float()
        union[cls] = (pred_mask | label_mask).sum().float()
        
    return intersection, union

def stratified_pixel_sample(
    true_classes: np.ndarray, 
    prob_path: np.ndarray, 
    prob_obstacle: np.ndarray, 
    prob_bg: np.ndarray,
    samples_per_class: int = 300
) -> Dict[str, List[Any]]:
    """
    Samples an equal number of pixels from each ground-truth class to prevent 
    the massive 'Background' class from washing out our probability distributions.
    """
    sampled_data: Dict[str, List[Any]] = {
        "true_class": [],
        "prob_path": [],
        "prob_obstacle": [],
        "prob_bg": []
    }

    for class_idx in [0, 1, 2]:
        indices = np.where(true_classes == class_idx)[0]
        
        if len(indices) == 0:
            continue
            
        replace = len(indices) < samples_per_class
        sampled_indices = np.random.choice(indices, size=samples_per_class, replace=replace)

        sampled_data["true_class"].extend(true_classes[sampled_indices].tolist())
        sampled_data["prob_path"].extend(prob_path[sampled_indices].tolist())
        sampled_data["prob_obstacle"].extend(prob_obstacle[sampled_indices].tolist())
        sampled_data["prob_bg"].extend(prob_bg[sampled_indices].tolist())

    return sampled_data

def evaluate_and_export(
    model: SAM3Segmenter, 
    val_dataloader: DataLoader, 
    val_dataset: ConcatDataset,
    device: torch.device,
    pixel_csv_path: str,
    iou_csv_path: str
) -> None:
    """
    Iterates through the validation dataset, generates predictions, calculates global IoU, 
    samples pixels, and saves the aggregated data to CSVs.
    """
    model.eval() 
    
    master_data: Dict[str, List[Any]] = {
        "image_name": [],
        "true_class": [],
        "prob_path": [],
        "prob_obstacle": [],
        "prob_bg": []
    }

    # IoU Tracking Accumulators
    global_intersection = torch.zeros(3, device=device)
    global_union = torch.zeros(3, device=device)

    all_filenames = []
    for ds in val_dataset.datasets:
        if hasattr(ds, 'images'):
            all_filenames.extend(ds.images)
        else:
            all_filenames.extend([f"unknown_file_{i}" for i in range(len(ds))])

    pbar = tqdm(val_dataloader, desc="Evaluating Semantic Masking")

    for batch_idx, (images, masks) in enumerate(pbar):
        image_tensor = images.to(device)
        mask_tensor = masks.to(device)
        
        filename = all_filenames[batch_idx] if batch_idx < len(all_filenames) else f"batch_{batch_idx}"

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits: torch.Tensor = model(image_tensor)
            
            # Step out of bfloat16 for mathematical stability during Softmax and Argmax
            logits = logits.to(torch.float32)
            probs: torch.Tensor = F.softmax(logits, dim=1)
            preds: torch.Tensor = torch.argmax(logits, dim=1)

        # ------------------------------------------------------------------
        # CALCULATE INTERSECTION & UNION
        # ------------------------------------------------------------------
        intersect, union = update_iou_metrics(preds, mask_tensor, num_classes=3)
        global_intersection += intersect
        global_union += union

        # Extract values for pixel sampling
        prob_path = probs[0, 0, :, :].flatten().cpu().numpy()
        prob_obstacle = probs[0, 1, :, :].flatten().cpu().numpy()
        prob_bg = probs[0, 2, :, :].flatten().cpu().numpy()
        true_classes = mask_tensor.flatten().cpu().numpy()

        sampled_data = stratified_pixel_sample(true_classes, prob_path, prob_obstacle, prob_bg)

        num_extracted = len(sampled_data["true_class"])
        master_data["image_name"].extend([filename] * num_extracted)
        master_data["true_class"].extend(sampled_data["true_class"])
        master_data["prob_path"].extend(sampled_data["prob_path"])
        master_data["prob_obstacle"].extend(sampled_data["prob_obstacle"])
        master_data["prob_bg"].extend(sampled_data["prob_bg"])

    # ------------------------------------------------------------------
    # FINALIZE IOU CALCULATIONS
    # ------------------------------------------------------------------
    # Add 1e-6 to prevent division by zero in case a class was totally absent
    final_ious = global_intersection / (global_union + 1e-6)
    
    path_iou = final_ious[0].item()
    obstacle_iou = final_ious[1].item()
    bg_iou = final_ious[2].item()
    mIoU = final_ious.mean().item()

    print("\n" + "="*40)
    print("🚦 GLOBAL VALIDATION IOU SCORES 🚦")
    print("="*40)
    print(f"Path (Class 0):      {path_iou:.4f}")
    print(f"Obstacle (Class 1):  {obstacle_iou:.4f}")
    print(f"Background (Class 2):{bg_iou:.4f}")
    print("-" * 40)
    print(f"Mean IoU (mIoU):     {mIoU:.4f}")
    print("="*40 + "\n")

    # Export IoU Summary
    iou_df = pd.DataFrame([{
        "Path_IoU": path_iou,
        "Obstacle_IoU": obstacle_iou,
        "Background_IoU": bg_iou,
        "mIoU": mIoU
    }])
    iou_df.to_csv(iou_csv_path, index=False)
    print(f"IoU summary saved to {iou_csv_path}")

    # Export Sampled Pixels
    print(f"Exporting {len(master_data['true_class'])} sampled pixels to CSV...")
    df = pd.DataFrame(master_data)
    df.to_csv(pixel_csv_path, index=False)
    print(f"Pixel probability data saved to {pixel_csv_path}")

def main() -> None:
    """Main execution block to set up paths and trigger the evaluation pipeline."""
    load_dotenv()
    ROOT_PATH: str | None = os.getenv("ROOT_PATH")
    if not ROOT_PATH:
        raise ValueError("ROOT_PATH environment variable is not set.")

    BDD_PATH: str | None = os.getenv("BDD_RELATIVE_PATH")
    CITY_PATH: str | None = os.getenv("CITYSCAPES_RELATIVE_PATH")
    MAP_PATH: str | None = os.getenv("MAPILLARY_RELATIVE_PATH")
    
    VALIDATION_RELATIVE_PATH: str | None = os.getenv("VALIDATION_RELATIVE_PATH")
    WEIGHTS_RELATIVE_PATH: str | None = os.getenv("WEIGHTS_RELATIVE_PATH")
    SAM3_RELATIVE_PATH: str | None = os.getenv("SAM3_RELATIVE_PATH")
    
    if not all([BDD_PATH, CITY_PATH, MAP_PATH, WEIGHTS_RELATIVE_PATH, SAM3_RELATIVE_PATH, VALIDATION_RELATIVE_PATH]):
        raise ValueError("Missing relative paths in .env file.")
        
    BDD_PATH = os.path.join(ROOT_PATH, BDD_PATH)
    CITY_PATH = os.path.join(ROOT_PATH, CITY_PATH)
    MAP_PATH = os.path.join(ROOT_PATH, MAP_PATH)

    # Target the specific epoch you want to measure
    WEIGHTS_PATH: str = os.path.join(ROOT_PATH, WEIGHTS_RELATIVE_PATH, "full_tune_epoch_x.pt")
    SAM3_PATH: str = os.path.join(ROOT_PATH, SAM3_RELATIVE_PATH)
    
    # Dual outputs
    pixel_csv_path: str = os.path.join(ROOT_PATH, VALIDATION_RELATIVE_PATH, "segmentation_pixel_metrics.csv")
    iou_csv_path: str = os.path.join(ROOT_PATH, VALIDATION_RELATIVE_PATH, "segmentation_iou_summary.csv")

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device.type.upper()}...")

    print("Loading base SAM 3 backbone...")
    base_model = build_sam3_image_model(checkpoint_path=SAM3_PATH)
    model = SAM3Segmenter(base_model=base_model, num_classes=3).to(device)
    
    print("Loading trained weights...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Initializing datasets...")
    bdd_dataset = BDD100KSemanticDataset(
        image_dir=os.path.join(BDD_PATH, "10k/val"),
        mask_dir=os.path.join(BDD_PATH, "labels/val")
    )
    cityscapes_dataset = CityscapesSemanticDataset(
        image_dir=os.path.join(CITY_PATH, "leftImg8bit/val"),
        mask_dir=os.path.join(CITY_PATH, "gtFine/val")
    )
    mapillary_dataset = MapillarySemanticDataset(
        image_dir=os.path.join(MAP_PATH, "validation/images"), 
        mask_dir=os.path.join(MAP_PATH, "validation/v2.0/labels")
    )

    val_dataset = ConcatDataset([bdd_dataset, cityscapes_dataset, mapillary_dataset])

    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=1, 
        shuffle=False,      
        num_workers=4,
        pin_memory=True,
        drop_last=False     
    )

    evaluate_and_export(model, val_dataloader, val_dataset, device, pixel_csv_path, iou_csv_path)

if __name__ == "__main__":
    main()