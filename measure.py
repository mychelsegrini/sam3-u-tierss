"""
Evaluates the custom SAM 3 Semantic Segmenter on the validation dataset.
Uses Stratified Pixel Sampling to extract raw class probabilities and ground truth 
labels without overloading RAM, exporting them to a lightweight CSV for downstream plotting.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any
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

def stratified_pixel_sample(
    true_classes: np.ndarray, 
    prob_path: np.ndarray, 
    prob_obstacle: np.ndarray, 
    prob_bg: np.ndarray,
    samples_per_class: int = 300
) -> Dict[str, List[Any]]:
    """
    Samples an equal number of pixels from each ground-truth class to prevent 
    the massive 'Background' class from washing out our safety metrics.
    """
    sampled_data: Dict[str, List[Any]] = {
        "true_class": [],
        "prob_path": [],
        "prob_obstacle": [],
        "prob_bg": []
    }

    # Loop through our 3 target classes: 0 (Path), 1 (Obstacle), 2 (Background)
    for class_idx in [0, 1, 2]:
        # Find all pixel indices belonging to this ground-truth class
        indices = np.where(true_classes == class_idx)[0]
        
        # If the image doesn't have this class (e.g., no obstacles), skip sampling
        if len(indices) == 0:
            continue
            
        # If we have fewer pixels than requested, just take all of them
        replace = len(indices) < samples_per_class
        sampled_indices = np.random.choice(indices, size=samples_per_class, replace=replace)

        # Extract the data for these specific pixels
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
    output_csv: str
) -> None:
    """
    Iterates through the validation dataset, generates predictions, samples pixels,
    and saves the aggregated data to a CSV.
    """
    model.eval() # Lock batch norm and dropouts
    
    master_data: Dict[str, List[Any]] = {
        "image_name": [],
        "true_class": [],
        "prob_path": [],
        "prob_obstacle": [],
        "prob_bg": []
    }

    # Reconstruct a unified list of filenames from the concatenated datasets
    # This requires shuffle=False in the DataLoader to map correctly
    all_filenames = []
    for ds in val_dataset.datasets:
        if hasattr(ds, 'images'):
            all_filenames.extend(ds.images)
        else:
            all_filenames.extend([f"unknown_file_{i}" for i in range(len(ds))])

    pbar = tqdm(val_dataloader, desc="Evaluating Semantic Masking")

    for batch_idx, (images, masks) in enumerate(pbar):
        # The DataLoader automatically batches our inputs
        image_tensor = images.to(device)
        mask_tensor = masks.to(device)
        
        # Map back to the original filename
        filename = all_filenames[batch_idx] if batch_idx < len(all_filenames) else f"batch_{batch_idx}"

        with torch.no_grad():
            # Use bfloat16 to prevent dtype mismatch crashes with SAM 3
                logits: torch.Tensor = model(image_tensor)
                # Convert raw logits into percentages (0.0 to 1.0)
                probs: torch.Tensor = F.softmax(logits, dim=1).to(torch.float32)

        # Extract values (assuming batch_size=1, index 0 is safe)
        prob_path = probs[0, 0, :, :].flatten().cpu().numpy()
        prob_obstacle = probs[0, 1, :, :].flatten().cpu().numpy()
        prob_bg = probs[0, 2, :, :].flatten().cpu().numpy()
        true_classes = mask_tensor.flatten().cpu().numpy()

        # Sample the pixels
        sampled_data = stratified_pixel_sample(true_classes, prob_path, prob_obstacle, prob_bg)

        # Append to our master tracking dictionary
        num_extracted = len(sampled_data["true_class"])
        master_data["image_name"].extend([filename] * num_extracted)
        master_data["true_class"].extend(sampled_data["true_class"])
        master_data["prob_path"].extend(sampled_data["prob_path"])
        master_data["prob_obstacle"].extend(sampled_data["prob_obstacle"])
        master_data["prob_bg"].extend(sampled_data["prob_bg"])

    print(f"\nExporting {len(master_data['true_class'])} sampled pixels to CSV...")
    df = pd.DataFrame(master_data)
    df.to_csv(output_csv, index=False)
    print(f"Evaluation complete! Saved to {output_csv}")

def main() -> None:
    """Main execution block to set up paths and trigger the evaluation pipeline."""
    load_dotenv()
    ROOT_PATH: str | None = os.getenv("ROOT_PATH")
    if not ROOT_PATH:
        raise ValueError("ROOT_PATH environment variable is not set.")

    # Ensure your .env file has these paths pointing to the parent folder of each dataset's 'images' and 'masks' subdirectories
    BDD_PATH: str | None = os.getenv("BDD_RELATIVE_PATH")
    CITY_PATH: str | None = os.getenv("CITYSCAPES_RELATIVE_PATH")
    MAP_PATH: str | None = os.getenv("MAPILLARY_RELATIVE_PATH")
    
    VALIDATION_RELATIVE_PATH: str | None = os.getenv("VALIDATION_RELATIVE_PATH")
    WEIGHTS_RELATIVE_PATH: str | None = os.getenv("WEIGHTS_RELATIVE_PATH")
    SAM3_RELATIVE_PATH: str | None = os.getenv("SAM3_RELATIVE_PATH")
    
    if not BDD_PATH or not CITY_PATH or not MAP_PATH or not WEIGHTS_RELATIVE_PATH or not SAM3_RELATIVE_PATH or not VALIDATION_RELATIVE_PATH:
        raise ValueError("Missing relative paths in .env file.")
        
    BDD_PATH = os.path.join(ROOT_PATH, BDD_PATH)
    CITY_PATH = os.path.join(ROOT_PATH, CITY_PATH)
    MAP_PATH = os.path.join(ROOT_PATH, MAP_PATH)

    # Target the best epoch you found earlier
    WEIGHTS_PATH: str = os.path.join(ROOT_PATH, WEIGHTS_RELATIVE_PATH, "epoch_1.pt")
    SAM3_PATH: str = os.path.join(ROOT_PATH, SAM3_RELATIVE_PATH)
    output_csv: str = os.path.join(ROOT_PATH, VALIDATION_RELATIVE_PATH, "segmentation_metrics.csv")

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device.type.upper()}...")

    # Initialize Base Architecture
    print("Loading base SAM 3 backbone...")
    base_model = build_sam3_image_model(checkpoint_path=SAM3_PATH)
    model = SAM3Segmenter(base_model=base_model, num_classes=3).to(device)
    
    print("Loading trained weights...")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    # Load the entire model state at once
    model.load_state_dict(checkpoint['model_state_dict'])

    # ------------------------------------------------------------------
    # UNIFIED VALIDATION DATASET INITIALIZATION
    # ------------------------------------------------------------------
    print("Initializing datasets...")
    
    bdd_dataset = BDD100KSemanticDataset(
        image_dir=os.path.join(BDD_PATH, "10k/val"),
        mask_dir=os.path.join(BDD_PATH, "labels/val")
    )
    
    cityscapes_dataset = CityscapesSemanticDataset(
        image_dir=os.path.join(CITY_PATH, "leftImg8bit/val"),
        mask_dir=os.path.join(CITY_PATH, "gtFine/val")
    )
    
    # Update Mapillary paths according to your specific folder structure for validation
    mapillary_dataset = MapillarySemanticDataset(
        image_dir=os.path.join(MAP_PATH, "validation/images"), 
        mask_dir=os.path.join(MAP_PATH, "validation/v2.0/labels")
    )

    # Combine all individual datasets into one continuous iterable
    val_dataset = ConcatDataset([bdd_dataset, cityscapes_dataset, mapillary_dataset])

    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=1, 
        shuffle=False,      # Must be False to accurately map filenames
        num_workers=4,
        pin_memory=True,
        drop_last=False     # We want to evaluate every single image
    )

    evaluate_and_export(model, val_dataloader, val_dataset, device, output_csv)

if __name__ == "__main__":
    main()