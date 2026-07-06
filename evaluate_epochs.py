'''
Runs inference on a folder of images with a folder of .pt files, getting the 
validation error across different epochs of training.
'''

import torch
import torch.nn as nn
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Import your custom modules
from sam3.model_builder import build_sam3_image_model
from segmenter import SAM3Segmenter, SegmentationLoss
from dataset import (
    BDD100KSemanticDataset, 
    CityscapesSemanticDataset, 
    MapillarySemanticDataset
)
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

def evaluate_single_epoch(
    model: SAM3Segmenter, 
    dataloader: DataLoader, 
    criterion: SegmentationLoss, 
    device: torch.device, 
    epoch: int, 
    num_epochs: int
) -> float:
    """
    Runs the validation set through the model and calculates the average loss for a single epoch.
    """
    model.eval() # Lock batch norm and dropouts
    total_loss: float = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    
    with torch.no_grad(): # Turn off gradient tracking to save VRAM
        for images, masks in pbar:
            images = images.to(device)
            # Masks must be long integers (0, 1, 2) for CrossEntropyLoss/FocalLoss
            masks = masks.to(device, dtype=torch.long)
            
            with torch.autocast(device_type=device.type):
                # 1. Forward Pass
                logits: torch.Tensor = model(images)
                
                # 2. Calculate Error 
                loss: torch.Tensor = criterion(logits, masks)
                
            total_loss += loss.item()
            
    # Return the average loss across all batches
    return total_loss / len(dataloader)

def main() -> None:
    """
    Main execution pipeline for model evaluation. 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}...")
    
    load_dotenv()
    ROOT_PATH: str = os.getenv("ROOT_PATH")
    
    # Load dataset paths
    BDD_PATH: str = os.path.join(ROOT_PATH, os.getenv("BDD_RELATIVE_PATH"))
    CITY_PATH: str = os.path.join(ROOT_PATH, os.getenv("CITYSCAPES_RELATIVE_PATH"))
    MAP_PATH: str = os.path.join(ROOT_PATH, os.getenv("MAPILLARY_RELATIVE_PATH"))

    WEIGHTS_RELATIVE_PATH: str = os.getenv("WEIGHTS_RELATIVE_PATH")
    WEIGHTS_FOLDER: str = os.path.join(ROOT_PATH, WEIGHTS_RELATIVE_PATH)
    SAM3_RELATIVE_PATH: str = os.getenv("SAM3_RELATIVE_PATH")
    MODEL_PATH: str = os.path.join(ROOT_PATH, SAM3_RELATIVE_PATH)

    # 1. Setup Validation Dataloader (Unified Datasets)
    print("Initializing validation datasets...")
    bdd_val = BDD100KSemanticDataset(
        image_dir=os.path.join(BDD_PATH, "10k/val"),
        mask_dir=os.path.join(BDD_PATH, "labels/val")
    )
    cityscapes_val = CityscapesSemanticDataset(
        image_dir=os.path.join(CITY_PATH, "leftImg8bit/val"),
        mask_dir=os.path.join(CITY_PATH, "gtFine/val")
    )
    mapillary_val = MapillarySemanticDataset(
        image_dir=os.path.join(MAP_PATH, "images"),
        mask_dir=os.path.join(MAP_PATH, "v2.0/labels")
    )

    val_dataset = ConcatDataset([bdd_val, cityscapes_val, mapillary_val])
    print(f"Total validation images: {len(val_dataset)}")
    
    # Batch size can be larger than training since we aren't storing gradients
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Initialize Base Architecture
    print("Loading base SAM 3 backbone...")
    base_model = build_sam3_image_model(checkpoint_path=MODEL_PATH)
    model = SAM3Segmenter(base_model=base_model, num_classes=3).to(device)
    
    # Focal + Dice Loss matching your train.py configuration
    class_weights: torch.Tensor = torch.tensor([1.5, 5.0, 1.0], dtype=torch.float).to(device)
    criterion = SegmentationLoss(dice_weight=0.6, focal_weight=0.4, class_weights=class_weights)
    
    # 3. Iterate through all saved checkpoints
    epochs: list[int] = []
    val_errors: list[float] = []
    
    # Scan up to 100 epochs based on the new training script
    TOTAL_EPOCHS: int = 100 
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        weight_path: str = os.path.join(WEIGHTS_FOLDER, f"full_tune_epoch_{epoch}.pt")
        
        if not os.path.exists(weight_path):
            # Fail silently on missing epochs to allow mid-training evaluation
            continue
            
        print(f"\nEvaluating Epoch {epoch}...")
        
        # Load the full end-to-end model weights
        checkpoint: dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Grade the model
        avg_loss: float = evaluate_single_epoch(model, val_dataloader, criterion, device, epoch, TOTAL_EPOCHS)
        
        epochs.append(epoch)
        val_errors.append(avg_loss)
        print(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")

    if not epochs:
        print("No trained epoch checkpoints found in the weights folder. Exiting...")
        return

    # 4. Find the best model (Minimum Error)
    best_epoch_index: int = np.argmin(val_errors)
    best_epoch: int = epochs[best_epoch_index]
    min_loss: float = val_errors[best_epoch_index]
    
    print("-" * 30)
    print(f"🏆 BEST MODEL: Epoch {best_epoch} with a validation loss of {min_loss:.4f}")
    print("-" * 30)

    # 5. Plotting the Automation
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_errors, marker='o', linestyle='-', color='b', linewidth=2)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    
    plt.title('Validation Error across Training Epochs (Unified Domain)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (Focal + Dice)')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the graph
    image_name: str = "validation_loss_curve"
    plt.savefig(f'{image_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {image_name}.png")

if __name__ == "__main__":
    main()