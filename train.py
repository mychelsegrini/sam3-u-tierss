"""
End-to-End Fine-Tuning for SAM3-TierSS using Layer-wise Learning Rate Decay.

This script executes a robust, hardware-aware training pipeline that fine-tunes a pre-trained 
Vision Transformer (SAM 3 Backbone) alongside a custom ASPP Neck and Decoder Head.
It trains across multiple autonomous navigation datasets (BDD100K, Cityscapes, Mapillary)
using Gradient Accumulation, Automatic Mixed Precision (AMP), and Intra-epoch Checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import glob
import os
import csv 

# Import custom architecture and dataset modules
from sam3.model_builder import build_sam3_image_model
from segmenter import SAM3Segmenter
from dataset import (
    BDD100KSemanticDataset, 
    CityscapesSemanticDataset, 
    MapillarySemanticDataset
)
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List, Dict, Any

class RobustSegmentationLoss(nn.Module):
    """
    A mathematically armored combination of Focal Loss and Dice Loss.
    Forces all probability math into float32 and clamps extremes to prevent NaN corruption.
    """
    def __init__(self, dice_weight=0.6, focal_weight=0.4, class_weights=None, epsilon=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.class_weights = class_weights 
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. UPCAST TO FLOAT32: Prevents bfloat16 Softmax overflow
        logits = logits.to(torch.float32)
        
        # 2. CALCULATE PROBABILITIES & CLAMP: Prevents log(0) in Focal Loss
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
        
        # Prepare targets (Shape: [B, C, H, W])
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Device-agnostic class weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
        else:
            self.class_weights = torch.ones(num_classes, device=logits.device)
            
        # --- FOCAL LOSS ---
        ce_loss = -targets_one_hot * torch.log(probs)
        gamma = 2.0
        focal_term = (1.0 - probs) ** gamma
        focal_loss = focal_term * ce_loss
        focal_loss = focal_loss.mean(dim=(0, 2, 3)) * self.class_weights
        focal_loss = focal_loss.mean()
        
        # --- DICE LOSS ---
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(probs, dim=(0, 2, 3)) + torch.sum(targets_one_hot, dim=(0, 2, 3))
        
        dice_score = (2. * intersection + self.epsilon) / (union + self.epsilon)
        dice_loss = 1.0 - dice_score
        dice_loss = (dice_loss * self.class_weights).mean()
        
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

def get_parameter_groups(model: nn.Module, base_lr: float, weight_decay: float = 1e-4) -> List[Dict[str, Any]]:
    """
    Implements Layer-wise Learning Rate Decay (LLRD) for the Vision Transformer.
    """
    parameter_groups: List[Dict[str, Any]] = []
    
    # 1. NEW LAYERS (Highest LR)
    head_neck_params = list(model.neck.parameters()) + list(model.head.parameters())
    parameter_groups.append({'params': head_neck_params, 'lr': base_lr, 'weight_decay': weight_decay})
    
    # 2. TRANSFORMER BLOCK CRAWLER
    blocks = None
    for name, module in model.backbone.named_modules():
        if (name.endswith('blocks') or name.endswith('layers')) and isinstance(module, nn.ModuleList):
            if len(module) > 0:  
                blocks = module
                print(f"Success: Found {len(blocks)} Transformer blocks located at 'backbone.{name}'!")
                break
                
    if blocks is None:
        print("Warning: Could not automatically detect Transformer blocks. Using a uniform low LR for the backbone.")
        parameter_groups.append({'params': model.backbone.parameters(), 'lr': base_lr * 0.01, 'weight_decay': weight_decay})
        return parameter_groups

    # 3. EXPONENTIAL DECAY APPLICATION
    lr_decay_factor = 0.8 
    num_blocks = len(blocks)
    
    for idx, block in enumerate(reversed(blocks)):
        block_lr = (base_lr * 0.1) * (lr_decay_factor ** idx)
        parameter_groups.append({'params': block.parameters(), 'lr': block_lr, 'weight_decay': weight_decay})
        
    # 4. PATCH EMBEDDINGS (Lowest LR)
    for name, module in model.backbone.named_modules():
        if 'patch_embed' in name:
            earliest_lr = (base_lr * 0.1) * (lr_decay_factor ** num_blocks)
            parameter_groups.append({'params': module.parameters(), 'lr': earliest_lr, 'weight_decay': weight_decay})
            print(f"Success: Applied lowest LR to Patch Embeddings at 'backbone.{name}'.")
            break
            
    return parameter_groups

def main() -> None:
    # ------------------------------------------------------------------
    # ENVIRONMENT & PATH SETUP
    # ------------------------------------------------------------------
    load_dotenv()
    ROOT_PATH: str = os.getenv("ROOT_PATH")

    BDD_PATH: str = os.path.join(ROOT_PATH, os.getenv("BDD_RELATIVE_PATH"))
    CITY_PATH: str = os.path.join(ROOT_PATH, os.getenv("CITYSCAPES_RELATIVE_PATH"))
    MAP_PATH: str = os.path.join(ROOT_PATH, os.getenv("MAPILLARY_RELATIVE_PATH"))

    WEIGHTS_RELATIVE_PATH: str = os.getenv("WEIGHTS_RELATIVE_PATH")
    WEIGHTS_FOLDER: str = os.path.join(ROOT_PATH, WEIGHTS_RELATIVE_PATH)
    SAM3_RELATIVE_PATH: str = os.getenv("SAM3_RELATIVE_PATH")
    MODEL_PATH: str = os.path.join(ROOT_PATH, SAM3_RELATIVE_PATH)
    
    CSV_LOG_PATH: str = os.path.join(ROOT_PATH, "metrics/epoch_losses.csv")

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing pipeline on: {device.type.upper()}")

    # ------------------------------------------------------------------
    # MODEL, OPTIMIZER & LOSS INITIALIZATION
    # ------------------------------------------------------------------
    base_model = build_sam3_image_model(checkpoint_path=MODEL_PATH)
    model: SAM3Segmenter = SAM3Segmenter(base_model=base_model, num_classes=3).to(device)

    base_lr = 1e-4
    param_groups = get_parameter_groups(model, base_lr)
    optimizer: optim.AdamW = optim.AdamW(param_groups)

    class_weights: torch.Tensor = torch.tensor([1.5, 5.0, 1.0], dtype=torch.float).to(device)
    # Using the new robust loss class defined above
    criterion: RobustSegmentationLoss = RobustSegmentationLoss(dice_weight=0.6, focal_weight=0.4, class_weights=class_weights)

    scaler: GradScaler = GradScaler(device)

    # ------------------------------------------------------------------
    # CHECKPOINT RECOVERY SYSTEM
    # ------------------------------------------------------------------
    resume_checkpoint_path_from_recovery = None
    resume_checkpoint_path_from_epoch = os.path.join(WEIGHTS_FOLDER, "full_tune_epoch_17.pt")
    start_epoch = 0
    start_batch = 0

    if resume_checkpoint_path_from_recovery and os.path.exists(resume_checkpoint_path_from_recovery):
        print(f"Resuming training from {resume_checkpoint_path_from_recovery}...")
        checkpoint = torch.load(resume_checkpoint_path_from_recovery, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] - 1 
        start_batch = checkpoint['batch']
        print(f"Successfully loaded checkpoint. Resuming at epoch {start_epoch + 1}, batch {start_batch}.")

    elif resume_checkpoint_path_from_epoch and os.path.exists(resume_checkpoint_path_from_epoch):
        print(f"Loading checkpoint from {resume_checkpoint_path_from_epoch}...")
        checkpoint = torch.load(resume_checkpoint_path_from_epoch, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Successfully loaded. Resuming training at Epoch {start_epoch + 1}...")

    # ------------------------------------------------------------------
    # TRAINING HYPERPARAMETERS & BATCH CONFIGURATION
    # ------------------------------------------------------------------
    num_epochs: int = 100
    
    physical_batch_size = 2
    accumulation_steps = 8  

    # ------------------------------------------------------------------
    # UNIFIED MULTI-DOMAIN DATASET INITIALIZATION
    # ------------------------------------------------------------------
    print("Initializing datasets...")
    bdd_dataset = BDD100KSemanticDataset(
        image_dir=os.path.join(BDD_PATH, "10k/train"),
        mask_dir=os.path.join(BDD_PATH, "labels/train")
    )
    
    cityscapes_dataset = CityscapesSemanticDataset(
        image_dir=os.path.join(CITY_PATH, "leftImg8bit/train"),
        mask_dir=os.path.join(CITY_PATH, "gtFine/train")
    )
    
    mapillary_dataset = MapillarySemanticDataset(
        image_dir=os.path.join(MAP_PATH, "training/images"),
        mask_dir=os.path.join(MAP_PATH, "training/v2.0/labels")
    )

    train_dataset = ConcatDataset([bdd_dataset, cityscapes_dataset, mapillary_dataset])
    print(f"Total training images after aggregation: {len(train_dataset)}")

    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=physical_batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True, 
        drop_last=True   
    )

    tracked_epochs: list[int] = []
    tracked_losses: list[float] = []
    save_frequency = 3000  
    
    # ==================================================================
    # CORE TRAINING LOOP
    # ==================================================================
    for epoch in range(start_epoch, num_epochs):
        model.train() 
        running_loss: float = 0.0
        
        optimizer.zero_grad(set_to_none=True) 
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
                
            images = images.to(device)
            masks = masks.to(device).long()
            
            # 1. MIXED PRECISION FORWARD PASS (LOGITS ONLY)
            with torch.autocast(device_type=device.type):
                logits: torch.Tensor = model(images)
                
            # 2. LOSS CALCULATION OUTSIDE AUTOCAST
            # This ensures Softmax and log(p) are done in safe float32 math
            loss: torch.Tensor = criterion(logits, masks)
            loss = loss / accumulation_steps
                
            # 3. SCALED BACKWARD PASS
            scaler.scale(loss).backward()
            
            # 4. GRADIENT ACCUMULATION & OPTIMIZER STEP
            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # --------------------------------------------------------------
            # INTRA-EPOCH EMERGENCY BACKUP
            # --------------------------------------------------------------
            if (batch_idx + 1) % save_frequency == 0:
                mid_epoch_save_path = os.path.join(WEIGHTS_FOLDER, f"recovery_epoch_{epoch+1}_batch_{batch_idx+1}.pt")
                checkpoint = {
                    'epoch': epoch + 1,           
                    'batch': batch_idx + 1,       
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                }
                torch.save(checkpoint, mid_epoch_save_path)

            running_loss += (loss.item() * accumulation_steps)
            pbar.set_postfix({"Loss": f"{(loss.item() * accumulation_steps):.4f}"})
            
        # ==================================================================
        # EPOCH FINALIZATION
        # ==================================================================
        epoch_loss: float = running_loss / len(train_dataloader)
        print(f"--> Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f}")
        
        tracked_epochs.append(epoch + 1)
        tracked_losses.append(epoch_loss)
        
        file_exists = os.path.isfile(CSV_LOG_PATH)
        with open(CSV_LOG_PATH, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['Epoch', 'Average_Loss'])
            writer.writerow([epoch + 1, f"{epoch_loss:.6f}"])

        pt_save_path = os.path.join(WEIGHTS_FOLDER, f"full_tune_epoch_{epoch+1}.pt")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(checkpoint, pt_save_path)
        
        recovery_pattern = os.path.join(WEIGHTS_FOLDER, f"recovery_epoch_{epoch+1}_batch_*.pt")
        recovery_files = glob.glob(recovery_pattern)
        
        for file_path in recovery_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not automatically delete {file_path}. Error: {e}")
                
        if recovery_files:
            print(f"Cleaned up {len(recovery_files)} mid-epoch recovery files.")

        print(f"Epoch {epoch+1} saved to {pt_save_path}")

    # ==================================================================
    # TRAINING COMPLETE: GENERATE PLOTS
    # ==================================================================
    print("\nTraining Complete! Generating Loss Curve...")

    plt.figure(figsize=(10, 6))
    plt.plot(tracked_epochs, tracked_losses, marker='o', linestyle='-', color='indigo', linewidth=2)
    plt.title('Training Loss over Epochs (Full End-to-End Fine-Tuning)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Weighted Cross Entropy + Dice)')
    plt.xticks(tracked_epochs)
    plt.grid(True, linestyle='--', alpha=0.7)

    image_name: str = 'full_tuning_loss_curve'
    plt.savefig(f'{image_name}.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to '{image_name}.png'!")

if __name__ == "__main__":
    main()