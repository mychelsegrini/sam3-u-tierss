"""
Core Architecture and Loss Definitions for the SAM 3 Semantic Segmenter.

This module defines the physical neural network pathways and the mathematical objective
functions required to transform a generalized Vision Transformer into a highly 
calibrated, domain-aware robotic vision system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Tuple

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    
    Why this is needed:
    In autonomous driving datasets (like Cityscapes or BDD100K), target objects vary 
    massively in size—a pedestrian 100 meters away requires fine, local details, while a 
    bus directly in front of the camera requires massive global context. 
    Standard pooling layers destroy resolution. ASPP uses "dilated" (atrous) convolutions 
    to physically expand the model's receptive field to capture this multi-scale context 
    simultaneously, without ever downsampling the image tensor.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        # Branch 1: 1x1 Convolution. 
        # Captures strict, highly localized pixel relationships.
        self.branch1: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branches 2-4: Dilated Convolutions (Rates 6, 12, 18).
        # The 'dilation' parameter spaces out the kernel's focus. A 3x3 kernel with 
        # dilation=18 actually looks at a 39x39 pixel area, capturing massive context 
        # while only paying the computational cost of a 3x3 operation.
        self.branch2: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 5: Global Average Pooling.
        # Squashes the entire image into a single point to understand the absolute global 
        # environment (e.g., "Is this an urban highway or a narrow cobblestone street?").
        self.branch5: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final Bottleneck
        # Fuses all 5 branches (local to global) back into a standardized feature map.
        self.out_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3) # Regularization to prevent overfitting on specific features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size: torch.Size = x.shape[-2:]
        b1: torch.Tensor = self.branch1(x)
        b2: torch.Tensor = self.branch2(x)
        b3: torch.Tensor = self.branch3(x)
        b4: torch.Tensor = self.branch4(x)
        
        # The global branch must be mathematically scaled back up to match the H, W 
        # dimensions of the other branches before concatenation.
        b5: torch.Tensor = F.interpolate(self.branch5(x), size=size, mode='bilinear', align_corners=False)
        
        # Stack the features depth-wise and fuse them
        concat: torch.Tensor = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.out_conv(concat)

class SAM3Segmenter(nn.Module):
    """
    Advanced semantic segmenter designed for full end-to-end fine-tuning.
    
    Architecture Flow:
    1. Backbone (ViT): Extracts deep, generalized spatial embeddings.
    2. ASPP Neck: Standardizes the embeddings across multiple spatial scales.
    3. Decoder Head: Projects the features back to the original image resolution.
    """
    def __init__(self, base_model: nn.Module, num_classes: int = 3) -> None:
        super().__init__()
        
        # ==========================================
        # 1. THE BACKBONE (Unfrozen)
        # By explicitly setting requires_grad=True, we allow the optimizer to fundamentally 
        # alter SAM 3's pre-trained weights, forcing it to learn European street textures 
        # and autonomous driving geometries instead of just general internet images.
        # ==========================================
        self.backbone: nn.Module = base_model.backbone
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        # ==========================================
        # 2. THE NECK
        # ==========================================
        self.neck: ASPP = ASPP(in_channels=256, out_channels=128)
        
        # ==========================================
        # 3. THE DECODER HEAD
        # Uses transposed convolutions (often called de-convolutions) to double the 
        # spatial resolution twice (2x * 2x = 4x scale up), mapping the dense 
        # conceptual features back into physical coordinate space.
        # ==========================================
        self.head: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size: torch.Size = x.shape[-2:]
        
        # Forward pass through the Transformer. 
        # Note: torch.no_grad() is entirely removed here to allow backpropagation.
        features: Union[torch.Tensor, Tuple, Dict] = self.backbone.forward_image(x)
        
        # Safety Wrapper: SAM 3's output formats can vary depending on the specific PyTorch 
        # wrapping or hook injections. This ensures we always extract the raw embedding tensor.
        if isinstance(features, dict):
            first_key: str = list(features.keys())[0]
            spatial_embeddings: torch.Tensor = features[first_key]
        elif isinstance(features, tuple):
            spatial_embeddings: torch.Tensor = features[0]
        else:
            spatial_embeddings: torch.Tensor = features
                
        refined_features: torch.Tensor = self.neck(spatial_embeddings)
        decoded_features: torch.Tensor = self.head(refined_features)
        
        # Final physical interpolation to perfectly match the [1008, 1008] input image
        logits: torch.Tensor = F.interpolate(decoded_features, size=original_size, mode='bilinear', align_corners=False)
        return logits

class SegmentationLoss(nn.Module):
    """
    Combines Focal Loss (with Label Smoothing) and Dice Loss.
    
    Why this specific combination:
    1. Focal Loss handles 'Confidence Calibration'. Standard Cross-Entropy wastes optimizer 
       time punishing the model for easy background pixels. Focal Loss dynamically scales 
       down the loss for easy pixels and heavily penalizes overconfidence on hard edge-cases.
    2. Dice Loss handles 'Structural Calibration'. It directly optimizes the Intersection 
       over Union (IoU) of the masks, forcing the model to care about the contiguous shapes 
       and boundaries of obstacles rather than just guessing isolated pixel probabilities.
    """
    def __init__(
        self, 
        dice_weight: float = 0.5, 
        focal_weight: float = 0.5, 
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        gamma: float = 2.0
    ) -> None:
        super().__init__()
        self.dice_weight: float = dice_weight
        self.focal_weight: float = focal_weight
        self.class_weights: Optional[torch.Tensor] = class_weights
        self.gamma: float = gamma
        
        # Label Smoothing acts as a mathematical guardrail. Instead of targeting exactly 
        # [0, 1, 0] (which causes logits to explode toward Infinity, triggering NaN crashes 
        # in 16-bit precision), it smooths the target to [0.05, 0.90, 0.05].
        # reduction='none' allows us to intercept the loss and apply the Focal factor manually.
        self.ce: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=label_smoothing,
            reduction='none' 
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # 1. FOCAL LOSS (Probability Calibration)
        # ==========================================
        # Calculate raw cross entropy loss per pixel
        ce_loss_unreduced: torch.Tensor = self.ce(inputs, targets)
        
        # Convert raw logits to probabilities (0.0 to 1.0)
        probs: torch.Tensor = F.softmax(inputs, dim=1)
        
        # For every single pixel, gather the model's predicted probability for the TRUE class.
        # If the model is highly confident and correct, 'pt' is close to 1.0.
        pt: torch.Tensor = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply the focal modulation factor: (1 - pt)^gamma
        # If pt is 0.95, (1 - 0.95)^2 is a tiny number, almost eliminating the loss.
        # If pt is 0.10, (1 - 0.10)^2 remains high, forcing the optimizer to fix it.
        focal_loss: torch.Tensor = ((1 - pt) ** self.gamma) * ce_loss_unreduced
        focal_loss = focal_loss.mean() # Reduce the matrix to a single scalar for backprop
        
        # ==========================================
        # 2. DICE LOSS (Structural Calibration)
        # ==========================================
        num_classes: int = inputs.shape[1]
        
        # Convert the flat [Batch, H, W] targets into a one-hot spatial grid [Batch, Classes, H, W]
        targets_one_hot: torch.Tensor = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Compute the Intersection and the Cardinality (Union) across the spatial dimensions
        dims: Tuple[int, int, int] = (0, 2, 3) 
        intersection: torch.Tensor = torch.sum(probs * targets_one_hot, dims)
        cardinality: torch.Tensor = torch.sum(probs + targets_one_hot, dims)
        
        # Dice Score Formula: (2 * Intersection) / (Prediction + Ground Truth)
        # The + 1e-6 prevents zero-division errors if a class is entirely missing from a batch
        dice_score: torch.Tensor = (2. * intersection + 1e-6) / (cardinality + 1e-6)
        
        if self.class_weights is not None:
            # Shift the dice weights to the active GPU to prevent device mismatch crashes
            weights: torch.Tensor = self.class_weights.to(dice_score.device)
            # Subtract from 1.0 because we want to MINIMIZE the loss (maximize the score)
            dice_loss: torch.Tensor = 1.0 - torch.sum(dice_score * weights) / torch.sum(weights)
        else:
            dice_loss: torch.Tensor = 1.0 - dice_score.mean()
        
        # ==========================================
        # 3. COMBINATION & FINAL OUTPUT
        # ==========================================
        return (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)