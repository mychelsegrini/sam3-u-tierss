import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Tuple

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.branch1: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
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
        
        self.branch5: nn.Sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.out_conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size: torch.Size = x.shape[-2:]
        b1: torch.Tensor = self.branch1(x)
        b2: torch.Tensor = self.branch2(x)
        b3: torch.Tensor = self.branch3(x)
        b4: torch.Tensor = self.branch4(x)
        b5: torch.Tensor = F.interpolate(self.branch5(x), size=size, mode='bilinear', align_corners=False)
        
        concat: torch.Tensor = torch.cat([b1, b2, b3, b4, b5], dim=1)
        return self.out_conv(concat)

class SAM3Segmenter(nn.Module):
    """
    Advanced semantic segmenter designed for full end-to-end fine-tuning.
    The entire SAM 3 backbone is unfrozen and optimized alongside the custom Neck and Head.
    """
    def __init__(self, base_model: nn.Module, num_classes: int = 3) -> None:
        super().__init__()
        
        # 1. UNFREEZE THE ENTIRE BACKBONE
        self.backbone: nn.Module = base_model.backbone
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        # 2. THE CUSTOM NECK (ASPP)
        self.neck: ASPP = ASPP(in_channels=256, out_channels=128)
        
        # 3. THE CUSTOM HEAD (Decoder)
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
        
        # Standard forward pass with gradients enabled
        features: Union[torch.Tensor, Tuple, Dict] = self.backbone.forward_image(x)
        
        if isinstance(features, dict):
            first_key: str = list(features.keys())[0]
            spatial_embeddings: torch.Tensor = features[first_key]
        elif isinstance(features, tuple):
            spatial_embeddings: torch.Tensor = features[0]
        else:
            spatial_embeddings: torch.Tensor = features
                
        refined_features: torch.Tensor = self.neck(spatial_embeddings)
        decoded_features: torch.Tensor = self.head(refined_features)
        
        logits: torch.Tensor = F.interpolate(decoded_features, size=original_size, mode='bilinear', align_corners=False)
        return logits

class SegmentationLoss(nn.Module):
    """
    Combines Focal Loss (with Label Smoothing) and Dice Loss.
    Designed to heavily penalize overconfidence and aggressively target hard-to-classify pixels.
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
        
        # Use PyTorch's built-in CE with label smoothing, but set reduction to 'none' 
        # to manually apply the Focal factor pixel-by-pixel.

        self.ce: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=label_smoothing,
            reduction='none' 
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ==========================================
        # 1. FOCAL LOSS (Replaces standard CE)
        # ==========================================
        # Calculate raw cross entropy loss per pixel
        ce_loss_unreduced: torch.Tensor = self.ce(inputs, targets)
        
        # Extract the probabilities for the correct target classes
        probs: torch.Tensor = F.softmax(inputs, dim=1)
        # Gather the probability of the true class for every pixel
        pt: torch.Tensor = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply the focal modulation: (1 - pt)^gamma
        focal_loss: torch.Tensor = ((1 - pt) ** self.gamma) * ce_loss_unreduced
        focal_loss = focal_loss.mean() # Reduce to a scalar
        
        # ==========================================
        # 2. DICE LOSS
        # ==========================================
        num_classes: int = inputs.shape[1]
        targets_one_hot: torch.Tensor = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        dims: Tuple[int, int, int] = (0, 2, 3) 
        intersection: torch.Tensor = torch.sum(probs * targets_one_hot, dims)
        cardinality: torch.Tensor = torch.sum(probs + targets_one_hot, dims)
        
        dice_score: torch.Tensor = (2. * intersection + 1e-6) / (cardinality + 1e-6)
        
        if self.class_weights is not None:
            weights: torch.Tensor = self.class_weights.to(dice_score.device)
            dice_loss: torch.Tensor = 1.0 - torch.sum(dice_score * weights) / torch.sum(weights)
        else:
            dice_loss: torch.Tensor = 1.0 - dice_score.mean()
        
        # ==========================================
        # 3. COMBINE
        # ==========================================
        return (self.focal_weight * focal_loss) + (self.dice_weight * dice_loss)