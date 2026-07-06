import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from typing import Dict, List, Tuple

# ==========================================
# 1. BDD100K
# ==========================================
class BDD100KSemanticDataset(Dataset):
    """BDD100K mapped from 19 classes to 3."""
    def __init__(self, image_dir: str, mask_dir: str) -> None:
        super().__init__()
        self.image_dir: str = image_dir
        self.mask_dir: str = mask_dir
        
        self.images: List[str] = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
        self.masks: List[str] = sorted([f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))])
        
        # 0: Path | 1: Obstacle | 2: Background
        self.class_map: Dict[int, int] = {
            0: 0, 1: 0, 
            11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 4: 1, 5: 1, 6: 1, 7: 1, 3: 1, 
            2: 2, 8: 2, 9: 2, 10: 2, 255: 2
        }

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path: str = os.path.join(self.image_dir, self.images[idx])
        mask_path: str = os.path.join(self.mask_dir, self.masks[idx])
        
        return _process_image_and_mask(img_path, mask_path, self.class_map)


# ==========================================
# 2. Cityscapes
# ==========================================
class CityscapesSemanticDataset(Dataset):
    """Cityscapes mapped from TrainIDs (0-18) to 3 classes. Recursively searches city subfolders."""
    def __init__(self, image_dir: str, mask_dir: str) -> None:
        super().__init__()
        self.image_dir: str = image_dir
        self.mask_dir: str = mask_dir
        
        # glob recursively searches through the city subfolders and returns the FULL paths
        self.images: List[str] = sorted(glob.glob(os.path.join(image_dir, '**', '*_leftImg8bit.png'), recursive=True))
        self.masks: List[str] = sorted(glob.glob(os.path.join(mask_dir, '**', '*_labelIds.png'), recursive=True))
        
        # Safety check to ensure subfolder structures match perfectly
        if len(self.images) != len(self.masks):
            print(f"WARNING: Cityscapes image count ({len(self.images)}) does not match mask count ({len(self.masks)})!")

        # Cityscapes TrainIDs -> 0: Path | 1: Obstacle | 2: Background
        self.class_map: Dict[int, int] = {
            0: 0, 1: 0, # Road, Sidewalk
            11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, # Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle
        }
        # Anything not explicitly defined in the map above will be defaulted to 2 (Background)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # glob already provides the absolute path, so os.path.join is no longer needed here
        img_path: str = self.images[idx]
        mask_path: str = self.masks[idx]
        
        return _process_image_and_mask(img_path, mask_path, self.class_map)


# ==========================================
# 3. Mapillary Vistas
# ==========================================
class MapillarySemanticDataset(Dataset):
    """Mapillary Vistas mapped from v2.0 to 3 classes."""
    def __init__(self, image_dir: str, mask_dir: str) -> None:
        super().__init__()
        self.image_dir: str = image_dir
        self.mask_dir: str = mask_dir
        
        self.images: List[str] = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
        self.masks: List[str] = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        self.class_map = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 1, 21: 0, 22: 0, 23: 0, 24: 0, 25: 1, 26: 2, 27: 2, 28: 2, 29: 2, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 2, 60: 2, 61: 2, 62: 2, 63: 2, 64: 2, 65: 2, 66: 2, 67: 1, 68: 1, 69: 1, 70: 1, 71: 1, 72: 1, 73: 1, 74: 1, 75: 1, 76: 1, 77: 1, 78: 2, 79: 2, 80: 2, 81: 2, 82: 2, 83: 2, 84: 2, 85: 2, 86: 2, 87: 2, 88: 2, 89: 1, 90: 2, 91: 2, 92: 2, 93: 2, 94: 2, 95: 2, 96: 2, 97: 2, 98: 2, 99: 2, 100: 2, 101: 2, 102: 2, 103: 2, 104: 1, 105: 1, 106: 1, 107: 1, 108: 1, 109: 1, 110: 1, 111: 1, 112: 1, 113: 1, 114: 1, 115: 1, 116: 1, 117: 1, 118: 2, 119: 2, 120: 2, 121: 2, 122: 2, 123: 2}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path: str = os.path.join(self.image_dir, self.images[idx])
        mask_path: str = os.path.join(self.mask_dir, self.masks[idx])
        
        return _process_image_and_mask(img_path, mask_path, self.class_map)


# ==========================================
# 4. Custom Local Dataset (Oversampled)
# ==========================================
class LisbonSidewalkDataset(Dataset):
    """
    Custom dataset with built-in oversampling to prevent the 
    massive global datasets from drowning out these local features.
    """
    def __init__(self, image_dir: str, mask_dir: str, multiplier: int = 100) -> None:
        super().__init__()
        self.image_dir: str = image_dir
        self.mask_dir: str = mask_dir
        
        base_images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
        base_masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        # Artificially expand the dataset by duplicating the file references
        self.images: List[str] = base_images * multiplier
        self.masks: List[str] = base_masks * multiplier
        
        self.class_map: Dict[int, int] = {0: 0, 1: 1, 2: 2}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path: str = os.path.join(self.image_dir, self.images[idx])
        mask_path: str = os.path.join(self.mask_dir, self.masks[idx])
        
        return _process_image_and_mask(img_path, mask_path, self.class_map)


# ==========================================
# Helper Function (DRY Principle)
# ==========================================
def _process_image_and_mask(img_path: str, mask_path: str, class_map: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Handles the repetitive image loading, resizing, and tensor mapping for all datasets."""
    image: Image.Image = Image.open(img_path).convert("RGB")
    mask: Image.Image = Image.open(mask_path)

    # 1. Resize
    image = TF.resize(image, (1008, 1008), interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (1008, 1008), interpolation=TF.InterpolationMode.NEAREST)

    # 2. To Tensor & Normalize Image
    image_tensor: torch.Tensor = TF.to_tensor(image)
    image_tensor = TF.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    mask_tensor: torch.Tensor = torch.as_tensor(np.array(mask), dtype=torch.long)

    # 3. Fast mapping (defaulting unmapped pixels to 2: Background)
    mapped_mask: torch.Tensor = torch.full_like(mask_tensor, fill_value=2)
    for old_val, new_val in class_map.items():
        mapped_mask[mask_tensor == old_val] = new_val

    return image_tensor, mapped_mask