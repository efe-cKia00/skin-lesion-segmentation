import os
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class SegmentationDatasets(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Ensure alphabetical alignment between images and masks
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # This line creates the full path (relative or absolute, depending on self.image_dir)
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        # Load as PIL images and convert to numpy arrays
        image = np.array(Image.open(img_path).convert("RGB"))  # RGB for images
        mask = np.array(Image.open(mask_path).convert("L"))    # Grayscale for masks, 0-255

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # IMPORTANT: Normalize mask to binary [0, 1] AFTER transform pipeline
        # This ensures the mask is always in the correct range regardless of transform
        import torch
        if isinstance(mask, torch.Tensor):
            # If already a tensor, threshold at 0.5 (works for both [0,1] and [0,255])
            mask = (mask > 0.5).float()
            # Ensure mask has shape [1, H, W]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            # If still numpy, convert and normalize
            mask = (mask > 127.5).astype(np.float32)

        return image, mask
    

# Define the transform pipeline
train_transform = A.Compose([
    # 1. Spatial Transforms (Applied to both Image + Mask)
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),

    # 2. Pixel-level Transforms (Only applied to Image)
    A.RandomBrightnessContrast(p=0.2),

    # 3. Normalization (Image only)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
    # 4. Convert to tensor
    # Note: Mask normalization happens in __getitem__ after transforms
    ToTensorV2(),
], is_check_shapes=False)

# Validation transform (No random flips/distortion!)
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # Note: Mask normalization happens in __getitem__ after transforms
    ToTensorV2(),
], is_check_shapes=False)