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
        mask = np.array(Image.open(mask_path).convert("L"))    # Grayscale for masks

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask
    

# Custom Albumentations transform to threshold mask to binary values
class BinaryMaskThreshold(A.ImageOnlyTransform):
    def __init__(self, threshold=0.5, always_apply=False, p=1.0):
        super(BinaryMaskThreshold, self).__init__(always_apply, p)
        self.threshold = threshold
    
    def apply(self, img, **params):
        return (img > self.threshold).astype(np.float32)
    
    def get_transform_init_args_names(self):
        return ("threshold",)


# Define the transform pipeline
train_transform = A.Compose([
    # 1. Spatial Transforms (Applied to both Image + Mask)
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),

    # 2. Pixel-level Transforms (Only applied to Image)
    A.RandomBrightnessContrast(p=0.2),

    # 3. Normalization (Image only, mask handled separately)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    
    # 4. Threshold mask to binary and convert to tensor
    A.Compose([
        BinaryMaskThreshold(threshold=0.5, p=1.0)
    ], is_check_shapes=False),
    ToTensorV2(),
], is_check_shapes=False)

# Validation transform (No random flips/distortion!)
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.Compose([
        BinaryMaskThreshold(threshold=0.5, p=1.0)
    ], is_check_shapes=False),
    ToTensorV2(),
], is_check_shapes=False)