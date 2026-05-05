import matplotlib.pyplot as plt
import torch
import numpy as np

def show_samples(dataset, n=6):
    fig, axes = plt.subplots(n, 2, figsize=(6, n*2))
    for i in range(n):
        image, mask = dataset[i]
        
        # Handle both tensor and numpy array inputs
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(mask, cmap="gray")
    plt.tight_layout()
    plt.show()