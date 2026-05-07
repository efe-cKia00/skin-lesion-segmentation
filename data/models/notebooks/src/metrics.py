import torch
import torch.nn.functional as F


def dice_score(pred, target, smooth=1.0, eps=1e-7):
    """
    Compute Dice Score for binary segmentation.
    
    Dice Score = 2 * |X ∩ Y| / (|X| + |Y|)
    
    Range: [0, 1], where 1 is perfect segmentation
    
    Args:
        pred: Model predictions with shape (batch_size, 1, height, width) or logits
        target: Ground truth masks with shape (batch_size, 1, height, width)
        smooth: Smoothing factor to prevent division by zero
        eps: Small epsilon value for numerical stability
    
    Returns:
        Dice score as a float value
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1).float()
    
    # Calculate Dice score
    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth + eps)
    
    return dice.item()


def iou_score(pred, target, smooth=1.0, eps=1e-7):
    """
    Compute Intersection over Union (IoU / Jaccard Index) for binary segmentation.
    
    IoU = |X ∩ Y| / |X ∪ Y|
    
    Range: [0, 1], where 1 is perfect segmentation
    
    Args:
        pred: Model predictions with shape (batch_size, 1, height, width) or logits
        target: Ground truth masks with shape (batch_size, 1, height, width)
        smooth: Smoothing factor to prevent division by zero
        eps: Small epsilon value for numerical stability
    
    Returns:
        IoU score as a float value
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)
    
    # Flatten tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1).float()
    
    # Calculate IoU
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth + eps)
    
    return iou.item()
