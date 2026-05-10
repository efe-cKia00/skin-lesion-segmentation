import torch
import torch.nn.functional as F
import numpy as np


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


def precision_score(pred, target, threshold=0.5, eps=1e-7):
    """
    Compute Precision for binary segmentation.
    
    Precision = TP / (TP + FP)
    
    Args:
        pred: Model predictions (logits) with shape (batch_size, 1, height, width)
        target: Ground truth masks with shape (batch_size, 1, height, width)
        threshold: Threshold for converting probabilities to binary predictions
        eps: Small epsilon value for numerical stability
    
    Returns:
        Precision as a float value
    """
    # Convert logits to probabilities
    pred_prob = torch.sigmoid(pred)
    
    # Binarize predictions
    pred_binary = (pred_prob > threshold).float()
    target_binary = (target > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate TP and FP
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    
    # Precision
    precision = (tp + eps) / (tp + fp + eps)
    return precision.item()


def recall_score(pred, target, threshold=0.5, eps=1e-7):
    """
    Compute Recall (Sensitivity) for binary segmentation.
    
    Recall = TP / (TP + FN)
    
    Args:
        pred: Model predictions (logits) with shape (batch_size, 1, height, width)
        target: Ground truth masks with shape (batch_size, 1, height, width)
        threshold: Threshold for converting probabilities to binary predictions
        eps: Small epsilon value for numerical stability
    
    Returns:
        Recall as a float value
    """
    # Convert logits to probabilities
    pred_prob = torch.sigmoid(pred)
    
    # Binarize predictions
    pred_binary = (pred_prob > threshold).float()
    target_binary = (target > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate TP and FN
    tp = (pred_flat * target_flat).sum()
    fn = ((1 - pred_flat) * target_flat).sum()
    
    # Recall
    recall = (tp + eps) / (tp + fn + eps)
    return recall.item()


def f1_score(pred, target, threshold=0.5, eps=1e-7):
    """
    Compute F1-Score for binary segmentation.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        pred: Model predictions (logits) with shape (batch_size, 1, height, width)
        target: Ground truth masks with shape (batch_size, 1, height, width)
        threshold: Threshold for converting probabilities to binary predictions
        eps: Small epsilon value for numerical stability
    
    Returns:
        F1-Score as a float value
    """
    prec = precision_score(pred, target, threshold, eps)
    rec = recall_score(pred, target, threshold, eps)
    
    f1 = 2 * (prec * rec) / (prec + rec + eps)
    return f1


def evaluate_model(model, data_loader, device, criterion=None, threshold=0.5):
    """
    Comprehensive evaluation of model on a dataset.
    
    Computes multiple metrics: Dice, IoU, Precision, Recall, F1-Score, and optionally Loss.
    
    Args:
        model: Trained model (should be in eval mode)
        data_loader: DataLoader for evaluation
        device: torch device (cuda or cpu)
        criterion: Loss function (optional)
        threshold: Threshold for binary predictions
    
    Returns:
        Dictionary containing all computed metrics
    """
    model.eval()
    
    metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'loss': [] if criterion else None
    }
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Ensure masks have correct shape
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            # Forward pass
            outputs = model(images)
            
            # Compute metrics
            metrics['dice'].append(dice_score(outputs, masks))
            metrics['iou'].append(iou_score(outputs, masks))
            metrics['precision'].append(precision_score(outputs, masks, threshold))
            metrics['recall'].append(recall_score(outputs, masks, threshold))
            metrics['f1'].append(f1_score(outputs, masks, threshold))
            
            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, masks)
                metrics['loss'].append(loss.item())
    
    # Average metrics
    results = {
        'dice': np.mean(metrics['dice']),
        'iou': np.mean(metrics['iou']),
        'precision': np.mean(metrics['precision']),
        'recall': np.mean(metrics['recall']),
        'f1': np.mean(metrics['f1']),
    }
    
    if metrics['loss'] is not None:
        results['loss'] = np.mean(metrics['loss'])
    
    return results
