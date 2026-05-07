import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    
    Dice Score = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice Score
    
    Works well for imbalanced datasets where background dominates.
    """
    def __init__(self, smooth=1.0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions with shape (batch_size, 1, height, width)
                  or (batch_size, height, width) after squeezing
            target: Ground truth masks with shape (batch_size, 1, height, width)
                    or (batch_size, height, width)
        
        Returns:
            Dice loss value
        """
        # Ensure predictions are in [0, 1] range
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1).float()
        
        # Calculate Dice score
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth + self.eps)
        
        # Return Dice loss
        return 1 - dice_score


class BCEWithDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    
    Total Loss = α * BCE_Loss + (1 - α) * Dice_Loss
    
    This combination works particularly well for imbalanced segmentation tasks
    where the foreground (lesion) is much smaller than the background.
    - BCE handles pixel-level classification errors
    - Dice focuses on overall shape and overlap
    
    Typical α value: 0.5 (equal weighting)
    """
    def __init__(self, alpha=0.5, smooth=1.0):
        """
        Args:
            alpha: Weight for BCE loss (1 - alpha is weight for Dice loss)
                   Default: 0.5 (equal weighting)
            smooth: Smoothing factor for Dice loss
        """
        super(BCEWithDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
    
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (logits) with shape (batch_size, 1, height, width)
            target: Ground truth masks with shape (batch_size, 1, height, width)
        
        Returns:
            Combined loss value
        """
        # BCE loss (input should be logits)
        bce = self.bce_loss(pred, target.float())
        
        # Dice loss
        dice = self.dice_loss(pred, target)
        
        # Combined loss
        combined_loss = self.alpha * bce + (1 - self.alpha) * dice
        
        return combined_loss
