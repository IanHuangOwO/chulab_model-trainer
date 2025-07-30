import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

def dice_bce_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)

    dice_loss = 1 - dice
    bce_loss = F.binary_cross_entropy(pred, target)
    
    return 0.7 * dice_loss + 0.3 * bce_loss