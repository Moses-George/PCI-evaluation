import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W) logits
        # target: (B, H, W) indices
        pred = F.softmax(pred, dim=1)
        target_one_hot = (
            F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        )
        dims = (2, 3)
        intersection = (pred * target_one_hot).sum(dim=dims)
        union = pred.sum(dim=dims) + target_one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_loss(loss_type, focal_gamma=2.0):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "focal":
        return FocalLoss(gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")
