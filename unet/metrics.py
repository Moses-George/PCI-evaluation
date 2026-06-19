import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

def compute_multiclass_metrics(pred, target, num_classes):
    """
    Args:
        pred: logits (B, C, H, W)
        target: indices (B, H, W)
    Returns:
        dict with metrics
    """
    pred = torch.argmax(pred, dim=1).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    valid = target < num_classes  # ignore invalid labels
    pred = pred[valid]
    target = target[valid]

    # Per-class IoU
    class_iou = []
    for c in range(num_classes):
        tp = np.sum((pred == c) & (target == c))
        fp = np.sum((pred == c) & (target != c))
        fn = np.sum((pred != c) & (target == c))
        iou = tp / (tp + fp + fn + 1e-8)
        class_iou.append(iou)
    mIoU = np.mean(class_iou)

    # Overall accuracy (micro)
    accuracy = accuracy_score(target, pred)

    # Macro precision, recall, F1
    precision = precision_score(target, pred, average="macro", zero_division=0)
    recall = recall_score(target, pred, average="macro", zero_division=0)
    f1 = f1_score(target, pred, average="macro", zero_division=0)

    return {
        "mIoU": mIoU,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_iou": class_iou,
    }
