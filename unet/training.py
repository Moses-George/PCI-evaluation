import os
import time
import torch
import tqdm
from torch.optim import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
from loss import get_loss
from metrics import compute_multiclass_metrics
from config import cfg



def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Training"):
        t0 = time.time()
        imgs, masks = imgs.to(device), masks.to(device)
        t1 = time.time()
        optimizer.zero_grad()
        t2 = time.time()
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(imgs)
            loss = criterion(outputs, masks)

        t3 = time.time()
        # Backward with scaler
        scaler.scale(loss).backward()
        t4 = time.time()
        scaler.step(optimizer)
        scaler.update()
        t5 = time.time()
        print(
            f"to(device): {t1-t0:.3f}s, zero_grad: {t2-t1:.3f}s, forward: {t3-t2:.3f}s, backward: {t4-t3:.3f}s, step: {t5-t4:.3f}s"
        )
        # outputs = model(imgs)
        # loss = criterion(outputs, masks)
        # loss.backward()
        # optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_metrics = []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        total_loss += loss.item()
        metrics = compute_multiclass_metrics(outputs, masks, num_classes)
        all_metrics.append(metrics)
    # Average metrics over batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key == "class_iou":
            avg_metrics[key] = np.mean([m[key] for m in all_metrics], axis=0)
        else:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    return total_loss / len(loader), avg_metrics


def train_model(model, train_loader, val_loader, cfg, model_name):
    # model.to(device)
    criterion = get_loss(cfg.loss_type, cfg.focal_gamma)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    print("Model param device:", next(model.parameters()).device)
    print("Optimizer param device:", optimizer.param_groups[0]["params"][0].device)
    best_mIoU = 0.0
    train_losses, val_losses = [], []
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.device, scaler
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, cfg.device, cfg.num_classes
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {val_metrics['mIoU']:.4f}"
        )
        if val_metrics["mIoU"] > best_mIoU:
            best_mIoU = val_metrics["mIoU"]
            torch.save(
                model.state_dict(), os.path.join(cfg.save_dir, f"{model_name}_best.pth")
            )
    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{model_name} Training Loss")
    plt.savefig(os.path.join(cfg.log_dir, f"{model_name}_loss.png"))
    plt.close()
    return best_mIoU
