import os
import torch
from datasets import get_dataloaders
from config import cfg
from training import train_model, evaluate
from quantification import quantify_all_classes
from loss import get_loss
from networks import UNet, AttUNet, R2AttUNet, R2UNet

if __name__ == "__main__":
    # 1. Load data
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # # 2. Choose model
    # if cfg.model_name == "UNet":
    #     # model = UNet(in_ch=3, num_classes=cfg.num_classes, dropout=cfg.dropout)
    #     model = UNET(in_channels=3, out_channels=cfg.num_classes)
    # elif cfg.model_name == "AttentionUNet":
    #     model = AttentionUNet(in_ch=3, num_classes=cfg.num_classes, dropout=cfg.dropout)
    # elif cfg.model_name == "RAUNet":
    #     model = RAUNet(in_ch=3, num_classes=cfg.num_classes, dropout=cfg.dropout)
    # elif cfg.model_name == "TransUNet":
    #     model = TransUNet(
    #         in_ch=3,
    #         num_classes=cfg.num_classes,
    #         img_size=cfg.img_size[0],
    #         dropout=cfg.dropout,
    #     )
    # elif cfg.model_name == "SwinUnet":
    #     cfg.batch_size = 8  # as per Swin-Unet memory requirements
    #     model = SwinUnet(in_ch=3, num_classes=cfg.num_classes, dropout=cfg.dropout)
    # else:
    #     raise ValueError(f"Unknown model: {cfg.model_name}")

    # 2. Choose model
    if cfg.model_name == "UNet":
        model = UNet(img_ch=3, output_ch=cfg.num_classes)
    elif cfg.model_name == "AttUNet":
        model = AttUNet(img_ch=3, output_ch=cfg.num_classes)
    elif cfg.model_name == "R2AttUNet":
        model = R2AttUNet(img_ch=3, output_ch=cfg.num_classes, t=cfg.t)
    elif cfg.model_name == "R2UNet":
        model = R2UNet(img_ch=3, output_ch=cfg.num_classes, t=cfg.t)
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")

    model = model.to(cfg.device)

    print("Device:", cfg.device)
    print("Model on GPU:", next(model.parameters()).is_cuda)
    print("CUDA available:", torch.cuda.is_available())
    print(len(list(model.parameters())))

    # 3. Train (or load pretrained weights)
    best_mIoU = train_model(model, train_loader, val_loader, cfg, cfg.model_name)
    print(f"Best validation mIoU: {best_mIoU:.4f}")

    # 4. Load best model and evaluate on test set
    model.load_state_dict(
        torch.load(os.path.join(cfg.save_dir, f"{cfg.model_name}_best.pth"))
    )
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        get_loss(cfg.loss_type, cfg.focal_gamma),
        cfg.device,
        cfg.num_classes,
    )
    print("Test metrics:", test_metrics)

    # 5. Quantification on a test sample
    # (Use a sample from test_loader)
    for img, mask in test_loader:
        img, mask = img.to(cfg.device), mask.to(cfg.device)
        logits = model(img)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
        mask_np = mask.cpu().numpy()[0]
        # Example class mapping (index -> name)
        class_names = {
            0: "background",
            1: "longitudinal/transverse crack",
            2: "alligator cracking",
            3: "pothole",
            4: "pavement marking",
            5: "drains",
            6: "patching",
        }
        quant_results = quantify_all_classes(pred, class_names)
        print("Quantification results:", quant_results)
        break
