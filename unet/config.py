import os
import torch


# Configuration class for all hyperparameters
class Config:
    # Data
    data_root = "/content/datasets"
    dataset_name = "PavementDamagesG-7"  # Options: PavementDamagesG-7, RoadExtraction, Crack500, DeepCrack
    img_size = (256, 256)
    num_classes = 7  # For PavementDamagesG-7 (6 damages + background)
    batch_size = 32
    num_workers = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    epochs = 5
    lr = 0.0001
    dropout = 0.3
    t = 3
    beta1 = 0.5
    beta2 = 0.999

    # Loss
    loss_type = "ce"  # Options: "ce", "dice", "focal"
    focal_gamma = 2.0

    # Model selection
    # model_name = (
    #     "UNet"  # Options: "UNet", "AttentionUNet", "RAUNet", "TransUNet", "SwinUnet"
    # )

    # Model selection
    model_name = (
        "UNet"  # Options: "UNet", "R2UNet", "R2AttUNet", "AttUNet"
    )

    # Ensemble
    ensemble = False
    ensemble_method = "weighted_avg"  # "stochastic" or "weighted_avg"

    # Quantification
    skeleton_method = "medial_axis"

    # Paths
    save_dir = "/content/output/checkpoints"
    log_dir = "/content/output/logs"
    result_dir = "/content/output/results"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.result_dir, exist_ok=True)
print(f"Using device: {cfg.device}")
