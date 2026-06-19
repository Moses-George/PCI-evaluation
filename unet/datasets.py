import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch

class MulticlassCrackDataset(Dataset):
    """Generic multiclass dataset for road damage segmentation.
    Expects folder structure:
        data_root/dataset_name/images/xxx.png
        data_root/dataset_name/masks/xxx.png   (palette image or single-channel index)
    """

    def __init__(
        self,
        root_dir,
        dataset_name,
        img_size=(256, 256),
        class_mapping=None,
        transform=None,
    ):
        self.root = os.path.join(root_dir, dataset_name)
        self.img_size = img_size
        self.class_mapping = class_mapping
        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(os.path.join(self.root, "images"))
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.masks = sorted(
            [
                f
                for f in os.listdir(os.path.join(self.root, "masks"))
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        assert len(self.images) == len(self.masks), "Mismatch between images and masks"

    def __len__(self):
        return len(self.images)

    def _convert_palette_to_index(self, mask_img):
        """Convert a palette image (RGB) to a single-channel index mask.
        Example mapping for PavementDamagesG-7:
            (0,0,0) -> 0 (background)
            (250,50,83) -> 1 (longitudinal/transverse crack)
            (61,245,61) -> 2 (alligator cracking)
            (42,125,209) -> 3 (pothole)
            (245,147,49) -> 4 (pavement marking)
            (115,51,128) -> 5 (drains)
            (250,250,55) -> 6 (patching)
        """
        mask_np = np.array(mask_img)
        if mask_np.ndim == 3 and mask_np.shape[2] == 3:
            index_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
            if self.class_mapping:
                for idx, rgb in self.class_mapping.items():
                    if isinstance(rgb, tuple) and len(rgb) == 3:
                        match = (mask_np == rgb).all(axis=2)
                        index_mask[match] = idx
            else:
                # Fallback: use unique colours as indices (not robust)
                unique_colors = np.unique(mask_np.reshape(-1, mask_np.shape[2]), axis=0)
                for i, color in enumerate(unique_colors):
                    mask = (mask_np == color).all(axis=2)
                    index_mask[mask] = i
            return index_mask
        else:
            # Already a single-channel mask
            return mask_np

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Convert mask to index map
        index_mask = self._convert_palette_to_index(mask)

        # Resize
        image = image.resize(self.img_size, Image.BILINEAR)
        index_mask = Image.fromarray(index_mask).resize(self.img_size, Image.NEAREST)

        # To tensor and normalize
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # (C,H,W)
        mask = np.array(index_mask, dtype=np.int64)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return torch.from_numpy(image), torch.from_numpy(mask)


def get_dataloaders(cfg):
    # Example class mapping for PavementDamagesG-7 (RGB -> class index)
    class_mapping = {
        0: (0, 0, 0),  # background
        1: (250, 50, 83),  # longitudinal/transverse crack
        2: (61, 245, 61),  # alligator cracking
        3: (42, 125, 209),  # pothole
        4: (245, 147, 49),  # pavement marking
        5: (115, 51, 128),  # drains
        6: (250, 250, 55),  # patching
    }
    full_dataset = MulticlassCrackDataset(
        cfg.data_root,
        cfg.dataset_name,
        img_size=cfg.img_size,
        class_mapping=class_mapping,
    )
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
