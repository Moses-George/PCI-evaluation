# import matplotlib.pyplot as plt


# def mask_to_rgb(mask, class_mapping):
#     """
#     mask: (H,W) class indices
#     class_mapping: {(R,G,B): class_index}
#     """

#     index_to_rgb = {v: k for k, v in class_mapping.items()}

#     rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

#     for idx, colour in index_to_rgb.items():
#         rgb[mask == idx] = colour

#     return rgb


# @torch.no_grad()
# def plot_test_predictions(model, loader, device, class_mapping, num_images=5):
#     model.eval()

#     fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

#     shown = 0

#     for imgs, masks in loader:

#         imgs = imgs.to(device)
#         masks = masks.to(device)

#         logits = model(imgs)
#         preds = torch.argmax(logits, dim=1)

#         batch_size = imgs.shape[0]

#         for i in range(batch_size):

#             if shown >= num_images:
#                 break

#             ########################
#             # Original image
#             ########################
#             img = imgs[i].cpu().permute(1, 2, 0).numpy()

#             img = np.clip(img, 0, 1)

#             ########################
#             # Ground truth
#             ########################
#             gt = masks[i].cpu().numpy()

#             ########################
#             # Prediction
#             ########################
#             pred = preds[i].cpu().numpy()

#             gt_rgb = mask_to_rgb(gt, class_mapping)

#             pred_rgb = mask_to_rgb(pred, class_mapping)

#             axes[shown, 0].imshow(img)
#             axes[shown, 0].set_title("Original")
#             axes[shown, 0].axis("off")

#             axes[shown, 1].imshow(gt_rgb)
#             axes[shown, 1].set_title("Ground Truth")
#             axes[shown, 1].axis("off")

#             axes[shown, 2].imshow(pred_rgb)
#             axes[shown, 2].set_title("Prediction")
#             axes[shown, 2].axis("off")

#             quant = quantify_all_classes(pred, class_names)
#             axes[shown, 2].set_title(f"Prediction\n{quant}")

#             shown += 1

#         if shown >= num_images:
#             break

#     plt.tight_layout()
#     plt.show()


# plot_test_predictions(
#     model=model,
#     loader=test_loader,
#     device=device,
#     class_mapping=class_mapping,
#     num_images=6,
# )


if __name__ == "__main__":
    # 1. Load data
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    # 2. Choose model
    if cfg.model_name == "UNet":
        model = UNet(img_ch=3, output_ch=cfg.num_classes)
    elif cfg.model_name == "UNetSkip":
        model = UNETSKIP(in_channels=3, out_channels=1)
    elif cfg.model_name == "AttUNet":
        model = AttUNet(img_ch=3, output_ch=cfg.num_classes)
    elif cfg.model_name == "R2AttUNet":
        model = R2AttUNet(img_ch=3, output_ch=cfg.num_classes, t=cfg.t)
    elif cfg.model_name == "R2UNet":
        model = R2UNet(img_ch=3, output_ch=cfg.num_classes, t=cfg.t)
    else:
        raise ValueError(f"Unknown model: {cfg.model_name}")

    model = model.to(device)

    print("Device:", device)
    print("Model on GPU:", next(model.parameters()).is_cuda)
    print("CUDA available:", torch.cuda.is_available())
    print(len(list(model.parameters())))

    # 3. Train (or load pretrained weights)
    best_mIoU = train_model(model, train_loader, val_loader, cfg, cfg.model_name)
    print(f"Best validation mIoU: {best_mIoU:.4f}")

    # 4. Load best model and evaluate on test set
    checkpoint_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_best.pth")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    # for img, mask in test_loader:
    #     img, mask = img.to(device), mask.to(device)
    #     logits = model(img)
    #     pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    #     mask_np = mask.cpu().numpy()[0]
    #     # Example class mapping (index -> name)
    #     class_names = {
    #         0: "background",
    #         1: "longitudinal/transverse crack",
    #         2: "alligator cracking",
    #         3: "pothole",
    #         4: "pavement marking",
    #         5: "drains",
    #         6: "patching",
    #     }
    #     quant_results = quantify_all_classes(pred, class_names)
    #     print("Quantification results:", quant_results)
    #     break

    def mask_to_rgb(mask, class_mapping):
        """
        mask: (H,W) class indices
        class_mapping: {(R,G,B): class_index}
        """

        index_to_rgb = {k: v for k, v in class_mapping.items()}

        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

        for idx, colour in index_to_rgb.items():
            rgb[mask == idx] = colour

        return rgb

    @torch.no_grad()
    def plot_test_predictions(model, loader, device, class_mapping, num_images=5):
        model.eval()

        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

        shown = 0

        for imgs, masks in loader:

            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

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
            quant_results = quantify_all_classes(pred.cpu().numpy()[0], class_names)
            print("Quantification results:", quant_results)

            batch_size = imgs.shape[0]

            for i in range(batch_size):

                if shown >= num_images:
                    break

                ########################
                # Original image
                ########################
                img = imgs[i].cpu().permute(1, 2, 0).numpy()

                img = np.clip(img, 0, 1)

                ########################
                # Ground truth
                ########################
                gt = masks[i].cpu().numpy()

                ########################
                # Prediction
                ########################
                pred = preds[i].cpu().numpy()

                gt_rgb = mask_to_rgb(gt, class_mapping)

                pred_rgb = mask_to_rgb(pred, class_mapping)

                axes[shown, 0].imshow(img)
                axes[shown, 0].set_title("Original")
                axes[shown, 0].axis("off")

                axes[shown, 1].imshow(gt_rgb)
                axes[shown, 1].set_title("Ground Truth")
                axes[shown, 1].axis("off")

                axes[shown, 2].imshow(pred_rgb)
                axes[shown, 2].set_title("Prediction")
                axes[shown, 2].axis("off")

                quant = quantify_all_classes(pred, class_names)
                axes[shown, 2].set_title(f"Prediction\n{quant}")

                shown += 1

            if shown >= num_images:
                break

        plt.tight_layout()
        plt.show()

    plot_test_predictions(
        model=model,
        loader=test_loader,
        device=device,
        class_mapping=cfg.class_color_mapping,
        num_images=6,
    )
