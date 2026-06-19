# import cv2
# import numpy as np
# from skimage.filters.ridges import sato
# from skimage.morphology import skeletonize
# from ultralytics import YOLO
# from scipy.spatial import ConvexHull

# CLASS_NAMES = ["Crack", "pothole"]  # Only two classes


# class PavementAnalyzer:
#     def __init__(self, model_weights, pixels_per_mm=1.0):
#         self.model = YOLO(model_weights)
#         self.pixels_per_mm = pixels_per_mm

#     def predict_masks(self, image):
#         results = self.model(image)[0]
#         if results.masks is None:
#             return []
#         masks = []
#         for seg, cls in zip(results.masks.data, results.boxes.cls):
#             cls_id = int(cls)
#             mask_np = (seg.cpu().numpy() > 0.5).astype(np.uint8) * 255
#             masks.append((cls_id, mask_np))
#         return masks

#     @staticmethod
#     def refine_mask_sato(mask, original_gray):
#         tubeness = sato(
#             original_gray / 255.0, sigmas=range(3, 15, 2), black_ridges=True
#         )
#         tubeness = (tubeness > 0.2).astype(np.uint8) * 255
#         refined = cv2.bitwise_and(tubeness, mask)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#         refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
#         refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
#         return refined

#     @staticmethod
#     def skeletonize_mask(mask):
#         bin_img = mask // 255
#         sk = skeletonize(bin_img).astype(np.uint8) * 255
#         return sk

#     @staticmethod
#     def compute_shape_features(mask_binary):
#         """Calculate area, perimeter, circularity, equivalent diameter, min Feret diameter."""
#         # Area
#         area = np.sum(mask_binary // 255)
#         # Perimeter
#         contours, _ = cv2.findContours(
#             mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#         if not contours:
#             return None
#         cnt = max(contours, key=cv2.contourArea)
#         perimeter = cv2.arcLength(cnt, True)
#         # Circularity
#         circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
#         # Equivalent diameter (diameter of circle with same area)
#         equiv_diameter = 2 * np.sqrt(area / np.pi)
#         # Minimum Feret diameter (using minAreaRect)
#         rect = cv2.minAreaRect(cnt)
#         w, h = rect[1]
#         min_feret = min(w, h) if w > 0 and h > 0 else 0
#         return {
#             "area_px": area,
#             "perimeter_px": perimeter,
#             "circularity": circularity,
#             "equiv_diameter_px": equiv_diameter,
#             "min_feret_px": min_feret,
#         }

#     def detect_alligator(self, mask_binary, skeleton, branch_threshold=3):
#         """
#         Heuristic to decide if a crack is alligator-like.
#         A crack is alligator if:
#         - The skeleton has many branches (endpoints) relative to length.
#         - The area is spread out (low compactness).
#         """
#         # Count endpoints and branch points on skeleton
#         skel = skeleton // 255
#         if np.sum(skel) < 10:
#             return False
#         # Use OpenCV to find skeleton endpoints and branches
#         # (simplified: count number of neighbours)
#         from skimage.morphology import skeletonize
#         from skimage.filters import convolve

#         kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
#         neighbours = convolve(skel, kernel, mode="constant")
#         endpoints = np.sum((neighbours == 1) & (skel == 1))
#         branches = np.sum((neighbours >= 3) & (skel == 1))
#         total_skeleton_pixels = np.sum(skel)
#         # Ratio of branch points to skeleton length
#         branch_ratio = branches / (total_skeleton_pixels + 1e-6)
#         # Also area / (convex hull area) -> low ratio means spread out
#         if np.sum(mask_binary) < 100:
#             return False
#         hull = ConvexHull(np.column_stack(np.where(mask_binary > 0)))
#         hull_area = hull.volume  # for 2D, this is area
#         area_ratio = np.sum(mask_binary // 255) / (hull_area + 1e-6)
#         # Alligator usually has many branches and low area ratio
#         return (branch_ratio > 0.01) and (area_ratio < 0.4)

#     def compute_crack_metrics(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         masks = self.predict_masks(image)
#         metrics = []
#         for cls_id, raw_mask in masks:
#             if np.sum(raw_mask) < 100:
#                 continue
#             # Refine with Sato
#             refined = self.refine_mask_sato(raw_mask, gray)
#             skeleton = self.skeletonize_mask(refined)
#             # Basic metrics
#             area_px = np.sum(refined // 255)
#             skel_px = np.sum(skeleton // 255)
#             if skel_px == 0:
#                 continue
#             avg_width_px = area_px / skel_px
#             avg_width_mm = avg_width_px / self.pixels_per_mm
#             length_m = (skel_px / self.pixels_per_mm) / 1000.0

#             # Shape features (for potholes and alligator detection)
#             shape = self.compute_shape_features(refined)
#             if shape is None:
#                 continue

#             class_name = CLASS_NAMES[cls_id]

#             # --- Decide if a 'Crack' is alligator-like ---
#             is_alligator = False
#             if class_name == "Crack":
#                 is_alligator = self.detect_alligator(refined, skeleton)
#                 # Override class name for PCI deduction if alligator
#                 distress_type = (
#                     "Alligator crack" if is_alligator else "Longitudinal crack"
#                 )
#             else:
#                 distress_type = "Pothole"

#             # --- Severity determination ---
#             if distress_type == "Longitudinal crack":
#                 # Based on width
#                 if avg_width_mm < 5:
#                     severity = "Low"
#                 elif avg_width_mm < 20:
#                     severity = "Medium"
#                 else:
#                     severity = "High"
#             elif distress_type == "Alligator crack":
#                 # Severity based on density and width; we'll set later from density
#                 severity = None  # will be assigned later
#             elif distress_type == "Pothole":
#                 # For pothole, severity is usually based on depth and area.
#                 # Since depth is unknown, we use area/density as proxy.
#                 # We'll set severity based on density later.
#                 severity = None

#             metrics.append(
#                 {
#                     "class": distress_type,  # actual distress type for PCI
#                     "class_id": cls_id,
#                     "width_mm": avg_width_mm,
#                     "severity": severity,
#                     "length_m": length_m,
#                     "area_px": area_px,
#                     "perimeter_px": shape["perimeter_px"],
#                     "equiv_diameter_px": shape["equiv_diameter_px"],
#                     "min_feret_px": shape["min_feret_px"],
#                     "circularity": shape["circularity"],
#                     "is_alligator": is_alligator,
#                     "mask": refined,
#                 }
#             )
#         return metrics


import cv2
import numpy as np
from skimage.filters.ridges import sato
from skimage.morphology import skeletonize
from ultralytics import YOLO
from scipy.spatial import ConvexHull
from skimage.filters import convolve

CLASS_NAMES = ["Crack", "pothole"]  # Match your model's class order


class PavementAnalyzer:
    def __init__(self, model_weights, pixels_per_mm=1.0):
        self.model = YOLO(model_weights)
        self.pixels_per_mm = pixels_per_mm
        self.annotated_image = None

    def predict_masks(self, image):
        """Run YOLOv8-seg and return masks resized to original image size."""
        results = self.model(image)[0]
        annotated_image = results.plot()
        self.annotated_image = annotated_image

        if results.masks is None:
            return []

        h, w = image.shape[:2]  # original image size
        masks = []
        for seg, cls in zip(results.masks.data, results.boxes.cls):
            cls_id = int(cls)
            # Mask from model is in input space (e.g., 448x640)
            mask_np = (seg.cpu().numpy() > 0.5).astype(np.uint8) * 255
            # Resize to original image dimensions (using nearest neighbour to keep binary)
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append((cls_id, mask_resized))
        return masks

    @staticmethod
    def refine_mask_sato(mask, original_gray):
        """Apply Sato filter only inside the mask region (same size)."""
        tubeness = sato(
            original_gray / 255.0, sigmas=range(3, 15, 2), black_ridges=True
        )
        tubeness = (tubeness > 0.2).astype(np.uint8) * 255
        # Ensure both are same dtype (uint8) and size
        mask = mask.astype(np.uint8)
        refined = cv2.bitwise_and(tubeness, mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        return refined

    @staticmethod
    def skeletonize_mask(mask):
        bin_img = mask // 255
        sk = skeletonize(bin_img).astype(np.uint8) * 255
        return sk

    @staticmethod
    def compute_shape_features(mask_binary):
        """Calculate area, perimeter, circularity, equivalent diameter, min Feret diameter."""
        area = np.sum(mask_binary // 255)
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)
        equiv_diameter = 2 * np.sqrt(area / np.pi)
        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        min_feret = min(w, h) if w > 0 and h > 0 else 0
        return {
            "area_px": area,
            "perimeter_px": perimeter,
            "circularity": circularity,
            "equiv_diameter_px": equiv_diameter,
            "min_feret_px": min_feret,
        }

    def detect_alligator(self, mask_binary, skeleton, branch_threshold=3):
        """Heuristic to decide if a crack is alligator-like."""
        skel = skeleton // 255
        if np.sum(skel) < 10:
            return False
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbours = convolve(skel, kernel, mode="constant")
        branches = np.sum((neighbours >= 3) & (skel == 1))
        total_skeleton_pixels = np.sum(skel)
        branch_ratio = branches / (total_skeleton_pixels + 1e-6)
        if np.sum(mask_binary) < 100:
            return False
        hull = ConvexHull(np.column_stack(np.where(mask_binary > 0)))
        hull_area = hull.volume
        area_ratio = np.sum(mask_binary // 255) / (hull_area + 1e-6)
        return (branch_ratio > 0.01) and (area_ratio < 0.4)

    def compute_crack_metrics(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masks = self.predict_masks(image)
        metrics = []
        for cls_id, raw_mask in masks:
            if np.sum(raw_mask) < 100:
                continue
            # Refine with Sato
            refined = self.refine_mask_sato(raw_mask, gray)
            skeleton = self.skeletonize_mask(refined)
            # Basic metrics
            area_px = np.sum(refined // 255)
            skel_px = np.sum(skeleton // 255)
            if skel_px == 0:
                continue
            avg_width_px = area_px / skel_px
            avg_width_mm = avg_width_px / self.pixels_per_mm
            length_m = (skel_px / self.pixels_per_mm) / 1000.0

            shape = self.compute_shape_features(refined)
            if shape is None:
                continue

            class_name = CLASS_NAMES[cls_id]  # use global list

            # Decide if a 'Crack' is alligator-like
            is_alligator = False
            if class_name == "Crack":
                is_alligator = self.detect_alligator(refined, skeleton)
                distress_type = (
                    "Alligator crack" if is_alligator else "Longitudinal crack"
                )
            else:
                distress_type = "Pothole"

            # Severity (will be refined later using density)
            if distress_type == "Longitudinal crack":
                if avg_width_mm < 5:
                    severity = "Low"
                elif avg_width_mm < 20:
                    severity = "Medium"
                else:
                    severity = "High"
            else:
                severity = None  # set later based on density

            metrics.append(
                {
                    "class": distress_type,
                    "class_id": cls_id,
                    "width_mm": avg_width_mm,
                    "severity": severity,
                    "length_m": length_m,
                    "area_px": area_px,
                    "perimeter_px": shape["perimeter_px"],
                    "equiv_diameter_px": shape["equiv_diameter_px"],
                    "min_feret_px": shape["min_feret_px"],
                    "circularity": shape["circularity"],
                    "is_alligator": is_alligator,
                    "mask": refined,
                }
            )
        return metrics
