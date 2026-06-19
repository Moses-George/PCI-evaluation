# import cv2
# import numpy as np
# from skimage.filters.ridges import sato
# from skimage.morphology import skeletonize
# from ultralytics import YOLO

# # CLASS_NAMES = ['Alligator crack', 'Longitudinal crack', 'Oblique crack',
# #                'Pothole', 'Repair', 'Transverse crack']

# class PavementAnalyzer:
#     def __init__(self, model_weights, pixels_per_mm=1.0):
#         self.model = YOLO(model_weights)
#         self.pixels_per_mm = pixels_per_mm

#     def predict_masks(self, image):
#         """Run YOLOv8-seg, return list of (class_id, mask)."""
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
#         """Apply Sato filter only inside the predicted mask region."""
#         tubeness = sato(original_gray / 255.0, sigmas=range(3,15,2), black_ridges=True)
#         tubeness = (tubeness > 0.2).astype(np.uint8) * 255
#         refined = cv2.bitwise_and(tubeness, mask)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#         refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
#         refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
#         return refined

#     @staticmethod
#     def skeletonize_mask(mask):
#         bin_img = mask // 255
#         sk = skeletonize(bin_img).astype(np.uint8) * 255
#         return sk

#     def compute_crack_metrics(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         masks = self.predict_masks(image)
#         metrics = []
#         for cls_id, raw_mask in masks:
#             if np.sum(raw_mask) < 100:
#                 continue
#             refined = self.refine_mask_sato(raw_mask, gray)
#             skeleton = self.skeletonize_mask(refined)
#             area_px = np.sum(refined // 255)
#             skel_px = np.sum(skeleton // 255)
#             if skel_px == 0:
#                 continue
#             avg_width_px = area_px / skel_px
#             avg_width_mm = avg_width_px / self.pixels_per_mm
#             length_m = (skel_px / self.pixels_per_mm) / 1000.0

#             # Determine severity based on class and width (for long/trans)
#             class_name = CLASS_NAMES[cls_id]
#             if class_name in ['Longitudinal crack', 'Transverse crack']:
#                 if avg_width_mm < 5:
#                     severity = 'Low'
#                 elif avg_width_mm < 20:
#                     severity = 'Medium'
#                 else:
#                     severity = 'High'
#             else:
#                 # For alligator, pothole, repair, oblique: severity based on density later
#                 severity = None   # will be set during PCI calc from area

#             metrics.append({
#                 'class': class_name,
#                 'class_id': cls_id,
#                 'width_mm': avg_width_mm,
#                 'severity': severity,
#                 'length_m': length_m,
#                 'area_px': area_px,
#                 'mask': refined   # optional, for debug
#             })
#         return metrics

# # ------------------- Deduct Value Curves (digitised from ASTM D6433) -------------------
# # These are simplified; in practice you would fit polynomials to the official curves.

# def get_deduct_value(distress_type, severity, density):
#     """
#     density = (area of distress / section area) * 100   (Equation 8)
#     Returns deduct value (0-100).
#     """
#     if distress_type in ['Longitudinal crack', 'Transverse crack', 'Oblique crack']:
#         if severity == 'Low':
#             return min(8, 0.15 * density)
#         elif severity == 'Medium':
#             return min(15, 0.30 * density)
#         else:  # High
#             return min(25, 0.50 * density)
#     elif distress_type == 'Alligator crack':
#         if severity == 'Low':
#             return min(20, 0.60 * density)
#         elif severity == 'Medium':
#             return min(40, 1.20 * density)
#         else:
#             return 60
#     elif distress_type == 'Pothole':
#         return min(45, 1.5 * density)   # potholes are severe
#     elif distress_type == 'Repair':
#         return min(12, 0.25 * density)
#     else:
#         return 0

# def compute_pci(crack_metrics_list, section_area_m2=100, pixels_per_mm=1.0):
#     """
#     crack_metrics_list: output from compute_crack_metrics()
#     Returns PCI (0-100) and rating.
#     """
#     if not crack_metrics_list:
#         return 100, 'Good'

#     deduct_values = []
#     for cm in crack_metrics_list:
#         # Convert area from pixels to m²
#         area_m2 = (cm['area_px'] / (pixels_per_mm ** 2)) / 1e6
#         density = (area_m2 / section_area_m2) * 100

#         # For classes that didn't have severity from width, assign based on density
#         if cm['severity'] is None:
#             if density < 10:
#                 sev = 'Low'
#             elif density < 30:
#                 sev = 'Medium'
#             else:
#                 sev = 'High'
#         else:
#             sev = cm['severity']

#         dv = get_deduct_value(cm['class'], sev, density)
#         deduct_values.append(dv)

#     # ASTM correction for multiple distresses (simplified)
#     deduct_values.sort(reverse=True)
#     q = len(deduct_values)
#     if q > 2:
#         cdv = sum(deduct_values) * (1 - 0.02 * q)
#         cdv = min(100, max(0, cdv))
#     else:
#         cdv = sum(deduct_values)

#     pci = max(0, 100 - cdv)

#     if pci >= 86:
#         rating = 'Good'
#     elif pci >= 71:
#         rating = 'Satisfactory'
#     elif pci >= 56:
#         rating = 'Fair'
#     elif pci >= 41:
#         rating = 'Poor'
#     elif pci >= 26:
#         rating = 'Very Poor'
#     elif pci >= 11:
#         rating = 'Serious'
#     else:
#         rating = 'Failed'
#     return pci, rating

# CLASSES = [
#     "Alligator crack",  # id 1
#     "Longitudinal crack",  # id 2
#     "Oblique crack",  # id 3
#     "Pothole",  # id 4
#     "Repair",  # id 5
#     "Transverse crack",  # id 6
# ]
