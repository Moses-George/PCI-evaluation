from skimage.morphology import medial_axis, skeletonize
from skimage.measure import regionprops
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import label
import cv2
from config import cfg


def skeletonize_crack(mask, method="medial_axis"):
    """mask: binary numpy array (H,W) with crack pixels = 1"""
    if method == "medial_axis":
        skeleton = medial_axis(mask).astype(np.uint8)
    elif method == "zhang_suen":
        skeleton = skeletonize(mask).astype(np.uint8)
    else:
        raise ValueError
    return skeleton


def compute_crack_length(skeleton):
    points = np.argwhere(skeleton > 0)
    if len(points) < 2:
        return 0.0
    # Simple nearest-neighbour ordering (use graph for complex cracks)
    visited = [False] * len(points)
    current = 0
    visited[current] = True
    total_length = 0.0
    for _ in range(len(points) - 1):
        dists = cdist([points[current]], points, "euclidean")[0]
        dists[visited] = np.inf
        next_idx = np.argmin(dists)
        total_length += dists[next_idx]
        visited[next_idx] = True
        current = next_idx
    return total_length


def compute_crack_width(mask, skeleton):
    edges = cv2.Canny((mask * 255).astype(np.uint8), 100, 200)
    skeleton_points = np.argwhere(skeleton > 0)
    widths = []
    for y, x in skeleton_points:
        dist = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 3)
        width = 2 * dist[y, x]
        widths.append(width)
    if len(widths) == 0:
        return 0, 0
    return np.mean(widths), np.max(widths)


def quantify_pothole(mask):
    """Compute area, perimeter, circularity, approximate depth for a pothole."""
    # Ensure mask is binary
    mask_bin = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0, 0
    # Take largest contour as the pothole
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    # Circularity: 4π * area / perimeter^2 (1 = perfect circle)
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-8)
    # Approximate depth using shadow/height estimation from intensity variation
    # This is a placeholder; real depth requires depth maps or perspective geometry
    depth_estimate = 0.0
    return area, perimeter, circularity, depth_estimate


def quantify_all_classes(multiclass_mask, class_mapping):
    """Run quantification for each class of interest."""
    results = {}
    # Crack classes: longitudinal/transverse crack, alligator cracking
    crack_classes = [1, 2]  # adjust based on your class indices
    pothole_class = 3
    for cls, name in class_mapping.items():
        binary_mask = (multiclass_mask == cls).astype(np.uint8)
        if cls in crack_classes:
            skeleton = skeletonize_crack(binary_mask, cfg.skeleton_method)
            length = compute_crack_length(skeleton)
            mean_width, max_width = compute_crack_width(binary_mask, skeleton)
            results[name] = {
                "length (px)": length,
                "mean_width (px)": mean_width,
                "max_width (px)": max_width,
            }
        elif cls == pothole_class:
            area, perimeter, circularity, depth = quantify_pothole(binary_mask)
            results[name] = {
                "area (px)": area,
                "perimeter (px)": perimeter,
                "circularity": circularity,
                "depth (estimate)": depth,
            }
        else:
            # Other classes (marking, drains, patching) – only area
            results[name] = {"area (px)": binary_mask.sum()}
    return results
