"""Leaf segmentation and the 38 handcrafted measurements.

This is a faithful copy of `segment_leaf` and `extract_all_features` from
full_pipeline/08_OLID_Full_Feature_Extraction.ipynb. The trained models expect
exactly these measurements, so any change here must also be made in NB08 and the
models retrained.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

SHAPE = ['area', 'perimeter', 'aspect_ratio', 'circularity', 'extent', 'solidity',
         'convex_area', 'major_axis_length', 'minor_axis_length', 'eccentricity']
RGB = ['mean_R', 'std_R', 'mean_G', 'std_G', 'mean_B', 'std_B']
HSV = ['mean_H', 'std_H', 'mean_S', 'std_S', 'mean_V', 'std_V']
LAB = ['mean_L', 'std_L', 'mean_a', 'std_a', 'mean_b', 'std_b']
DERIVED = ['green_ratio', 'yellow_ratio']
TEXTURE = ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation']
SPATIAL = ['core_mean_H', 'core_yellow_ratio', 'edge_mean_H', 'edge_yellow_ratio']

FEATURES = SHAPE + RGB + HSV + LAB + DERIVED + TEXTURE + SPATIAL  # 38, same order as the CSV
FEATURE_FAMILIES = {'Shape': SHAPE, 'RGB': RGB, 'HSV': HSV, 'Lab': LAB,
                    'DerivedColor': DERIVED, 'Texture': TEXTURE, 'Spatial': SPATIAL}


def segment_leaf(img_bgr: np.ndarray):
    """ExG + Otsu segmentation. Returns (segmented_rgb, mask, mask_area_ratio, quality_flag)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)

    denom = R + G + B
    denom[denom == 0] = 1

    exg = 2 * (G / denom) - (R / denom) - (B / denom)
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_mask = np.zeros_like(mask)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [main_contour], -1, 255, cv2.FILLED)

    mask_ratio = np.sum(final_mask > 0) / final_mask.size

    if mask_ratio == 0:
        quality_flag = 'FAILED'
    elif mask_ratio > 0.98:
        quality_flag = 'SUSPICIOUS'
    elif mask_ratio < 0.01:
        quality_flag = 'FAILED'
    else:
        quality_flag = 'ACCEPT'

    segmented_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=final_mask)
    return segmented_rgb, final_mask, float(mask_ratio), quality_flag


def extract_all_features(img_rgb: np.ndarray, mask: np.ndarray):
    """Shape, colour, texture and core/edge measurements of a segmented leaf (dict), or None."""
    features = {}

    # 1. Shape & geometry
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)

    features['area'] = area
    features['perimeter'] = perimeter
    features['aspect_ratio'] = float(w) / h if h > 0 else 0
    features['circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    features['extent'] = float(area) / (w * h) if w * h > 0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    features['solidity'] = float(area) / hull_area if hull_area > 0 else 0
    features['convex_area'] = hull_area

    if len(cnt) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        features['major_axis_length'] = max(MA, ma)
        features['minor_axis_length'] = min(MA, ma)
        a, b = max(MA, ma) / 2, min(MA, ma) / 2
        features['eccentricity'] = np.sqrt(1 - (b ** 2) / (a ** 2)) if a > 0 else 0
    else:
        features['major_axis_length'] = 0
        features['minor_axis_length'] = 0
        features['eccentricity'] = 0

    # 2. Colour (global)
    pixels = img_rgb[mask > 0]
    if len(pixels) == 0:
        return None

    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    features['mean_R'], features['std_R'] = np.mean(R), np.std(R)
    features['mean_G'], features['std_G'] = np.mean(G), np.std(G)
    features['mean_B'], features['std_B'] = np.mean(B), np.std(B)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hsv_pixels = img_hsv[mask > 0]
    features['mean_H'], features['std_H'] = np.mean(hsv_pixels[:, 0]), np.std(hsv_pixels[:, 0])
    features['mean_S'], features['std_S'] = np.mean(hsv_pixels[:, 1]), np.std(hsv_pixels[:, 1])
    features['mean_V'], features['std_V'] = np.mean(hsv_pixels[:, 2]), np.std(hsv_pixels[:, 2])

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab_pixels = img_lab[mask > 0]
    features['mean_L'], features['std_L'] = np.mean(lab_pixels[:, 0]), np.std(lab_pixels[:, 0])
    features['mean_a'], features['std_a'] = np.mean(lab_pixels[:, 1]), np.std(lab_pixels[:, 1])
    features['mean_b'], features['std_b'] = np.mean(lab_pixels[:, 2]), np.std(lab_pixels[:, 2])

    denom = (R.astype(float) + G.astype(float) + B.astype(float))
    denom[denom == 0] = 1
    r, g, b = R / denom, G / denom, B / denom
    features['green_ratio'] = np.mean(g)
    features['yellow_ratio'] = np.mean((r + g) / 2)

    # 3. Texture (GLCM on the leaf bounding box)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_crop = gray[y:y + h, x:x + w]
    mask_crop = mask[y:y + h, x:x + w]

    glcm = graycomatrix(gray_crop, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
    features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
    features['glcm_energy'] = np.mean(graycoprops(glcm, 'energy'))
    features['glcm_correlation'] = np.mean(graycoprops(glcm, 'correlation'))

    # 4. Core / edge regions (central third of the bounding box vs the rest)
    cell_w, cell_h = w // 3, h // 3
    core_x1, core_x2 = cell_w, 2 * cell_w
    core_y1, core_y2 = cell_h, 2 * cell_h

    core_mask = np.zeros_like(mask_crop)
    core_mask[core_y1:core_y2, core_x1:core_x2] = 255
    core_mask = cv2.bitwise_and(core_mask, mask_crop)

    edge_mask = cv2.subtract(mask_crop, core_mask)

    hsv_crop = img_hsv[y:y + h, x:x + w]
    rgb_crop = img_rgb[y:y + h, x:x + w]

    def get_region_stats(region_mask, prefix):
        if np.sum(region_mask) == 0:
            return {f'{prefix}_mean_H': 0, f'{prefix}_yellow_ratio': 0}
        reg_hsv = hsv_crop[region_mask > 0]
        reg_rgb = rgb_crop[region_mask > 0]
        mean_H = np.mean(reg_hsv[:, 0])

        R, G, B = reg_rgb[:, 0].astype(float), reg_rgb[:, 1].astype(float), reg_rgb[:, 2].astype(float)
        denom = R + G + B
        denom[denom == 0] = 1
        y_ratio = np.mean((R / denom + G / denom) / 2)
        return {f'{prefix}_mean_H': mean_H, f'{prefix}_yellow_ratio': y_ratio}

    features.update(get_region_stats(core_mask, 'core'))
    features.update(get_region_stats(edge_mask, 'edge'))

    return {k: float(v) for k, v in features.items()}


def measure_leaf(img_bgr: np.ndarray):
    """Full measurement step for one photo.

    Returns dict with keys: quality_flag, mask_ratio, mask (uint8), features (list of 38 floats or None).
    """
    segmented_rgb, mask, ratio, flag = segment_leaf(img_bgr)
    feats = extract_all_features(segmented_rgb, mask) if flag == 'ACCEPT' else None
    if flag == 'ACCEPT' and feats is None:
        flag = 'FAILED'
    return {
        'quality_flag': flag,
        'mask_ratio': ratio,
        'mask': mask,
        'features': [feats[f] for f in FEATURES] if feats else None,
    }
