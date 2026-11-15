"""Spot check on synthetic leaves: real spots are found, shading and holes are not.

Run from the repository root:  python -m pytest Application/backend/tests -q
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.features import segment_leaf  # noqa: E402
from app.lesions import detect_spots  # noqa: E402

GREEN = (90, 150, 95)       # BGR leaf colour
SPOT = (175, 160, 205)      # pale purple-brown centre, like the tomato leaf-spot photo
RING = (70, 90, 140)        # brown rim


def leaf(seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((800, 1200, 3), np.uint8)
    cv2.ellipse(img, (600, 400), (480, 260), 0, 0, 360, GREEN, -1)
    noise = rng.normal(0, 4, img.shape)
    img = np.where(img > 0, np.clip(img + noise, 0, 255), 0).astype(np.uint8)
    return img


def run(img):
    _, mask, _, flag = segment_leaf(img)
    assert flag == 'ACCEPT'
    return detect_spots(img, mask)[0]


def test_clean_leaf_has_no_spots():
    s = run(leaf())
    assert s['count'] == 0 and not s['flagged']


def test_spotted_leaf_is_flagged():
    img = leaf()
    for x, y in [(400, 300), (520, 450), (700, 330), (820, 480), (650, 560), (900, 380)]:
        cv2.circle(img, (x, y), 11, RING, -1)
        cv2.circle(img, (x, y), 7, SPOT, -1)
    s = run(img)
    assert s['count'] >= 5 and s['flagged']


def test_shadows_and_holes_are_ignored():
    img = leaf()
    # dark-green shading (brightness changes, colour stays green)
    shade = img.copy()
    cv2.circle(shade, (450, 350), 40, (45, 80, 50), -1)
    cv2.circle(shade, (750, 450), 35, (45, 80, 50), -1)
    img = np.where(img > 0, shade, img)
    # holes showing the black background inside the leaf outline
    cv2.circle(img, (600, 300), 25, (0, 0, 0), -1)
    cv2.circle(img, (850, 420), 18, (0, 0, 0), -1)
    s = run(img)
    assert not s['flagged'], s
