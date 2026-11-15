"""Spot (lesion) check: finds small brown / dark spots on the segmented leaf.

The 38 research measurements are averages over the whole leaf, so a few small spots hardly
change them and the WL models cannot see them. This rule-based check runs next to the models
(it does not change their output) and warns the user when a leaf has clear spots.

Method: the leaf is scaled to a standard size; each pixel's green-red value (a* in Lab colour
space) is compared with the median of the leaf around it. Spots are pixels that have clearly
lost their green (a* much higher than the surrounding leaf and close to neutral). Shadows,
veins and bumps change brightness but stay green, so they do not pass. Background showing
through gaps and holes is ignored. Connected regions of a plausible spot size and shape are
counted.
"""
from __future__ import annotations

import cv2
import numpy as np

WORK_SIDE = 600          # leaf bounding box is scaled so its longer side has this many pixels
RED_DELTA = 14.0         # a* above the local leaf median (OpenCV Lab units, 0-255 scale, 128 = neutral)
MIN_A = 122              # and a* close to or above neutral: the spot is no longer green
BLACK_L = 40             # L* below this is background showing through gaps, not leaf
MIN_SPOT_PX = 12         # at WORK_SIDE scale
MAX_SPOT_FRACTION = 0.04  # a single region larger than this share of the leaf is not a "spot"
MAX_ELONGATION = 4.0     # long thin regions are veins / edges, not spots

# When to warn. Tuned on 40 OLID-I leaves per class (Nov 2026): flags 1 of 320 healthy leaves,
# 62% of eggplant beetle damage, ~22% of leaf miner / bitter-gourd leaf spot / downy mildew, and
# the tomato leaf-spot photo that prompted this check. Kept conservative on purpose: few false
# alarms, but small dark gourd leaf spots are often missed.
FLAG_MIN_SPOTS = 3
FLAG_MIN_AREA = 0.002


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 else n + 1


def detect_spots(img_bgr: np.ndarray, mask: np.ndarray):
    """Returns (summary dict, picture with the spots circled)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    empty = {'count': 0, 'area_ratio': 0.0, 'flagged': False, 'spots': []}
    if not contours:
        return empty, img_bgr
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    img = img_bgr[y:y + h, x:x + w]
    m = mask[y:y + h, x:x + w]
    s = WORK_SIDE / max(w, h)
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    m = (cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A = lab[:, :, 0], lab[:, :, 1]

    # the segmentation mask is filled, so gaps between lobes and holes show black background
    # inside it; drop near-black pixels and a rim around them, plus a rim at the leaf border
    ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    background = cv2.dilate(((L < BLACK_L) | (m == 0)).astype(np.uint8), ring) > 0
    leaf = ~background
    leaf_px = int(leaf.sum())
    if leaf_px < 500:
        return empty, img

    def local_median(ch):
        filled = ch.copy()
        filled[~leaf] = int(np.median(ch[leaf]))  # background must not pull the median
        return cv2.medianBlur(filled, _odd(WORK_SIDE // 10)).astype(np.float32)

    # spots (brown, purple, grey-centred) lose the green: a* jumps towards red. Shading, veins
    # and leaf bumps change brightness but stay green, so brightness is not used.
    da = A.astype(np.float32) - local_median(A)
    cand = leaf & (da >= RED_DELTA) & (A >= MIN_A)
    cand = cv2.morphologyEx(cand.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    spots = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < MIN_SPOT_PX or area > MAX_SPOT_FRACTION * leaf_px:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > MAX_ELONGATION or area / (bw * bh) < 0.25:
            continue
        spots.append({'x': round(float(cents[i][0]) / size[0], 4), 'y': round(float(cents[i][1]) / size[1], 4),
                      'area': area, 'r': max(bw, bh) / 2})

    area_ratio = sum(sp['area'] for sp in spots) / leaf_px
    flagged = len(spots) >= FLAG_MIN_SPOTS and area_ratio >= FLAG_MIN_AREA

    pic = img.copy()
    pic[m == 0] = (pic[m == 0] * 0.25).astype(np.uint8)
    t = max(2, WORK_SIDE // 250)
    for sp in spots:
        c = (int(sp['x'] * size[0]), int(sp['y'] * size[1]))
        cv2.circle(pic, c, int(sp['r'] + 6), (40, 40, 255), t)
    summary = {'count': len(spots), 'area_ratio': round(float(area_ratio), 5), 'flagged': bool(flagged),
               'spots': [{k: sp[k] for k in ('x', 'y', 'area')} for sp in spots[:200]]}
    return summary, pic
