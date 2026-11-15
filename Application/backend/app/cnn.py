"""Hybrid decision: a frozen ImageNet CNN decides, the WL model explains.

EfficientNet-B0 (ImageNet weights, NOT trained on OLID-I) turns the photo into 1,280 numbers. Small
logistic-regression classifiers on those numbers (trained by train_cnn.py) decide the crop and whether the
leaf is healthy or stressed. In the benchmark (New_Models_Baselines/03_Benchmark_Suite) this was the best
full pipeline (macro-F1 0.739 vs 0.706 for WL), identified the crop for 99% of leaves, and was unaffected by
changes in light, blur, compression and resolution.

Optional: if PyTorch / torchvision or models/cnn_heads.joblib are missing, the app falls back to WL only.
"""
from __future__ import annotations

import os
import time

import cv2
import joblib
import numpy as np

SIDE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BACKBONE = 'efficientnet_b0'
BACKBONE_NAME = 'EfficientNet-B0'


def load_backbone():
    import torch
    import torchvision.models as tvm
    weights = tvm.get_model_weights(BACKBONE).DEFAULT
    model = tvm.get_model(BACKBONE, weights=weights).eval()
    model.classifier = torch.nn.Identity()  # pooled embedding: 1,280 numbers per photo
    return model


def to_input(img_bgr: np.ndarray, reduced: bool = False) -> np.ndarray:
    """Photo -> normalised 224x224 RGB array (C, H, W). The whole photo is resized (not cropped), as in training."""
    img = cv2.resize(img_bgr, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


class CNNDecider:
    def __init__(self, models_dir: str):
        path = os.path.join(models_dir, 'cnn_heads.joblib')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found. Run "python train_cnn.py" first.')
        self.heads = joblib.load(path)
        import torch
        self.torch = torch
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.model = load_backbone()

    def embed(self, img_bgr: np.ndarray) -> np.ndarray:
        x = self.torch.from_numpy(to_input(img_bgr)[None].copy())
        with self.torch.inference_mode():
            return self.model(x).numpy()

    def decide(self, img_bgr: np.ndarray, crop_hint: str | None = None) -> dict:
        t0 = time.perf_counter()
        E = self.embed(img_bgr)
        t1 = time.perf_counter()
        router = self.heads['router']
        cp = router.predict_proba(E)[0]
        classes = list(router.classes_)
        order = np.argsort(-cp)
        crop = crop_hint if crop_hint else classes[order[0]]
        stress = self.heads['stress'][crop]
        sp = stress.predict_proba(E)[0]
        p_stressed = float(sp[list(stress.classes_).index(1)])
        return {
            'crop': crop, 'crop_confidence': float(cp[classes.index(crop)]),
            'crop_scores': [{'code': classes[i], 'probability': round(float(cp[i]), 4)} for i in order],
            'p_stressed': p_stressed,
            'timings_ms': {'cnn': round((t1 - t0) * 1000, 1), 'heads': round((time.perf_counter() - t1) * 1000, 1)},
        }
