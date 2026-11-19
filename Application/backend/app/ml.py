"""Loads the trained models and turns one leaf photo into a diagnosis.

Besides the diagnosis, every result carries a `trace`: what each pipeline step actually
produced for this photo (segmentation numbers, all 38 measurements, the crop forest's votes,
the feature graph with its WL labels, and how the SVM weighed the training leaves). The
frontend uses it to replay the pipeline step by step.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import joblib
import numpy as np

from .conditions import CONDITIONS, CROPS, FEATURE_NAMES
from .features import FEATURE_FAMILIES, FEATURES, measure_leaf
from .lesions import detect_spots
from .wl import GRAPH_NAMES, cosine_kernel

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
LOW_CROP_CONFIDENCE = 0.5
FAMILY_OF = {f: fam for fam, fl in FEATURE_FAMILIES.items() for f in fl}
STEP_IMAGE_SIDE = 640
SIZE_FEATURES = ('area', 'perimeter', 'convex_area', 'major_axis_length', 'minor_axis_length')  # in pixels


class ImageError(ValueError):
    """The upload is not a usable leaf photo (message is shown to the user)."""


class Predictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        meta_path = os.path.join(models_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'No trained models in {models_dir}. Run "python train.py" first.')
        with open(meta_path, encoding='utf-8') as f:
            self.meta = json.load(f)
        self.router = joblib.load(os.path.join(models_dir, 'crop_router.joblib'))
        self.crop_models = {c: joblib.load(os.path.join(models_dir, f'{c}.joblib')) for c in self.meta['crops']}
        for crop, bundle in self.crop_models.items():
            model = bundle['stress']
            info = self.meta['crops'][crop]
            combined = getattr(model, 'measurements_', None) is not None
            if (combined != (info.get('stress_kernel') == 'wl_plus_measurements') or
                    (model.graph, model.h) != (info['graph'], info['wl_depth'])):
                raise ValueError(f'Model files and metadata differ for {crop}. Run python train.py to rebuild them together.')

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def decode(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
        if img is None:
            raise ImageError('The file is not a readable image. Please upload a JPG or PNG photo of a leaf.')
        return img

    @staticmethod
    def overlay(img_bgr: np.ndarray, mask: np.ndarray, max_side: int = 900) -> np.ndarray:
        """Photo with the background dimmed and the detected leaf outlined."""
        out = img_bgr.copy()
        bg = mask == 0
        out[bg] = (out[bg] * 0.25).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = max(2, int(round(max(img_bgr.shape[:2]) / 300)))
        cv2.drawContours(out, contours, -1, (80, 220, 120), thickness)
        return Predictor.resize(out, max_side)

    @staticmethod
    def resize(img: np.ndarray, max_side: int) -> np.ndarray:
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img

    @staticmethod
    def step_images(img_bgr: np.ndarray, mask: np.ndarray):
        """Pictures of the segmentation and region steps, plus the Otsu threshold that was used.

        Recomputes the ExG map the same way as features.segment_leaf, only to show it.
        """
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        denom = rgb.sum(axis=2)
        denom[denom == 0] = 1
        exg = 2 * (rgb[:, :, 1] / denom) - rgb[:, :, 0] / denom - rgb[:, :, 2] / denom
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        otsu, _ = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contrast-stretched copy for display only (the threshold above uses the exact map)
        lo, hi = np.percentile(exg_norm, (1, 99.5))
        shown = np.clip((exg_norm.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
        exg_img = cv2.applyColorMap(shown, cv2.COLORMAP_VIRIDIS)

        # leaf bounding box (edge region, blue tint) and its central third (core region, green tint)
        regions = img_bgr.copy()
        bg = mask == 0
        regions[bg] = (regions[bg] * 0.25).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t = max(2, int(round(max(img_bgr.shape[:2]) / 250)))
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cw, ch = w // 3, h // 3
            leaf = mask[..., None] > 0
            regions = np.where(leaf, cv2.addWeighted(regions, 0.7, np.full_like(regions, (40, 170, 255)), 0.35, 0), regions)
            core_mask = np.zeros_like(mask)
            core_mask[y + ch:y + 2 * ch, x + cw:x + 2 * cw] = 255
            core = cv2.bitwise_and(core_mask, mask)[..., None] > 0
            regions = np.where(core, cv2.addWeighted(img_bgr, 0.55, np.full_like(regions, (80, 220, 120)), 0.45, 0), regions)
            cv2.rectangle(regions, (x, y), (x + w, y + h), (255, 255, 255), t)
            cv2.rectangle(regions, (x + cw, y + ch), (x + 2 * cw, y + 2 * ch), (80, 220, 120), t)

        side = STEP_IMAGE_SIDE
        return {
            'exg': Predictor.resize(exg_img, side),
            'mask': Predictor.resize(mask, side),
            'regions': Predictor.resize(regions, side),
        }, float(otsu)

    def explain(self, crop: str, x: np.ndarray, top: int = 5):
        """Measurements that differ most from healthy leaves of the same crop."""
        m = self.crop_models[crop]
        z = (x - m['healthy_mean']) / m['healthy_std']
        order = np.argsort(-np.abs(z))[:top]
        return [{
            'feature': FEATURES[i], 'name': FEATURE_NAMES[FEATURES[i]],
            'value': round(float(x[i]), 4), 'healthy_mean': round(float(m['healthy_mean'][i]), 4),
            'z': round(float(z[i]), 2), 'direction': 'higher' if z[i] > 0 else 'lower',
        } for i in order]

    def router_trace(self, X: np.ndarray, proba: np.ndarray):
        """Every crop's probability and how many trees of the forest voted for it."""
        classes = list(self.router.classes_)
        votes = np.zeros(len(classes), int)
        for tree in self.router.estimators_:
            votes[int(tree.predict(X)[0])] += 1
        order = np.argsort(-proba)
        return {'trees': len(self.router.estimators_),
                'scores': [{'code': classes[i], 'name': CROPS[classes[i]], 'probability': round(float(proba[i]), 4),
                            'votes': int(votes[i])} for i in order]}

    @staticmethod
    def graph_trace(model, X: np.ndarray, x: np.ndarray):
        """The leaf's feature graph and its WL node labels at every iteration."""
        feat = model.feat_
        levels = feat.labels(X)
        # share of training leaves that have each (iteration, node, label) pattern; the columns of
        # the training WL vectors are laid out iteration by iteration, node by node
        freq = np.asarray(model.Ptr_.sum(0)).ravel() / model.Ptr_.shape[0]
        offsets = np.concatenate([[0], np.cumsum([s for sizes in feat.sizes_ for s in sizes])])
        iterations = []
        for k, L in enumerate(levels):
            lab = L[0]
            common = [round(float(freq[offsets[k * len(lab) + v] + l]), 4) if l >= 0 else 0.0 for v, l in enumerate(lab)]
            iterations.append({
                'labels': [int(v) for v in lab],
                'common': common,
                'seen': int((lab >= 0).sum()),
                'rare': int(sum(c < 0.05 for c in common)),
                'vocabulary': int(sum(feat.sizes_[k])),
            })
        P = feat.encode_levels(levels)
        ii, jj = np.where(np.triu(feat.A_, 1))
        return {
            'nodes': [{'feature': f, 'name': FEATURE_NAMES[f], 'family': FAMILY_OF[f], 'value': round(float(x[i]), 4),
                       'level': int(levels[0][0][i]),
                       'cuts': [round(float(c), 4) for c in feat.disc_.bin_edges_[i][1:-1]]}
                      for i, f in enumerate(FEATURES)],
            'edges': [[int(a), int(b)] for a, b in zip(ii, jj)],
            'iterations': iterations,
            'vector_size': int(P.shape[1]),
            'vector_nonzero': int(P.nnz),
        }, P

    @staticmethod
    def svm_trace(model, kernel, p_stress: float, top: int = 6):
        """How the healthy/stressed SVM reached its answer for this leaf.

        decision = sum over support vectors of (weight x WL-kernel similarity) + intercept;
        positive values point to "stressed".
        """
        svc = model.svc_
        k = kernel[0]  # WL-kernel similarity to every training leaf
        ksv = k[svc.support_]
        coef = svc.dual_coef_[0].copy()
        intercept = float(svc.intercept_[0])
        decision = float(svc.decision_function(k[None, :])[0])
        raw = float((coef * ksv).sum() + intercept)
        if abs(raw + decision) < abs(raw - decision):  # align sign with decision_function
            coef, intercept = -coef, -intercept
        # decision_function > 0 means classes_[1]; make "positive = stressed"
        if int(svc.classes_[1]) != 1:
            coef, intercept, decision = -coef, -intercept, -decision
        contrib = coef * ksv
        order = np.argsort(-np.abs(contrib))[:top]
        hist, _ = np.histogram(k, bins=10, range=(0.0, 1.0))
        return {
            'training_leaves': int(len(k)),
            'similarity': {'mean': round(float(k.mean()), 4), 'max': round(float(k.max()), 4),
                           'min': round(float(k.min()), 4), 'histogram': [int(v) for v in hist]},
            'support_vectors': int(len(svc.support_)),
            'push_stressed': round(float(contrib[contrib > 0].sum()), 4),
            'push_healthy': round(float(contrib[contrib < 0].sum()), 4),
            'intercept': round(float(intercept), 4),
            'decision': round(float(decision), 4),
            'probability_stressed': round(p_stress, 4),
            'top_support': [{'label': 'stressed' if coef[i] > 0 else 'healthy', 'similarity': round(float(ksv[i]), 4),
                             'weight': round(float(abs(coef[i])), 4), 'contribution': round(float(contrib[i]), 4)}
                            for i in order],
        }

    def reliability_check(self, crop: str, crop_source: str, crop_conf: float, x: np.ndarray, best_sim: float, spots: dict):
        """Is this photo like the training photos, and are there signs the models cannot see?

        Limits come from 5-fold CV on the training leaves (train.py: calibrate_familiarity): a photo
        is flagged when it is more unusual than 99% of held-out training leaves.
        """
        info = self.meta['crops'][crop]
        fam = info.get('familiarity')
        known = [CONDITIONS[c]['name'] for c in info['conditions']]
        checks = []
        if crop_source == 'model':
            checks.append({'id': 'crop', 'label': 'Crop recognised clearly', 'passed': crop_conf >= LOW_CROP_CONFIDENCE,
                           'detail': f'{crop_conf:.0%} sure (needs at least {LOW_CROP_CONFIDENCE:.0%}).'})
        if fam:
            lo, hi = np.asarray(fam['range_low']), np.asarray(fam['range_high'])
            out = [i for i in range(len(FEATURES)) if x[i] < lo[i] or x[i] > hi[i]]
            limit = int(fam['out_of_range_p99'])
            names = ', '.join(FEATURE_NAMES[FEATURES[i]].lower() for i in out[:6]) + (' …' if len(out) > 6 else '')
            checks.append({'id': 'range', 'label': 'Measurements within the training range', 'passed': len(out) <= limit,
                           'detail': f'{len(out)} of {len(FEATURES)} outside the range of training leaves'
                                     f' (normal: up to {limit}).' + (f' Outside: {names}.' if out else '')})
            size_out = [FEATURES[i] for i in out if FEATURES[i] in SIZE_FEATURES]
            checks.append({'id': 'scale', 'label': 'Leaf size similar to training photos', 'passed': not size_out,
                           'detail': 'Leaf size in pixels is in the usual range.' if not size_out else
                           'The leaf is much smaller or larger in pixels than in the training photos (different '
                           'resolution or zoom), so the shape measurements are not comparable.'})
            checks.append({'id': 'similar', 'label': 'Similar to at least one training leaf',
                           'passed': best_sim >= fam['best_similarity_p1'],
                           'detail': f'Closest training leaf: similarity {best_sim:.2f} '
                                     f'(99% of new training-like leaves reach {fam["best_similarity_p1"]:.2f}).'})
        checks.append({'id': 'spots', 'label': 'No visible spots the model cannot detect', 'passed': not spots['flagged'],
                       'detail': f'{spots["count"]} spot(s) on {spots["area_ratio"]:.1%} of the leaf.'})

        failed = [c for c in checks if not c['passed'] and c['id'] != 'spots']
        status = 'ok' if not failed else 'caution' if len(failed) == 1 else 'unreliable'
        return {'status': status, 'checks': checks, 'known_conditions': known}

    @staticmethod
    def spot_message(spots: dict, crop: str, known_codes: list[str]) -> str | None:
        if not spots['flagged']:
            return None
        msg = (f'{spots["count"]} spots were found on {spots["area_ratio"]:.1%} of the leaf. Spots like these can mean '
               'a leaf spot, blight or other fungal or bacterial disease, or insect damage.')
        if 'LS' in known_codes:
            return msg + ' Compare with the "Leaf spot" score below.'
        return (msg + f' The {CROPS[crop].lower()} model was not trained on leaf-spot diseases, so it cannot '
                'name them. Please confirm with an expert.')

    # ------------------------------------------------------------------ main entry
    def predict(self, image_bytes: bytes, crop_hint: str | None = None):
        """Returns (result dict, step images {name: array}, original image as BGR array)."""
        t0 = time.perf_counter()
        img = self.decode(image_bytes)
        measured = measure_leaf(img)
        if measured['quality_flag'] != 'ACCEPT':
            if measured['quality_flag'] == 'SUSPICIOUS':
                raise ImageError('The leaf could not be separated from the background (it seems to fill the whole '
                                 'photo). Photograph a single leaf on a plain, dark background.')
            raise ImageError('No leaf was found in this photo. Photograph a single leaf on a plain, dark background.')

        x = np.asarray(measured['features'], float)
        X = x[None, :]
        spots, spots_img = detect_spots(img, measured['mask'])
        t1 = time.perf_counter()

        # 1. crop
        proba = self.router.predict_proba(X)[0]
        order = np.argsort(-proba)
        crop_alts = [{'code': self.router.classes_[i], 'name': CROPS[self.router.classes_[i]],
                      'probability': round(float(proba[i]), 4)} for i in order[:3]]
        if crop_hint:
            if crop_hint not in self.crop_models:
                raise ImageError(f'Unknown crop "{crop_hint}".')
            crop, crop_source = crop_hint, 'user'
            crop_conf = float(proba[list(self.router.classes_).index(crop_hint)])
        else:
            crop, crop_source = crop_alts[0]['code'], 'model'
            crop_conf = crop_alts[0]['probability']
        router = self.router_trace(X, proba)
        t2 = time.perf_counter()

        # 2. feature graph + WL relabelling
        cm = self.crop_models[crop]
        graph, P = self.graph_trace(cm['stress'], X, x)
        t3 = time.perf_counter()

        # 3. healthy vs stressed, 4. ranked conditions
        stress_model = cm['stress']
        wl_kernel = cosine_kernel(stress_model.Ptr_, P)
        kernel = stress_model.combine_kernel(wl_kernel, X)
        # CV scores SVC.predict. Probability calibration can put 50% away from the SVM boundary.
        stressed = bool(stress_model.svc_.predict(kernel)[0])
        p_stress = float(stress_model.svc_.predict_proba(kernel)[0][list(stress_model.classes_).index(1)])
        cond_p = cm['condition'].predict_proba(X)[0]
        cond_order = np.argsort(-cond_p)
        conditions = [{'code': cm['condition'].classes_[i], **CONDITIONS[cm['condition'].classes_[i]],
                       'probability': round(float(cond_p[i]), 4)} for i in cond_order[:3]]
        svm = self.svm_trace(stress_model, kernel, p_stress)
        svm['conditions'] = [{'code': cm['condition'].classes_[i], 'name': CONDITIONS[cm['condition'].classes_[i]]['name'],
                              'probability': round(float(cond_p[i]), 4)} for i in cond_order]
        t4 = time.perf_counter()

        info = self.meta['crops'][crop]
        warnings = []
        if crop_source == 'model' and crop_conf < LOW_CROP_CONFIDENCE:
            warnings.append('The crop is uncertain. If you know the crop, choose it and analyse again.')
        if crop_source == 'user' and crop_alts[0]['code'] != crop:
            warnings.append(f'You selected {CROPS[crop]}, but the leaf looks most like {crop_alts[0]["name"]}.')
        check = self.reliability_check(crop, crop_source, crop_conf, x, float(wl_kernel.max()), spots)
        spot_msg = self.spot_message(spots, crop, info['conditions'])
        if check['status'] != 'ok':
            warnings.insert(0, 'This photo is unlike the photos the model was trained on, so the result may be wrong. '
                               'See "Reliability check" for why.')
        if spot_msg:
            warnings.insert(0, spot_msg)

        images, otsu = self.step_images(img, measured['mask'])
        images['overlay'] = self.overlay(img, measured['mask'])
        images['spots'] = self.resize(spots_img, STEP_IMAGE_SIDE)
        z = (x - cm['healthy_mean']) / cm['healthy_std']
        ms = lambda a, b: round((b - a) * 1000, 1)
        trace = {
            'segmentation': {'otsu_threshold': round(otsu, 1), 'mask_ratio': round(measured['mask_ratio'], 4),
                             'leaf_pixels': int((measured['mask'] > 0).sum()),
                             'image_size': [int(img.shape[1]), int(img.shape[0])], 'accept_range': [0.01, 0.98]},
            'features': [{'feature': f, 'name': FEATURE_NAMES[f], 'family': FAMILY_OF[f], 'value': round(float(x[i]), 4),
                          'healthy_mean': round(float(cm['healthy_mean'][i]), 4), 'z': round(float(z[i]), 2)}
                         for i, f in enumerate(FEATURES)],
            'crop_router': router,
            'graph': graph,
            'svm': svm,
            'timings_ms': {'measure': ms(t0, t1), 'crop_routing': ms(t1, t2), 'wl_graph': ms(t2, t3), 'svm': ms(t3, t4)},
            'images': {},  # filled in by the API once the pictures are saved
        }

        result = {
            'quality': {'flag': measured['quality_flag'], 'mask_ratio': round(measured['mask_ratio'], 4)},
            'crop': {'code': crop, 'name': CROPS[crop], 'confidence': round(crop_conf, 4), 'source': crop_source,
                     'alternatives': crop_alts, 'reliability': self.meta['crop_router']['cv_accuracy']},
            'stress': {'status': 'stressed' if stressed else 'healthy', 'decision_rule': 'svm_predict',
                       'probability_stressed': round(p_stress, 4),
                       'confidence': round(p_stress if stressed else 1 - p_stress, 4),
                       # balanced accuracy is the headline: 84% of leaves are stressed, so plain accuracy
                       # looks good even for a model that always answers "stressed"
                       'reliability': {'accuracy': info['stress_accuracy'], 'macro_f1': info['stress_macro_f1'],
                                       'balanced_accuracy': info.get('stress_balanced_accuracy'),
                                       'healthy_found': info.get('stress_healthy_found'),
                                       'always_stressed_accuracy': info.get('always_stressed_accuracy')}},
            'conditions': conditions,
            'conditions_reliability': {'top1': info['condition_top1_accuracy'], 'top3': info['condition_top3_accuracy']},
            'model': {'graph': stress_model.graph, 'graph_name': GRAPH_NAMES[stress_model.graph], 'wl_depth': stress_model.h,
                      'kernel': 'wl_plus_measurements' if getattr(stress_model, 'measurements_', None) is not None else 'wl'},
            'explanation': self.explain(crop, x),
            'warnings': warnings,
            'spots': {**{k: spots[k] for k in ('count', 'area_ratio', 'flagged')}, 'message': spot_msg},
            'reliability_check': check,
            'features': {f: round(float(v), 6) for f, v in zip(FEATURES, x)},
            'trace': trace,
        }
        return result, images, img
