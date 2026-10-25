"""Build the PlantCare AI data bundle from the WL-LeafKernel experiment outputs.

Reads the CSV/JSON results under full_pipeline/ and writes
  src/plantcare/data/research.ts           summary data used by every PlantCare page
  public/plantcare/oof_level3.json         out-of-fold log for all accepted images (History & Logs)
  public/plantcare/samples/*.jpg           two thumbnails per crop-condition class

Run from Application/:
  python scripts/build_plantcare_data.py
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

FRONTEND = Path(__file__).resolve().parents[1]   # Application/
ROOT = FRONTEND.parent                           # repository root
FP = ROOT / 'full_pipeline'
RES = FP / 'results'
DEC = FP / 'decomposition' / 'results'
ARCHIVE = ROOT.parent / 'archive (3)'           # raw OLID-I images
OUT_TS = FRONTEND / 'src' / 'plantcare' / 'data' / 'research.ts'
OUT_ASSETS = FRONTEND / 'public' / 'plantcare'
THUMB = 360

# ── vocabulary ────────────────────────────────────────────────────────────────
CROPS = {
    'ash_gourd': ('Ash gourd', 'Benincasa hispida'),
    'bitter_gourd': ('Bitter gourd', 'Momordica charantia'),
    'bottle_gourd': ('Bottle gourd', 'Lagenaria siceraria'),
    'cucumber': ('Cucumber', 'Cucumis sativus'),
    'eggplant': ('Eggplant', 'Solanum melongena'),
    'ridge_gourd': ('Ridge gourd', 'Luffa acutangula'),
    'snake_gourd': ('Snake gourd', 'Trichosanthes cucumerina'),
    'tomato': ('Tomato', 'Solanum lycopersicum'),
}

# Definitions follow Task_Direction_Audit/AUDIT_REPORT.md (corrected OLID-I codes).
CONDITIONS = {
    'healthy': ('Healthy control', 'No documented stress', 'Control'),
    'N': ('Nitrogen deficiency', 'Macronutrient deficiency', 'Nutrient deficiency'),
    'K': ('Potassium deficiency', 'Macronutrient deficiency', 'Nutrient deficiency'),
    'N_K': ('Nitrogen + potassium deficiency', 'Compound macronutrient deficiency', 'Nutrient deficiency'),
    'N_Mg': ('Nitrogen + magnesium deficiency', 'Compound nutrient deficiency', 'Nutrient deficiency'),
    'K_Mg': ('Potassium + magnesium deficiency', 'Compound nutrient deficiency', 'Nutrient deficiency'),
    'MIT': ('Mite', 'Tetranychus spp.', 'Pest'),
    'JAS': ('Jassid', 'Amrasca biguttula', 'Pest'),
    'JAS_MIT': ('Jassid + mite', 'Amrasca biguttula & Tetranychus spp.', 'Pest'),
    'EB': ('Epilachna beetle', 'Henosepilachna spp.', 'Pest'),
    'MIT_EB': ('Mite + Epilachna beetle', 'Tetranychus & Henosepilachna spp.', 'Pest'),
    'FB': ('Flea beetle', 'Phyllotreta spp.', 'Pest'),
    'LM': ('Leaf miner', 'Liriomyza spp.', 'Pest'),
    'PLEI': ('Pumpkin leaf-eating insect', 'Aulacophora spp.', 'Pest'),
    'PLEI_IEM': ('Pumpkin leaf-eating insect + insect egg mass', 'Aulacophora spp. with oviposition', 'Pest'),
    'PLEI_MIT': ('Pumpkin leaf-eating insect + mite', 'Aulacophora & Tetranychus spp.', 'Pest'),
    'PC': ('Pumpkin caterpillar', 'Diaphania indica', 'Pest'),
    'PM': ('Powdery mildew', 'Podosphaera / Erysiphe spp.', 'Disease'),
    'DM': ('Downy mildew', 'Pseudoperonospora cubensis', 'Disease'),
    'LS': ('Leaf spot', 'Cercospora / Alternaria spp.', 'Disease'),
}

FAMILIES = {
    'Morphology': ['area', 'perimeter', 'aspect_ratio', 'circularity', 'extent', 'solidity', 'convex_area',
                   'major_axis_length', 'minor_axis_length', 'eccentricity'],
    'Colour': ['mean_R', 'std_R', 'mean_G', 'std_G', 'mean_B', 'std_B', 'mean_H', 'std_H', 'mean_S', 'std_S',
               'mean_V', 'std_V', 'mean_L', 'std_L', 'mean_a', 'std_a', 'mean_b', 'std_b', 'green_ratio', 'yellow_ratio'],
    'Texture': ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation'],
    'Spatial': ['core_mean_H', 'core_yellow_ratio', 'edge_mean_H', 'edge_yellow_ratio'],
}
FEATURES = [f for fs in FAMILIES.values() for f in fs]
assert len(FEATURES) == 38


def r(x, n=4):
    return None if pd.isna(x) else round(float(x), n)


def key(path):
    """Normalise an image path to '<class>/<file>' so every results file joins on it."""
    p = str(path).replace('\\', '/').split('/')
    return f'{p[-2]}/{p[-1]}'


# ── load ──────────────────────────────────────────────────────────────────────
feats = pd.read_csv(FP / 'olid_full_features.csv')
feats = feats[feats.quality_flag == 'ACCEPT'].copy()
feats['key'] = feats.image_path.map(key)

l3 = pd.read_csv(DEC / 'level3_hierarchical' / 'level3_oof_predictions.csv')
l3['key'] = l3.sample_id.map(key)

l1 = pd.read_csv(DEC / 'level1_binary_wl' / 'level1_oof_predictions.csv')
l1 = l1[(l1['Graph Type'] == 'Hybrid') & (l1.Height == 3)].copy()
l1['key'] = l1.image_path.map(key)

c57 = pd.read_csv(RES / 'continuous_graph_svm' / 'continuous_graph_svm_oof_predictions.csv')
c57 = c57[c57.method == 'Flat RBF SVM'].copy()
c57['key'] = c57.image_path.map(key)

df = (feats.merge(l3[['key', 'fold', 'predicted_crop', 'predicted_stress', 'true_stress', 'crop_routing_correct',
                      'stress_prediction_correct', 'selected_graph', 'selected_height']], on='key', how='inner')
      .merge(l1[['key', 'pred_binary', 'pred_prob']], on='key', how='left')
      .merge(c57[['key', 'pred_label']], on='key', how='left'))
df['img_no'] = df.image_name.str.extract(r'\((\d+)\)').astype(float).fillna(0).astype(int)
print('joined images:', len(df), 'of', len(feats))

# tertile cut points on the full accepted set — display only (experiments fit bins per training fold)
tertiles = {f: [r(feats[f].quantile(1 / 3), 6), r(feats[f].quantile(2 / 3), 6)] for f in FEATURES}
stats = {f: [r(feats[f].mean(), 6), r(feats[f].std(), 6)] for f in FEATURES}

# ── per-crop and per-condition summaries ──────────────────────────────────────
l1_crop = pd.read_csv(DEC / 'level1_binary_wl' / 'level1_crop_subgroup_results.csv').set_index('Crop')
l2_best = pd.read_csv(DEC / 'level2_crop_wl' / 'level2_best_crop_results.csv').set_index('Crop')
rf_class = pd.read_csv(RES / 'classical_baselines' / 'best_model_per_class.csv').set_index('class')

crops = []
for c, (name, sci) in CROPS.items():
    d = df[df.crop == c]
    b = l2_best.loc[c]
    crops.append({
        'id': c, 'name': name, 'scientific': sci, 'images': int(len(d)),
        'healthy': int((d.condition == 'healthy').sum()), 'stressed': int((d.condition != 'healthy').sum()),
        'conditions': sorted(d.condition.unique().tolist(), key=lambda k: list(CONDITIONS).index(k)),
        'routingAccuracy': r(d.crop_routing_correct.mean()),
        'endToEndAccuracy': r(d.stress_prediction_correct.mean()),
        'healthyMeans': [r(x, 6) for x in feats[(feats.crop == c) & (feats.condition == 'healthy')][FEATURES].mean()],
        'globalWl': {'accuracy': r(l1_crop.loc[c, 'Accuracy']), 'f1': r(l1_crop.loc[c, 'F1'])},
        'cropSpecific': {
            'graph': b['Graph Type'], 'height': int(b.Height),
            'accuracy': r(b.accuracy), 'accuracySd': r(b.accuracy_sd), 'balancedAccuracy': r(b.balanced_accuracy),
            'macroF1': r(b.macro_f1), 'macroF1Sd': r(b.macro_f1_sd), 'rocAuc': r(b.roc_auc),
        },
    })

conditions = []
for k, (name, sci, kind) in CONDITIONS.items():
    d = df[df.condition == k]
    hit = (d.predicted_stress == (0 if k == 'healthy' else 1)).mean()
    classes = sorted(d['class'].unique())
    conditions.append({
        'code': k, 'name': name, 'scientific': sci, 'type': kind, 'images': int(len(d)),
        'crops': sorted(d.crop.unique().tolist()),
        'detectionRate': r(hit),  # healthy: share kept as healthy; stresses: share flagged as stressed
        'classes': [{'id': cl, 'crop': cl.split('__')[0], 'images': int((d['class'] == cl).sum()),
                     'rfRecall': r(rf_class.loc[cl, 'recall']) if cl in rf_class.index else None,
                     'rfF1': r(rf_class.loc[cl, 'f1']) if cl in rf_class.index else None} for cl in classes],
    })

# ── sample images (two per class: one the pipeline got right, one it got wrong when available) ──
samples = []
thumbs = OUT_ASSETS / 'samples'
thumbs.mkdir(parents=True, exist_ok=True)
for cl, d in df.sort_values(['class', 'img_no']).groupby('class'):
    right, wrong = d[d.stress_prediction_correct], d[~d.stress_prediction_correct]
    picks = [x for x in (right.head(1), wrong.head(1)) if len(x)]
    pick = pd.concat(picks)
    if len(pick) < 2:
        pick = pd.concat([pick, d[~d.key.isin(pick.key)].head(2 - len(pick))])
    for _, s in pick.iterrows():
        sid = f"{cl}-{s.img_no}"
        out = thumbs / f'{sid}.jpg'
        if not out.exists():
            im = ImageOps.exif_transpose(Image.open(ARCHIVE / s.key)).convert('RGB')
            im.thumbnail((THUMB, THUMB))
            im.save(out, quality=82)
        samples.append({
            'id': sid, 'image': f'/plantcare/samples/{sid}.jpg', 'file': s.image_name,
            'crop': s.crop, 'condition': s.condition, 'classLabel': cl, 'fold': int(s.fold),
            'trueStress': int(s.true_stress), 'predictedCrop': s.predicted_crop,
            'predictedStress': int(s.predicted_stress), 'routingCorrect': bool(s.crop_routing_correct),
            'stressCorrect': bool(s.stress_prediction_correct),
            'graph': s.selected_graph, 'height': int(s.selected_height),
            'globalPrediction': None if pd.isna(s.pred_binary) else int(s.pred_binary),
            'globalProbability': r(s.pred_prob),
            'fineGrainedPrediction': None if pd.isna(s.pred_label) else s.pred_label,
            'maskAreaRatio': r(s.mask_area_ratio),
            'features': [r(s[f], 6) for f in FEATURES],
        })
print('samples:', len(samples))

# ── experiment results ────────────────────────────────────────────────────────
g = pd.read_csv(DEC / 'level1_binary_wl' / 'level1_wl_summary.csv')
level1_grid = [{'graph': x['Graph Type'], 'height': int(x.Height), 'accuracy': r(x.accuracy),
                'balancedAccuracy': r(x.balanced_accuracy), 'macroF1': r(x.macro_f1), 'rocAuc': r(x.roc_auc)}
               for _, x in g[g.Method == 'WL + SVM'].iterrows()]
level1_baselines = [{'method': x.Method, 'accuracy': r(x.accuracy), 'balancedAccuracy': r(x.balanced_accuracy),
                     'macroF1': r(x.macro_f1), 'rocAuc': r(x.roc_auc)} for _, x in g[g.Method != 'WL + SVM'].iterrows()]

l3s = pd.read_csv(DEC / 'level3_hierarchical' / 'level3_end_to_end_summary.csv')
def pm(v):
    """'0.8160 +/- 0.0160' -> [0.816, 0.016]"""
    return [r(t) for t in str(v).split('+/-')]


hier = [{'method': x.Method, 'accuracy': pm(x.accuracy), 'macroF1': pm(x.macro_f1), 'balancedAccuracy': pm(x.balanced_accuracy)}
        for _, x in l3s.iterrows()]
crop_clf = pd.read_csv(DEC / 'level3_hierarchical' / 'level3_crop_classifier_summary.csv')
crop_clf = [{'model': x.Model, 'accuracy': pm(x.accuracy), 'macroF1': pm(x.macro_f1)} for _, x in crop_clf.iterrows()]

cmp57 = pd.read_csv(RES / 'classical_baselines' / 'graph_vs_baseline_comparison.csv')
fine57 = [{'method': x.Method, 'representation': x.Representation, 'accuracy': r(x.Accuracy), 'macroF1': r(x['Macro-F1'])}
          for _, x in cmp57.iterrows()]

graphs = pd.read_csv(RES / 'final_graph_design' / 'graph_design_comparison.csv')
graphs = [{'graph': x.graph_type, 'edges': int(x.edge_count), 'density': r(x.density), 'avgDegree': r(x.avg_degree, 2),
           'components': int(x.connected_components), 'isolated': int(x.isolated_nodes), 'source': x.learned_or_fixed,
           'meaning': x.interpretation} for _, x in graphs.iterrows()]
edges = pd.read_csv(RES / 'final_graph_design' / 'graph_edge_lists.csv')
edge_lists = {gt: [[FEATURES.index(a), FEATURES.index(b)] for a, b in zip(d.node_1, d.node_2)]
              for gt, d in edges.groupby('graph_type')}

imp = pd.read_csv(RES / 'classical_baselines' / 'rf_feature_importances.csv').head(12)
importances = [{'feature': x.feature, 'importance': r(x.importance)} for _, x in imp.iterrows()]

wl_cfg = json.loads((RES / 'wl_engine' / 'wl_engine_config.json').read_text())

data = {
    'dataset': {'name': 'OLID-I', 'rawImages': int(len(pd.read_csv(FP / 'olid_full_features.csv'))),
                'accepted': int(len(feats)), 'crops': len(CROPS), 'classes': int(feats['class'].nunique()),
                'conditions': len(CONDITIONS), 'healthy': int((feats.condition == 'healthy').sum()),
                'stressed': int((feats.condition != 'healthy').sum()), 'folds': 5},
    'config': {**{k: wl_cfg[k] for k in ('n_bins', 'bin_labels', 'corr_thresh', 'knn_k', 'wl_heights', 'random_state')}},
    'features': [{'id': f, 'family': fam} for fam, fs in FAMILIES.items() for f in fs],
    'featureStats': [stats[f] for f in FEATURES],
    'featureTertiles': [tertiles[f] for f in FEATURES],
    'crops': crops, 'conditions': conditions, 'samples': samples,
    'graphs': graphs, 'edgeLists': edge_lists,
    'level1Grid': level1_grid, 'level1Baselines': level1_baselines,
    'hierarchical': hier, 'cropClassifier': crop_clf, 'fineGrained': fine57,
    'rfImportances': importances,
}

OUT_TS.parent.mkdir(parents=True, exist_ok=True)
OUT_TS.write_text(
    '// Generated by scripts/build_plantcare_data.py from the full_pipeline experiment outputs. Do not edit by hand.\n'
    "import type { ResearchData } from '../types';\n\n"
    f'export const RESEARCH: ResearchData = {json.dumps(data, separators=(",", ":"))};\n',
    encoding='utf-8')

# ── full out-of-fold log: [crop, condition, image no., predicted crop, true stress, predicted stress, fold, graph, height]
crop_ix = {c: i for i, c in enumerate(CROPS)}
cond_ix = {c: i for i, c in enumerate(CONDITIONS)}
graph_names = sorted(df.selected_graph.unique())
rows = [[crop_ix[x.crop], cond_ix[x.condition], int(x.img_no), crop_ix[x.predicted_crop], int(x.true_stress),
         int(x.predicted_stress), int(x.fold), graph_names.index(x.selected_graph), int(x.selected_height)]
        for x in df.sort_values(['crop', 'condition', 'img_no']).itertuples()]
(OUT_ASSETS / 'oof_level3.json').write_text(json.dumps(
    {'crops': list(CROPS), 'conditions': list(CONDITIONS), 'graphs': graph_names, 'rows': rows}, separators=(',', ':')))
print('wrote', OUT_TS, 'and', len(rows), 'log rows')
