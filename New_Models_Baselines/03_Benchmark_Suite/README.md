# 03 · Multi-dimensional baseline benchmark (final)

## Draft branch revision to the WL stress model

The local binary classifier uses a correlation graph with 38 measurement nodes, three training-quantile labels and one WL refinement per crop. The SVM (`C=4`, balanced class weights) averages the cosine-normalized WL subtree kernel and a Laplacian measurement kernel. The second kernel compares training percentiles of the 38 values plus 11 quantities derived from the same photo: three RGB proportions, six coefficients of variation and two core-to-edge colour differences. Its similarity is `exp(-4 × mean absolute percentile difference)` with percentiles on [0, 1]. Both kernels have a unit training diagonal. The crop router and the specific-condition ranking model retain their original roles.

| Task, mean fold macro-F1 | Original WL | Revised WL + measurements |
|---|---:|---:|
| Pooled binary (Level 1) | 0.6969 | 0.7168 |
| Known crop, equal weight per crop (Level 2) | 0.7165 | 0.7229 |
| Automatically routed crop (Level 3) | 0.7064 | 0.7257 |

On the original five routed folds, accuracy rose from 82.81% to 85.14%, stressed recall from 87.86% to 91.24%, and balanced accuracy from 72.34% to 72.49%. Healthy recall fell from 56.82% to 53.74%. Among 4,387 predictions this is 124 additional stressed leaves detected and 22 additional false alarms. In known-crop testing cucumber rose from 0.8521 to 0.9060 macro-F1, while ridge gourd fell from 0.5478 to 0.5167.

**How the design was chosen.** Original fold 4 was excluded from every fit and score in the revision's four-fold development comparison. We compared subtree resolutions, graph depth and topology, supervised node bins, tree proximity, measurement similarities and SVM settings. We chose a shallow model that improved development macro-F1 and retained the original healthy recall there. We then froze the setting and checked fold 4 (877 leaves). Its macro-F1 improved from 0.7170 to 0.7549. The all-fold figures include the four development folds. Fold 4 is internal confirmation on OLID-I, which the project has used in earlier studies. Generalization to independently collected plants is still unmeasured.

**What WL contributes.** With the same router, folds, SVM settings and training examples, the two single-kernel controls scored 0.6712 (shallow WL alone) and 0.7324 (measurements alone). Adding WL to the measurement kernel raised healthy recall from 50.92% to 53.74% and lowered macro-F1 from 0.7324 to 0.7257. The archived tuned Extra Trees and frozen EfficientNet-B0 routed results are 0.7287 and 0.7392. The CNN benchmark uses its own image-embedding crop router.

**Resource use.** The eight serialized stress classifiers together averaged 5.01 MiB per outer fold, down from 10.40 MiB. The benchmark times feature-vector prediction with probability estimation disabled and excludes photo processing, routing, condition ranking and explanation. Both variants still construct a dense per-crop training Gram matrix with quadratic memory cost in that crop's training count.

**Photo changes.** Both routed classifiers were also tested on the cached perturbations of 240 original fold-0 photos. The revision scored higher on six of seven perturbations. Under a warm colour cast its macro-F1 fell to 0.5406 versus 0.5732 for original WL. Its mean paired drop from clean macro-F1 was 0.0862 versus 0.0749. Each perturbation is compared with the clean predictions for the same accepted leaves, with quality-control rejections reported separately.

Run these commands from this directory using the app backend requirements. The checked-in manifest in `Research_Paper/analysis` supplies the image-level folds. The runs used Python 3.12.0, NumPy 2.5.4, SciPy 1.16.2, pandas 2.2.3 and scikit-learn 1.8.0.

```bash
python run_wl_revision.py --development
python run_wl_revision.py
python run_wl_revision.py --unrouted
python run_wl_revision.py --robustness
```

The default run also regenerates the local app comparison. [Development scores](results/wl_revision_development.csv) · [Routed folds and resources](results/wl_revision_outer.csv) · [Per-leaf predictions](results/wl_revision_predictions.csv) · [Pooled and known-crop results](results/wl_revision_unrouted.csv) · [Perturbations and rejections](results/wl_revision_robustness.csv).

## Original benchmark

**Question:** Is the WL graph-kernel model better than standard models for plant-stress detection? We
measure it on several dimensions, not only accuracy: quality, speed, size, memory, data needs and
robustness to different photo conditions.

**Scope:** the main study is healthy-vs-stressed detection (Levels 1–3). The 57-class task
(crop × condition) is set aside as future work; its results are in the appendix of
[`results/REPORT_TABLES.md`](results/REPORT_TABLES.md).

All numbers are produced by the scripts in this folder and in `../04_CNN_Frozen_Features/`, and can be
regenerated (see **How to reproduce**).

---

## 1. Setup

| | |
|---|---|
| Data | 4,387 quality-accepted OLID-I leaves; 38 research measurements (`full_pipeline/olid_full_features.csv`); 84% of leaves are stressed |
| **Level 1** | healthy vs stressed, all crops in one model. Fixed folds `cv_fold_assignments.csv` (as NB21) |
| **Level 2** | healthy vs stressed, one model per crop. `StratifiedKFold(5, shuffle, 42)` per crop (as NB22) |
| **Level 3** | the full app pipeline: identify the crop (Random Forest, 300 trees), then use that crop's stress model. Fixed folds (as NB23) |
| WL + SVM | (a) the best research setting per task (NB21/NB22): slightly optimistic, because that setting was chosen on all data. (b) **tuned by nested CV** over graph type × WL depth inside each training fold: the fair version |
| Feature baselines | Logistic Regression, Linear SVM, RBF SVM, k-NN, Random Forest, Extra Trees, Gradient Boosting (HistGB), XGBoost, LightGBM, AdaBoost, Decision Tree, Naive Bayes, MLP; defaults, plus nested-CV **tuned** versions of the 5 strongest |
| CNN baselines | ImageNet-pretrained **MobileNetV3-Large** and **EfficientNet-B0**, frozen (not trained on OLID-I); logistic regression on their embeddings, regularisation tuned by nested CV (`../04_CNN_Frozen_Features`) |
| Fairness | same folds for every model; everything learned on the training fold only; one CPU core for training |

**Reproduction checks passed:** NB21 (Level 1 WL 0.6969), NB22 (e.g. cucumber WL 0.8521, RF 0.8038) and
NB23 (Level 3 WL with nested selection 0.6827 vs NB23's 0.6825) are reproduced. Re-extracting features
from the photos reproduces the research CSV exactly.

---

## 2. Why accuracy is not the headline metric

84% of the leaves are stressed, so a model that **always answers "stressed"** gets **83.8% accuracy**
while finding no healthy leaf at all. Several models reach high accuracy mostly this way (Random Forest
finds only 25% of healthy leaves). We therefore report **macro-F1** (main), **balanced accuracy**
and **specificity** (the share of healthy leaves found), and always show the "always stressed" line.

![Accuracy trap](figures/fig7_level1_accuracy_trap.png)

---

## 3. Results

### 3.1 Quality (macro-F1, healthy vs stressed)

| Model | Level 1 | Level 2 (mean of 8 crops) | **Level 3 (full pipeline)** |
|---|---|---|---|
| CNN EfficientNet-B0 (frozen) | 0.705 | 0.729 | **0.739** |
| CNN MobileNetV3-Large (frozen) | 0.702 | 0.734 | 0.734 |
| Extra Trees (tuned) | 0.722 | **0.742** | 0.729 |
| LightGBM | 0.720 | 0.714 | 0.726 |
| Gradient Boosting (tuned) | 0.728 | 0.727 | 0.721 |
| XGBoost | 0.723 | 0.721 | 0.719 |
| Gradient Boosting (default) | **0.729** | 0.717 | 0.714 |
| **WL + SVM, best research setting** | 0.697 | 0.716 | 0.706 |
| **WL + SVM, tuned (fair)** | n/a | 0.685 | 0.683 |
| Random Forest (default) | 0.653 | 0.662 | 0.682 |
| *Always "stressed" (reference)* | *0.456* | *0.455* | *0.456* |

![Level 3](figures/fig6_level3_macro_f1.png)

**Reading:** WL is clearly better than the "always stressed" reference and than several standard
models (default Random Forest, Logistic Regression, Linear SVM, k-NN, Naive Bayes, MLP, Decision Tree).
It is **not** the most accurate: tuned tree ensembles, gradient boosting and the frozen CNNs score
0.02–0.05 higher. With fair (nested) tuning WL loses a further ~0.03, which shows that part of its earlier
advantage came from choosing its setting on all the data.

**Crop identification (Level 3 Stage 1):** Random Forest on the 38 measurements identifies the crop
correctly for 91.9% of leaves; the CNN embedding does so for **99.2%**. For WL, a wrong crop costs only
about 0.003 macro-F1 (oracle 0.709 vs 0.706).

**Finding healthy leaves (specificity, Level 3):** WL 57%, tuned Extra Trees 55%, CNNs 62–65%, default
Random Forest 31%, LightGBM 47%. Linear models and the RBF SVM find more (69–75%) but miss more
stressed leaves.

### 3.2 Robustness: photos taken differently

240 test photos changed 7 ways (darker, brighter, warm cast, blur, heavy JPEG, noise, half resolution),
changes applied at full size, then each model's normal pipeline.

| Model | Clean F1 | Mean F1 change | 95% CI |
|---|---|---|---|
| CNN EfficientNet-B0 | 0.705 | **+0.001** | [−0.027, +0.027] |
| CNN MobileNetV3-Large | 0.689 | **−0.006** | [−0.038, +0.021] |
| **WL + SVM** | 0.680 | **−0.053** | [−0.098, −0.009] |
| k-NN | 0.662 | −0.070 | |
| Gradient Boosting (tuned) | 0.718 | −0.075 | |
| Extra Trees (tuned) | 0.755 | −0.100 | |
| XGBoost / LightGBM | 0.750 / 0.760 | −0.117 / −0.126 | |
| Random Forest | 0.717 | −0.176 | |

![Robustness](figures/fig5_robustness_f1_change.png)

* The frozen CNNs are essentially unaffected by these changes.
* **WL is the most robust feature-based model.** It loses significantly less than 9 feature models,
  including XGBoost, LightGBM, Random Forest, Extra Trees and tuned RBF SVM (bootstrap CIs exclude 0). It
  also loses less than the tuned tree ensembles (−0.075 to −0.113), but those differences are not significant.
* Why: WL uses only whether each measurement is Low, Medium or High, so small shifts in the measurements
  often do not change the graph. The feature pipeline as a whole is less robust than a CNN because the
  segmentation and colour measurements themselves move when lighting changes.

### 3.3 Speed, size, training

| | WL + SVM | Tree ensembles / boosting | Frozen CNN |
|---|---|---|---|
| **Photo → answer, one leaf** | ~1,080 ms | ~1,065–1,080 ms | **118–143 ms** |
| of which image processing | ~1,060 ms (full-size segmentation + 38 measurements) | same | 72–78 ms (read at 1/4 size, resize) |
| Model step | 18 ms | 2–8 ms | 39–70 ms (CNN) + < 1 ms |
| Model size (all 8 crop models) | 10.7 MB | 0.6–43 MB | 11.9–16.0 MB (one shared network + 8 small classifiers) |
| Training (8 crops, 1 core) | 1.1 s | 3–6 s (defaults), minutes (tuned) | seconds (classifier only; network not trained) |
| Peak training memory | 26 MB per crop, 392 MB for Level 1 (grows with n²) | < 6 MB | small (embeddings pre-computed) |

* End to end, the **frozen CNN is about 8× faster** than every feature-based model, because the research
  feature pipeline segments and measures the full 3,024 px photo. Speed is therefore not an argument for WL.
* WL's training memory grows with the square of the training set; this is its main scaling cost.

![Speed](figures/fig2_tradeoff_latency_level1.png)

### 3.4 Data efficiency

| Level 2, macro-F1 with … of the training data | 10% | 20% | 50% | 100% |
|---|---|---|---|---|
| CNN MobileNetV3 | **0.628** | **0.648** | **0.698** | 0.734 |
| RBF SVM | 0.622 | 0.632 | 0.657 | 0.679 |
| XGBoost | 0.586 | 0.619 | 0.674 | 0.721 |
| WL + SVM | 0.555 | 0.593 | 0.650 | 0.716 |

WL needs more data than most models; it is still improving at 100%, so more labelled leaves would help it.

![Learning curve](figures/fig4_learning_curve_level2.png)

### 3.5 Explainability (qualitative)

| Model | What it can show for one leaf |
|---|---|
| **WL + SVM** | the leaf's feature graph, each measurement's Low/Medium/High level, which neighbourhood patterns are rare in training, and which training leaves pulled the answer each way, in terms of named, physical leaf measurements (shown live in the app) |
| Trees / boosting | global feature importance; per-leaf needs an extra tool (e.g. SHAP) |
| Frozen CNN | 960–1,280 abstract numbers; per-leaf needs saliency maps (e.g. Grad-CAM) that highlight pixels, not named measurements |

---

## 4. Honest conclusion: the trade-off

| If the priority is… | Best choice (measured) |
|---|---|
| Highest quality on the full pipeline, robustness to photo conditions, speed | **Frozen CNN** (EfficientNet-B0 / MobileNetV3): Level 3 0.739, unaffected by photo changes, ~120–140 ms per photo, 99% crop identification |
| Highest quality using only the 38 interpretable measurements | Tuned Extra Trees / gradient boosting (Level 2 up to 0.742) |
| An explanation in terms of named leaf measurements and their relations, with the most stable feature-based behaviour | **WL + SVM**: competitive (Level 3 0.706), the most robust feature-based model, structural explanation built in |
| Very small model / very little data | Logistic Regression / RBF SVM |

**Claims supported by the data:**
* WL graph kernels on handcrafted leaf measurements are competitive for stress detection: clearly above
  the trivial baseline and above several standard models.
* WL is the most robust feature-based model to changes in photo conditions.
* WL gives a structural, measurement-level explanation that the stronger models do not.

**Claims NOT supported (do not write):** WL is the most accurate model; WL is the fastest; WL beats CNNs;
WL is data-efficient; "~90% accuracy" as evidence of quality (83.8% is reached by always answering
"stressed").

**Suggested positioning for the report:** the research question is *how graph construction influences WL
kernels for plant-stress classification*. The benchmark shows where that approach stands: an
interpretable, structure-aware method that is competitive with classical models and more robust than them,
while pretrained CNN features remain stronger on raw performance. That is a legitimate and defensible
research result.

---

## 5. Files

| File | What |
|---|---|
| `bench.py` | shared protocol: data, splits, model zoo (defaults + nested-CV tuned), metrics, timing, size |
| `run_benchmark.py` | Levels 1, 2 (and 57-class appendix): quality, speed, size; `--models round2/tuned` |
| `run_level3.py` | Level 3, full pipeline (crop identification → crop stress model) |
| `run_learning_curves.py` | data efficiency (10–100% of training data) |
| `run_robustness.py` | `extract` (changed photos → measurements) and `evaluate` |
| `make_report.py` | tables, statistics (Friedman, Wilcoxon + Holm, bootstrap CIs), figures |
| `run_remaining.sh` | runs Level 3, the CNN steps, learning curves, robustness and the report in order |
| `../04_CNN_Frozen_Features/cnn_pipeline.py` | CNN baseline: photo cache, embeddings, all evaluations |
| `results/` | per-fold CSVs, summaries, statistics, `REPORT_TABLES.md` |
| `figures/` | report figures (PNG, 200 dpi) |

## How to reproduce (from this folder)

```bash
python run_benchmark.py level2 level1          # defaults (~10 min)
python run_benchmark.py level2 level1 --models round2
python run_benchmark.py level2 level1 --models tuned   # ~40 min
python run_learning_curves.py
python run_robustness.py extract 240 3          # ~15 min, needs ~2 GB free memory
bash run_remaining.sh                           # Level 3 + CNN + curves + robustness + report (~1.5 h)
```

Windows note: LightGBM must be imported before scikit-learn (`bench.py` does this); otherwise it crashes
with an access violation.

## Not yet done

* **CNN fine-tuning** (option B, ~6–8 h overnight): would likely raise the CNN's scores further.
* **57-class task** for the new models and the CNN (future work).
* The research paper's 57-class WL number (NB14: 0.391) uses an older labelling; with the corrected
  `feature::level` labels it is 0.423 (see appendix).
