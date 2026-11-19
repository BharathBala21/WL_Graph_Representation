# How the revised WL model compares

Weisfeiler-Lehman graph kernels for binary stress detection in 4,387 quality-accepted OLID-I leaf images from eight crops.

### What changed on the held-out leaves

Across five held-out folds with automatic crop identification, the revised classifier detected 124 more stressed leaves and flagged 22 more healthy leaves. Macro-F1 improved on every fold. EfficientNet-B0 retains the highest average macro-F1 among the benchmarked pipelines.

| Routed metric | Original WL | Revised WL | Change |
| --- | --- | --- | --- |
| Macro-F1 | 0.7064 | 0.7257 | +0.0192 |
| Accuracy | 82.81% | 85.14% | +2.33 pp |
| Balanced accuracy | 72.34% | 72.49% | +0.15 pp |
| Healthy leaves recognized | 56.82% | 53.74% | -3.08 pp |
| Stressed leaves detected | 87.86% | 91.24% | +3.37 pp |

### What changed in the classifier

Both systems identify the crop with the same Random Forest. The original SVM compares discrete WL subtree counts using a graph type and depth previously chosen for each crop. The revision uses a correlation graph of 38 measurements and one WL refinement for every crop. Its SVM averages the WL similarity and a continuous kernel over the measurements and 11 derived color and region contrasts.

### How to read the scores

Macro-F1 gives the healthy and stressed classes equal weight. Accuracy counts all correct predictions. Because 3,674 of the 4,387 leaves are stressed, predicting stress every time already yields 83.75% accuracy. Healthy recall measures how often a healthy leaf avoids a false alarm. Changes marked pp are percentage-point differences.

### What the comparison establishes

Four original folds informed model development. After fixing the design, the team evaluated it on the reserved fifth fold of 877 leaves. Its macro-F1 rose from 0.7170 to 0.7549. The five-fold averages also include the development folds. All evidence here comes from OLID-I. Performance on independent field photos remains unknown.

Data: wl_revision_outer.csv, wl_revision_predictions.csv, summary_level3.csv.


## All routed classifiers

Level 3. Each pipeline identifies the crop before classifying the leaf. Rows are ordered by five-fold mean macro-F1. The two single-kernel controls use the revised SVM settings.

| Classifier | Macro-F1 | Accuracy | Bal. acc. | Healthy recall | Stress recall |
| --- | --- | --- | --- | --- | --- |
| EfficientNet-B0 + LR | 0.7392 | 84.84% | 75.64% | 62.01% | 89.28% |
| MobileNetV3-Large + LR | 0.7339 | 83.93% | 76.11% | 64.52% | 87.70% |
| Measurement kernel only (C = 4) | 0.7324 | 86.23% | 72.00% | 50.92% | 93.09% |
| Extra Trees (tuned) | 0.7287 | 85.12% | 73.03% | 55.13% | 90.94% |
| LightGBM | 0.7260 | 86.53% | 70.55% | 46.86% | 94.23% |
| Revised WL + measurements | 0.7257 | 85.14% | 72.49% | 53.74% | 91.24% |
| Histogram gradient boosting (tuned) | 0.7213 | 84.57% | 72.54% | 54.72% | 90.36% |
| XGBoost | 0.7186 | 85.80% | 70.33% | 47.42% | 93.25% |
| RBF SVM (tuned) | 0.7156 | 82.88% | 74.08% | 61.03% | 87.13% |
| Random Forest (tuned) | 0.7144 | 85.80% | 69.72% | 45.90% | 93.55% |
| Histogram gradient boosting | 0.7139 | 85.75% | 69.69% | 45.88% | 93.49% |
| Original WL | 0.7064 | 82.81% | 72.34% | 56.82% | 87.86% |
| Extra Trees | 0.6998 | 87.74% | 65.84% | 33.39% | 98.29% |
| AdaBoost | 0.6938 | 85.87% | 66.82% | 38.59% | 95.05% |
| RBF SVM | 0.6828 | 78.09% | 74.49% | 69.15% | 79.83% |
| Original WL (nested tuning) | 0.6827 | 81.17% | 70.18% | 53.89% | 86.47% |
| Random Forest | 0.6818 | 86.98% | 64.54% | 31.29% | 97.80% |
| k-NN (k=5) | 0.6775 | 85.41% | 65.13% | 35.08% | 95.18% |
| Shallow WL only (C = 4) | 0.6712 | 78.37% | 71.44% | 61.17% | 81.71% |
| Logistic Regression (tuned) | 0.6707 | 76.50% | 74.16% | 70.69% | 77.63% |
| Decision Tree | 0.6679 | 81.76% | 67.03% | 45.19% | 88.87% |
| Logistic Regression | 0.6662 | 75.34% | 75.06% | 74.64% | 75.48% |
| Linear SVM | 0.6568 | 74.17% | 74.59% | 75.21% | 73.98% |
| MLP (128-64) | 0.6096 | 84.82% | 58.95% | 20.62% | 97.28% |
| Naive Bayes (Gaussian) | 0.6043 | 76.27% | 61.71% | 40.13% | 83.29% |
| Always predicts stressed | 0.4558 | 83.75% | 50.00% | 0.00% | 100.00% |

### Where the revised WL model sits

Its macro-F1 is 0.0030 below tuned Extra Trees and 0.0135 below EfficientNet-B0. LightGBM is only 0.0003 higher to four decimal places. These are descriptive gaps. The frozen CNNs use logistic regression on image embeddings and identify crops at about 99.2% accuracy; the feature-based router reaches 91.9%.

Data: summary_level3.csv and wl_revision_outer.csv. Bal. acc. averages the two class recalls.


## Performance across the three tasks

Mean macro-F1. Level 1 pools all crops into one classifier. Level 2 gives the true crop to the classifier and weights the eight crops equally. Level 3 includes crop-identification errors.

| Classifier | Level 1 / pooled | Level 2 / known crop | Level 3 / routed |
| --- | --- | --- | --- |
| EfficientNet-B0 + LR | 0.7053 | 0.7289 | 0.7392 |
| MobileNetV3-Large + LR | 0.7017 | 0.7337 | 0.7339 |
| Extra Trees (tuned) | 0.7217 | 0.7421 | 0.7287 |
| LightGBM | 0.7201 | 0.7138 | 0.7260 |
| Revised WL + measurements | 0.7168 | 0.7229 | 0.7257 |
| Histogram gradient boosting (tuned) | 0.7283 | 0.7274 | 0.7213 |
| XGBoost | 0.7233 | 0.7211 | 0.7186 |
| RBF SVM (tuned) | 0.7066 | 0.7175 | 0.7156 |
| Random Forest (tuned) | 0.7101 | 0.7071 | 0.7144 |
| Histogram gradient boosting | 0.7288 | 0.7167 | 0.7139 |
| Original WL | 0.6969 | 0.7165 | 0.7064 |
| Extra Trees | 0.6803 | 0.6833 | 0.6998 |
| AdaBoost | 0.6193 | 0.6784 | 0.6938 |
| RBF SVM | 0.6642 | 0.6787 | 0.6828 |
| Original WL (nested tuning) | n/a | 0.6852 | 0.6827 |
| Random Forest | 0.6532 | 0.6624 | 0.6818 |
| k-NN (k=5) | 0.6776 | 0.6298 | 0.6775 |
| Logistic Regression (tuned) | 0.6152 | 0.6672 | 0.6707 |
| Decision Tree | 0.6523 | 0.6655 | 0.6679 |
| Logistic Regression | 0.6154 | 0.6539 | 0.6662 |
| Linear SVM | 0.6123 | 0.6528 | 0.6568 |
| MLP (128-64) | 0.6622 | 0.5349 | 0.6096 |
| Naive Bayes (Gaussian) | 0.5969 | 0.5614 | 0.6043 |
| Always predicts stressed | 0.4558 | 0.4553 | 0.4558 |

### Which model leads depends on the task

Histogram gradient boosting leads on pooled leaves (0.7288), tuned Extra Trees leads with a known crop (0.7421), and EfficientNet-B0 leads with automatic routing (0.7392). Original WL refers to the fixed research setting selected in earlier experiments. The nested-tuning row selects graph settings inside training folds. The archive has no pooled score for this variant.

Data: summary_level1.csv, summary_level2.csv, summary_level3.csv, wl_revision_unrouted.csv and wl_revision_outer.csv.


## Which leaves changed the result?

Per-crop macro-F1 uses the known-crop protocol. The fold comparison below it uses automatic routing. The two protocols answer different questions.

| Crop | Original | Revised | Change | Best archived model | F1 |
| --- | --- | --- | --- | --- | --- |
| Ash gourd | 0.6462 | 0.6390 | -0.0072 | EfficientNet-B0 + LR | 0.703 |
| Bitter gourd | 0.6624 | 0.6528 | -0.0096 | EfficientNet-B0 + LR | 0.753 |
| Bottle gourd | 0.7189 | 0.7575 | +0.0386 | Extra Trees (tuned) | 0.830 |
| Cucumber | 0.8521 | 0.9060 | +0.0539 | Extra Trees | 0.925 |
| Eggplant | 0.7504 | 0.7497 | -0.0007 | Original WL | 0.750 |
| Ridge gourd | 0.5478 | 0.5167 | -0.0311 | EfficientNet-B0 + LR | 0.720 |
| Snake gourd | 0.7775 | 0.7740 | -0.0035 | Extra Trees (tuned) | 0.817 |
| Tomato | 0.7762 | 0.7875 | +0.0113 | Extra Trees | 0.803 |

Cucumber and bottle gourd gained the most. Ridge gourd declined from 0.5478 to 0.5167. The equal-weight mean across crops improved from 0.7165 to 0.7229. Five of the eight crops scored lower under this protocol. The archived per-crop leaders are reported to three decimals.

### Paired results on the original routed folds

| Original fold ID | Leaves | Original WL | Revised WL | Change |
| --- | --- | --- | --- | --- |
| 0 | 878 | 0.7165 | 0.7241 | +0.0076 |
| 1 | 878 | 0.6981 | 0.7127 | +0.0146 |
| 2 | 877 | 0.6871 | 0.7020 | +0.0150 |
| 3 | 877 | 0.7135 | 0.7346 | +0.0211 |
| 4 (reserved) | 877 | 0.7170 | 0.7549 | +0.0379 |

### Error counts across all five folds

| Classifier | Missed stress | False alarms | Stress detected | Healthy correct |
| --- | --- | --- | --- | --- |
| Original WL | 446 | 308 | 3228 | 405 |
| Revised WL + measurements | 322 | 330 | 3352 | 383 |

Data: wl_revision_unrouted.csv, summary_level2_per_crop.csv, wl_revision_outer.csv and wl_revision_predictions.csv.


## Model cost and photo robustness

Resource measurements cover the eight crop-specific stress classifiers. The photo checks reroute each altered image and measure the complete binary classification pipeline.

### Stage 2 resource measurements

| Mean across folds | Original WL | Revised WL |
| --- | --- | --- |
| Serialized classifiers (MiB) | 10.40 | 5.01 |
| Total fit time (seconds) | 1.57 | 0.97 |
| Single-leaf model latency (ms) | 7.65 | 6.58 |

The revision roughly halves stress-classifier storage. These timings start with extracted features and omit routing, photo processing, condition ranking and explanations. The archived end-to-end estimate is about 1,077 ms/photo for original WL and 143 ms for EfficientNet-B0. A corresponding end-to-end measurement for revised WL is unavailable. Both WL variants still require a dense training kernel with quadratic memory in the per-crop sample count.

### Changes to 240 held-out fold-0 photos

| Photo condition | Accepted | Original F1 | Revised F1 | Change |
| --- | --- | --- | --- | --- |
| Original photo | 240 | 0.6949 | 0.7348 | +0.0399 |
| Blur | 240 | 0.6140 | 0.6250 | +0.0110 |
| Brighter | 225 | 0.6157 | 0.6572 | +0.0414 |
| Darker | 235 | 0.6430 | 0.6953 | +0.0522 |
| Half resolution | 240 | 0.6497 | 0.6955 | +0.0458 |
| Heavy JPEG | 240 | 0.6479 | 0.6919 | +0.0439 |
| Sensor noise | 237 | 0.6143 | 0.6195 | +0.0052 |
| Warm color cast | 228 | 0.5732 | 0.5406 | -0.0326 |

The revision scored higher on six of seven alterations. The exception was warm color cast (0.5406 versus 0.5732). Across alterations its average loss from the matched clean photos was larger, 0.0862 versus 0.0749. Quality control rejected up to 15 photos under a single alteration.

### What the WL component contributes

At the same SVM settings, the continuous kernel alone scored 0.7324 macro-F1. Adding WL raised healthy recall from 50.92% to 53.74% and lowered stress recall from 93.09% to 91.24%. This supports a trade-off in which leaves are caught, without establishing a macro-F1 advantage over the continuous control.

Data: wl_revision_outer.csv, wl_revision_robustness.csv and summary_level3.csv.


## Appendix / exact-condition classification

The archived 57-class task predicts a crop-condition combination. The revised binary stress classifier has no 57-class evaluation. These scores describe the earlier benchmark.

| Archived classifier | Macro-F1 | Accuracy | Balanced acc. |
| --- | --- | --- | --- |
| Extra Trees | 0.5700 | 59.38% | 56.78% |
| Random Forest | 0.5535 | 58.74% | 54.89% |
| LightGBM | 0.5286 | 56.48% | 53.08% |
| Histogram gradient boosting | 0.5255 | 55.60% | 52.96% |
| XGBoost | 0.5180 | 55.25% | 52.41% |
| Linear SVM | 0.4720 | 48.55% | 52.46% |
| RBF SVM | 0.4307 | 45.54% | 49.08% |
| Original WL | 0.4233 | 44.15% | 47.23% |
| Logistic Regression | 0.4196 | 44.54% | 48.11% |
| MLP (128-64) | 0.4178 | 52.02% | 41.89% |
| k-NN (k=5) | 0.4102 | 48.76% | 41.76% |
| Decision Tree | 0.3799 | 40.23% | 39.59% |
| Naive Bayes (Gaussian) | 0.1013 | 20.95% | 13.59% |
| AdaBoost | 0.0691 | 22.95% | 9.67% |

### Evaluation record

All tables use the project snapshot fd1b6be on the draftCompletion branch. Tables on pages 1-5 cover binary stress. This report summarizes the saved fold results and predictions. The models were last evaluated at that project snapshot.

Level 1 and Level 3 use the original five image-level folds. Level 2 uses five stratified binary-label folds within each crop. Means can differ from metrics computed after pooling every prediction. The experiments provide internal evidence and do not establish that small rank differences will persist on new plants.

### Where to audit the numbers

Open New_Models_Baselines/03_Benchmark_Suite/results/ in the repository. The summary_level1, summary_level2, summary_level3 and summary_57-class CSVs contain archived model comparisons. Files beginning with wl_revision_ contain the revised runs. The predictions CSV has one row for every accepted image. run_wl_revision.py and the benchmark README describe the splits, settings and reproduction commands.

Data: summary_57-class.csv for the appendix. Fold manifest: Research_Paper/analysis/cv_fold_assignments.csv.
