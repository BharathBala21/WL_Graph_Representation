# Level 3: Hierarchical Crop Stress Classification - Metrics Summary

This document summarizes the accuracy and F1 score metrics extracted from the `23_Level3_Hierarchical_Crop_Stress_Classification.ipynb` notebook.

## 1. Level-1 Baseline Metrics
* **Accuracy:** 0.8160 +/- 0.0160
* **Macro-F1:** 0.6969 +/- 0.0256
* **Weighted-F1:** 0.8251 +/- 0.0150
* **Balanced Acc:** 0.7219 +/- 0.0302

## 2. Overall Stage-1 Crop Identification Performance (Pooled OOF)
* **Overall Crop Accuracy:** 0.9184
* **Macro-F1:** 0.8950
* **Weighted-F1:** 0.9172

## 3. End-to-End Stress Classification (Level 3 vs Oracle vs Level 1)
* **L3 Real Macro-F1:** 0.6825 +/- 0.0132
* **L3 Oracle Macro-F1:** 0.6864 +/- 0.0141
* **L3 Real Macro-F1 delta (vs L1 Baseline):** -0.0144
* **L3 Oracle Macro-F1 delta (vs L1 Baseline):** -0.0105
* **Oracle - Real Macro-F1 Gap:** +0.0039

## 4. Outer Fold Metrics Summary

### Fold 0
* **Stage-1 Crop Acc:** RF: 0.9260 | SVM: 0.9226
* **End-to-End Macro-F1:** Real: 0.6907 | Oracle: 0.6960

### Fold 1
* **Stage-1 Crop Acc:** RF: 0.9123 | SVM: 0.9032
* **End-to-End Macro-F1:** Real: 0.6851 | Oracle: 0.6960

### Fold 2
* **Stage-1 Crop Acc:** RF: 0.9282 | SVM: 0.9145
* **End-to-End Macro-F1:** Real: 0.6620 | Oracle: 0.6702

### Fold 3
* **Stage-1 Crop Acc:** RF: 0.9247 | SVM: 0.9247
* **End-to-End Macro-F1:** Real: 0.6960 | Oracle: 0.6981

### Fold 4
* **Stage-1 Crop Acc:** RF: 0.9008 | SVM: 0.9247
* **End-to-End Macro-F1:** Real: 0.6788 | Oracle: 0.6719

## 5. Crop-Specific Routing Notes (Example: Tomato)
* **Tomato Routing Accuracy:** 0.9828 (98.28%)
* **Tomato End-to-End Stress Macro-F1:** 0.7413
* **Tomato Oracle Stress Macro-F1:** 0.7429
* **Tomato Correctly-Routed Stress Macro-F1:** 0.7433

## Key Takeaways
- Measured routing accuracy is **91.8%**. 
- The gap between Oracle (0.6864) and Real (0.6825) routing is only **0.0039 Macro-F1 points**, indicating that Stage-1 crop misclassification is a measurable but small bottleneck.
- The **Stress Macro-F1 = 0.6914** for correctly routed samples.
