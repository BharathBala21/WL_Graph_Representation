# Evaluating Graph Representations using Weisfeiler–Lehman Graph Kernels for Plant-Stress Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()

> **Project Code:** `WL-LeafKernel`  
> **Investigation:** *How Does Graph Construction Influence the Effectiveness of Weisfeiler–Lehman Graph Kernels for Fine-Grained Plant-Stress Classification from Leaf Images?*  
> **Academic Affiliation:** Amrita Vishwa Vidyapeetham, School of Computing, Coimbatore — Final Year Capstone Project (Team 49)

---

## Table of Contents

- [Overview & Research Motivation](#overview--research-motivation)
- [Key Contributions](#key-contributions)
- [Repository Structure](#repository-structure)
- [Dataset: OLID-I Benchmark](#dataset-olid-i-benchmark)
- [Feature Engineering Pipeline (38 Handcrafted Features)](#feature-engineering-pipeline-38-handcrafted-features)
- [Graph Construction Topologies & WL Kernel](#graph-construction-topologies--wl-kernel)
- [Task Decomposition & Hierarchical Modeling](#task-decomposition--hierarchical-modeling)
- [Experimental Results & Benchmarks](#experimental-results--benchmarks)
- [PlantCare AI Web Application](#plantcare-ai-web-application)
- [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
- [Academic Presentation & Literature](#academic-presentation--literature)
- [Research Team & Supervision](#research-team--supervision)
- [Citation](#citation)

---

## Overview & Research Motivation

Accurate, automated plant-stress detection from leaf imagery is vital for precision agriculture, food security, and timely crop protection. While deep learning architectures require extensive labelled data and act as uninterpretable black boxes, classical machine learning approaches often collapse rich morphological, color, and texture interactions into flat tabular vectors, ignoring topological correlations.

The **Weisfeiler–Lehman (WL) Graph Kernel** is renowned for its expressive power in bioinformatics and chemoinformatics. However, applying WL graph kernels to computer vision and plant phenotyping requires transforming unstructured image measurements into meaningful graphs. **The choice of graph topology directly dictates which neighborhood labels the WL kernel aggregates.**

This repository presents an end-to-end empirical study investigating:
1. **Graph Construction Strategies:** Systematically evaluating 5 engineered graph topologies (Correlation, Feature-Similarity kNN, Feature-Family, Spatial/Regional, and Hybrid).
2. **WL Subtree Heights:** Analyzing iteration depths $h \in \{0, 1, 2, 3\}$ to isolate structural topology gains from raw feature histogram baselines.
3. **Leakage-Safe Hierarchical Architecture:** Implementing a 3-level task decomposition (Binary Stress $\to$ Crop-Specific Multi-Class $\to$ End-to-End Hierarchical Routing) using nested 5-fold cross-validation.
4. **Interactive Translation:** Translating research outcomes into **PlantCare AI**, an interactive diagnostic web portal.

---

## Key Contributions

- **Systematic Topology Benchmarking:** First comprehensive ablation comparing five engineered feature-graph construction strategies for WL kernels on plant leaf phenotyping.
- **Topological vs. Tabular Diagnosis:** Direct benchmarking of WL kernel SVMs against Random Forest, Support Vector Machines (RBF), Logistic Regression, and Extra Trees under strictly identical, leakage-safe nested CV.
- **Unsupervised Segmentation & Feature Extraction:** Automated ExG (Excess Green) + Otsu thresholding across 4,749 leaf images with a rigorous 3-tier quality audit (`ACCEPT`, `FAILED`, `SUSPICIOUS`).
- **Hierarchical Routing Architecture:** High-accuracy Stage-1 crop routing (91.84% accuracy) delivering seamless routing to specialized Stage-2 crop-stress models.
- **Open Interactive Research Application:** A production-ready web platform (PlantCare AI) featuring interactive leaf analytics, out-of-fold prediction exploration, and stress taxonomy.

---

## Repository Structure

```plaintext
WL_Graph_Representation/
├── Application/                     # PlantCare AI Web Application (React + Vite + Tailwind)
│   ├── public/                      # Static assets, team photos, sample leaf thumbnails
│   ├── src/                         # React components, pages (Home, Dashboard, Analyzer, etc.)
│   │   ├── plantcare/               # PlantCare AI diagnostic suite, crop cards, logs
│   │   └── lib/content.js           # Project metadata, findings, and documentation copy
│   ├── scripts/                     # Python scripts to compile research data for frontend
│   │   └── build_plantcare_data.py  # Generates research.ts & out-of-fold predictions
│   ├── package.json                 # Node dependencies
│   ├── tailwind.config.js           # Tailwind CSS configuration
│   ├── vercel.json                  # Vercel deployment rewrite rules
│   └── README.md                    # Application-specific documentation
│
├── full_pipeline/                   # Core Sequential Research Pipeline (Jupyter Notebooks)
│   ├── 08_OLID_Full_Feature_Extraction.ipynb        # ExG + Otsu segmentation & 38 feature extraction
│   ├── 09_OLID_Full_Feature_Validation.ipynb        # Feature distributions and data integrity checks
│   ├── 10_OLID_Segmentation_Bias_Audit.ipynb        # Audit of segmentation quality & background bias
│   ├── 11_OLID_Final_Graph_Construction.ipynb       # 5 graph topology builders (nodes = feature channels)
│   ├── 12_OLID_WL_Subtree_Engine.ipynb              # Weisfeiler-Lehman subtree kernel engine
│   ├── 13_OLID_WL_Kernel_Construction.ipynb         # Gram matrix generation across heights h=0..3
│   ├── 14_OLID_WL_SVM_Classification.ipynb          # SVM classification over WL kernel matrices
│   ├── 15_OLID_WL_Representation_Diagnosis.ipynb   # Kernel matrix rank and similarity diagnosis
│   ├── 16_OLID_Attribute_Aware_Graph_Representation.ipynb # Continuous/attribute-aware graph nodes
│   ├── 17_OLID_Continuous_Graph_SVM.ipynb           # SVM evaluation with continuous graph kernels
│   ├── 18_OLID_Classical_Baselines.ipynb            # Tabular baselines: Random Forest & RBF SVM
│   ├── olid_full_features.csv                       # Extracted 38-feature matrix for 4,749 images
│   ├── leaf_segmentation_algorithm_rationale.md     # Rationale and mathematical basis for ExG + Otsu
│   │
│   ├── decomposition/               # Hierarchical 3-Level Task Decomposition
│   │   ├── 19_OLID_Task_Decomposition.ipynb         # Formulation of hierarchical stages
│   │   ├── 21_Level1_Binary_Stress_Detection.ipynb  # Level 1: Global binary stress classification
│   │   ├── 22_Level2_Crop_Specific_Stress_Detection.ipynb # Level 2: Crop-specific stress models
│   │   ├── 23_Level3_Hierarchical_Crop_Stress_Classification.ipynb # Level 3: Hierarchical pipeline
│   │   ├── 24_High_Accuracy_Graph_Comparison.ipynb # Comprehensive graph benchmark
│   │   ├── condition_to_stress_mapping.csv          # 20 conditions to binary stress mapping
│   │   └── ACCURACY_METRICS.md                      # Detailed level-wise metric tables
│   │
│   └── feature_analysis/            # Feature Importance & Ablation Studies
│       └── 20_OLID_Feature_Importance_Ablation.ipynb # Trait ablation (Morphology vs Color vs Texture)
│
├── New_Models_Baselines/            # Supplementary Classical ML Baselines & Benchmarks
│   ├── 01_Logistic_Regression/      # Regularized Logistic Regression fold results & latency
│   ├── 02_Extra_Trees/              # Extra Trees ensemble benchmark
│   ├── figures/                     # Latency, training time, and macro-F1 comparison figures
│   └── run_lr_experiment.py         # Automated runner for logistic regression experiment
│
├── ppt/                             # Academic Presentation Materials
│   └── presentation.tex             # LaTeX Beamer presentation with TikZ diagrams
│
├── final_year_project_pdfs/         # Foundational Literature & Reference Papers
│   ├── NIPS-2009-fast-subtree-kernels-on-graphs-Paper.pdf
│   ├── OLDI_DATASET_OLID I an open leaf image dataset for plant stress recognition.pdf
│   ├── SRIRAM_Perceptual graph kernels for image-derived plant trait interaction analysis.pdf
│   └── shervashidze11a.pdf          # Shervashidze et al. (JMLR 2011)
│
├── build_and_run_lr.py              # Root automation script for LR baseline execution
├── build_nb22.py                    # Generator script for Level 2 experiment notebook
├── build_nb23.py                    # Generator script for Level 3 experiment notebook
├── requirements.txt                 # Complete Python dependencies
└── README.md                        # Project root documentation (this file)
```

---

## Dataset: OLID-I Benchmark

The research utilizes **OLID-I (Open Leaf Image Dataset I)**, comprising field and greenhouse leaf imagery across 8 economically significant crops and 20 condition states (57 distinct crop-condition classes).

### Crop Distribution & Quality Control

| Crop | Botanical Name | Raw Images | QC Accepted | Healthy | Stressed | Primary Stresses |
|---|---|---|---|---|---|---|
| **Ash gourd** | *Benincasa hispida* | 997 | 996 | 83 | 913 | N, K, N+K, N+Mg, K+Mg, Powdery Mildew |
| **Tomato** | *Solanum lycopersicum* | 798 | 580 | 184 | 396 | N, K, N+K, Mite, Jassid+Mite, Leaf Miner |
| **Bitter gourd** | *Momordica charantia* | 785 | 742 | 173 | 569 | N, K, N+K, N+Mg, K+Mg, Jassid, Downy Mildew, Leaf Spot |
| **Eggplant** | *Solanum melongena* | 685 | 662 | 91 | 571 | N, K, N+K, Mite, Jassid, Epilachna Beetle, Flea Beetle |
| **Snake gourd** | *Trichosanthes cucumerina* | 456 | 452 | 58 | 394 | N, K, N+K, Leaf Spot |
| **Ridge gourd** | *Luffa acutangula* | 434 | 383 | 59 | 324 | N, N+Mg, Pumpkin Leaf-Eating Insect, Caterpillar |
| **Bottle gourd** | *Lagenaria siceraria* | 345 | 332 | 31 | 301 | N, K, N+K, N+Mg, Jassid, Downy Mildew, Leaf Spot |
| **Cucumber** | *Cucumis sativus* | 249 | 240 | 34 | 206 | N, K, N+K |
| **Total** | — | **4,749** | **4,387** | **713** | **3,674** | **57 Fine-Grained Classes** |

### Segmentation Quality Control
Each leaf is segmented using an unsupervised **Excess Green (ExG) + Otsu** pipeline:
$$\mathrm{ExG} = 2 \cdot \frac{G}{R+G+B} - \frac{R}{R+G+B} - \frac{B}{R+G+B}$$
- Masks with foreground area $< 1\%$ are flagged as `FAILED` (349 images).
- Masks covering $> 98\%$ of the frame are flagged as `SUSPICIOUS` (13 images).
- Only the **4,387 `ACCEPT`** images enter downstream feature extraction.

---

## Feature Engineering Pipeline (38 Handcrafted Features)

Rather than feeding raw pixels into high-dimensional black-box encoders, the pipeline extracts **38 interpretable descriptors** grouped into four computational families:

```plaintext
38 Handcrafted Feature Channels (Graph Nodes)
├── Morphology & Geometry (10)
│   └── Area, Perimeter, Aspect Ratio, Circularity, Extent, Solidity, Convex Area,
│       Major Axis Length, Minor Axis Length, Eccentricity
├── Color Statistics (18)
│   ├── RGB: Mean & Std (R, G, B)
│   ├── HSV: Mean & Std (H, S, V)
│   └── CIELAB: Mean & Std (L*, a*, b*)
├── Color Chromaticity Ratios (2)
│   └── Green Ratio, Yellow Ratio
├── Texture via GLCM (4)
│   └── GLCM Contrast, Homogeneity, Energy, Correlation
└── Spatial & Regional Mask Descriptors (4)
    └── Core Mean Hue, Core Yellow Ratio, Edge Mean Hue, Edge Yellow Ratio
```

All continuous features are converted to discrete node labels using training-fold quantile binning (Low, Medium, High) to guarantee strict zero-leakage cross-validation.

---

## Graph Construction Topologies & WL Kernel

Each leaf image is represented as a graph where **nodes correspond to the 38 feature channels**, and edges define relational paths between traits.

### 1. Evaluated Graph Topologies

```
     [Star Topology]                 [Feature-Family Topology]            [Correlation / kNN]
         (Plant)                          (Morphology)                         (Area)
       /    |    \                        /     |    \                        /      \
    (LA)  (LC)   (VP)                 (Area) (Perim) (Circ)               (GLCM-C)  (Mean-G)
 (Hub-and-spoke connectivity)      (Interactions within families)      (Empirical feature covariance)
```

1. **Correlation Graph:** Connects pairs of feature channels whose absolute Pearson correlation $|r| > \tau$ on the training fold ($\tau = 0.70$).
2. **Feature-Similarity kNN:** Connects each feature node to its $k=3$ nearest neighbors based on feature value similarity.
3. **Feature-Family Graph:** Retains dense intra-family edges among domain-specific clusters (Morphology, Color, Texture, Spatial) without inter-family cross-talk.
4. **Spatial / Regional Graph:** Connects spatially co-located descriptors and boundary/core features.
5. **Hybrid Topology:** Unions domain-specific Feature-Family structures with empirically verified correlation edges.
6. **Star vs. Fully Connected (Ablation):** Compares a centralized hub-and-spoke root vs. a fully connected clique to test information diffusion limits.

### 2. Weisfeiler–Lehman Subtree Kernel

The WL subtree kernel iteratively updates the label of node $v$ by hashing its current label with the multiset of its neighbors' labels:
$$l_i(v) = f\left(l_{i-1}(v),\ \{\!\{\, l_{i-1}(u) : u \in \mathcal{N}(v) \,\}\!\}\right)$$

For iterations $h \in \{0, 1, 2, 3\}$, the label frequency vectors across all iterations are concatenated to form the graph feature vector $\phi_{\text{WL}}(G)$:
$$K_{\text{WL}}(G, G') = \langle \phi_{\text{WL}}(G), \phi_{\text{WL}}(G') \rangle$$

The resulting normalized Gram matrix is passed to a C-Support Vector Machine (SVM).

---

## Task Decomposition & Hierarchical Modeling

To tackle extreme class imbalance and fine-grained visual confusion, the 57-class problem is structured into a 3-level hierarchical system:

```mermaid
flowchart TD
    IMG["Input Leaf Image (38 Handcrafted Features)"] --> L1["Level 1: Binary Stress Detector<br/>(Healthy Control vs. Stressed)"]
    IMG --> S1["Stage-1 Crop Router<br/>(Identifies 1 of 8 Crops)"]
    
    S1 -->|Ash Gourd (93.8%)| L2_AG["Stage-2 Ash Gourd Classifier<br/>(6 Stresses + Healthy)"]
    S1 -->|Tomato (98.3%)| L2_TOM["Stage-2 Tomato Classifier<br/>(6 Stresses + Healthy)"]
    S1 -->|Bitter Gourd (99.1%)| L2_BG["Stage-2 Bitter Gourd Classifier<br/>(8 Stresses + Healthy)"]
    S1 -->|Eggplant (95.9%)| L2_EP["Stage-2 Eggplant Classifier<br/>(8 Stresses + Healthy)"]
    S1 -->|Other Crops (82-86%)| L2_OTH["Crop-Specific Specialized Classifiers"]

    L2_AG --> OUT["Final Fine-Grained Diagnosis & Stress Class"]
    L2_TOM --> OUT
    L2_BG --> OUT
    L2_EP --> OUT
    L2_OTH --> OUT
```

- **Level 1 (Binary Stress):** Classifies whether a leaf exhibits any stress (3,674 images) or is healthy (713 images).
- **Level 2 (Crop-Specific Models):** Independent multi-class classifiers tuned individually per crop using optimal graph topologies.
- **Level 3 (Hierarchical End-to-End):** Samples are routed by an automated Stage-1 crop identification model to the appropriate Level-2 specialist.

---

## Experimental Results & Benchmarks

All models are evaluated using a strictly leakage-free **nested 5-fold cross-validation** scheme (outer folds evaluate generalization; inner folds tune topology thresholds, WL heights, and SVM hyperparameters).

### 1. Hierarchical Routing Performance (Level 3)

| Evaluation Stage | Metric | Score / Margin | Notes |
|---|---|---|---|
| **Stage-1 Crop Identification** | Accuracy | **91.84%** | Pooled out-of-fold accuracy across all 8 crops |
| Stage-1 Crop Identification | Macro-F1 | **0.8950** | Balanced across varying crop sample sizes |
| Stage-1 Crop Identification | Weighted-F1 | **0.9172** | Reliable routing foundation |
| **End-to-End Stress (Real Routing)** | Macro-F1 | **0.6825 ± 0.0132** | Realistic pipeline with Stage-1 routing errors |
| **End-to-End Stress (Oracle Routing)** | Macro-F1 | **0.6864 ± 0.0141** | Theoretical upper bound with perfect crop ID |
| **Routing Bottleneck Gap** | $\Delta$ Macro-F1 | **-0.0039** | Stage-1 routing causes less than 0.4% F1 drop |

### 2. Crop-Specific Routing & Stress Classification (Level 2 Highlights)

| Crop | Routing Accuracy | Best Graph Topology | WL Height ($h$) | End-to-End Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| **Bitter gourd** | 99.06% | Feature-Similarity kNN | $h=3$ | 71.97% | 0.6624 |
| **Tomato** | 98.28% | Correlation Graph | $h=2$ | 77.59% | 0.7762 |
| **Eggplant** | 95.92% | Hybrid Graph | $h=3$ | 85.65% | 0.7504 |
| **Ash gourd** | 93.78% | Feature-Family | $h=2$ | 82.03% | 0.6462 |
| **Snake gourd** | 86.28% | Hybrid Graph | $h=3$ | 88.94% | 0.7775 |
| **Ridge gourd** | 84.33% | Hybrid Graph | $h=1$ | 75.98% | 0.5478 |
| **Bottle gourd** | 82.23% | Hybrid Graph | $h=3$ | 87.65% | 0.7189 |
| **Cucumber** | 70.42% | Feature-Family | $h=2$ | 86.67% | 0.8521 |

### 3. Model Benchmark on Tomato Binary Task (Nested 5-Fold CV)

| Model Architecture | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Healthy Recall (%) |
|---|---|---|---|---|
| **Nested WL Graph Kernel (Ours)** | 78.97 ± 6.38 | 75.33 ± 7.78 | 75.38 ± 7.64 | **65.32%** |
| Random Forest Baseline | **83.45 ± 4.58** | 76.56 ± 6.06 | **78.66 ± 6.37** | 57.67% |
| RBF Support Vector Machine | 82.59 ± 5.04 | **77.24 ± 6.08** | 78.57 ± 6.33 | 60.12% |
| Extra Trees Classifier | 82.11 ± 4.90 | 76.10 ± 6.15 | 77.80 ± 6.20 | 58.40% |
| Regularized Logistic Regression | 77.20 ± 5.80 | 73.10 ± 6.90 | 73.50 ± 7.10 | 54.20% |

> [!NOTE]
> While aggregate accuracy is slightly higher in ensemble trees due to majority-class dominance (stressed samples), the **WL Graph Kernel demonstrates higher sensitivity on minority healthy controls (65.32% recall)**, mitigating dangerous false-positive alarms in agricultural diagnosis.

---

## PlantCare AI Web Application

The `Application/` directory houses **PlantCare AI**, a full-featured web platform built to demonstrate the research findings and provide interactive diagnostics.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PlantCare AI Research Portal                    │
├─────────────────┬──────────────────────────────────────────────────────┤
│ 🌿 Dashboard    │ Real-time metrics: 4,387 images, 8 crops, 57 classes │
│ 🔬 Analyzer     │ Upload leaf image, extract features, infer diagnosis │
│ 📊 Diagnostics  │ Topology comparisons, WL height sweeps, kernel ranks │
│ 📚 Stress Index │ Taxonomy of 20 deficiencies, pests, and pathologies  │
│ 📋 History/OOF  │ Searchable out-of-fold log with thumbnail previews   │
│ ⚙️ ML Pipeline  │ Visual walkthrough of the 5-stage research pipeline  │
└─────────────────┴──────────────────────────────────────────────────────┘
```

- **Frontend Tech Stack:** React 18, Vite 5, Tailwind CSS, Recharts, Framer Motion, Lucide Icons, jsPDF.
- **Data Regeneration:** Automatically updates its research bundle directly from pipeline outputs via `python scripts/build_plantcare_data.py`.
- **Deployment Ready:** Configured for Vercel with clean client-side routing via `vercel.json`.

---

## Quickstart & Reproduction Guide

### Prerequisites
- Linux / macOS / Windows with WSL
- Python 3.9+
- Node.js 18+ and npm

### 1. Python Environment Setup

Clone the repository and install the dependencies:
```bash
git clone https://github.com/BharathBala21/WL_Graph_Representation.git
cd WL_Graph_Representation

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Research Pipeline Notebooks

Launch Jupyter to explore or re-execute the numbered pipeline:
```bash
jupyter notebook
```
Follow the sequential execution order:
1. `full_pipeline/08_OLID_Full_Feature_Extraction.ipynb`: Feature extraction (reads images, writes `solid_full_features.csv`).
2. `full_pipeline/11_OLID_Final_Graph_Construction.ipynb`: Graph generation.
3. `full_pipeline/12_OLID_WL_Subtree_Engine.ipynb` & `13_OLID_WL_Kernel_Construction.ipynb`: Computes WL subtree kernels.
4. `full_pipeline/14_OLID_WL_SVM_Classification.ipynb`: Evaluates WL kernel SVM.
5. `full_pipeline/decomposition/23_Level3_Hierarchical_Crop_Stress_Classification.ipynb`: End-to-end hierarchical benchmark.

### 3. Running Baseline Experiments

To run the automated baseline experiments and generate comparison plots:
```bash
# Run Logistic Regression benchmark
python New_Models_Baselines/run_lr_experiment.py

# Run Extra Trees / baseline build scripts
python build_and_run_lr.py
```

### 4. Running the PlantCare AI Web Application

```bash
cd Application

# Install node dependencies
npm install

# Run local development server
npm run dev
# The portal will be available at: http://localhost:5173

# Create production build
npm run build
```

---

## Academic Presentation & Literature

- **Presentation Slides:** A comprehensive LaTeX Beamer presentation is available at [`ppt/presentation.tex`](ppt/presentation.tex), containing vector TikZ diagrams illustrating graph aggregation steps, neighborhood hashing, and topological comparisons. Compile with:
  ```bash
  cd ppt
  pdflatex presentation.tex
  ```
- **Foundational Literature:** Reference literature included in [`final_year_project_pdfs/`](final_year_project_pdfs/) covers:
  - *Weisfeiler-Lehman Graph Kernels* (Shervashidze et al., JMLR 2011)
  - *OLID-I: An Open Leaf Image Dataset for Plant Stress Recognition*
  - *Perceptual Graph Kernels for Image-Derived Plant Trait Interaction Analysis*

---

## Research Team & Supervision

**Department:** School of Computing  
**Institution:** Amrita Vishwa Vidyapeetham, Coimbatore, India  
**Project Group:** Final Year B.Tech CSE — Team 49

### Project Guide
- **Mr. Sriram S.**  
  *Assistant Professor (OC), School of Computing, Amrita Vishwa Vidyapeetham*  
  Email: `s_sriram@cb.amrita.edu`  
  Research Interests: Big Data Analytics, IoT, Applied Graph Theory, Cybersecurity

### Student Research Team
- **Merugumala Sandeep** (Roll No: 23CSE664) — [GitHub](https://github.com/Sandeep-Merugumala) • [LinkedIn](https://www.linkedin.com/in/sandeep-merugumala)
- **Chilukuri Subrahmanya Sri Prakash** (Roll No: 23CSE610) — [GitHub](https://github.com/prakashchilukuri) • [LinkedIn](https://www.linkedin.com/in/chilukuriprakash)
- **Bala Bharath Chandra Sai** (Roll No: 23CSE609) — [GitHub](https://github.com/BharathBala21) • [LinkedIn](https://www.linkedin.com/in/bharathbala21/)
- **Muppagowni Vamsidhar** (Roll No: 23CSE613) — [GitHub](https://github.com/Vamsidhar-Muppagowni) • [LinkedIn](https://www.linkedin.com/in/vamsidhar-muppagowni-723797382)
- **Yerneni Vishnu Vardhan** (Roll No: 23CSE154) — [GitHub](https://github.com/vishnuyerneni-maker) • [LinkedIn](https://www.linkedin.com/in/vishnu-vardhan-yerneni-1158a1298)

---

## Citation

If you use this codebase, methodology, or experimental findings in your research, please cite:

```bibtex
@misc{wlleafkernel2026,
  title={Evaluating Graph Representations using Weisfeiler-Lehman Graph Kernels for Plant-Stress Classification},
  author={Sandeep, Merugumala and Sri Prakash, Chilukuri Subrahmanya and Bharath Chandra Sai, Bala and Vamsidhar, Muppagowni and Vishnu Vardhan, Yerneni and Sriram, S.},
  institution={Amrita Vishwa Vidyapeetham, School of Computing, Coimbatore},
  year={2026},
  url={https://github.com/BharathBala21/WL_Graph_Representation}
}
```

---

*Academic research repository maintained by [Team 49](https://github.com/BharathBala21/WL_Graph_Representation).*
