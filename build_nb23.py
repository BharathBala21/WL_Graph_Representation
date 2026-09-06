"""
Script to build NB23 Jupyter notebook from 23_source.py.
Run after 23_source.py is complete.
"""
import json, re, os, uuid

NOTEBOOK_PATH = r'full_pipeline/decomposition/23_Level3_Hierarchical_Crop_Stress_Classification.ipynb'
SOURCE_PATH   = r'full_pipeline/decomposition/23_source.py'

with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
    source = f.read()

# Define cell boundaries: each SECTION becomes a markdown + code pair
section_pattern = re.compile(
    r'# =+\r?\n# SECTION ([\d\.]+):.*?\r?\n# =+',
    re.MULTILINE
)

SECTION_TITLES = {
    '0':  'Setup, Configuration & Environment Verification',
    '1':  'Data Loading & Feature Verification (38 Handcrafted Features)',
    '2':  'Binary Target Construction: Condition -> Stressed / Non-Stressed',
    '3':  'WL Engine & Graph Constructors',
    '4':  'Stage 1 Crop Classifier & Stage 2 Hierarchical Routing',
    '5':  'Aggregation, Summaries & Error Decomposition',
    '6':  'Visualizations',
    '7':  'FINAL RESEARCH CONCLUSION (Level 3)',
}

SECTION_DESCRIPTIONS = {
    '0':  """Import required machine learning, graph theory, and visualization packages. 
Set random seed (42) for strict reproducibility. Verify project assertions on `GRAPH_TYPES` and `WL_HEIGHTS`.""",
    '1':  """Load the OLID-I dataset (4,387 quality-accepted samples, 38 validated handcrafted features).
Assert that exactly 38 numerical features are present and feature families match NB12.""",
    '2':  """Construct the binary classification target:
- `healthy` -> **NON-STRESSED** (0): Explicit control condition with no documented symptoms.
- All remaining 19 conditions -> **STRESSED** (1).""",
    '3':  """Preserve the original validated WL Subtree Engine and Graph Constructors verbatim from NB12.
- 5 Graph constructions: Correlation ($|r| \\ge 0.70$), Feature-Similarity kNN ($k=3$), Feature-Family, Spatial/Regional, and Hybrid.
- Normalized WL kernel: $K(i, j) / \\sqrt{K(i, i) K(j, j)}$.""",
    '4':  """Execute the primary Level-3 experiment: **Hierarchical Crop-Specific Stress Detection**.
1. Dynamically read the optimal crop-specific WL configurations verified in Level 2.
2. Stratified 5-Fold Cross Validation.
3. Train Stage-1 Crop Classifier (Flat Random Forest & RBF SVM) on $X_{train}$ and predict crop on $X_{val}$.
4. Train Stage-2 crop-specific WL stress models using **only** $X_{train}$ matching that true crop.
5. End-to-End Routing: Route $X_{val}$ samples to the crop-specific models using their *predicted crop*.
6. Oracle Routing: Route $X_{val}$ samples using their *true crop* to evaluate upper-bound potential.""",
    '5':  """Compute mean ± SD validation metrics across the 5 CV folds for all configurations.
Compare End-to-End Hierarchical WL with the Oracle WL and the global Level 1 baseline.
Decompose errors by separating stress performance into Correctly vs Incorrectly routed crops.
Isolate the Tomato case study.""",
    '6':  """Generate visual analysis artifacts:
1. 8-class Stage-1 Crop Confusion Matrix.
2. Stage-1 Crop Routing Accuracy bar chart.
3. Level-1 vs Level-3 (Real vs Oracle) Macro-F1 comparison.
4. Correct vs Incorrect Routing Impact on Stress Classification.
5. Tomato Case Study metrics.""",
    '7':  """**Final Research Conclusion**:
Dynamically generates answers to the 8 core Level 3 research questions (Routing accuracy, Oracle vs Real performance, Routing degradation impact) based strictly on the measured empirical evidence, avoiding overstrong biological claims."""
}

def make_markdown_cell(sec_num, title, description):
    lines = [
        f"## Section {sec_num}: {title}",
        "",
        description,
    ]
    return {
        "id": uuid.uuid4().hex[:8],
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines]
    }

def make_code_cell(code):
    lines = code.split('\n')
    source_lines = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {
        "id": uuid.uuid4().hex[:8],
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": source_lines
    }

# Split source into sections
splits = list(section_pattern.finditer(source))
cells = []

# Title / intro cell
intro_md = {
    "id": uuid.uuid4().hex[:8],
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# NB23: Level 3 – Hierarchical Crop Stress Classification\n",
        "\n",
        "## Core Research Question\n",
        "> **\"Can hierarchical crop identification followed by crop-specific Weisfeiler–Lehman stress classification improve end-to-end plant-stress recognition compared with direct global stress classification?\"**\n",
        "\n",
        "## Pipeline Architecture\n",
        "```\n",
        "OLID-I\n",
        "  ↓\n",
        "4387 accepted images\n",
        "  ↓\n",
        "38 handcrafted features\n",
        "  ↓\n",
        "5 fixed CV folds\n",
        "  ↓\n",
        "─────────────────────────────────────\n",
        "Stage 1: Crop Identification\n",
        "  ├── RF (PRIMARY)\n",
        "  └── RBF SVM (SECONDARY)\n",
        "─────────────────────────────────────\n",
        "  ↓\n",
        "Predicted crop\n",
        "  ↓\n",
        "Crop-specific WL model\n",
        "  ├── graph construction\n",
        "  ├── training-only discretization\n",
        "  ├── training-only WL mapping\n",
        "  └── preselected / nested graph+h\n",
        "  ↓\n",
        "Stress / Non-Stress\n",
        "```\n",
        "\n",
        "This experiment evaluates a realistic deployable application pipeline. It measures whether the advantages of crop-specific graph modeling (Level 2) are robust against real-world errors in upstream crop identification (Stage 1).\n"
    ]
}
cells.append(intro_md)

# Process each section
for idx, match in enumerate(splits):
    sec_num = match.group(1)
    title   = SECTION_TITLES.get(sec_num, f'Section {sec_num}')
    desc    = SECTION_DESCRIPTIONS.get(sec_num, '')

    code_start = match.end()
    code_end   = splits[idx + 1].start() if idx + 1 < len(splits) else len(source)
    code_block = source[code_start:code_end].strip()

    cells.append(make_markdown_cell(sec_num, title, desc))
    if code_block:
        cells.append(make_code_cell(code_block))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells
}

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generated successfully: {NOTEBOOK_PATH}")
print(f"Total cells: {len(cells)}")
