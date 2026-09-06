"""
Script to build NB21 Jupyter notebook from 21_source.py.
Run after 21_source.py is complete.
"""
import json, re, os, uuid

NOTEBOOK_PATH = r'full_pipeline/decomposition/22_Level2_Crop_Specific_Stress_Detection.ipynb'
SOURCE_PATH   = r'full_pipeline/decomposition/22_source.py'

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
    '3':  'WL Engine & Graph Constructors (Preserving Validated Architecture)',
    '4':  'Crop Analysis & Stratified Cross-Validation',
    '5':  'Aggregation & Summaries (with Statistical SD and Tomato Case Study)',
    '6':  'Visualizations & Confusion Matrices',
    '7':  'FINAL RESEARCH CONCLUSION (Level 2)',
}

SECTION_DESCRIPTIONS = {
    '0':  """Import required machine learning, graph theory, and visualization packages. 
Set random seed (42) for strict reproducibility. Verify project assertions on `GRAPH_TYPES` and `WL_HEIGHTS`.""",
    '1':  """Load the OLID-I dataset (4,387 quality-accepted samples, 38 validated handcrafted features).
Assert that exactly 38 numerical features are present and feature families match NB12.""",
    '2':  """Construct the binary classification target:
- `healthy` -> **NON-STRESSED** (0): Explicit control condition with no documented symptoms.
- All remaining 19 conditions -> **STRESSED** (1): Documented fungal/bacterial/oomycete pathologies, pest/mite damage, or nutrient deficiencies.""",
    '3':  """Preserve the original validated WL Subtree Engine and Graph Constructors verbatim from NB12:
- 5 Graph constructions: Correlation ($|r| \\ge 0.70$), Feature-Similarity kNN ($k=3$), Feature-Family, Spatial/Regional, and Hybrid.
- `FoldSafeDiscretizer`: 3 quantile bins fit strictly on training fold.
- Initial node labels preserve feature identity: `feature_name::High`.
- `WLLabelMapper`: strict inductive training mapping with OOV token handling for validation graphs.
- Normalized WL kernel: $K(i, j) / \\sqrt{K(i, i) K(j, j)}$.""",
    '4':  """Execute the primary Level-2 experiment: **Crop-Specific Binary Stress Detection**.
For each crop independently:
1. Filter dataset to the crop.
2. Determine sample class balance. If minority class < 2 samples, skip the crop.
3. Dynamically set StratifiedKFold splits to `min(5, minority_class_count)` to maximize data retention.
4. Run all 20 configurations (5 graph types × 4 WL heights) using strict inductive protocol.
5. Run flat baselines (Random Forest, RBF SVM).""",
    '5':  """Compute mean ± SD validation metrics across the 5 CV folds for all configurations for all crops.
Generate the complete comparison table tracking Best WL, Flat RF, and Flat RBF performance with Deltas.
Extract the Tomato Case Study and save class balance metrics.
Save results and Out-of-Fold predictions for the best WL and flat baselines to the `level2` directory.""",
    '6':  """Generate visual analysis artifacts:
1. Confusion matrix heatmaps for the best WL configuration of each crop.
2. Grouped bar chart comparing Best WL vs Flat RF vs Flat RBF.
3. Delta bar chart mapping WL advantage over Flat RF.
4. Pie chart showing optimal Graph Constructor frequency.
5. Heatmap mapping Crop vs WL Height vs Macro-F1.
6. Tomato Case Study comparison bar chart.""",
    '7':  """**Final Research Conclusion**:
Dynamically generates answers to the 8 core Level 2 research questions (Task Restriction vs WL Advantage, Tomato comparison, Graph variability, etc.) based strictly on the measured empirical evidence, avoiding overstrong biological claims."""
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
        "# NB22: Level 2 – Crop-Specific Stress Detection via WL Graph Kernel + SVM\n",
        "\n",
        "## Research Question\n",
        "> **\"Does restricting the problem to a single crop make graph-based stress recognition more effective?\"**\n",
        "\n",
        "## Pipeline Architecture\n",
        "```\n",
        "OLID-I image\n",
        "→ filter to individual crop (e.g. Tomato, Eggplant)\n",
        "→ preprocessing / segmentation\n",
        "→ 38 validated handcrafted features\n",
        "→ graph construction (5 types)\n",
        "→ WL subtree transformation (h = 0, 1, 2, 3)\n",
        "→ normalized WL kernel matrix\n",
        "→ precomputed SVM (C = 1.0, balanced class weight)\n",
        "→ binary stress prediction (STRESSED vs. NON-STRESSED)\n",
        "```\n",
        "\n",
        "This Level-2 experiment tests whether removing cross-crop morphological variance provides a cleaner topology for the WL Graph Kernel.\n"
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
