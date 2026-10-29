# IEEE-style full-dataset 

## Correct interpretation of the image subset

The six photographs used in Figure 1 come from the small image ZIP supplied by the team. They are **illustrative figures only**. They are not entered as an image table, are not a separate 20-image modelling study, and are not used to calculate the paper's classification results. The paper's quantitative analyses use the supplied full derived-feature table containing **4,749 records** across the 57 OLID-I crop-condition classes. Of these, **4,387 records** have `quality_flag=ACCEPT`; model evaluation is performed on those accepted records.

The team reports that the complete local raw-image archive is approximately 14 GB. The raw archive is not included in this package. The included feature table is only about 4 MB and is enough to inspect or rerun the reported feature-based evaluation. A complete end-to-end image-to-feature rerun still requires retrieving the original OLID-I archive.

## Main findings retained in the draft

- Best graph-kernel setting in the included comparison: hybrid topology, WL height 2; accuracy 44.31% ± 0.72%, macro-F1 42.79% ± 0.95%.
- Random Forest baseline: accuracy 58.54% ± 1.47%, macro-F1 54.99% ± 1.35%.
- 362 of 4,749 rows (7.62%) were filtered out; retention varies materially by crop and condition.

These are results from re-analysis of the stored derived-feature table and saved project code, not a fresh run of feature extraction on the 14 GB image archive. Fold standard deviations are not confidence intervals. The selected graph height was selected by comparing the same five folds, so nested/repeated validation is needed for an unbiased estimate of a fully selected configuration.

## Build the manuscript

Run `pdflatex main.tex` twice from this folder. The PDF currently included is the verified 11-page IEEE conference-style layout for professor review. IEEE conferences can have specific templates and page limits, so check the target venue before submission.

## Reproduce or inspect figures/results

- `analysis/input/olid_full_features.csv`: full derived-feature table used for the analysis. Absolute local paths in `image_path` were normalised to file names for portability; labels, flags, and numeric descriptors were preserved.
- `analysis/recompute_models.py`: recomputes the baseline and WL comparisons from the full feature table. It may take substantial time because of graph-kernel calculations.
- `analysis/generate_figures.py`: regenerates the core workflow, topology, model-performance and acceptance figures from included analysis outputs.
- `analysis/generate_descriptive_figures.py`: regenerates the full-feature distributions, full 38-feature correlation map, and full-table mask-ratio histogram.
- `analysis/create_representative_panel.py`: regenerates the Figure 1 image montage from `data/illustrative_images/`.
- CSV outputs in `analysis/` and `supplementary/` preserve fold-level scores, split assignments, graph edges, feature dictionary and acceptance audits.

## Before external submission

1. Replace `[Institution Name]` with the real affiliation and confirm author order/spelling.
2. Complete the author-contribution, funding, and competing-interest statements accurately.
3. Add the final public repository URL/DOI if a repository is created.
4. Independently rerun the full image processing pipeline and verify the filter thresholds/results.
5. Verify image/data redistribution terms and the target IEEE venue's manuscript requirements.
6. Review the overlap with the related 2026 perceptual graph-kernel paper noted in the integrity audit; establish the distinct contribution and disclose any shared authorship, code, text, data, or prior results.

This package is a **professor-review draft**, not a representation that peer review, publication checks, plagiarism scanning, or full pipeline verification have been completed.

## Dataset and format references

- OLID-I dataset paper: https://doi.org/10.3389/fpls.2023.1251888
- Dataset repository: https://zenodo.org/records/8105154
- IEEE conference templates: https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/
