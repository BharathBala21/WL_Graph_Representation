# IEEE-style full-dataset

## Correct interpretation of the image subset

The six photographs used in Figure 1 come from the small image ZIP supplied by the team. They are **illustrative figures only**. They are not entered as an image table, are not a separate 20-image modelling study, and are not used to calculate the paper's classification results. The paper's quantitative analyses use the supplied full derived-feature table containing **4,749 records** across the 57 OLID-I crop-condition classes. Of these, **4,387 records** have `quality_flag=ACCEPT`; model evaluation is performed on those accepted records.

The team reports that the complete local raw-image archive is approximately 14 GB. The raw archive is not included in this package. The included feature table is only about 4 MB and is enough to inspect or rerun the reported feature-based evaluation. A complete end-to-end image-to-feature rerun still requires retrieving the original OLID-I archive.

## Main findings retained in the draft

- Best graph-kernel setting in the included comparison: hybrid topology, WL height 2; accuracy 44.31% ± 0.72%, macro-F1 42.79% ± 0.95%.
- Random Forest baseline: accuracy 58.54% ± 1.47%, macro-F1 54.99% ± 1.35%.
- 362 of 4,749 rows (7.62%) were filtered out; retention varies materially by crop and condition.

These are results from re-analysis of the stored derived-feature table and saved project code, not a fresh run of feature extraction on the 14 GB image archive. Fold standard deviations are not confidence intervals. The selected graph height was selected by comparing the same five folds, so nested/repeated validation is needed for an unbiased estimate of a fully selected configuration.


## Dataset and format references

- OLID-I dataset paper: https://doi.org/10.3389/fpls.2023.1251888
- Dataset repository: https://zenodo.org/records/8105154
- IEEE conference templates: https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/
