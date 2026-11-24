# Manuscript and submission notes

Open **main.pdf** for the reading edition with exhibits near their discussion. **submission.pdf** places exhibits after the references for journal upload. Both editions use the same manuscript source. Edit **article.tex** for the scientific text, **main.tex** for the title and proposed byline, and **exhibits.tex** for figure and table captions. In Overleaf select pdfLaTeX and set **main.tex** (reading) or **submission.tex** (journal layout) as the root. An Overleaf source ZIP is provided in this directory. Local compilation uses:

    pdflatex main
    bibtex main
    pdflatex main
    pdflatex main

The class and author–date bibliography style come from the [official Frontiers template](https://www.frontiersin.org/design/zip/Frontiers_LaTeX_Templates.zip). The supplied PDF equivalent of the publisher's EPS logo allows standard pdfLaTeX builds. Source, finished PDF, separate RGB JPEG figures (350 dpi export), editable tables, bibliography and fourteen supplementary CSVs are included. For the journal, supply the manuscript files and PDF, upload figures individually in numerical order and identify S1–S14 as supplementary data sheets. For a Git checkout, the source files and the PDF are sufficient to rebuild the Overleaf archive.

## Venue and article type

**Frontiers in Plant Science, Technical Advances in Plant Science, Original Research.** The practical research question concerns which visible leaf measurements retain screening information across crops and which symptoms a segmentation-dependent system underrepresents. The section expects a technical comparison, enough experimental detail to enable reuse and a substantive contribution to plant science. An image-level, single-collection benchmark presents a real editorial-fit risk. Independent crop/site validation or expert mask assessment would strengthen the biological claim. The project currently supports a comparative measurement study. Original Research admits disconfirming results, uses IMRaD and has a 12,000-word ceiling. An A-type publication charge applies if accepted; the team should verify institutional fee support.

Checked 22 November 2026: [section scope](https://www.frontiersin.org/journals/plant-science/sections/technical-advances-in-plant-science/about), [article types](https://www.frontiersin.org/journals/plant-science/for-authors/article-types), [manuscript guidelines](https://www.frontiersin.org/journals/plant-science/for-authors/author-guidelines), [submission checklist](https://www.frontiersin.org/journals/plant-science/for-authors/submission-checklist), [publishing fees](https://www.frontiersin.org/journals/plant-science/for-authors/publishing-fees), [AI disclosure policy](https://www.frontiersin.org/guidelines/policies-and-publication-ethics).

## Proposed submission scope statement

This study evaluates what image-derived leaf traits contribute to stress screening across eight vegetable crops. By comparing graph neighborhoods at fixed descriptor resolution, we isolate the information gained from relationships among color, geometry and regional measurements. Crop-specific model comparisons and acquisition perturbations expose conditions under which those measurements remain informative. A coverage audit identifies which phenotypes are disproportionately lost before classification. Together these results connect an interpretable computational method to a practical question in phenotyping: which leaves can be measured reliably, and how accurately can their recorded stress status be recognized?

## Author decisions before upload

- The byline follows the team roster and citation order in the project README. All authors must confirm spelling, family names (including the running header), author eligibility, order, affiliation, corresponding author, full mailing address and email. The project guide has not been designated corresponding author by default.
- Complete the marked CRediT, funding and conflict-of-interest statements from the team's actual information. Obtain all authors' approval of the final text, citations, figure labels and AI disclosure. Check that the repository URL is accessible to reviewers and archive the release used for submission. Confirm original publication status and eligibility with the journal.
- Figure 1 uses the team's three illustrative ash-gourd JPEGs. Labels follow the supplied image manifest. Source attribution appears in the caption. The [OLID-I dataset DOI](https://doi.org/10.5281/zenodo.8105154) is registered under CC BY 4.0. Verify the three image identities against the release when signing off the final figures. No image was synthesized.
- The study reports recorded conditions in OLID-I. New sites, plant-grouped splits and independent expert segmentation masks would support stronger generalization claims. No such validation is reported here.

## Source record and updates after a Git pull

Initial evidence snapshot: 3a4e2575c3856753f60d04149274fb73c2f82339. The earlier IEEE manuscript covers a different target (57 crop–condition classes). The new primary endpoint is binary stress recognition. Metric values from those tasks should remain separately labeled.

| Manuscript evidence | Tracked source relative to repository root |
| --- | --- |
| Cohort, binary status, counts and mask retention | full_pipeline/olid_full_features.csv |
| Image folds, decoded using the exporter schema | Application/public/plantcare/oof_level3.json and Application/scripts/build_plantcare_data.py |
| Pooled topology/depth grid | Saved comparison table in full_pipeline/decomposition/21_Level1_Binary_Stress_Detection_FIXED.ipynb |
| Segmentation, features and WL engine | Application/backend/app/features.py and wl.py |
| Comparators, folds and routing | New_Models_Baselines/03_Benchmark_Suite/bench.py and run_level3.py |
| Fold scores, robustness, timing and learning curves | New_Models_Baselines/03_Benchmark_Suite/results/ and make_report.py |
| Frozen features and preprocessing | New_Models_Baselines/04_CNN_Frozen_Features/cnn_pipeline.py |
| Photos and descriptor definitions | Research_Paper/data/illustrative_images/ and Research_Paper/data/feature_dictionary.csv |

After new experiment results are pulled, if the cohort or folds change, regenerate S1 with **python frontires_draft/refresh_exhibits.py --folds-only** before fitting. Run **python frontires_draft/topology_control.py** to refresh the 35 pooled fits and per-image predictions, then **python frontires_draft/trace_leaf.py** for the worked example. Finally run **python frontires_draft/refresh_exhibits.py** from the repository root (Python with NumPy, pandas, matplotlib, Pillow, SciPy, scikit-learn, scikit-image and OpenCV). It refreshes the tables, figures and derived data sheets and checks the identity joins and routed scores. It uses existing experiments; editing a notebook without updating its saved outputs does not refresh a result. Review all numbers embedded in the title-page abstract, article and captions against refreshed tables, update the snapshot hash and author declarations, rebuild the PDF and repackage the sources.

S1 recovers the existing image-to-fold mapping by joining the archived index (crop, condition, image number) to accepted source filenames. This matched all 4,387 identifiers exactly once. To run the original benchmark loader, join S1 to the supplied feature table on image_name and export image_path and fold to full_pipeline/cv_fold_assignments.csv. For image-based reruns, map dataset paths to the local OLID-I archive consistently. Full cohort matching and identical held-out sizes were verified. The crop router and fixed-setting WL pipeline were refitted on all five folds during preparation; their per-fold macro-F1 scores matched the archived results. All 110 routed fold records were used to check reported means. Other model fits, image extraction and hardware timings rely on the saved studies.

S9 is an additional analysis of the tracked individual robustness predictions. It intersects photo identifiers across every tested model and all eight image versions and reports macro-F1 and paired-cohort change for each model and perturbation. The complete-case cohort contains 217 images (181 stressed and 36 healthy).


## Revised analysis, 24 November 2026

Seven figures accompany five tables. S10 contains 35 independently fitted pooled models: five depth-zero fits, five original hybrid depth-three fits and five permutations on each of five folds. S11 contains 4,387 held-out calls of the original hybrid model. S12 describes 38 within-crop descriptor contrasts in eight crops. S13 copies summary values from the separate multiclass reanalysis with the source filenames recorded. S14 traces cucumber image 15, selected by filename and fold before prediction. The permutation study is exploratory. Its five seeds are not independent replications of the collection. Check both the signs and the denominators before updating any result in the abstract.

When compiling the submission edition locally, substitute submission for main in all four build commands. The ZIP includes both entry points, figures, generated tables and bibliography. Upload the supplementary CSVs separately. Changes to captions and prose apply to both editions.
