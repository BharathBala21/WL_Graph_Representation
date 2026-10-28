# Scientific scope, reference and integrity audit

## Scope correction

The image ZIP supplied for figures contains a small selection of real OLID-I images. The images are used only in Figure 1 as examples of image variety. The paper no longer reports a 20-image or three-group classification experiment. All numerical results in the manuscript are derived from the 4,749-row stored feature table; 4,387 `ACCEPT` rows are used for model evaluation across the project's 57 crop-condition labels.

The OLID-I dataset paper reports 4,749 high-resolution images in 57 categories across eight crops. It also describes a consistent black background and notes that segmentation is not required for the dataset as released. The current study's ExG/Otsu stage is therefore a project-pipeline choice. The observed exclusion rate is an audit of this pipeline—not proof that the raw OLID-I images are invalid.

## Reported results and correct limitations

- Best graph-kernel setting reported: hybrid topology, WL height 2, accuracy 44.31% ± 0.72%, macro-F1 42.79% ± 0.95%.
- Random Forest baseline: accuracy 58.54% ± 1.47%, macro-F1 54.99% ± 1.35%.
- Filter: 4,387 accepted, 362 excluded (7.62%). The reported crop/condition differences can mean the downstream evaluation is conditional on a selected subset.
- These are fold means ± fold standard deviations, not confidence intervals. Same five folds were used to compare/select WL heights, so the chosen configuration score may be optimistic.
- The project table stores derived image features; the full raw-image-to-feature pipeline has not been independently rerun as part of this draft.

## Reference verification

The reference list includes core methodological sources for OLID-I, Weisfeiler--Lehman graph kernels, Otsu thresholding, GLCM texture, SVMs, random forests, and high-throughput plant phenotyping. The 2023 OLID-I dataset paper and the current IEEE conference-template guidance were checked against their public source pages. The two 2026 Frontiers papers cited in Related Work are verifiable public articles:

- S. Sriram and T. Ramraj, “Perceptual graph kernels for image-derived plant trait interaction analysis in precision agriculture,” *Frontiers in Plant Science*, 2026, DOI: https://doi.org/10.3389/fpls.2026.1822052.
- A. Karthikeyan and S. Sundar, “Q-TriLSTM-Vision: a quantum-interference-augmented tri-stream LSTM for multi-label plant stress recognition on the OLID-I benchmark,” *Frontiers in Plant Science*, 2026, DOI: https://doi.org/10.3389/fpls.2026.1903924.

## Important novelty/authorship check

The team author list includes “S Sriram”, and a cited 2026 graph-kernel article lists an author named “Sriram S.” This is a name match, not proof of identity. The team and supervisor should confirm whether this is the same person and, if so, explicitly check for any reuse of ideas, text, code, data, images, results, or thesis material. Proper citation is necessary but does not by itself resolve self-plagiarism, duplicate-publication, or novelty questions. The current manuscript positions itself as a controlled comparison of five feature-graph topologies and WL heights against flat-feature baselines, but the team must substantiate that distinction and disclose relevant prior work.

## Plagiarism status

No Turnitin/iThenticate report is included, and no percentage similarity can be truthfully certified. This is a reference/integrity review, not a plagiarism clearance. Before submission, independently inspect quoted/paraphrased text, figure provenance, code licences, and any reused text or results.

## Items deliberately left as placeholders

Institution, author contributions, funding, competing interests, and final public repository URL/DOI must be completed from verified facts. Do not replace these with guessed declarations.
