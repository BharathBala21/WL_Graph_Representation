import React from 'react';
import { RESEARCH } from '../data/research';
import { CONDITION_BY_CODE, Card, Chip, SectionTitle, num, pct } from '../lib';

const c = RESEARCH.config;
const d = RESEARCH.dataset;

const SETTINGS: [string, string][] = [
  ['Dataset', `${d.name} · ${num(d.accepted)} of ${num(d.rawImages)} images accepted`],
  ['Classes', `${d.crops} crops · ${d.conditions} conditions · ${d.classes} crop-condition classes`],
  ['Binary labels', `healthy = non-stressed (${num(d.healthy)}), all other conditions = stressed (${num(d.stressed)})`],
  ['Cross-validation', `${d.folds} fixed folds (cv_fold_assignments.csv), shared by every experiment`],
  ['Node labels', `${c.n_bins} quantile bins: ${c.bin_labels.join(', ')}`],
  ['Correlation edges', `|Pearson r| > ${c.corr_thresh} on the training fold`],
  ['Feature-similarity kNN', `k = ${c.knn_k} nearest features`],
  ['WL heights', c.wl_heights.join(', ')],
  ['Random seed', String(c.random_state)],
];

const LEAKAGE = [
  'Bin edges, correlation thresholds and kNN neighbourhoods are fitted on the training fold and only applied to the held-out fold.',
  "In the hierarchical pipeline, each crop's graph topology and WL height are chosen by inner CV on the training fold; the held-out fold is scored once.",
  'Every prediction in History & Logs is out-of-fold: the image was never seen by the model that scored it.',
  'Segmentation quality control is applied once, before the fold split, so every experiment sees the same accepted images.',
];

const SOURCES = [
  ['Per-image features', 'full_pipeline/olid_full_features.csv'],
  ['Hierarchical predictions', 'full_pipeline/decomposition/results/level3_hierarchical/level3_oof_predictions.csv'],
  ['Global WL predictions', 'full_pipeline/decomposition/results/level1_binary_wl/level1_oof_predictions.csv'],
  ['Topology × height grid', 'full_pipeline/decomposition/results/level1_binary_wl/level1_wl_summary.csv'],
  ['Per-crop best models', 'full_pipeline/decomposition/results/level2_crop_wl/level2_best_crop_results.csv'],
  ['57-class comparison', 'full_pipeline/results/classical_baselines/graph_vs_baseline_comparison.csv'],
  ['Graph edge lists', 'full_pipeline/results/final_graph_design/graph_edge_lists.csv'],
];

export const ProtocolView: React.FC = () => (
  <div className="flex flex-col gap-6 w-full max-w-[1200px] mx-auto">
    <div>
      <Chip className="bg-pc-mint text-pc-mint-ink">Reproducibility</Chip>
      <h1 className="font-headline-lg text-pc-deep-ink mt-2">Experiment Protocol</h1>
      <p className="font-body-md text-pc-ink2 mt-1 max-w-3xl">
        The settings behind every number in PlantCare AI, the safeguards against data leakage, and where each figure comes from.
      </p>
    </div>

    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <SectionTitle icon="settings" title="Configuration" sub="From the WL engine and experiment configs" />
        <dl className="flex flex-col">
          {SETTINGS.map(([k, v]) => (
            <div key={k} className="flex flex-col sm:flex-row sm:justify-between gap-1 py-2 border-b border-pc-line/50 font-body-sm text-[13px]">
              <dt className="text-pc-muted">{k}</dt>
              <dd className="text-pc-ink sm:text-right font-medium">{v}</dd>
            </div>
          ))}
        </dl>
      </Card>

      <div className="flex flex-col gap-6">
        <Card>
          <SectionTitle icon="shield" title="Leakage controls" />
          <ul className="flex flex-col gap-2.5">
            {LEAKAGE.map((t) => (
              <li key={t} className="flex gap-2 font-body-sm text-[13px] text-pc-ink2 leading-relaxed">
                <span className="material-symbols-outlined text-[18px] text-pc-green">check_circle</span>
                {t}
              </li>
            ))}
          </ul>
        </Card>
        <Card>
          <SectionTitle icon="warning" title="Known limitations" />
          <ul className="flex flex-col gap-2.5 font-body-sm text-[13px] text-pc-ink2 leading-relaxed">
            <li>
              Healthy leaves are the minority class ({pct(d.healthy / d.accepted)} of images); only {pct(CONDITION_BY_CODE.healthy.detectionRate)} of them are
              kept as healthy by the hierarchical pipeline.
            </li>
            <li>
              The per-crop "best configuration" figures on the Crops page are the best of 20 topology × height settings by
              out-of-fold score, so they are slightly optimistic; the nested hierarchical results are the unbiased estimate.
            </li>
            <li>Results come from a single public dataset; evaluation on farm-collected images is planned but not yet done.</li>
            <li>New uploads cannot be scored in the browser — the feature extraction and WL kernels run in the Python pipeline.</li>
          </ul>
        </Card>
      </div>
    </div>

    <Card>
      <SectionTitle icon="dataset" title="Data sources" sub="Regenerate with: python scripts/build_plantcare_data.py (from Application/)" />
      <dl className="flex flex-col">
        {SOURCES.map(([k, v]) => (
          <div key={k} className="flex flex-col md:flex-row md:justify-between gap-1 py-2 border-b border-pc-line/50 font-body-sm text-[13px]">
            <dt className="text-pc-ink">{k}</dt>
            <dd className="font-code-sm text-pc-muted break-all">{v}</dd>
          </div>
        ))}
      </dl>
    </Card>
  </div>
);
