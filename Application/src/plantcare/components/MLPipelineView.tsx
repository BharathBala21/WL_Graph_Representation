import React, { useState } from 'react';
import { RESEARCH } from '../data/research';
import { Bar, Card, Chip, SectionTitle, featureLabel, num, pct } from '../lib';

const STAGES = [
  { n: '01', icon: 'crop_free', name: 'Segmentation & QC', text: `${num(RESEARCH.dataset.rawImages)} images → ${num(RESEARCH.dataset.accepted)} accepted leaf masks` },
  { n: '02', icon: 'tune', name: 'Feature extraction', text: '38 features: 10 morphology, 20 colour, 4 GLCM texture, 4 core/edge spatial' },
  { n: '03', icon: 'stacked_bar_chart', name: 'Discretisation', text: `${RESEARCH.config.n_bins} bins (${RESEARCH.config.bin_labels.join(' / ')}) fitted on the training fold only` },
  { n: '04', icon: 'hub', name: 'Graph construction', text: `5 topologies over the 38 feature nodes (|r| > ${RESEARCH.config.corr_thresh}, k = ${RESEARCH.config.knn_k})` },
  { n: '05', icon: 'account_tree', name: 'WL subtree kernel', text: `h ∈ {${RESEARCH.config.wl_heights.join(', ')}} relabelling iterations → kernel matrix` },
  { n: '06', icon: 'model_training', name: 'SVM & evaluation', text: `Precomputed-kernel SVM, ${RESEARCH.dataset.folds}-fold CV, macro-F1 / balanced accuracy` },
];

const GRAPHS = RESEARCH.graphs.map((g) => g.graph);
const HEIGHTS = RESEARCH.config.wl_heights;
type Metric = 'macroF1' | 'accuracy' | 'balancedAccuracy' | 'rocAuc';
const METRICS: [Metric, string][] = [
  ['macroF1', 'Macro-F1'],
  ['balancedAccuracy', 'Balanced acc.'],
  ['accuracy', 'Accuracy'],
  ['rocAuc', 'ROC-AUC'],
];

const bestWl = [...RESEARCH.level1Grid].sort((a, b) => b.macroF1 - a.macroF1)[0];

export const MLPipelineView: React.FC = () => {
  const [metric, setMetric] = useState<Metric>('macroF1');
  const cells = RESEARCH.level1Grid;
  const values = cells.map((c) => c[metric]);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const cell = (g: string, h: number) => cells.find((c) => c.graph === g && c.height === h);
  const maxFine = Math.max(...RESEARCH.fineGrained.map((f) => f.macroF1));
  const maxImp = RESEARCH.rfImportances[0].importance;

  const binary = [
    { method: `WL + SVM (${bestWl.graph}, h = ${bestWl.height})`, ...bestWl, ours: true },
    ...RESEARCH.level1Baselines.map((b) => ({ ...b, ours: false })),
  ];

  return (
    <div className="flex flex-col gap-6 w-full max-w-[1440px] mx-auto">
      <div>
        <Chip className="bg-pc-mint text-pc-mint-ink">Leakage-safe cross-validation</Chip>
        <h1 className="font-headline-lg text-pc-deep-ink mt-2">ML Pipeline &amp; Results</h1>
        <p className="font-body-md text-pc-ink2 mt-1 max-w-3xl">
          How a leaf image becomes a WL kernel prediction, and how each design choice performed. All numbers are out-of-fold
          results read from the experiment outputs.
        </p>
      </div>

      {/* Stages */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-6 gap-3">
        {STAGES.map((s) => (
          <div key={s.n} className="p-4 rounded-2xl bg-pc-card border border-pc-line/70">
            <div className="flex items-center justify-between">
              <span className="material-symbols-outlined text-pc-green">{s.icon}</span>
              <span className="font-code-sm text-pc-muted">{s.n}</span>
            </div>
            <p className="font-title-sm text-pc-ink mt-2">{s.name}</p>
            <p className="font-body-sm text-[11px] text-pc-muted mt-1 leading-relaxed">{s.text}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Heatmap */}
        <Card>
          <SectionTitle icon="grid_on" title="Topology × WL height" sub="Global binary stress detection (all crops pooled)" />
          <div className="flex flex-wrap gap-1.5 mb-4">
            {METRICS.map(([m, l]) => (
              <button
                key={m}
                onClick={() => setMetric(m)}
                className={`px-2.5 py-1 rounded-md font-body-sm text-[12px] ${metric === m ? 'bg-pc-deep text-white' : 'bg-pc-low text-pc-ink2 hover:bg-pc-cont'}`}
              >
                {l}
              </button>
            ))}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full font-body-sm text-[12px] border-separate border-spacing-1">
              <thead>
                <tr className="text-pc-muted">
                  <th className="text-left font-normal">Graph</th>
                  {HEIGHTS.map((h) => (
                    <th key={h} className="font-normal">h = {h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {GRAPHS.map((g) => (
                  <tr key={g}>
                    <td className="text-pc-ink pr-2 whitespace-nowrap">{g}</td>
                    {HEIGHTS.map((h) => {
                      const c = cell(g, h);
                      const v = c ? c[metric] : null;
                      const t = v === null ? 0 : (v - lo) / (hi - lo || 1);
                      return (
                        <td
                          key={h}
                          className="text-center font-code-sm rounded-md py-2"
                          style={{ background: `rgba(22, 163, 74, ${0.08 + t * 0.62})`, color: t > 0.6 ? '#fff' : undefined }}
                        >
                          {v === null ? '—' : pct(v)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="font-body-sm text-[11px] text-pc-muted mt-3">
            At h = 0 the kernel only counts node labels, so every topology scores the same; structure starts to matter from h = 1.
          </p>
        </Card>

        {/* Binary comparison */}
        <Card>
          <SectionTitle icon="compare_arrows" title="WL vs. baselines" sub="Binary stress detection, same folds" />
          <table className="w-full font-body-sm text-[12px]">
            <thead className="text-pc-muted">
              <tr className="text-left">
                <th className="font-normal py-1.5">Method</th>
                {METRICS.map(([, l]) => (
                  <th key={l} className="font-normal py-1.5 text-right">{l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {binary.map((b) => (
                <tr key={b.method} className={`border-t border-pc-line/50 ${b.ours ? 'text-pc-green font-semibold' : 'text-pc-ink'}`}>
                  <td className="py-2">{b.method}</td>
                  {METRICS.map(([m]) => (
                    <td key={m} className="py-2 text-right font-code-sm">{pct(b[m])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-4 p-3 rounded-xl bg-pc-low font-body-sm text-[12px] text-pc-ink2 leading-relaxed">
            The flat Random Forest has the highest raw accuracy but mostly predicts the majority "stressed" class (balanced accuracy
            {' '}{pct(RESEARCH.level1Baselines.find((b) => b.method === 'Flat RF')?.balancedAccuracy)}). The WL kernel gives the best
            macro-F1; the flat RBF SVM has the best balanced accuracy.
          </div>

          <div className="mt-6" />
          <SectionTitle icon="account_tree" title="Hierarchical pipeline" sub="Stage 1 crop routing → crop-specific Stage 2" />
          <table className="w-full font-body-sm text-[12px]">
            <thead className="text-pc-muted">
              <tr className="text-left">
                <th className="font-normal py-1.5">Setup</th>
                <th className="font-normal py-1.5 text-right">Accuracy</th>
                <th className="font-normal py-1.5 text-right">Macro-F1</th>
                <th className="font-normal py-1.5 text-right">Bal. acc.</th>
              </tr>
            </thead>
            <tbody>
              {RESEARCH.hierarchical.map((h) => (
                <tr key={h.method} className="border-t border-pc-line/50 text-pc-ink">
                  <td className="py-2">{h.method}</td>
                  {[h.accuracy, h.macroF1, h.balancedAccuracy].map(([m, sd], i) => (
                    <td key={i} className="py-2 text-right font-code-sm">
                      {pct(m)} <span className="text-pc-muted">± {pct(sd)}</span>
                    </td>
                  ))}
                </tr>
              ))}
              {RESEARCH.cropClassifier.map((c) => (
                <tr key={c.model} className="border-t border-pc-line/50 text-pc-ink2">
                  <td className="py-2">Stage 1 crop classifier · {c.model}</td>
                  <td className="py-2 text-right font-code-sm">{pct(c.accuracy[0])}</td>
                  <td className="py-2 text-right font-code-sm">{pct(c.macroF1[0])}</td>
                  <td className="py-2 text-right text-pc-muted">—</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* 57-class */}
        <Card className="xl:col-span-2">
          <SectionTitle icon="category" title={`Fine-grained ${RESEARCH.dataset.classes}-class task`} sub="Crop × condition classification, macro-F1" />
          <div className="flex flex-col gap-2.5">
            {[...RESEARCH.fineGrained]
              .sort((a, b) => b.macroF1 - a.macroF1)
              .map((f) => (
                <div key={f.method}>
                  <div className="flex justify-between gap-2 font-body-sm text-[12px]">
                    <span className="text-pc-ink truncate">
                      {f.method.replace(/^Best NB\d+ classical \((.+)\)$/, '$1').replace(/\s*\(NB\d+\)/, '')} <span className="text-pc-muted">· {f.representation}</span>
                    </span>
                    <span className="font-code-sm text-pc-muted shrink-0">
                      F1 {pct(f.macroF1)} · acc {pct(f.accuracy)}
                    </span>
                  </div>
                  <Bar value={f.macroF1} max={maxFine} className={f.representation === 'WL graph' ? 'bg-pc-green-bg' : 'bg-pc-outline'} />
                </div>
              ))}
          </div>
          <p className="font-body-sm text-[12px] text-pc-muted mt-3">
            On the 57-class task the flat Random Forest is clearly ahead of every graph representation — the reason the project
            decomposes the problem into crop routing plus per-crop binary stress detection.
          </p>
        </Card>

        {/* RF importances */}
        <Card>
          <SectionTitle icon="leaderboard" title="Most informative features" sub="Random Forest importance, 57-class task" />
          <div className="flex flex-col gap-2">
            {RESEARCH.rfImportances.map((f) => (
              <div key={f.feature}>
                <div className="flex justify-between font-body-sm text-[12px]">
                  <span className="text-pc-ink">{featureLabel(f.feature)}</span>
                  <span className="font-code-sm text-pc-muted">{f.importance.toFixed(3)}</span>
                </div>
                <Bar value={f.importance} max={maxImp} />
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Graph statistics */}
      <Card>
        <SectionTitle icon="hub" title="Graph topologies" sub="Structure of each feature graph over the 38 nodes" />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] font-body-sm text-[12px]">
            <thead className="text-pc-muted">
              <tr className="text-left">
                {['Topology', 'Edges', 'Density', 'Avg. degree', 'Components', 'Isolated', 'Built from', 'What edges mean'].map((h) => (
                  <th key={h} className="font-normal py-1.5 pr-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {RESEARCH.graphs.map((g) => (
                <tr key={g.graph} className="border-t border-pc-line/50 text-pc-ink align-top">
                  <td className="py-2 pr-3 font-semibold whitespace-nowrap">{g.graph}</td>
                  <td className="py-2 pr-3 font-code-sm">{g.edges}</td>
                  <td className="py-2 pr-3 font-code-sm">{g.density.toFixed(3)}</td>
                  <td className="py-2 pr-3 font-code-sm">{g.avgDegree}</td>
                  <td className="py-2 pr-3 font-code-sm">{g.components}</td>
                  <td className="py-2 pr-3 font-code-sm">{g.isolated}</td>
                  <td className="py-2 pr-3 text-pc-ink2">{g.source}</td>
                  <td className="py-2 text-pc-ink2">{g.meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
};
