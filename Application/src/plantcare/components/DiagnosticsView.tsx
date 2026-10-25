import React, { useMemo, useState } from 'react';
import { PageView, Sample } from '../types';
import { RESEARCH } from '../data/research';
import {
  CROP_BY_ID,
  Card,
  Chip,
  SampleThumb,
  SectionTitle,
  TIER_COLORS,
  Verdict,
  className as classLabelName,
  conditionName,
  cropName,
  featureLabel,
  pct,
  stressLabel,
  tertile,
} from '../lib';

interface DiagnosticsViewProps {
  sample: Sample;
  onNavigate: (page: PageView) => void;
  onOpenSample: (id: string) => void;
}

const FAMILIES = ['Morphology', 'Colour', 'Texture', 'Spatial'] as const;
const TIERS = RESEARCH.config.bin_labels;

/** Circular layout of the 38 feature nodes, grouped by family with a gap between groups. */
function nodePositions(size: number) {
  const n = RESEARCH.features.length;
  const gap = 1.5;
  const slots = n + gap * FAMILIES.length;
  let k = 0;
  let last = '';
  return RESEARCH.features.map((f) => {
    if (f.family !== last) {
      if (last) k += gap;
      last = f.family;
    }
    const a = (k++ / slots) * Math.PI * 2 - Math.PI / 2;
    const rad = size * 0.4;
    return { x: size / 2 + rad * Math.cos(a), y: size / 2 + rad * Math.sin(a), a };
  });
}

const FeatureGraph: React.FC<{ sample: Sample; graph: string }> = ({ sample, graph }) => {
  const size = 420;
  const pos = useMemo(() => nodePositions(size), []);
  const [hover, setHover] = useState<number | null>(null);
  const edges = RESEARCH.edgeLists[graph] ?? [];
  const labels = sample.features.map((v, i) => tertile(i, v));
  const near = new Set(hover === null ? [] : edges.filter(([a, b]) => a === hover || b === hover).flat());

  return (
    <div className="relative">
      <svg viewBox={`-60 -10 ${size + 120} ${size + 20}`} className="w-full max-w-[520px] mx-auto block" role="img" aria-label={`${graph} feature graph`}>
        {edges.map(([a, b], i) => {
          const on = hover !== null && (a === hover || b === hover);
          return (
            <line
              key={i}
              x1={pos[a].x}
              y1={pos[a].y}
              x2={pos[b].x}
              y2={pos[b].y}
              stroke="currentColor"
              className={on ? 'text-pc-green' : 'text-pc-outline'}
              strokeOpacity={hover === null ? 0.55 : on ? 1 : 0.15}
              strokeWidth={on ? 1.6 : 0.9}
            />
          );
        })}
        {pos.map((p, i) => (
          <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} className="cursor-pointer">
            <circle
              cx={p.x}
              cy={p.y}
              r={hover === i ? 9 : 7}
              fill={TIER_COLORS[labels[i]]}
              stroke="currentColor"
              className="text-pc-card"
              strokeWidth={2}
              opacity={hover === null || hover === i || near.has(i) ? 1 : 0.35}
            />
          </g>
        ))}
        {FAMILIES.map((fam) => {
          const idx = RESEARCH.features.map((f, i) => (f.family === fam ? i : -1)).filter((i) => i >= 0);
          const a = (pos[idx[0]].a + pos[idx[idx.length - 1]].a) / 2;
          return (
            <text
              key={fam}
              x={size / 2 + size * 0.45 * Math.cos(a)}
              y={size / 2 + size * 0.46 * Math.sin(a)}
              textAnchor={Math.cos(a) > 0.35 ? 'start' : Math.cos(a) < -0.35 ? 'end' : 'middle'}
              dominantBaseline="middle"
              className="fill-pc-muted"
              style={{ font: '600 11px Inter, sans-serif' }}
            >
              {fam}
            </text>
          );
        })}
      </svg>
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none w-40">
        {hover === null ? (
          <>
            <p className="font-title-sm text-pc-ink">{graph}</p>
            <p className="font-body-sm text-[11px] text-pc-muted">{edges.length} edges · hover a node</p>
          </>
        ) : (
          <>
            <p className="font-title-sm text-[13px] text-pc-ink">{featureLabel(RESEARCH.features[hover].id)}</p>
            <p className="font-code-sm text-pc-muted">{sample.features[hover].toPrecision(4)}</p>
            <p className="font-body-sm text-[11px] font-semibold" style={{ color: TIER_COLORS[labels[hover]] }}>
              {TIERS[labels[hover]]} · {near.size ? near.size - 1 : 0} neighbours
            </p>
          </>
        )}
      </div>
    </div>
  );
};

export const DiagnosticsView: React.FC<DiagnosticsViewProps> = ({ sample, onNavigate, onOpenSample }) => {
  const [graph, setGraph] = useState(sample.graph);
  const [sortByDeviation, setSortByDeviation] = useState(true);
  const crop = CROP_BY_ID[sample.crop];
  const others = RESEARCH.samples.filter((s) => s.classLabel === sample.classLabel && s.id !== sample.id);
  const fineOk = sample.fineGrainedPrediction === sample.classLabel;

  // deviation from the same crop's healthy leaves, in units of the feature's dataset SD
  const rows = RESEARCH.features.map((f, i) => {
    const sd = RESEARCH.featureStats[i][1] || 1;
    return { ...f, i, v: sample.features[i], z: (sample.features[i] - crop.healthyMeans[i]) / sd, tier: tertile(i, sample.features[i]) };
  });

  const stages = [
    {
      icon: 'alt_route',
      title: 'Stage 1 · crop routing',
      model: 'Random Forest on 38 features',
      result: cropName(sample.predictedCrop),
      truth: cropName(sample.crop),
      ok: sample.routingCorrect,
    },
    {
      icon: 'hub',
      title: 'Stage 2 · stress detection',
      model: `${cropName(sample.predictedCrop)} WL kernel SVM · ${sample.graph}, h = ${sample.height}`,
      result: stressLabel(sample.predictedStress),
      truth: stressLabel(sample.trueStress),
      ok: sample.stressCorrect,
    },
    {
      icon: 'public',
      title: 'Global WL model (reference)',
      model: 'All crops pooled · Hybrid graph, h = 3',
      result: `${stressLabel(sample.globalPrediction)}${sample.globalProbability !== null ? ` · P(stress) ${sample.globalProbability.toFixed(2)}` : ''}`,
      truth: stressLabel(sample.trueStress),
      ok: sample.globalPrediction === sample.trueStress,
    },
    {
      icon: 'category',
      title: '57-class reference',
      model: 'Flat RBF SVM on 38 features',
      result: sample.fineGrainedPrediction ? classLabelName(sample.fineGrainedPrediction) : '—',
      truth: classLabelName(sample.classLabel),
      ok: fineOk,
    },
  ];

  return (
    <div className="flex flex-col gap-6 w-full max-w-[1440px] mx-auto">
      {/* Header */}
      <Card className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <Chip className="bg-pc-low text-pc-ink2 font-code-sm">{sample.file}</Chip>
            <Chip className="bg-pc-high text-pc-ink2">Fold {sample.fold + 1} of 5 held out</Chip>
            <Verdict ok={sample.stressCorrect} label={sample.stressCorrect ? 'Stress call correct' : 'Stress call wrong'} />
          </div>
          <h1 className="font-headline-md text-pc-ink mt-2">
            {crop.name} · {conditionName(sample.condition)}
          </h1>
          <p className="font-body-sm text-pc-muted mt-0.5">
            <i>{crop.scientific}</i> · class <span className="font-code-sm">{sample.classLabel}</span> · leaf mask covers {pct(sample.maskAreaRatio)} of the frame
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => onNavigate('analyze-plant')}
            className="inline-flex items-center gap-2 bg-pc-deep text-white px-4 py-2 rounded-lg hover:bg-pc-forest font-title-sm"
          >
            <span className="material-symbols-outlined text-[18px]">document_scanner</span> Analyze another
          </button>
          <button
            onClick={() => window.print()}
            className="inline-flex items-center gap-2 bg-pc-card border border-pc-line text-pc-ink px-4 py-2 rounded-lg hover:bg-pc-low font-title-sm"
          >
            <span className="material-symbols-outlined text-[18px]">print</span> Print / PDF
          </button>
        </div>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Image + decisions */}
        <div className="xl:col-span-2 flex flex-col gap-6">
          <Card className="p-3">
            <img src={sample.image} alt={sample.classLabel} className="w-full rounded-xl object-cover aspect-[4/3]" />
          </Card>
          <Card>
            <SectionTitle icon="fact_check" title="Decisions" sub="Recorded out-of-fold predictions for this image" />
            <div className="flex flex-col gap-3">
              {stages.map((s) => (
                <div key={s.title} className="p-3 rounded-xl border border-pc-line/60 bg-pc-low">
                  <div className="flex items-start justify-between gap-2">
                    <p className="font-title-sm text-[13px] text-pc-ink flex items-center gap-1.5">
                      <span className="material-symbols-outlined text-[18px] text-pc-green">{s.icon}</span>
                      {s.title}
                    </p>
                    <Verdict ok={s.ok} />
                  </div>
                  <p className="font-body-sm text-[11px] text-pc-muted mt-0.5">{s.model}</p>
                  <div className="grid grid-cols-2 gap-2 mt-2 font-body-sm text-[12px]">
                    <div>
                      <p className="text-pc-muted">Predicted</p>
                      <p className="text-pc-ink font-semibold">{s.result}</p>
                    </div>
                    <div>
                      <p className="text-pc-muted">Ground truth</p>
                      <p className="text-pc-ink font-semibold">{s.truth}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Graph + features */}
        <div className="xl:col-span-3 flex flex-col gap-6">
          <Card>
            <SectionTitle
              icon="hub"
              title="Feature Graph"
              sub="Nodes are the 38 features, coloured by their discretised WL label; edges come from the chosen topology"
            />
            <div className="flex flex-wrap gap-1.5 mb-3">
              {RESEARCH.graphs.map((g) => (
                <button
                  key={g.graph}
                  onClick={() => setGraph(g.graph)}
                  className={`px-2.5 py-1 rounded-md font-body-sm text-[12px] transition-colors ${
                    graph === g.graph ? 'bg-pc-deep text-white' : 'bg-pc-low text-pc-ink2 hover:bg-pc-cont'
                  }`}
                >
                  {g.graph}
                  {g.graph === sample.graph && ' ★'}
                </button>
              ))}
            </div>
            <FeatureGraph sample={sample} graph={graph} />
            <div className="flex flex-wrap justify-center gap-4 mt-2 font-body-sm text-[11px] text-pc-muted">
              {TIERS.map((t, i) => (
                <span key={t} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: TIER_COLORS[i] }} /> {t}
                </span>
              ))}
              <span>★ topology selected for this crop &amp; fold</span>
            </div>
            <p className="font-body-sm text-[11px] text-pc-muted mt-2 text-center">
              Labels here use dataset-wide tertiles for display; in the experiments the bins were fitted on each training fold only.
            </p>
          </Card>

          <Card>
            <SectionTitle
              icon="tune"
              title="Feature Profile"
              sub={`Deviation from healthy ${crop.name.toLowerCase()} leaves (mean of ${crop.healthy}), in dataset standard deviations`}
              right={
                <button
                  onClick={() => setSortByDeviation((v) => !v)}
                  className="font-label-sm text-[11px] text-pc-green uppercase tracking-wider hover:underline shrink-0"
                >
                  {sortByDeviation ? 'Group by family' : 'Sort by deviation'}
                </button>
              }
            />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
              {(sortByDeviation ? [...rows].sort((a, b) => Math.abs(b.z) - Math.abs(a.z)) : rows).map((r) => (
                <div key={r.id} className="flex items-center gap-2 py-1.5 border-b border-pc-line/40">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ background: TIER_COLORS[r.tier] }} title={TIERS[r.tier]} />
                  <span className="font-body-sm text-[12px] text-pc-ink w-36 truncate" title={`${r.family} · ${r.id}`}>
                    {featureLabel(r.id)}
                  </span>
                  <div className="flex-1 h-2 relative bg-pc-cont rounded-full">
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-pc-outline" />
                    <div
                      className={`absolute top-0 bottom-0 rounded-full ${r.z >= 0 ? 'bg-[#f59e0b]' : 'bg-[#60a5fa]'}`}
                      style={
                        r.z >= 0
                          ? { left: '50%', width: `${Math.min(Math.abs(r.z), 3) / 6 * 100}%` }
                          : { right: '50%', width: `${Math.min(Math.abs(r.z), 3) / 6 * 100}%` }
                      }
                    />
                  </div>
                  <span className="font-code-sm text-[11px] text-pc-muted w-12 text-right">
                    {r.z > 0 ? '+' : ''}
                    {r.z.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {others.length > 0 && (
        <Card>
          <SectionTitle icon="photo_library" title="Same class" sub={classLabelName(sample.classLabel)} />
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
            {others.map((s) => (
              <SampleThumb key={s.id} sample={s} onClick={() => onOpenSample(s.id)} />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};
