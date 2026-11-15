import React, { useEffect, useMemo, useState } from 'react';
import { fileUrl } from '../../lib/api';
import { Card, TIER_COLORS, pct } from '../lib';
import type { DiagnosisRecord } from './DiagnosisResult';

/** What each pipeline step produced for one photo (`result.trace` from the backend). */
export interface PipelineTrace {
  segmentation: { otsu_threshold: number; mask_ratio: number; leaf_pixels: number; image_size: [number, number]; accept_range: [number, number] };
  features: { feature: string; name: string; family: string; value: number; healthy_mean: number; z: number }[];
  crop_router: { trees: number; scores: { code: string; name: string; probability: number; votes: number }[] };
  graph: {
    nodes: { feature: string; name: string; family: string; value: number; level: number; cuts: number[] }[];
    edges: [number, number][];
    iterations: { labels: number[]; common: number[]; seen: number; rare: number; vocabulary: number }[];
    vector_size: number;
    vector_nonzero: number;
  };
  svm: {
    training_leaves: number;
    similarity: { mean: number; max: number; min: number; histogram: number[] };
    support_vectors: number;
    push_stressed: number;
    push_healthy: number;
    intercept: number;
    decision: number;
    probability_stressed: number;
    top_support: { label: 'healthy' | 'stressed'; similarity: number; weight: number; contribution: number }[];
    conditions: { code: string; name: string; probability: number }[];
  };
  timings_ms: { measure: number; crop_routing: number; wl_graph: number; svm: number };
  images: { exg?: string; mask?: string; regions?: string; spots?: string };
}

export const PIPELINE_STEPS = [
  { id: 'segment', icon: 'crop_free', title: 'Leaf segmentation & QC', text: 'Foreground mask extracted; images with implausible mask area are rejected.' },
  { id: 'spots', icon: 'blur_on', title: 'Spot check', text: 'Looks for small brown or purple spots, which the averaged measurements cannot show.' },
  { id: 'features', icon: 'tune', title: 'Feature extraction', text: '38 handcrafted features: morphology, colour, GLCM texture, core/edge spatial.' },
  { id: 'crop', icon: 'alt_route', title: 'Stage 1 · crop routing', text: 'Random Forest identifies which of the 8 crops the leaf belongs to.' },
  { id: 'graph', icon: 'hub', title: 'Feature graph + WL relabelling', text: "38 feature nodes labelled Low/Medium/High, linked by the crop's selected topology, refined for h WL iterations." },
  { id: 'svm', icon: 'model_training', title: 'Stage 2 · WL kernel SVM', text: 'The crop-specific WL subtree kernel SVM decides stressed vs. non-stressed and ranks likely conditions.' },
  { id: 'check', icon: 'verified_user', title: 'Reliability check', text: 'Is this photo like the training photos? Can the model see everything that matters?' },
] as const;
type StepId = (typeof PIPELINE_STEPS)[number]['id'];
const GRAPH_STEP = PIPELINE_STEPS.findIndex((p) => p.id === 'graph');

const LEVELS = ['Low', 'Medium', 'High'];
const STEP_MS = 5000;
const ITER_MS = 3000;

const fmt = (v: number) => {
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(3);
};
const signed = (v: number, d = 2) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(d)}`;

/** One-line result of each step, for the compact step list. */
export function stepSummary(record: DiagnosisRecord, i: number): string | null {
  const t = record.result.trace;
  if (!t) return null;
  const id: StepId = PIPELINE_STEPS[i].id;
  const r = record.result;
  if (id === 'spots') return r.spots ? (r.spots.flagged ? `${r.spots.count} spots found · ${pct(r.spots.area_ratio, 1)} of leaf` : `${r.spots.count === 0 ? 'No' : r.spots.count} clear spots`) : null;
  if (id === 'check') {
    const c = r.reliability_check;
    if (!c) return null;
    const failed = c.checks.filter((k) => !k.passed).length;
    return c.status === 'ok' && !r.spots?.flagged ? 'All checks passed' : `${failed} of ${c.checks.length} checks failed · ${c.status === 'unreliable' ? 'result unreliable' : 'use with care'}`;
  }
  if (id === 'segment') return `Leaf = ${pct(t.segmentation.mask_ratio, 1)} of the photo · threshold ${t.segmentation.otsu_threshold}`;
  if (id === 'features') {
    const top = [...t.features].sort((a, b) => Math.abs(b.z) - Math.abs(a.z))[0];
    return `38 values measured · most unusual: ${top.name}`;
  }
  if (id === 'crop') {
    const c = t.crop_router.scores[0];
    return `${c.name} · ${c.votes} of ${t.crop_router.trees} trees`;
  }
  if (id === 'graph') return `${t.graph.edges.length} edges · ${t.graph.iterations.length - 1} WL iterations · ${t.graph.vector_nonzero} patterns`;
  const s = t.svm;
  return `${s.decision < 0 ? 'Healthy' : 'Stressed'} side · decision ${signed(s.decision)} · ${pct(s.probability_stressed, 0)} stressed`;
}

/** Bar that grows from zero when it first appears. */
const GrowBar: React.FC<{ value: number; className?: string; delay?: number }> = ({ value, className = 'bg-pc-green-bg', delay = 0 }) => {
  const [w, setW] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setW(Math.max(0, Math.min(1, value))), 60 + delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return (
    <div className="h-2 w-full rounded-full bg-pc-cont overflow-hidden">
      <div className={`h-full rounded-full transition-[width] duration-700 ease-out ${className}`} style={{ width: `${w * 100}%` }} />
    </div>
  );
};

const Fact: React.FC<{ label: string; value: React.ReactNode; sub?: string }> = ({ label, value, sub }) => (
  <div className="p-3 rounded-xl bg-pc-low border border-pc-line/60">
    <p className="font-label-sm text-[10px] uppercase tracking-wider text-pc-muted">{label}</p>
    <p className="font-title-md text-pc-ink tabular-nums">{value}</p>
    {sub && <p className="font-body-sm text-[11px] text-pc-muted">{sub}</p>}
  </div>
);

const Explain: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <p className="flex gap-2 items-start rounded-lg bg-pc-low border border-pc-line/60 px-3 py-2 font-body-sm text-[13px] text-pc-ink2">
    <span className="material-symbols-outlined text-[18px] text-pc-green">lightbulb</span>
    <span>{children}</span>
  </p>
);

// ───────────────────────────────────────────── step 1: segmentation
const SegmentationStep: React.FC<{ record: DiagnosisRecord; t: PipelineTrace }> = ({ record, t }) => {
  const s = t.segmentation;
  const frames = [
    { src: fileUrl(record.image_url), title: '1. Your photo', text: `${s.image_size[0]} × ${s.image_size[1]} px` },
    { src: fileUrl(t.images.exg), title: '2. Greenness map (ExG)', text: 'ExG = 2g − r − b. Bright = green pixels.' },
    { src: fileUrl(t.images.mask), title: '3. Otsu mask', text: `Pixels above threshold ${s.otsu_threshold} → leaf. Largest region kept.` },
    { src: fileUrl(record.overlay_url), title: '4. Leaf found', text: 'Background dimmed, outline drawn.' },
  ];
  const [lo, hi] = s.accept_range;
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {frames.map((f, i) => (
          <figure key={f.title} className="flex flex-col gap-1.5 page-enter" style={{ animationDelay: `${i * 350}ms`, animationFillMode: 'both' }}>
            <div className="relative aspect-square rounded-xl overflow-hidden bg-pc-cont">
              {f.src ? <img src={f.src} alt={f.title} className="w-full h-full object-cover" /> : null}
              {i < 3 && (
                <span className="hidden lg:grid absolute -right-3 top-1/2 -translate-y-1/2 z-10 w-6 h-6 place-items-center rounded-full bg-pc-deep text-white">
                  <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
                </span>
              )}
            </div>
            <figcaption>
              <p className="font-title-sm text-[13px] text-pc-ink">{f.title}</p>
              <p className="font-body-sm text-[11px] text-pc-muted">{f.text}</p>
            </figcaption>
          </figure>
        ))}
      </div>
      <div className="p-3 rounded-xl bg-pc-low border border-pc-line/60">
        <div className="flex justify-between font-body-sm text-[12px] text-pc-ink2 mb-2">
          <span>Quality check: how much of the photo is leaf</span>
          <span className="font-semibold text-pc-ink">{pct(s.mask_ratio, 1)} → accepted</span>
        </div>
        <div className="relative h-3 rounded-full bg-pc-err/30 overflow-hidden">
          <div className="absolute inset-y-0 bg-pc-green-bg/60" style={{ left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }} />
          <div className="absolute inset-y-[-2px] w-1 rounded bg-pc-ink" style={{ left: `calc(${s.mask_ratio * 100}% - 2px)` }} />
        </div>
        <div className="flex justify-between font-body-sm text-[10px] text-pc-muted mt-1">
          <span>0% (no leaf → rejected)</span>
          <span>allowed {pct(lo, 0)}–{pct(hi, 0)}</span>
          <span>100% (whole photo → rejected)</span>
        </div>
      </div>
      <Explain>
        The model only looks at the leaf. Each pixel gets a greenness score, and Otsu's method picks the cut-off that best separates the two
        groups of pixels. Here {s.leaf_pixels.toLocaleString()} pixels were kept as leaf.
      </Explain>
    </div>
  );
};

// ───────────────────────────────────────────── step 2: features
const FeatureStep: React.FC<{ t: PipelineTrace }> = ({ t }) => {
  const families = useMemo(() => {
    const m = new Map<string, PipelineTrace['features']>();
    t.features.forEach((f) => m.set(f.family, [...(m.get(f.family) ?? []), f]));
    return [...m.entries()];
  }, [t]);
  return (
    <div className="grid grid-cols-1 xl:grid-cols-[260px_1fr] gap-5">
      <div className="flex flex-col gap-2">
        <div className="aspect-square rounded-xl overflow-hidden bg-pc-cont">
          {t.images.regions && <img src={fileUrl(t.images.regions)} alt="Leaf regions" className="w-full h-full object-cover" />}
        </div>
        <p className="font-body-sm text-[11px] text-pc-muted">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-[#4ade80] mr-1" /> core = centre third of the box ·{' '}
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-[#ffaa28] mr-1" /> edge = rest of the leaf. White box = bounding box (shape).
        </p>
        <Explain>Each bar compares the value with healthy leaves of the same crop. Long red bars are the unusual measurements.</Explain>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
        {families.map(([fam, list], fi) => (
          <div key={fam} className="page-enter" style={{ animationDelay: `${fi * 120}ms`, animationFillMode: 'both' }}>
            <p className="font-label-sm text-[11px] uppercase tracking-wider text-pc-green mb-1.5">{fam}</p>
            <ul className="flex flex-col gap-1">
              {list.map((f) => {
                const z = Math.max(-4, Math.min(4, f.z));
                const strong = Math.abs(f.z) >= 2;
                return (
                  <li key={f.feature} className="grid grid-cols-[1fr_64px_90px] items-center gap-2" title={`healthy average ${fmt(f.healthy_mean)} · z = ${f.z}`}>
                    <span className="font-body-sm text-[12px] text-pc-ink2 truncate">{f.name}</span>
                    <span className="font-body-sm text-[12px] text-pc-ink tabular-nums text-right">{fmt(f.value)}</span>
                    <span className="relative h-2 rounded-full bg-pc-cont">
                      <span className="absolute left-1/2 top-[-2px] bottom-[-2px] w-px bg-pc-outline" />
                      <span
                        className={`absolute top-0 bottom-0 rounded-full ${strong ? 'bg-pc-err' : 'bg-pc-green-bg'}`}
                        style={z >= 0 ? { left: '50%', width: `${(z / 8) * 100}%` } : { right: '50%', width: `${(-z / 8) * 100}%` }}
                      />
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
        <p className="md:col-span-2 font-body-sm text-[11px] text-pc-muted">
          Bar: ◀ lower · centre = healthy average · higher ▶ (up to 4 standard deviations).
        </p>
      </div>
    </div>
  );
};

// ───────────────────────────────────────────── step 3: crop routing
const CropStep: React.FC<{ record: DiagnosisRecord; t: PipelineTrace }> = ({ record, t }) => {
  const r = t.crop_router;
  const chosen = record.result.crop;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-5">
      <ul className="flex flex-col gap-2.5">
        {r.scores.map((c, i) => (
          <li key={c.code} className="grid grid-cols-[120px_1fr_110px] items-center gap-3">
            <span className={`font-title-sm text-[13px] ${c.code === chosen.code ? 'text-pc-green' : 'text-pc-ink2'}`}>
              {c.code === chosen.code && <span className="material-symbols-outlined text-[14px] align-[-2px] mr-0.5">check</span>}
              {c.name}
            </span>
            <GrowBar value={c.probability} delay={i * 90} className={c.code === chosen.code ? 'bg-pc-green-bg' : 'bg-pc-outline'} />
            <span className="font-body-sm text-[12px] text-pc-ink2 tabular-nums text-right">
              {c.votes} trees · {pct(c.probability, 0)}
            </span>
          </li>
        ))}
      </ul>
      <div className="flex flex-col gap-3">
        <Fact label="Forest" value={`${r.trees} decision trees`} sub="Each tree looks at the 38 values and votes for one crop." />
        <Fact
          label="Crop used"
          value={chosen.name}
          sub={chosen.source === 'user' ? 'You chose this crop, so it was used directly.' : `Won ${r.scores[0].votes} of ${r.trees} votes.`}
        />
        <Explain>The crop decides which crop-specific graph and SVM are used in the next two steps.</Explain>
      </div>
    </div>
  );
};

// ───────────────────────────────────────────── step 4: feature graph + WL
const commonColor = (c: number, seen: boolean) => (!seen ? '#6b7280' : c >= 0.2 ? '#22c55e' : c >= 0.05 ? '#f59e0b' : '#ef4444');

const GraphStep: React.FC<{ record: DiagnosisRecord; t: PipelineTrace; iter: number; setIter: (k: number) => void }> = ({ record, t, iter, setIter }) => {
  const g = t.graph;
  const [hover, setHover] = useState<number | null>(null);
  const it = g.iterations[iter];
  const highlight = useMemo(() => new Set(record.result.explanation.map((e) => e.feature)), [record]);

  // nodes on a circle, families separated by a small gap
  const pos = useMemo(() => {
    const fams: string[] = [];
    g.nodes.forEach((n) => fams.includes(n.family) || fams.push(n.family));
    const slots = g.nodes.length + fams.length;
    let slot = 0;
    let last = '';
    const p = g.nodes.map((n) => {
      if (n.family !== last) (slot += 1), (last = n.family);
      const a = (2 * Math.PI * slot) / slots - Math.PI / 2;
      slot += 1;
      return { x: 260 + 190 * Math.cos(a), y: 260 + 190 * Math.sin(a), a };
    });
    const labels = fams.map((f) => {
      const idx = g.nodes.map((n, i) => (n.family === f ? i : -1)).filter((i) => i >= 0);
      const a = idx.reduce((s, i) => s + p[i].a, 0) / idx.length;
      return { f, x: 260 + 236 * Math.cos(a), y: 260 + 236 * Math.sin(a) };
    });
    return { p, labels };
  }, [g]);

  const neighbours = useMemo(() => {
    const nb: number[][] = g.nodes.map(() => []);
    g.edges.forEach(([a, b]) => (nb[a].push(b), nb[b].push(a)));
    return nb;
  }, [g]);

  const h = hover;
  const node = h !== null ? g.nodes[h] : null;
  const color = (i: number) => (iter === 0 ? TIER_COLORS[g.nodes[i].level] : commonColor(it.common[i], it.labels[i] >= 0));

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,560px)_1fr] gap-5">
      <div>
        <svg viewBox="0 0 520 520" className="w-full max-w-[560px] mx-auto" role="img" aria-label="Feature graph of this leaf">
          {g.edges.map(([a, b], i) => {
            const on = h !== null && (a === h || b === h);
            return (
              <line
                key={i}
                x1={pos.p[a].x}
                y1={pos.p[a].y}
                x2={pos.p[b].x}
                y2={pos.p[b].y}
                className={on ? 'stroke-pc-green' : 'stroke-pc-outline'}
                strokeWidth={on ? 2.2 : 1}
                strokeOpacity={h === null ? 0.45 : on ? 1 : 0.12}
              />
            );
          })}
          {pos.labels.map((l) => (
            <text key={l.f} x={l.x} y={l.y} textAnchor="middle" dominantBaseline="middle" className="fill-pc-muted" fontSize="11" fontWeight={600}>
              {l.f}
            </text>
          ))}
          {g.nodes.map((n, i) => (
            <g key={n.feature} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onClick={() => setHover(i)} className="cursor-pointer">
              {highlight.has(n.feature) && <circle cx={pos.p[i].x} cy={pos.p[i].y} r={14} fill="none" className="stroke-pc-err" strokeWidth={2} strokeDasharray="3 2" />}
              <circle
                cx={pos.p[i].x}
                cy={pos.p[i].y}
                r={h === i ? 11 : 9}
                fill={color(i)}
                className="stroke-pc-card transition-[fill] duration-500"
                strokeWidth={2}
              />
            </g>
          ))}
        </svg>
        <div className="flex flex-wrap justify-center gap-x-4 gap-y-1 font-body-sm text-[11px] text-pc-muted">
          {iter === 0 ? (
            LEVELS.map((l, i) => (
              <span key={l} className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: TIER_COLORS[i] }} /> {l}
              </span>
            ))
          ) : (
            <>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-[#22c55e]" /> common pattern (≥20% of training leaves)</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-[#f59e0b]" /> uncommon (5–20%)</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-[#ef4444]" /> rare (&lt;5%)</span>
              <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full bg-[#6b7280]" /> never seen</span>
            </>
          )}
          <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full border-2 border-dashed border-pc-err" /> in "What stands out"</span>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex gap-1 p-1 rounded-xl bg-pc-low border border-pc-line/60 w-fit">
          {g.iterations.map((_, k) => (
            <button
              key={k}
              type="button"
              onClick={() => setIter(k)}
              className={`px-3 py-1.5 rounded-lg font-title-sm text-[12px] ${iter === k ? 'bg-pc-card text-pc-ink shadow-xs' : 'text-pc-muted hover:text-pc-ink'}`}
            >
              {k === 0 ? 'Start: Low/Med/High' : `WL iteration ${k}`}
            </button>
          ))}
        </div>

        {iter === 0 ? (
          <Explain>
            Every measurement is a node. Its first label is <b>Low</b>, <b>Medium</b> or <b>High</b>, compared with the training leaves of this crop.
            Lines join measurements that the <b>{record.result.model.graph_name}</b> graph connects ({g.edges.length} edges).
          </Explain>
        ) : (
          <Explain>
            WL iteration {iter}: each node's new label is <b>its own label + the labels of all its neighbours</b>, so it now describes a
            neighbourhood {iter} step{iter > 1 ? 's' : ''} wide. Colour shows how common that neighbourhood pattern was among the training leaves.
            Rare patterns make this leaf look different.
          </Explain>
        )}

        <div className="grid grid-cols-3 gap-2">
          <Fact label="Known patterns" value={`${it.seen} / ${g.nodes.length}`} sub="seen in training" />
          <Fact label="Rare patterns" value={it.rare} sub="< 5% of training leaves" />
          <Fact label="Vocabulary" value={it.vocabulary.toLocaleString()} sub="labels at this step" />
        </div>

        <div className="p-3 rounded-xl border border-pc-line/60 min-h-[132px]">
          {node && h !== null ? (
            <div className="flex flex-col gap-1 font-body-sm text-[12px] text-pc-ink2">
              <p className="font-title-sm text-pc-ink">
                {node.name} <span className="text-pc-muted font-normal">· {node.family}</span>
              </p>
              <p>
                Value <b className="text-pc-ink">{fmt(node.value)}</b> → <b style={{ color: TIER_COLORS[node.level] }}>{LEVELS[node.level]}</b>
                {node.cuts.length === 2 && (
                  <span className="text-pc-muted"> (Low ≤ {fmt(node.cuts[0])} &lt; Medium ≤ {fmt(node.cuts[1])} &lt; High)</span>
                )}
              </p>
              <p>
                Neighbours ({neighbours[h].length}):{' '}
                {neighbours[h].length ? neighbours[h].map((j) => g.nodes[j].name).join(', ') : 'none, so its label never changes'}
              </p>
              {iter > 0 && (
                <p>
                  Pattern at this step:{' '}
                  {it.labels[h] < 0 ? <b className="text-pc-err">never seen in training</b> : <b className="text-pc-ink">{pct(it.common[h], 1)} of training leaves</b>}
                </p>
              )}
            </div>
          ) : (
            <p className="font-body-sm text-[12px] text-pc-muted">Point at (or tap) a node to see its value, its level and its neighbours.</p>
          )}
        </div>

        <Explain>
          Counting every node's label at every step turns the leaf into a vector of {g.vector_size.toLocaleString()} pattern counts ({g.vector_nonzero} of them
          non-zero). This vector is what the WL kernel compares in the next step.
        </Explain>
      </div>
    </div>
  );
};

// ───────────────────────────────────────────── step 5: SVM
const SvmStep: React.FC<{ record: DiagnosisRecord; t: PipelineTrace }> = ({ record, t }) => {
  const s = t.svm;
  const maxH = Math.max(...s.similarity.histogram, 1);
  const total = s.push_stressed - s.push_healthy + Math.abs(s.intercept) || 1;
  const stressed = record.result.stress.status === 'stressed';
  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div>
          <p className="font-title-sm text-[13px] text-pc-ink mb-1">A. Compare with every training leaf</p>
          <p className="font-body-sm text-[12px] text-pc-muted mb-3">
            WL kernel similarity (0 = nothing in common, 1 = identical patterns) to the {s.training_leaves} training leaves.
          </p>
          <div className="flex items-end gap-1 h-28">
            {s.similarity.histogram.map((v, i) => (
              <div key={i} className="flex-1 flex flex-col items-center justify-end h-full" title={`${v} leaves with similarity ${i / 10}–${(i + 1) / 10}`}>
                <span className="font-body-sm text-[10px] text-pc-muted">{v || ''}</span>
                <div className="w-full rounded-t bg-pc-green-bg/80 transition-[height] duration-700" style={{ height: `${(v / maxH) * 85}%` }} />
              </div>
            ))}
          </div>
          <div className="flex justify-between font-body-sm text-[10px] text-pc-muted mt-1">
            <span>0</span>
            <span>similarity</span>
            <span>1</span>
          </div>
          <div className="grid grid-cols-3 gap-2 mt-3">
            <Fact label="Most similar" value={s.similarity.max.toFixed(2)} />
            <Fact label="Average" value={s.similarity.mean.toFixed(2)} />
            <Fact label="Support vectors" value={s.support_vectors} sub="leaves the SVM uses" />
          </div>
        </div>

        <div>
          <p className="font-title-sm text-[13px] text-pc-ink mb-1">B. Weigh the votes</p>
          <p className="font-body-sm text-[12px] text-pc-muted mb-3">
            Each support vector pulls the answer towards its own label, as strongly as <i>weight × similarity</i>.
          </p>
          <div className="flex h-8 rounded-lg overflow-hidden font-body-sm text-[11px] text-white font-semibold">
            <div className="bg-pc-green-bg grid place-items-center transition-[width] duration-700" style={{ width: `${(-s.push_healthy / total) * 100}%` }}>
              healthy {(-s.push_healthy).toFixed(1)}
            </div>
            <div className="bg-pc-err grid place-items-center transition-[width] duration-700" style={{ width: `${(s.push_stressed / total) * 100}%` }}>
              stressed {s.push_stressed.toFixed(1)}
            </div>
          </div>
          <div className="mt-3 p-3 rounded-xl bg-pc-low border border-pc-line/60 font-body-sm text-[13px] text-pc-ink2 tabular-nums">
            <div className="flex justify-between"><span>Pull towards stressed</span><span>{signed(s.push_stressed)}</span></div>
            <div className="flex justify-between"><span>Pull towards healthy</span><span>{signed(s.push_healthy)}</span></div>
            <div className="flex justify-between"><span>Bias (intercept)</span><span>{signed(s.intercept)}</span></div>
            <div className="flex justify-between border-t border-pc-line/60 mt-1 pt-1 font-semibold text-pc-ink">
              <span>Decision value</span>
              <span className={s.decision < 0 ? 'text-pc-green' : 'text-pc-err'}>{signed(s.decision)}</span>
            </div>
          </div>
          <p className="font-body-sm text-[12px] text-pc-muted mt-2">
            Below 0 → healthy, above 0 → stressed. Converted to a probability: <b className="text-pc-ink">{pct(s.probability_stressed, 0)} stressed</b> →{' '}
            <b className={stressed ? 'text-pc-err' : 'text-pc-green'}>{stressed ? 'Stressed' : 'Healthy'}</b>.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div>
          <p className="font-title-sm text-[13px] text-pc-ink mb-2">Strongest pulls (training leaves)</p>
          <table className="w-full font-body-sm text-[12px] text-pc-ink2 tabular-nums">
            <thead>
              <tr className="text-pc-muted text-left">
                <th className="font-normal py-1">Training leaf</th>
                <th className="font-normal text-right">Similarity</th>
                <th className="font-normal text-right">Weight</th>
                <th className="font-normal text-right">Pull</th>
              </tr>
            </thead>
            <tbody>
              {s.top_support.map((v, i) => (
                <tr key={i} className="border-t border-pc-line/50">
                  <td className="py-1.5">
                    <span className={`inline-block w-2 h-2 rounded-full mr-1.5 ${v.label === 'stressed' ? 'bg-pc-err' : 'bg-pc-green-bg'}`} />
                    {v.label}
                  </td>
                  <td className="text-right">{v.similarity.toFixed(2)}</td>
                  <td className="text-right">{v.weight.toFixed(2)}</td>
                  <td className={`text-right font-semibold ${v.contribution > 0 ? 'text-pc-err' : 'text-pc-green'}`}>{signed(v.contribution)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <p className="font-title-sm text-[13px] text-pc-ink mb-2">C. Second SVM: which condition?</p>
          <ul className="flex flex-col gap-2">
            {s.conditions.map((c, i) => (
              <li key={c.code} className="grid grid-cols-[minmax(0,1fr)_120px_44px] items-center gap-2">
                <span className={`font-body-sm text-[12px] truncate ${i < 3 ? 'text-pc-ink font-semibold' : 'text-pc-ink2'}`}>
                  {i < 3 ? `${i + 1}. ` : ''}
                  {c.name}
                </span>
                <GrowBar value={c.probability} delay={i * 80} className={i < 3 ? 'bg-pc-green-bg' : 'bg-pc-outline'} />
                <span className="font-body-sm text-[12px] text-pc-ink2 tabular-nums text-right">{pct(c.probability, 0)}</span>
              </li>
            ))}
          </ul>
          <p className="font-body-sm text-[11px] text-pc-muted mt-2">The same WL vector goes into a second SVM that ranks every condition of this crop; the top 3 are reported.</p>
        </div>
      </div>
    </div>
  );
};

// ───────────────────────────────────────────── spot check
const NotRecorded: React.FC = () => (
  <p className="font-body-sm text-[13px] text-pc-muted">This step was added after this diagnosis was made. Analyse the photo again to see it.</p>
);

const SpotStep: React.FC<{ record: DiagnosisRecord; t: PipelineTrace }> = ({ record, t }) => {
  const sp = record.result.spots;
  if (!sp || !t.images.spots) return <NotRecorded />;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,420px)_1fr] gap-5">
      <figure className="flex flex-col gap-1.5">
        <div className="rounded-xl overflow-hidden bg-pc-cont">
          <img src={fileUrl(t.images.spots)} alt="Leaf with detected spots circled" className="w-full object-contain" />
        </div>
        <figcaption className="font-body-sm text-[11px] text-pc-muted">Red circles = spots found. The leaf is scaled to a standard size first.</figcaption>
      </figure>
      <div className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-2">
          <Fact label="Spots found" value={sp.count} />
          <Fact label="Leaf covered" value={pct(sp.area_ratio, 2)} />
        </div>
        <p
          className={`flex gap-2 items-start rounded-lg px-3 py-2 font-body-sm text-[13px] ${
            sp.flagged ? 'bg-pc-amber-bg text-pc-amber' : 'bg-pc-mint text-pc-mint-ink'
          }`}
        >
          <span className="material-symbols-outlined text-[18px]">{sp.flagged ? 'warning' : 'check_circle'}</span>
          {sp.flagged ? sp.message : 'No clear spots. (A warning needs at least 3 spots covering 0.2% of the leaf.)'}
        </p>
        <Explain>
          The 38 measurements are averages over the whole leaf, so a few small spots hardly change them and the WL models cannot see them.
          This check looks for them directly: pixels whose colour has turned from green towards brown or purple compared with the leaf
          around them. Shadows and veins change brightness but stay green, so they are ignored. It does not change the model&apos;s answer; it
          only warns.
        </Explain>
        <p className="font-body-sm text-[11px] text-pc-muted">
          Tested on OLID-I: raised a warning on 1 of 320 healthy leaves. Tuned to avoid false alarms, so very small or dark spots can be missed.
        </p>
      </div>
    </div>
  );
};

// ───────────────────────────────────────────── reliability check
export const ReliabilityChecks: React.FC<{ record: DiagnosisRecord }> = ({ record }) => {
  const c = record.result.reliability_check;
  if (!c) return <NotRecorded />;
  return (
    <div className="flex flex-col gap-3">
      <ul className="flex flex-col gap-2">
        {c.checks.map((k) => (
          <li key={k.id} className="flex gap-3 items-start p-3 rounded-xl border border-pc-line/60">
            <span className={`material-symbols-outlined text-[20px] ${k.passed ? 'text-pc-green' : k.id === 'spots' ? 'text-pc-amber' : 'text-pc-err'}`}>
              {k.passed ? 'check_circle' : k.id === 'spots' ? 'warning' : 'cancel'}
            </span>
            <div>
              <p className="font-title-sm text-[13px] text-pc-ink">{k.label}</p>
              <p className="font-body-sm text-[12px] text-pc-muted">{k.detail}</p>
            </div>
          </li>
        ))}
      </ul>
      <div className="p-3 rounded-xl bg-pc-low border border-pc-line/60">
        <p className="font-title-sm text-[13px] text-pc-ink">What the {record.result.crop.name.toLowerCase()} model can recognise</p>
        <div className="flex flex-wrap gap-1.5 mt-2">
          {c.known_conditions.map((n) => (
            <span key={n} className="px-2 py-0.5 rounded-md bg-pc-card border border-pc-line/60 font-body-sm text-[12px] text-pc-ink2">
              {n}
            </span>
          ))}
        </div>
        <p className="font-body-sm text-[11px] text-pc-muted mt-2">
          Any other problem (for example a disease not in this list) will be reported as the closest of these.
        </p>
      </div>
    </div>
  );
};

const CheckStep: React.FC<{ record: DiagnosisRecord }> = ({ record }) => {
  const c = record.result.reliability_check;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_300px] gap-5">
      <ReliabilityChecks record={record} />
      <div className="flex flex-col gap-3">
        {c && (
          <Fact
            label="Reliability"
            value={c.status === 'ok' ? 'Looks reliable' : c.status === 'caution' ? 'Use with care' : 'Unreliable'}
            sub={c.status === 'ok' ? 'The photo looks like the training photos.' : 'The photo differs from the training photos.'}
          />
        )}
        <Explain>
          The limits come from cross-validation on the training leaves: a check fails only when this photo is more unusual than 99% of
          leaves the model had not seen during training. One failed check means &quot;use with care&quot;; two or more mean &quot;unreliable&quot;.
        </Explain>
      </div>
    </div>
  );
};

// ───────────────────────────────────────────── walkthrough
interface Props {
  record: DiagnosisRecord;
  step: number;
  onStep: (i: number) => void;
  autoPlay?: boolean;
}

export const PipelineWalkthrough: React.FC<Props> = ({ record, step, onStep, autoPlay = false }) => {
  const t = record.result.trace;
  const [playing, setPlaying] = useState(autoPlay);
  const [iter, setIter] = useState(0);
  const lastIter = (t?.graph.iterations.length ?? 1) - 1;

  useEffect(() => setIter(0), [step]);
  useEffect(() => {
    if (!playing || !t) return;
    const inGraph = step === GRAPH_STEP && iter < lastIter;
    const timer = setTimeout(
      () => {
        if (inGraph) setIter(iter + 1);
        else if (step < PIPELINE_STEPS.length - 1) onStep(step + 1);
        else setPlaying(false);
      },
      step === GRAPH_STEP ? ITER_MS : STEP_MS,
    );
    return () => clearTimeout(timer);
  }, [playing, step, iter, lastIter, onStep, t]);

  if (!t) return null;
  const tm = t.timings_ms;
  const s = PIPELINE_STEPS[step];
  const times: Record<StepId, string> = {
    segment: `${tm.measure} ms (segmentation, spot check and measurements together)`,
    spots: `${tm.measure} ms (with segmentation and measurements)`,
    features: `${tm.measure} ms (with segmentation and spot check)`,
    crop: `${tm.crop_routing} ms`,
    graph: `${tm.wl_graph} ms`,
    svm: `${tm.svm} ms`,
    check: 'under 1 ms',
  };

  return (
    <Card className="flex flex-col gap-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="font-label-sm text-[11px] uppercase tracking-wider text-pc-muted">How the pipeline handled this leaf</p>
          <h2 className="font-headline-sm text-pc-ink flex items-center gap-2 mt-0.5">
            <span className="grid place-items-center w-8 h-8 rounded-full bg-pc-deep text-white text-[14px]">{step + 1}</span>
            {s.title}
          </h2>
          <p className="font-body-sm text-[12px] text-pc-muted mt-0.5">
            {s.text} · took {times[s.id]}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <button type="button" onClick={() => (setPlaying(false), onStep(Math.max(0, step - 1)))} disabled={step === 0} className="p-2 rounded-lg border border-pc-line text-pc-ink2 hover:bg-pc-low disabled:opacity-40" aria-label="Previous step">
            <span className="material-symbols-outlined text-[20px]">skip_previous</span>
          </button>
          <button
            type="button"
            onClick={() => {
              if (!playing && step === PIPELINE_STEPS.length - 1) onStep(0);
              setPlaying(!playing);
            }}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-pc-deep text-white font-title-sm text-[13px] hover:bg-pc-forest"
          >
            <span className="material-symbols-outlined text-[20px]">{playing ? 'pause' : 'play_arrow'}</span>
            {playing ? 'Pause' : step === PIPELINE_STEPS.length - 1 ? 'Replay' : 'Play'}
          </button>
          <button type="button" onClick={() => (setPlaying(false), onStep(Math.min(PIPELINE_STEPS.length - 1, step + 1)))} disabled={step === PIPELINE_STEPS.length - 1} className="p-2 rounded-lg border border-pc-line text-pc-ink2 hover:bg-pc-low disabled:opacity-40" aria-label="Next step">
            <span className="material-symbols-outlined text-[20px]">skip_next</span>
          </button>
        </div>
      </div>

      {/* step pills with progress */}
      <div className="grid grid-cols-7 gap-1.5">
        {PIPELINE_STEPS.map((p, i) => (
          <button key={p.title} type="button" onClick={() => (setPlaying(false), onStep(i))} className="flex flex-col gap-1 text-left group" aria-label={`Step ${i + 1}: ${p.title}`}>
            <span className="h-1.5 rounded-full bg-pc-cont overflow-hidden">
              <span
                key={`${i}-${step}-${playing}`}
                className={`block h-full rounded-full bg-pc-green-bg ${i === step && playing && i !== GRAPH_STEP ? 'pc-progress' : ''}`}
                style={{ width: i < step ? '100%' : i === step ? (playing && i !== GRAPH_STEP ? undefined : '100%') : '0%', animationDuration: `${STEP_MS}ms` }}
              />
            </span>
            <span className={`hidden md:flex items-center gap-1 font-body-sm text-[11px] ${i === step ? 'text-pc-ink font-semibold' : 'text-pc-muted group-hover:text-pc-ink'}`}>
              <span className="material-symbols-outlined text-[14px]">{p.icon}</span>
              <span className="truncate">{p.title}</span>
            </span>
          </button>
        ))}
      </div>

      <div key={step} className="page-enter">
        {s.id === 'segment' && <SegmentationStep record={record} t={t} />}
        {s.id === 'spots' && <SpotStep record={record} t={t} />}
        {s.id === 'features' && <FeatureStep t={t} />}
        {s.id === 'crop' && <CropStep record={record} t={t} />}
        {s.id === 'graph' && <GraphStep record={record} t={t} iter={iter} setIter={(k) => (setPlaying(false), setIter(k))} />}
        {s.id === 'svm' && <SvmStep record={record} t={t} />}
        {s.id === 'check' && <CheckStep record={record} />}
      </div>
    </Card>
  );
};
