import React from 'react';
import { RESEARCH } from './data/research';
import type { ConditionSummary, CropSummary, Sample } from './types';

export const CROP_BY_ID = Object.fromEntries(RESEARCH.crops.map((c) => [c.id, c])) as Record<string, CropSummary>;
export const CONDITION_BY_CODE = Object.fromEntries(RESEARCH.conditions.map((c) => [c.code, c])) as Record<string, ConditionSummary>;
export const SAMPLE_BY_ID = Object.fromEntries(RESEARCH.samples.map((s) => [s.id, s])) as Record<string, Sample>;

/** Default sample for Diagnostics: a correctly handled tomato leaf. */
export const DEFAULT_SAMPLE =
  RESEARCH.samples.find((s) => s.crop === 'tomato' && s.condition !== 'healthy' && s.stressCorrect) ?? RESEARCH.samples[0];

export const cropName = (id: string) => CROP_BY_ID[id]?.name ?? id;
export const conditionName = (code: string) => CONDITION_BY_CODE[code]?.name ?? code;
export const className = (cls: string) => {
  const [crop, cond] = cls.split('__');
  return `${cropName(crop)} · ${conditionName(cond)}`;
};
export const stressLabel = (v: number | null) => (v === null ? '—' : v === 1 ? 'Stressed' : 'Non-stressed');

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`;
export const num = (v: number) => v.toLocaleString('en-US');

/** Readable label for a feature id, e.g. `glcm_contrast` → `GLCM contrast`. */
export function featureLabel(id: string) {
  return id
    .replace(/^mean_/, 'mean ')
    .replace(/^std_/, 'std ')
    .replace(/^glcm_/, 'GLCM ')
    .replace(/^core_/, 'core ')
    .replace(/^edge_/, 'edge ')
    .replace(/_/g, ' ');
}

/** Low / Medium / High node label for feature i, from the dataset tertiles. */
export function tertile(i: number, v: number) {
  const [a, b] = RESEARCH.featureTertiles[i];
  return v <= a ? 0 : v <= b ? 1 : 2;
}

export const TYPE_STYLE: Record<string, { chip: string; icon: string }> = {
  Control: { chip: 'bg-pc-mint text-pc-mint-ink', icon: 'eco' },
  'Nutrient deficiency': { chip: 'bg-pc-amber-bg text-pc-amber', icon: 'science' },
  Pest: { chip: 'bg-pc-err-bg text-pc-err-ink', icon: 'pest_control' },
  Disease: { chip: 'bg-pc-high text-pc-ink2', icon: 'coronavirus' },
};

export const TIER_COLORS = ['#60a5fa', '#a3a3a3', '#f59e0b']; // Low, Medium, High node labels

export const Card: React.FC<{ className?: string; children: React.ReactNode }> = ({ className = '', children }) => (
  <section className={`bg-pc-card rounded-2xl border border-pc-line/70 p-5 shadow-xs ${className}`}>{children}</section>
);

export const SectionTitle: React.FC<{ icon: string; title: string; sub?: string; right?: React.ReactNode }> = ({
  icon,
  title,
  sub,
  right,
}) => (
  <div className="flex items-start justify-between gap-3 mb-4">
    <div>
      <h3 className="font-headline-sm text-pc-ink flex items-center gap-2">
        <span className="material-symbols-outlined text-pc-green text-[22px]">{icon}</span>
        {title}
      </h3>
      {sub && <p className="font-body-sm text-[12px] text-pc-muted mt-0.5">{sub}</p>}
    </div>
    {right}
  </div>
);

export const Chip: React.FC<{ className?: string; children: React.ReactNode }> = ({ className = '', children }) => (
  <span className={`inline-flex items-center gap-1 font-label-sm text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider ${className}`}>
    {children}
  </span>
);

/** Correct / wrong badge for a recorded prediction. */
export const Verdict: React.FC<{ ok: boolean; label?: string }> = ({ ok, label }) => (
  <Chip className={ok ? 'bg-pc-mint text-pc-mint-ink' : 'bg-pc-err-bg text-pc-err-ink'}>
    <span className="material-symbols-outlined text-[12px]">{ok ? 'check_circle' : 'cancel'}</span>
    {label ?? (ok ? 'Correct' : 'Misclassified')}
  </Chip>
);

export const Bar: React.FC<{ value: number; max?: number; className?: string }> = ({ value, max = 1, className = 'bg-pc-green-bg' }) => (
  <div className="h-1.5 w-full rounded-full bg-pc-cont overflow-hidden">
    <div className={`h-full rounded-full ${className}`} style={{ width: `${Math.max(0, Math.min(1, value / max)) * 100}%` }} />
  </div>
);

export const Stat: React.FC<{ label: string; value: string; sub?: string; icon: string }> = ({ label, value, sub, icon }) => (
  <div className="p-4 rounded-xl bg-pc-low border border-pc-line/60">
    <div className="flex items-center justify-between">
      <span className="font-label-sm text-[11px] uppercase tracking-wider text-pc-muted">{label}</span>
      <span className="material-symbols-outlined text-[18px] text-pc-green">{icon}</span>
    </div>
    <p className="font-headline-md text-pc-ink mt-1.5">{value}</p>
    {sub && <p className="font-body-sm text-[12px] text-pc-muted mt-0.5">{sub}</p>}
  </div>
);

/** Thumbnail of a dataset sample with its class caption. */
export const SampleThumb: React.FC<{ sample: Sample; onClick?: () => void; active?: boolean }> = ({ sample, onClick, active }) => (
  <button
    type="button"
    onClick={onClick}
    className={`group text-left rounded-xl overflow-hidden border transition-all bg-pc-card ${
      active ? 'border-pc-green ring-2 ring-pc-green/40' : 'border-pc-line/70 hover:border-pc-green/60'
    }`}
  >
    <div className="aspect-[4/3] bg-pc-cont overflow-hidden">
      <img src={sample.image} alt={sample.classLabel} loading="lazy" className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
    </div>
    <div className="p-2">
      <p className="font-title-sm text-[12px] text-pc-ink truncate">{cropName(sample.crop)}</p>
      <p className="font-body-sm text-[11px] text-pc-muted truncate">{conditionName(sample.condition)}</p>
    </div>
  </button>
);
