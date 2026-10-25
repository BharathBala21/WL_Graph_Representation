import React from 'react';
import { PageView } from '../types';
import { RESEARCH } from '../data/research';
import { Bar, Card, Chip, SectionTitle, Stat, TYPE_STYLE, Verdict, conditionName, cropName, num, pct } from '../lib';

interface DashboardViewProps {
  onNavigate: (page: PageView) => void;
  onOpenSample: (id: string) => void;
}

const best = [...RESEARCH.level1Grid].sort((a, b) => b.macroF1 - a.macroF1)[0];
const hier = RESEARCH.hierarchical.find((h) => h.method.startsWith('Level-3 Hierarchical'))!;
const routing = RESEARCH.cropClassifier.find((c) => c.model === 'Flat RF')!;
// a varied handful of recorded inspections: one per crop, alternating right / wrong
const recent = RESEARCH.crops
  .map((c, i) => RESEARCH.samples.filter((s) => s.crop === c.id && s.condition !== 'healthy' && s.stressCorrect === (i % 3 !== 2))[0])
  .filter(Boolean)
  .slice(0, 5);

export const DashboardView: React.FC<DashboardViewProps> = ({ onNavigate, onOpenSample }) => {
  const d = RESEARCH.dataset;
  return (
    <div className="flex flex-col gap-y-6 w-full max-w-[1440px] mx-auto">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-2xl bg-pc-card p-6 md:p-8 border border-pc-line/60 shadow-xs">
        <div className="absolute -right-16 -top-16 w-80 h-80 rounded-full bg-pc-green-bg/10 blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="max-w-2xl">
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <Chip className="bg-pc-mint text-pc-mint-ink">
                <span className="w-1.5 h-1.5 rounded-full bg-pc-green-bg animate-pulse" /> WL graph-kernel screening
              </Chip>
              <span className="font-code-sm text-pc-muted">{d.name} · {d.folds}-fold out-of-fold results</span>
            </div>
            <h1 className="font-headline-lg text-[28px] md:text-[34px] text-pc-deep-ink">Leaf Stress Screening</h1>
            <p className="font-body-md text-pc-ink2 mt-2 leading-relaxed">
              Each leaf is segmented, described by 38 handcrafted features, turned into a feature graph and classified with a
              Weisfeiler–Lehman subtree kernel SVM. Stage 1 identifies the crop; Stage 2 runs that crop's WL model to decide
              stressed vs. non-stressed.
            </p>
            <div className="flex flex-wrap items-center gap-3 mt-5">
              <button
                onClick={() => onNavigate('analyze-plant')}
                className="inline-flex items-center gap-2 bg-pc-deep text-white px-5 py-2.5 rounded-lg hover:bg-pc-forest transition-colors font-title-sm"
              >
                <span className="material-symbols-outlined text-[20px]">document_scanner</span> Analyze a leaf
              </button>
              <button
                onClick={() => onNavigate('model-pipeline')}
                className="inline-flex items-center gap-2 bg-pc-card border border-pc-line text-pc-ink px-5 py-2.5 rounded-lg hover:bg-pc-low transition-colors font-title-sm"
              >
                <span className="material-symbols-outlined text-[20px]">account_tree</span> Pipeline & results
              </button>
              <button
                onClick={() => onNavigate('history-logs')}
                className="inline-flex items-center gap-2 bg-pc-card border border-pc-line text-pc-ink px-5 py-2.5 rounded-lg hover:bg-pc-low transition-colors font-title-sm"
              >
                <span className="material-symbols-outlined text-[20px]">table_view</span> Prediction log
              </button>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-pc-low border border-pc-line/60 min-w-[260px]">
            <p className="font-label-sm text-[11px] uppercase tracking-wider text-pc-muted">Best global WL configuration</p>
            <p className="font-headline-sm text-pc-ink mt-1">{best.graph} · h = {best.height}</p>
            <div className="mt-3 grid grid-cols-3 gap-3 text-center">
              {[
                ['Accuracy', best.accuracy],
                ['Bal. acc.', best.balancedAccuracy],
                ['Macro-F1', best.macroF1],
              ].map(([l, v]) => (
                <div key={l as string}>
                  <p className="font-title-md text-pc-green">{pct(v as number)}</p>
                  <p className="font-body-sm text-[11px] text-pc-muted">{l}</p>
                </div>
              ))}
            </div>
            <p className="font-body-sm text-[11px] text-pc-muted mt-3">Binary stress detection, all 8 crops pooled.</p>
          </div>
        </div>

        <div className="relative z-10 grid grid-cols-2 lg:grid-cols-4 gap-4 mt-6 pt-6 border-t border-pc-line/60">
          <Stat icon="photo_library" label="Leaf images" value={num(d.accepted)} sub={`${num(d.rawImages - d.accepted)} removed by segmentation QC`} />
          <Stat icon="grass" label="Crops · classes" value={`${d.crops} · ${d.classes}`} sub={`${d.conditions} stress conditions incl. healthy`} />
          <Stat icon="alt_route" label="Stage 1 crop routing" value={pct(routing.accuracy[0])} sub="Random Forest crop classifier" />
          <Stat icon="verified" label="End-to-end stress acc." value={pct(hier.accuracy[0])} sub={`Macro-F1 ${pct(hier.macroF1[0])} (hierarchical WL)`} />
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Recent inspections */}
        <Card className="xl:col-span-2">
          <SectionTitle
            icon="history"
            title="Recorded Inspections"
            sub="Held-out (out-of-fold) predictions — each image was scored by a model that never saw it in training"
            right={
              <button onClick={() => onNavigate('history-logs')} className="font-label-sm text-[11px] text-pc-green uppercase tracking-wider hover:underline">
                All {num(d.accepted)}
              </button>
            }
          />
          <div className="flex flex-col gap-3">
            {recent.map((s) => (
              <button
                key={s.id}
                onClick={() => onOpenSample(s.id)}
                className="flex items-center gap-4 p-3 rounded-xl border border-pc-line/60 hover:border-pc-green/50 hover:bg-pc-low transition-colors text-left"
              >
                <img src={s.image} alt="" className="w-20 h-16 rounded-lg object-cover shrink-0" loading="lazy" />
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-title-md text-pc-ink">{cropName(s.crop)}</span>
                    <span className="font-body-sm text-pc-muted italic truncate">{s.file}</span>
                  </div>
                  <p className="font-body-sm text-[12px] text-pc-ink2 mt-0.5">
                    Ground truth: <b>{conditionName(s.condition)}</b> · Stage 2: {s.graph} graph, h = {s.height}
                  </p>
                </div>
                <div className="flex flex-col items-end gap-1 shrink-0">
                  <Verdict ok={s.stressCorrect} />
                  {s.globalProbability !== null && (
                    <span className="font-code-sm text-pc-muted">P(stress) {s.globalProbability.toFixed(2)}</span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </Card>

        {/* Crops */}
        <Card>
          <SectionTitle
            icon="grass"
            title="Crops"
            sub="Best per-crop WL setting (Level-2 sweep), macro-F1"
            right={
              <button onClick={() => onNavigate('crops')} className="font-label-sm text-[11px] text-pc-green uppercase tracking-wider hover:underline">
                Details
              </button>
            }
          />
          <div className="flex flex-col gap-3">
            {[...RESEARCH.crops]
              .sort((a, b) => b.cropSpecific.macroF1 - a.cropSpecific.macroF1)
              .map((c) => (
                <div key={c.id}>
                  <div className="flex justify-between font-body-sm text-[12px]">
                    <span className="text-pc-ink font-semibold">{c.name}</span>
                    <span className="font-code-sm text-pc-muted">
                      {c.cropSpecific.graph} h={c.cropSpecific.height} · {pct(c.cropSpecific.macroF1)}
                    </span>
                  </div>
                  <Bar value={c.cropSpecific.macroF1} />
                </div>
              ))}
          </div>
        </Card>
      </div>

      {/* Conditions overview */}
      <Card>
        <SectionTitle
          icon="menu_book"
          title="Stress Conditions"
          sub="Share of each condition's images the hierarchical pipeline handled correctly (healthy: kept as healthy)"
          right={
            <button onClick={() => onNavigate('plant-stress-library')} className="font-label-sm text-[11px] text-pc-green uppercase tracking-wider hover:underline">
              Stress library
            </button>
          }
        />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-6 gap-y-3">
          {RESEARCH.conditions.map((c) => (
            <div key={c.code}>
              <div className="flex items-center justify-between gap-2 font-body-sm text-[12px]">
                <span className="flex items-center gap-1.5 text-pc-ink truncate">
                  <span className={`material-symbols-outlined text-[15px] ${c.type === 'Control' ? 'text-pc-green' : 'text-pc-muted'}`}>
                    {TYPE_STYLE[c.type].icon}
                  </span>
                  <span className="truncate">{c.name}</span>
                </span>
                <span className="font-code-sm text-pc-muted shrink-0">{pct(c.detectionRate, 0)}</span>
              </div>
              <Bar value={c.detectionRate} className={c.detectionRate < 0.7 ? 'bg-[#f59e0b]' : 'bg-pc-green-bg'} />
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
