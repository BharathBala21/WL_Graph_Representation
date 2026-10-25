import React from 'react';
import { RESEARCH } from '../data/research';
import { Bar, Card, Chip, conditionName, num, pct } from '../lib';

interface CropsViewProps {
  onOpenSample: (id: string) => void;
}

export const CropsView: React.FC<CropsViewProps> = ({ onOpenSample }) => (
  <div className="flex flex-col gap-6 w-full max-w-[1440px] mx-auto">
    <div>
      <Chip className="bg-pc-mint text-pc-mint-ink">{RESEARCH.crops.length} crops</Chip>
      <h1 className="font-headline-lg text-pc-deep-ink mt-2">Crops</h1>
      <p className="font-body-md text-pc-ink2 mt-1 max-w-3xl">
        Each crop gets its own Stage-2 WL model. The topology, height and scores below are the best of the 20 topology × height
        settings for that crop by out-of-fold macro-F1 (Level-2 sweep), so they are slightly optimistic — in the hierarchical
        pipeline the setting is re-chosen by inner CV in every fold. Routing accuracy is how often Stage 1 sent the crop's
        leaves to the right model.
      </p>
    </div>

    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      {RESEARCH.crops.map((c) => {
        const examples = RESEARCH.samples.filter((s) => s.crop === c.id);
        const cs = c.cropSpecific;
        return (
          <Card key={c.id} className="flex flex-col gap-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="font-headline-sm text-pc-ink">{c.name}</h3>
                <p className="font-body-sm text-pc-muted italic">{c.scientific}</p>
              </div>
              <Chip className="bg-pc-high text-pc-ink2">
                {cs.graph} · h = {cs.height}
              </Chip>
            </div>

            <div className="flex gap-2 overflow-x-auto pb-1">
              {examples.slice(0, 6).map((s) => (
                <button key={s.id} onClick={() => onOpenSample(s.id)} className="shrink-0" title={conditionName(s.condition)}>
                  <img src={s.image} alt={s.classLabel} loading="lazy" className="w-20 h-16 rounded-lg object-cover hover:ring-2 hover:ring-pc-green" />
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                ['Images', num(c.images), `${c.healthy} healthy · ${c.stressed} stressed`],
                ['Stage-1 routing', pct(c.routingAccuracy), 'crop identified correctly'],
                ['Crop model acc.', pct(cs.accuracy), `± ${pct(cs.accuracySd)} over folds`],
                ['Macro-F1', pct(cs.macroF1), `balanced acc. ${pct(cs.balancedAccuracy)}`],
              ].map(([l, v, s]) => (
                <div key={l} className="p-3 rounded-xl bg-pc-low border border-pc-line/60">
                  <p className="font-label-sm text-[10px] uppercase tracking-wider text-pc-muted">{l}</p>
                  <p className="font-title-md text-pc-ink mt-0.5">{v}</p>
                  <p className="font-body-sm text-[11px] text-pc-muted">{s}</p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2 font-body-sm text-[12px]">
              <div>
                <div className="flex justify-between text-pc-ink2">
                  <span>Crop-specific WL macro-F1</span>
                  <span className="font-code-sm">{pct(cs.macroF1)}</span>
                </div>
                <Bar value={cs.macroF1} />
              </div>
              <div>
                <div className="flex justify-between text-pc-ink2">
                  <span>Global WL model F1 on this crop</span>
                  <span className="font-code-sm">{pct(c.globalWl.f1)}</span>
                </div>
                <Bar value={c.globalWl.f1} className="bg-pc-outline" />
              </div>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {c.conditions.map((k) => (
                <span key={k} className="px-2 py-0.5 rounded bg-pc-low text-pc-ink2 font-body-sm text-[11px]">
                  {conditionName(k)}
                </span>
              ))}
            </div>
          </Card>
        );
      })}
    </div>
    <p className="font-body-sm text-[12px] text-pc-muted">
      Global-model F1 is the stressed-class F1 of the pooled Hybrid h = 3 model; crop-model macro-F1 averages both classes, so the two are
      shown for orientation rather than as a like-for-like comparison.
    </p>
  </div>
);
