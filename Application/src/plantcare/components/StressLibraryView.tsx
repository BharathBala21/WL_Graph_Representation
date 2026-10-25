import React, { useState } from 'react';
import { RESEARCH } from '../data/research';
import { Bar, Card, Chip, TYPE_STYLE, cropName, num, pct } from '../lib';

interface StressLibraryViewProps {
  onOpenSample: (id: string) => void;
}

const TYPES = ['All', 'Control', 'Nutrient deficiency', 'Pest', 'Disease'] as const;

export const StressLibraryView: React.FC<StressLibraryViewProps> = ({ onOpenSample }) => {
  const [type, setType] = useState<(typeof TYPES)[number]>('All');
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState<string | null>(null);

  const q = query.trim().toLowerCase();
  const shown = RESEARCH.conditions.filter(
    (c) =>
      (type === 'All' || c.type === type) &&
      (!q || [c.code, c.name, c.scientific, ...c.crops.map(cropName)].some((t) => t.toLowerCase().includes(q))),
  );

  return (
    <div className="flex flex-col gap-6 w-full max-w-[1440px] mx-auto">
      <div>
        <Chip className="bg-pc-mint text-pc-mint-ink">{RESEARCH.conditions.length} conditions · {RESEARCH.dataset.classes} crop-condition classes</Chip>
        <h1 className="font-headline-lg text-pc-deep-ink mt-2">Stress Library</h1>
        <p className="font-body-md text-pc-ink2 mt-1 max-w-3xl">
          Every condition labelled in OLID-I, with how many accepted images it has and how reliably the hierarchical WL
          pipeline separated it from healthy leaves. Condition codes follow the corrected OLID-I definitions.
        </p>
      </div>

      <div className="flex flex-col md:flex-row md:items-center gap-3">
        <div className="flex flex-wrap gap-1.5">
          {TYPES.map((t) => (
            <button
              key={t}
              onClick={() => setType(t)}
              className={`px-3 py-1.5 rounded-full font-title-sm text-[12px] border transition-colors ${
                type === t ? 'bg-pc-deep text-white border-transparent' : 'border-pc-line text-pc-ink2 hover:bg-pc-low'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="relative md:ml-auto md:w-72">
          <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-pc-muted text-[18px]">search</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search condition, code or crop"
            className="w-full h-10 pl-9 pr-3 bg-pc-card text-pc-ink placeholder-pc-muted font-body-sm text-[13px] rounded-lg border border-pc-outline/50 focus:outline-none focus:ring-2 focus:ring-pc-green/30"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
        {shown.map((c) => {
          const example = RESEARCH.samples.find((s) => s.condition === c.code);
          const style = TYPE_STYLE[c.type];
          return (
            <Card key={c.code} className="p-0 overflow-hidden flex flex-col">
              {example && (
                <button onClick={() => onOpenSample(example.id)} className="relative aspect-[16/9] bg-pc-cont overflow-hidden group">
                  <img src={example.image} alt={c.name} loading="lazy" className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                  <span className="absolute left-3 top-3 font-code-sm px-2 py-0.5 rounded bg-black/60 text-white">{c.code}</span>
                </button>
              )}
              <div className="p-5 flex flex-col gap-3 flex-1">
                <div>
                  <Chip className={style.chip}>
                    <span className="material-symbols-outlined text-[12px]">{style.icon}</span>
                    {c.type}
                  </Chip>
                  <h3 className="font-headline-sm text-pc-ink mt-2">{c.name}</h3>
                  <p className="font-body-sm text-pc-muted italic">{c.scientific}</p>
                </div>
                <div className="grid grid-cols-2 gap-3 font-body-sm text-[12px]">
                  <div>
                    <p className="text-pc-muted">Images</p>
                    <p className="font-title-md text-pc-ink">{num(c.images)}</p>
                  </div>
                  <div>
                    <p className="text-pc-muted">{c.code === 'healthy' ? 'Kept as healthy' : 'Flagged as stressed'}</p>
                    <p className={`font-title-md ${c.detectionRate < 0.7 ? 'text-pc-amber' : 'text-pc-green'}`}>{pct(c.detectionRate)}</p>
                  </div>
                </div>
                <Bar value={c.detectionRate} className={c.detectionRate < 0.7 ? 'bg-[#f59e0b]' : 'bg-pc-green-bg'} />
                <div className="flex flex-wrap gap-1.5">
                  {c.crops.map((cr) => (
                    <span key={cr} className="px-2 py-0.5 rounded bg-pc-low text-pc-ink2 font-body-sm text-[11px]">
                      {cropName(cr)}
                    </span>
                  ))}
                </div>
                <button
                  onClick={() => setOpen(open === c.code ? null : c.code)}
                  className="mt-auto flex items-center gap-1 font-label-sm text-[11px] uppercase tracking-wider text-pc-green hover:underline w-fit"
                >
                  Per-class 57-way results
                  <span className={`material-symbols-outlined text-[16px] transition-transform ${open === c.code ? 'rotate-180' : ''}`}>expand_more</span>
                </button>
                {open === c.code && (
                  <table className="w-full font-body-sm text-[12px]">
                    <thead>
                      <tr className="text-pc-muted text-left">
                        <th className="font-normal py-1">Crop</th>
                        <th className="font-normal py-1 text-right">Images</th>
                        <th className="font-normal py-1 text-right">RF recall</th>
                        <th className="font-normal py-1 text-right">RF F1</th>
                      </tr>
                    </thead>
                    <tbody>
                      {c.classes.map((k) => (
                        <tr key={k.id} className="border-t border-pc-line/50 text-pc-ink">
                          <td className="py-1">{cropName(k.crop)}</td>
                          <td className="py-1 text-right font-code-sm">{k.images}</td>
                          <td className="py-1 text-right font-code-sm">{pct(k.rfRecall, 0)}</td>
                          <td className="py-1 text-right font-code-sm">{pct(k.rfF1, 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </Card>
          );
        })}
      </div>
      {shown.length === 0 && <p className="text-center text-pc-muted font-body-md py-10">No conditions match.</p>}
    </div>
  );
};
