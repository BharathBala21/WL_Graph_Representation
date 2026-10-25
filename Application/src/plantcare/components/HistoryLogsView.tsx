import React, { useEffect, useMemo, useState } from 'react';
import { toast } from 'sonner';
import { RESEARCH } from '../data/research';
import { Card, Chip, SAMPLE_BY_ID, Stat, Verdict, conditionName, cropName, num, pct, stressLabel } from '../lib';

interface HistoryLogsViewProps {
  onOpenSample: (id: string) => void;
}

interface OofLog {
  crops: string[];
  conditions: string[];
  graphs: string[];
  /** [crop, condition, image no., predicted crop, true stress, predicted stress, fold, graph, height] */
  rows: number[][];
}

interface Row {
  crop: string;
  condition: string;
  file: string;
  sampleId: string;
  predictedCrop: string;
  trueStress: number;
  predictedStress: number;
  fold: number;
  graph: string;
  height: number;
  routingOk: boolean;
  ok: boolean;
}

const PAGE = 25;

export const HistoryLogsView: React.FC<HistoryLogsViewProps> = ({ onOpenSample }) => {
  const [log, setLog] = useState<Row[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [crop, setCrop] = useState('all');
  const [condition, setCondition] = useState('all');
  const [outcome, setOutcome] = useState<'all' | 'correct' | 'wrong' | 'misrouted'>('all');
  const [fold, setFold] = useState('all');
  const [page, setPage] = useState(0);

  useEffect(() => {
    fetch('/plantcare/oof_level3.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d: OofLog) =>
        setLog(
          d.rows.map(([c, k, n, pc, ts, ps, f, g, h]) => {
            const cls = `${d.crops[c]}__${d.conditions[k]}`;
            return {
              crop: d.crops[c],
              condition: d.conditions[k],
              file: `${cls} (${n}).JPG`,
              sampleId: `${cls}-${n}`,
              predictedCrop: d.crops[pc],
              trueStress: ts,
              predictedStress: ps,
              fold: f,
              graph: d.graphs[g],
              height: h,
              routingOk: c === pc,
              ok: ts === ps,
            };
          }),
        ),
      )
      .catch((e) => setError(String(e.message || e)));
  }, []);

  const rows = useMemo(
    () =>
      (log ?? []).filter(
        (r) =>
          (crop === 'all' || r.crop === crop) &&
          (condition === 'all' || r.condition === condition) &&
          (fold === 'all' || r.fold === Number(fold)) &&
          (outcome === 'all' || (outcome === 'correct' ? r.ok : outcome === 'wrong' ? !r.ok : !r.routingOk)),
      ),
    [log, crop, condition, fold, outcome],
  );
  useEffect(() => setPage(0), [crop, condition, fold, outcome]);

  const acc = rows.length ? rows.filter((r) => r.ok).length / rows.length : null;
  const routing = rows.length ? rows.filter((r) => r.routingOk).length / rows.length : null;
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));

  const exportCsv = () => {
    const head = 'image,crop,condition,predicted_crop,true_stress,predicted_stress,correct,fold,graph,wl_height';
    const body = rows.map((r) =>
      [`"${r.file}"`, r.crop, r.condition, r.predictedCrop, r.trueStress, r.predictedStress, r.ok, r.fold, `"${r.graph}"`, r.height].join(','),
    );
    const url = URL.createObjectURL(new Blob([[head, ...body].join('\n')], { type: 'text/csv;charset=utf-8;' }));
    const a = Object.assign(document.createElement('a'), { href: url, download: `wl_level3_oof_predictions_${rows.length}.csv` });
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Exported ${num(rows.length)} predictions`);
  };

  const select = (value: string, set: (v: string) => void, options: [string, string][], label: string) => (
    <label className="flex flex-col gap-1">
      <span className="font-label-sm text-[10px] uppercase tracking-wider text-pc-muted">{label}</span>
      <select
        value={value}
        onChange={(e) => set(e.target.value)}
        className="h-9 px-2 rounded-lg bg-pc-card border border-pc-outline/50 text-pc-ink font-body-sm text-[13px] focus:outline-none focus:ring-2 focus:ring-pc-green/30"
      >
        {options.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="flex flex-col gap-6 w-full max-w-[1440px] mx-auto">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <Chip className="bg-pc-mint text-pc-mint-ink">Level-3 hierarchical WL · out-of-fold</Chip>
          <h1 className="font-headline-lg text-pc-deep-ink mt-2">History &amp; Logs</h1>
          <p className="font-body-md text-pc-ink2 mt-1 max-w-3xl">
            The recorded prediction for every accepted OLID-I image. Each image was scored once, by the models trained while its
            fold was held out.
          </p>
        </div>
        <button
          onClick={exportCsv}
          disabled={!rows.length}
          className="inline-flex items-center gap-2 bg-pc-deep text-white px-4 py-2.5 rounded-lg hover:bg-pc-forest font-title-sm disabled:opacity-40"
        >
          <span className="material-symbols-outlined text-[18px]">download</span> Export CSV
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Stat icon="filter_alt" label="Images in view" value={log ? num(rows.length) : '…'} sub={`of ${num(RESEARCH.dataset.accepted)}`} />
        <Stat icon="verified" label="Stress calls correct" value={pct(acc)} sub="Stage 2, end-to-end" />
        <Stat icon="alt_route" label="Routed to right crop" value={pct(routing)} sub="Stage 1" />
        <Stat icon="error" label="Misclassified" value={log ? num(rows.filter((r) => !r.ok).length) : '…'} sub="stress call wrong" />
      </div>

      <Card className="p-0 overflow-hidden">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4 border-b border-pc-line/60">
          {select(crop, (v) => (setCrop(v), setCondition('all')), [['all', 'All crops'], ...RESEARCH.crops.map((c) => [c.id, c.name] as [string, string])], 'Crop')}
          {select(
            condition,
            setCondition,
            [
              ['all', 'All conditions'],
              ...RESEARCH.conditions
                .filter((c) => crop === 'all' || c.crops.includes(crop))
                .map((c) => [c.code, c.name] as [string, string]),
            ],
            'Condition',
          )}
          {select(outcome, (v) => setOutcome(v as typeof outcome), [
            ['all', 'All outcomes'],
            ['correct', 'Correct stress call'],
            ['wrong', 'Wrong stress call'],
            ['misrouted', 'Wrong crop (Stage 1)'],
          ], 'Outcome')}
          {select(fold, setFold, [['all', 'All folds'], ...[0, 1, 2, 3, 4].map((f) => [String(f), `Fold ${f + 1}`] as [string, string])], 'Held-out fold')}
        </div>

        {error && <p className="p-6 text-pc-err font-body-md">Could not load the prediction log ({error}).</p>}
        {!log && !error && <p className="p-6 text-pc-muted font-body-md">Loading {num(RESEARCH.dataset.accepted)} predictions…</p>}

        {log && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] font-body-sm text-[12px]">
              <thead className="bg-pc-low text-pc-muted">
                <tr className="text-left">
                  {['Image', 'Ground truth', 'Stage 1 crop', 'Stage 2 stress', 'Model', 'Fold', 'Result'].map((h) => (
                    <th key={h} className="px-4 py-2.5 font-label-sm text-[10px] uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.slice(page * PAGE, page * PAGE + PAGE).map((r) => {
                  const hasSample = Boolean(SAMPLE_BY_ID[r.sampleId]);
                  return (
                    <tr key={r.file} className="border-t border-pc-line/50 text-pc-ink hover:bg-pc-low/60">
                      <td className="px-4 py-2 font-code-sm whitespace-nowrap">
                        {hasSample ? (
                          <button onClick={() => onOpenSample(r.sampleId)} className="text-pc-green hover:underline inline-flex items-center gap-1">
                            {r.file}
                            <span className="material-symbols-outlined text-[14px]">open_in_new</span>
                          </button>
                        ) : (
                          r.file
                        )}
                      </td>
                      <td className="px-4 py-2">
                        {cropName(r.crop)} · {conditionName(r.condition)}
                      </td>
                      <td className={`px-4 py-2 ${r.routingOk ? '' : 'text-pc-err font-semibold'}`}>{cropName(r.predictedCrop)}</td>
                      <td className="px-4 py-2">
                        {stressLabel(r.predictedStress)}
                        <span className="text-pc-muted"> / {stressLabel(r.trueStress)}</span>
                      </td>
                      <td className="px-4 py-2 font-code-sm text-pc-muted">
                        {r.graph} h={r.height}
                      </td>
                      <td className="px-4 py-2 font-code-sm">{r.fold + 1}</td>
                      <td className="px-4 py-2">
                        <Verdict ok={r.ok} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {log && (
          <div className="flex items-center justify-between gap-3 p-4 border-t border-pc-line/60 font-body-sm text-[12px] text-pc-muted">
            <span>
              {rows.length ? `${num(page * PAGE + 1)}–${num(Math.min(rows.length, (page + 1) * PAGE))} of ${num(rows.length)}` : 'No images match'}
              {' · '}linked file names open the image's diagnostics
            </span>
            <div className="flex items-center gap-1">
              <button disabled={page === 0} onClick={() => setPage((p) => p - 1)} className="p-1.5 rounded-lg hover:bg-pc-low disabled:opacity-30" aria-label="Previous page">
                <span className="material-symbols-outlined text-[18px]">chevron_left</span>
              </button>
              <span className="font-code-sm">
                {page + 1} / {pages}
              </span>
              <button disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)} className="p-1.5 rounded-lg hover:bg-pc-low disabled:opacity-30" aria-label="Next page">
                <span className="material-symbols-outlined text-[18px]">chevron_right</span>
              </button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};
