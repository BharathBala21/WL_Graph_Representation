import React, { useEffect, useState } from 'react';
import { fileUrl } from '../../lib/api';
import { Bar, Card, Chip, SectionTitle, TYPE_STYLE, pct } from '../lib';
import { PipelineTrace, PipelineWalkthrough, ReliabilityChecks } from './PipelineWalkthrough';

/** A saved diagnosis as returned by the backend (/api/predict, /api/predictions/:id). */
export interface DiagnosisRecord {
  id: string;
  created_at: string;
  filename: string;
  note: string | null;
  image_url: string;
  overlay_url: string;
  result: {
    crop: { code: string; name: string; confidence: number; source: 'model' | 'user'; reliability: number;
            alternatives: { code: string; name: string; probability: number }[] };
    stress: { status: 'healthy' | 'stressed'; probability_stressed: number; confidence: number;
              reliability: { accuracy: number; macro_f1: number; balanced_accuracy?: number | null;
                             healthy_found?: number | null; always_stressed_accuracy?: number | null } };
    conditions: { code: string; name: string; type: string; probability: number }[];
    conditions_reliability: { top1: number; top3: number };
    model: { graph: string; graph_name: string; wl_depth: number };
    explanation: { feature: string; name: string; value: number; healthy_mean: number; z: number; direction: 'higher' | 'lower' }[];
    warnings: string[];
    quality: { flag: string; mask_ratio: number };
    /** What every pipeline step produced (diagnoses made before this was added have none). */
    trace?: PipelineTrace;
    /** Who made the healthy/stressed call (hybrid: CNN decides, WL explains), and WL's own opinion. */
    decision?: {
      model: string; hybrid: boolean;
      cnn_status: 'healthy' | 'stressed' | null; cnn_probability_stressed: number | null;
      wl_status: 'healthy' | 'stressed'; wl_probability_stressed: number; agree: boolean;
      wl_reliability: { balanced_accuracy?: number | null; macro_f1: number };
    };
    /** Rule-based spot check that runs next to the models. */
    spots?: { count: number; area_ratio: number; flagged: boolean; message: string | null };
    /** Is the photo like the training photos? */
    reliability_check?: {
      status: 'ok' | 'caution' | 'unreliable';
      checks: { id: string; label: string; passed: boolean; detail: string }[];
      known_conditions: string[];
    };
  };
}

interface Props {
  record: DiagnosisRecord;
  onDelete?: () => void;
  onSaveNote?: (note: string) => Promise<void>;
  /** Offer the step-by-step pipeline replay (off where the page shows it separately). */
  showPipeline?: boolean;
}

export const DiagnosisResult: React.FC<Props> = ({ record, onDelete, onSaveNote, showPipeline = true }) => {
  const r = record.result;
  const stressed = r.stress.status === 'stressed';
  const check = r.reliability_check;
  const rel = r.stress.reliability;
  const hybrid = !!r.decision?.hybrid;
  const disagree = hybrid && r.decision!.agree === false;
  // below 0.6 balanced accuracy the healthy/stressed call for this crop is close to guessing
  const weakCrop = rel.balanced_accuracy != null && rel.balanced_accuracy < 0.6;
  // what the headline shows: the model's answer, or a hedge when the checks say it may be wrong
  const verdict: 'model' | 'unsure' | 'spots' =
    check?.status === 'unreliable' ? 'unsure' : r.spots?.flagged && !stressed ? 'spots' : 'model';
  const [note, setNote] = useState(record.note ?? '');
  const [saving, setSaving] = useState(false);
  const [pipelineStep, setPipelineStep] = useState<number | null>(null);
  useEffect(() => setNote(record.note ?? ''), [record.id, record.note]);
  useEffect(() => setPipelineStep(null), [record.id]);

  return (
    <div className="flex flex-col gap-5">
      {/* verdict */}
      <Card>
        <div className="grid grid-cols-1 md:grid-cols-[minmax(0,260px)_1fr] gap-5 items-start">
          <a href={fileUrl(record.image_url)} target="_blank" rel="noreferrer" title="Open original photo">
            <img src={fileUrl(record.overlay_url)} alt="Leaf with detected outline" className="w-full aspect-square object-cover rounded-xl bg-pc-cont" />
          </a>
          <div className="flex flex-col gap-4">
            <div>
              <p className="font-label-sm text-[11px] uppercase tracking-wider text-pc-muted">Diagnosis</p>
              <div className="flex flex-wrap items-center gap-3 mt-1">
                {verdict === 'model' ? (
                  <span
                    className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl font-headline-sm ${
                      stressed ? 'bg-pc-err-bg text-pc-err-ink' : 'bg-pc-mint text-pc-mint-ink'
                    }`}
                  >
                    <span className="material-symbols-outlined text-[22px]">{stressed ? 'warning' : 'check_circle'}</span>
                    {stressed ? 'Stressed' : 'Healthy'}
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-2 px-4 py-2 rounded-xl font-headline-sm bg-pc-amber-bg text-pc-amber">
                    <span className="material-symbols-outlined text-[22px]">{verdict === 'spots' ? 'search' : 'help'}</span>
                    {verdict === 'spots' ? 'Spots found: check' : 'Unsure'}
                  </span>
                )}
                <span className="font-body-md text-pc-ink2">
                  {verdict === 'model'
                    ? `${pct(r.stress.confidence, 0)} confident`
                    : `Model's best guess: ${stressed ? 'Stressed' : 'Healthy'} (${pct(r.stress.confidence, 0)})`}
                </span>
                {disagree && (
                  <Chip className="bg-pc-amber-bg text-pc-amber">
                    <span className="material-symbols-outlined text-[12px]">call_split</span> CNN and WL disagree
                  </Chip>
                )}
                {hybrid && !disagree && (
                  <Chip className="bg-pc-mint text-pc-mint-ink">
                    <span className="material-symbols-outlined text-[12px]">join</span> CNN and WL agree
                  </Chip>
                )}
                {weakCrop && (
                  <Chip className="bg-pc-amber-bg text-pc-amber">
                    <span className="material-symbols-outlined text-[12px]">info</span> weak for {r.crop.name.toLowerCase()}
                  </Chip>
                )}
                {check?.status === 'caution' && verdict === 'model' && (
                  <Chip className="bg-pc-amber-bg text-pc-amber">
                    <span className="material-symbols-outlined text-[12px]">info</span> use with care
                  </Chip>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-3 rounded-xl bg-pc-low border border-pc-line/60">
                <p className="font-label-sm text-[11px] uppercase tracking-wider text-pc-muted">Crop</p>
                <p className="font-title-md text-pc-ink mt-0.5">{r.crop.name}</p>
                <p className="font-body-sm text-[12px] text-pc-muted">
                  {r.crop.source === 'user' ? 'Chosen by you' : `Detected · ${pct(r.crop.confidence, 0)} sure`}
                </p>
              </div>
              <div className="p-3 rounded-xl bg-pc-low border border-pc-line/60">
                <p className="font-label-sm text-[11px] uppercase tracking-wider text-pc-muted">Likely conditions (shortlist)</p>
                <ol className="mt-0.5 font-body-sm text-[13px] text-pc-ink">
                  {r.conditions.map((c, i) => (
                    <li key={c.code} className={i === 0 ? 'font-title-sm' : 'text-pc-ink2'}>
                      {i + 1}. {c.name} <span className="text-pc-muted font-normal">· {pct(c.probability, 0)}</span>
                    </li>
                  ))}
                </ol>
                <p className="font-body-sm text-[12px] text-pc-muted mt-1">
                  In testing, the right condition was in this top 3 for {pct(r.conditions_reliability.top3, 0)} of{' '}
                  {r.crop.name.toLowerCase()} leaves.
                </p>
              </div>
            </div>

            {r.warnings.map((w) => (
              <p key={w} className="flex gap-2 items-start rounded-lg bg-pc-amber-bg text-pc-amber px-3 py-2 font-body-sm text-[13px]">
                <span className="material-symbols-outlined text-[18px]">info</span>
                {w}
              </p>
            ))}

            <p className="font-body-sm text-[12px] text-pc-muted">
              {record.filename} · {new Date(record.created_at).toLocaleString()}
            </p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {/* top-3 conditions */}
        <Card>
          <SectionTitle icon="format_list_numbered" title="Top 3 likely conditions" sub={`For ${r.crop.name.toLowerCase()} leaves`} />
          <ol className="flex flex-col gap-3">
            {r.conditions.map((c, i) => (
              <li key={c.code} className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between gap-3">
                  <span className="flex items-center gap-2 font-title-sm text-pc-ink">
                    <span className="grid place-items-center w-6 h-6 rounded-full bg-pc-deep text-white text-[11px]">{i + 1}</span>
                    {c.name}
                    <Chip className={TYPE_STYLE[c.type]?.chip ?? 'bg-pc-high text-pc-ink2'}>{c.type}</Chip>
                  </span>
                  <span className="font-body-sm text-pc-ink2 tabular-nums">{pct(c.probability, 0)}</span>
                </div>
                <Bar value={c.probability} />
              </li>
            ))}
          </ol>
          <p className="mt-4 flex gap-2 items-start rounded-lg bg-pc-low px-3 py-2 font-body-sm text-[12px] text-pc-ink2">
            <span className="material-symbols-outlined text-[16px] text-pc-muted">verified</span>
            In testing, the true condition was in this top-3 list for {pct(r.conditions_reliability.top3, 0)} of{' '}
            {r.crop.name.toLowerCase()} leaves, and ranked first for {pct(r.conditions_reliability.top1, 0)}. Treat it as a shortlist, not a final answer.
          </p>
        </Card>

        {/* explanation */}
        <Card>
          <SectionTitle icon="insights" title="What stands out" sub={`Compared with healthy ${r.crop.name.toLowerCase()} leaves`} />
          <ul className="flex flex-col gap-3">
            {r.explanation.map((e) => (
              <li key={e.feature} className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between gap-3">
                  <span className="font-title-sm text-pc-ink">{e.name}</span>
                  <span className={`font-body-sm text-[12px] ${Math.abs(e.z) >= 2 ? 'text-pc-err' : 'text-pc-ink2'}`}>
                    {e.direction === 'higher' ? '▲ higher' : '▼ lower'} than healthy
                  </span>
                </div>
                <Bar value={Math.min(Math.abs(e.z), 4)} max={4} className={Math.abs(e.z) >= 2 ? 'bg-pc-err' : 'bg-pc-green-bg'} />
              </li>
            ))}
          </ul>
          <p className="mt-4 font-body-sm text-[12px] text-pc-muted">Bar length shows how unusual the value is (up to 4 standard deviations).</p>
        </Card>
      </div>

      {check && (
        <Card>
          <SectionTitle
            icon="verified_user"
            title="Reliability check"
            sub={
              check.status === 'ok'
                ? 'This photo looks like the photos the model was trained on.'
                : check.status === 'caution'
                  ? 'One check failed: use the result with care.'
                  : 'Several checks failed: the result may be wrong.'
            }
          />
          <ReliabilityChecks record={record} />
        </Card>
      )}

      {showPipeline && r.trace && (
        pipelineStep === null ? (
          <button
            type="button"
            onClick={() => setPipelineStep(0)}
            className="self-start inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border border-pc-line text-pc-ink font-title-sm text-[13px] hover:bg-pc-low"
          >
            <span className="material-symbols-outlined text-[18px] text-pc-green">account_tree</span> Show step by step how this was decided
          </button>
        ) : (
          <PipelineWalkthrough record={record} step={pipelineStep} onStep={setPipelineStep} />
        )
      )}

      {/* model + note */}
      <Card>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div>
            <SectionTitle icon="hub" title="How this was decided" />
            <ul className="font-body-sm text-[13px] text-pc-ink2 flex flex-col gap-1.5">
              <li>• Leaf separated from the background and measured (38 measurements).</li>
              {hybrid ? (
                <li>
                  • <b>Hybrid:</b> a pretrained CNN ({r.decision!.model}) identified the crop (correct {pct(r.crop.reliability, 0)} of the time in
                  testing) and made the healthy/stressed call. The WL model explains it and gives its own opinion:{' '}
                  <b>{r.decision!.wl_status}</b> ({r.decision!.agree ? 'agrees' : 'disagrees'}).
                </li>
              ) : (
                <li>• Crop identified by a Random Forest (correct {pct(r.crop.reliability, 0)} of the time in testing).</li>
              )}
              <li>
                • Measurements turned into a <b>{r.model.graph_name}</b> graph and compared with the <b>WL graph kernel</b> (depth {r.model.wl_depth}).
              </li>
              <li>
                • {hybrid ? 'The CNN decided' : 'SVM decided'} healthy vs stressed.{' '}
                {rel.balanced_accuracy != null ? (
                  <>
                    For {r.crop.name.toLowerCase()} leaves in testing: balanced accuracy {pct(rel.balanced_accuracy, 0)} (healthy and stressed
                    leaves count equally), {rel.healthy_found != null && <>found {pct(rel.healthy_found, 0)} of healthy leaves, </>}
                    Macro-F1 {rel.macro_f1.toFixed(2)}.
                  </>
                ) : (
                  <>Accuracy {pct(rel.accuracy, 0)}, Macro-F1 {rel.macro_f1.toFixed(2)} for this crop.</>
                )}
              </li>
            </ul>
          </div>
          <div className="flex flex-col gap-2">
            {onSaveNote && (
              <>
                <label className="font-label-sm text-[12px] text-pc-ink2" htmlFor={`note-${record.id}`}>
                  Your note
                </label>
                <textarea
                  id={`note-${record.id}`}
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  maxLength={1000}
                  rows={3}
                  placeholder="e.g. field 3, east row, treated on Monday"
                  className="w-full rounded-lg border border-pc-line bg-pc-low px-3 py-2 font-body-sm text-pc-ink placeholder:text-pc-muted focus:outline-none focus:ring-2 focus:ring-pc-green/50"
                />
              </>
            )}
            <div className="flex flex-wrap gap-2 justify-end">
              {onSaveNote && (
                <button
                  type="button"
                  disabled={saving || note === (record.note ?? '')}
                  onClick={async () => {
                    setSaving(true);
                    try {
                      await onSaveNote(note);
                    } finally {
                      setSaving(false);
                    }
                  }}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-pc-deep text-white font-title-sm text-[13px] disabled:opacity-40"
                >
                  <span className="material-symbols-outlined text-[18px]">save</span> Save note
                </button>
              )}
              {onDelete && (
                <button
                  type="button"
                  onClick={onDelete}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border border-pc-line text-pc-err font-title-sm text-[13px] hover:bg-pc-err-bg"
                >
                  <span className="material-symbols-outlined text-[18px]">delete</span> Delete
                </button>
              )}
            </div>
          </div>
        </div>
        <p className="mt-4 font-body-sm text-[11px] text-pc-muted">
          Research prototype trained on OLID-I photos (single leaf on a dark background). Results on other photos may be less reliable. Not a
          substitute for an agronomist.
        </p>
      </Card>
    </div>
  );
};
