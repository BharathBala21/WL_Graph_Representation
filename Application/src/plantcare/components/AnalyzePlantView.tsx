import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { toast } from 'sonner';
import { RESEARCH } from '../data/research';
import { Card, Chip, SampleThumb, SectionTitle, conditionName, cropName, num } from '../lib';
import { api } from '../../lib/api';
import { useAuth, useBackend } from '../../lib/hooks';
import { DiagnosisRecord, DiagnosisResult } from './DiagnosisResult';
import { PIPELINE_STEPS as STEPS, PipelineWalkthrough, stepSummary } from './PipelineWalkthrough';

interface AnalyzePlantViewProps {
  onOpenSample: (id: string) => void;
}

export const AnalyzePlantView: React.FC<AnalyzePlantViewProps> = ({ onOpenSample }) => {
  const { status: authStatus } = useAuth();
  const signedIn = authStatus === 'signed-in';
  const { health, status: backendStatus, refresh } = useBackend();
  const liveReady = backendStatus === 'online' && health?.models_loaded === true;
  const [tab, setTab] = useState<'dataset' | 'upload'>('upload');
  const [crop, setCrop] = useState('tomato');
  const [condition, setCondition] = useState('all');
  const [selected, setSelected] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [cropHint, setCropHint] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<DiagnosisRecord | null>(null);
  const [walkStep, setWalkStep] = useState(0);
  const walkRef = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const cropInfo = RESEARCH.crops.find((c) => c.id === crop)!;
  const shown = useMemo(
    () => RESEARCH.samples.filter((s) => s.crop === crop && (condition === 'all' || s.condition === condition)),
    [crop, condition],
  );
  const sample = RESEARCH.samples.find((s) => s.id === selected);

  useEffect(() => () => void (preview && URL.revokeObjectURL(preview)), [preview]);

  const pickFile = (f: File | undefined) => {
    if (!f || busy) return;
    if (!f.size || f.size > 25 * 1024 * 1024) {
      setFile(null);
      setPreview(null);
      setResult(null);
      setError('Choose a non-empty image of 25 MB or less.');
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError('');
  };

  const runUpload = async () => {
    if (!file) return;
    setBusy(true);
    setError('');
    setResult(null);
    try {
      const rec = await api.predict(file, cropHint || undefined);
      setResult(rec);
      setWalkStep(0);
      toast.success('Diagnosis saved to My Diagnoses');
      setTimeout(() => (walkRef.current ?? resultRef.current)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (e: any) {
      setError(e?.message || 'The diagnosis failed.');
    } finally {
      setBusy(false);
    }
  };

  const canRun = tab === 'dataset' ? !!sample : !!file && signedIn && liveReady && !busy;

  return (
    <div className="flex flex-col gap-6 w-full max-w-[1440px] mx-auto">
      <div>
        <Chip className="bg-pc-mint text-pc-mint-ink">
          <span className="material-symbols-outlined text-[12px]">hub</span> Hierarchical WL pipeline
        </Chip>
        <h1 className="font-headline-lg text-pc-deep-ink mt-2">Analyze a Leaf</h1>
        <p className="font-body-md text-pc-ink2 mt-1 max-w-3xl">
          Upload a photo of a single leaf to get a live diagnosis, or pick an image from the OLID-I dataset to see the model's recorded
          out-of-fold prediction.
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <Card className="xl:col-span-2">
          <div className="flex gap-1 p-1 rounded-xl bg-pc-low border border-pc-line/60 w-fit mb-5">
            {(
              [
                ['upload', 'upload', 'Upload image'],
                ['dataset', 'photo_library', 'OLID-I dataset'],
              ] as const
            ).map(([id, icon, label]) => (
              <button
                key={id}
                disabled={busy}
                onClick={() => setTab(id)}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg font-title-sm transition-colors ${
                  tab === id ? 'bg-pc-card text-pc-ink shadow-xs' : 'text-pc-muted hover:text-pc-ink'
                }`}
              >
                <span className="material-symbols-outlined text-[18px]">{icon}</span>
                {label}
              </button>
            ))}
          </div>

          {tab === 'dataset' ? (
            <>
              <div className="flex flex-wrap gap-2 mb-3">
                {RESEARCH.crops.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => (setCrop(c.id), setCondition('all'), setSelected(null))}
                    className={`px-3 py-1.5 rounded-full font-title-sm text-[12px] border transition-colors ${
                      crop === c.id ? 'bg-pc-deep text-white border-transparent' : 'border-pc-line text-pc-ink2 hover:bg-pc-low'
                    }`}
                  >
                    {c.name}
                  </button>
                ))}
              </div>
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <span className="font-label-sm text-[11px] uppercase tracking-wider text-pc-muted mr-1">Condition</span>
                {['all', ...cropInfo.conditions].map((c) => (
                  <button
                    key={c}
                    onClick={() => (setCondition(c), setSelected(null))}
                    className={`px-2.5 py-1 rounded-md font-body-sm text-[12px] transition-colors ${
                      condition === c ? 'bg-pc-mint text-pc-mint-ink font-semibold' : 'bg-pc-low text-pc-ink2 hover:bg-pc-cont'
                    }`}
                  >
                    {c === 'all' ? 'All' : conditionName(c)}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                {shown.map((s) => (
                  <SampleThumb key={s.id} sample={s} active={s.id === selected} onClick={() => setSelected(s.id)} />
                ))}
              </div>
              <p className="font-body-sm text-[11px] text-pc-muted mt-3">
                Showing {shown.length} curated examples of {num(cropInfo.images)} {cropInfo.name.toLowerCase()} images (two per class). Every
                image's prediction is listed in History &amp; Logs.
              </p>
            </>
          ) : !liveReady && !busy ? (
            <div className="flex flex-col items-start gap-3 py-6" role="status">
              <p className="font-title-md text-pc-ink">{backendStatus === 'checking' ? 'Checking live diagnosis availability…' :
                backendStatus === 'offline' ? 'Live diagnosis is offline' : 'The diagnosis models are unavailable'}</p>
              <p className="font-body-sm text-pc-ink2">You can still explore the recorded OLID-I predictions.</p>
              <div className="flex gap-3">
                <button onClick={() => setTab('dataset')} className="rounded-lg bg-pc-deep text-white px-4 py-2">Browse dataset</button>
                <button onClick={() => refresh()} className="rounded-lg border border-pc-line px-4 py-2">Check again</button>
              </div>
            </div>
          ) : !signedIn ? (
            <div className="flex flex-col items-center text-center gap-3 py-10">
              <span className="material-symbols-outlined text-[44px] text-pc-green">lock</span>
              <p className="font-title-md text-pc-ink">Sign in to diagnose your own leaf photos</p>
              <p className="font-body-sm text-pc-ink2 max-w-md">Your diagnoses are saved to your account so you can come back to them later.</p>
              <div className="flex gap-2 mt-2">
                <Link to="/login?next=/plantcare/analyze" className="inline-flex items-center gap-1.5 bg-pc-deep text-white px-5 py-2.5 rounded-lg font-title-sm hover:bg-pc-forest">
                  <span className="material-symbols-outlined text-[18px]">login</span> Sign in
                </Link>
                <Link to="/login?mode=signup&next=/plantcare/analyze" className="inline-flex items-center gap-1.5 border border-pc-line text-pc-ink px-5 py-2.5 rounded-lg font-title-sm hover:bg-pc-low">
                  Create account
                </Link>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              <input
                ref={fileInput}
                type="file"
                disabled={busy}
                accept="image/jpeg,image/png,image/webp,image/bmp"
                className="hidden"
                onChange={(e) => { pickFile(e.target.files?.[0]); e.target.value = ''; }}
              />
              <button
                type="button"
                disabled={busy}
                onClick={() => fileInput.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => (e.preventDefault(), pickFile(e.dataTransfer.files?.[0]))}
                className="w-full border-2 border-dashed border-pc-line rounded-xl p-8 flex flex-col items-center gap-2 hover:border-pc-green/60 hover:bg-pc-low transition-colors"
              >
                {preview ? (
                  <img src={preview} alt={file?.name} className="max-h-72 rounded-lg object-contain" />
                ) : (
                  <span className="material-symbols-outlined text-[40px] text-pc-green">add_photo_alternate</span>
                )}
                <span className="font-title-sm text-pc-ink">{file ? file.name : 'Choose or drop a leaf photo'}</span>
                <span className="font-body-sm text-[12px] text-pc-muted">Best results: one leaf, plain dark background, good light.</span>
              </button>

              <label className="flex flex-col sm:flex-row sm:items-center gap-2">
                <span className="font-title-sm text-[13px] text-pc-ink2 sm:w-40">Crop (optional)</span>
                <select
                  value={cropHint}
                  disabled={busy}
                  onChange={(e) => { setCropHint(e.target.value); setResult(null); setError(''); }}
                  className="flex-1 rounded-lg border border-pc-line bg-pc-low px-3 py-2 font-body-md text-pc-ink focus:outline-none focus:ring-2 focus:ring-pc-green/50"
                >
                  <option value="">Detect automatically</option>
                  {RESEARCH.crops.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </label>

              {error && (
                <p role="alert" className="flex gap-2 items-start rounded-lg bg-pc-err-bg text-pc-err-ink px-3 py-2 font-body-sm text-[13px]">
                  <span className="material-symbols-outlined text-[18px]">error</span>
                  {error}
                </p>
              )}
            </div>
          )}
        </Card>

        {/* Pipeline run */}
        <Card className="flex flex-col">
          <SectionTitle
            icon="play_circle"
            title={tab === 'dataset' ? 'Recorded prediction' : 'Pipeline'}
            sub={tab === 'dataset' ? (sample ? `${cropName(sample.crop)} · ${sample.file}` : 'Select an image first') : file ? file.name : 'Choose a photo first'}
          />
          {tab === 'dataset' && sample && <img src={sample.image} alt="" className="w-full aspect-[4/3] object-cover rounded-xl mb-4" />}
          {tab === 'upload' && <ol className="flex flex-col gap-2 flex-1">
            {STEPS.map((s, i) => {
              const traced = tab === 'upload' && !!result?.result.trace;
              const current = traced && walkStep === i;
              const summary = traced && result ? stepSummary(result, i) : null;
              const Item = traced ? 'button' : 'div';
              return (
                <li key={s.title}>
                  <Item
                    {...(traced
                      ? { type: 'button' as const, onClick: () => (setWalkStep(i), walkRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })) }
                      : {})}
                    className={`w-full text-left flex gap-3 p-3 rounded-xl border transition-colors ${
                      current ? 'border-pc-green bg-pc-low' : 'border-pc-line/60'
                    } ${traced ? 'hover:border-pc-green/60 hover:bg-pc-low cursor-pointer' : ''}`}
                  >
                    <span
                      className={`material-symbols-outlined text-[20px] ${
                        traced ? 'text-pc-green' : 'text-pc-muted'
                      }`}
                    >
                      {traced ? 'check_circle' : s.icon}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="font-title-sm text-[13px] text-pc-ink">{s.title}</p>
                      {summary ? (
                        <p className="font-body-sm text-[12px] text-pc-green font-medium">{summary}</p>
                      ) : (
                        <p className="font-body-sm text-[11px] text-pc-muted">{s.text}</p>
                      )}
                    </div>
                    {traced && <span className="material-symbols-outlined text-[18px] text-pc-muted self-center">chevron_right</span>}
                  </Item>
                </li>
              );
            })}
          </ol>}
          <button
            disabled={!canRun}
            onClick={() => { if (tab === 'upload') void runUpload(); else if (sample) onOpenSample(sample.id); }}
            className="mt-4 inline-flex items-center justify-center gap-2 bg-pc-deep text-white px-5 py-3 rounded-lg hover:bg-pc-forest transition-colors font-title-sm disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <span className={`material-symbols-outlined text-[20px] ${busy ? 'animate-spin' : ''}`}>{busy ? 'progress_activity' : 'play_arrow'}</span>
            {busy ? 'Analyzing photo…' : tab === 'dataset' ? 'Open recorded prediction' : 'Diagnose this leaf'}
          </button>
        </Card>
      </div>

      {tab === 'upload' && result?.result.trace && (
        <div ref={walkRef} className="scroll-mt-24">
          <PipelineWalkthrough key={result.id} record={result} step={walkStep} onStep={setWalkStep} autoPlay />
        </div>
      )}

      {tab === 'upload' && result && (
        <div ref={resultRef} className="scroll-mt-24 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h2 className="font-headline-md text-pc-ink">Result</h2>
            <Link to="/plantcare/my-diagnoses" className="inline-flex items-center gap-1 font-title-sm text-[13px] text-pc-green hover:underline">
              Saved to My Diagnoses <span className="material-symbols-outlined text-[16px]">arrow_forward</span>
            </Link>
          </div>
          <DiagnosisResult record={result} showPipeline={false} />
        </div>
      )}
    </div>
  );
};
