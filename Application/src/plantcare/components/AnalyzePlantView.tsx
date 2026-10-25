import React, { useEffect, useMemo, useRef, useState } from 'react';
import { RESEARCH } from '../data/research';
import { Card, Chip, SampleThumb, SectionTitle, conditionName, cropName, num } from '../lib';

interface AnalyzePlantViewProps {
  onOpenSample: (id: string) => void;
}

const STEPS = [
  { icon: 'crop_free', title: 'Leaf segmentation & QC', text: 'Foreground mask extracted; images with implausible mask area are rejected.' },
  { icon: 'tune', title: 'Feature extraction', text: '38 handcrafted features: morphology, colour, GLCM texture, core/edge spatial.' },
  { icon: 'alt_route', title: 'Stage 1 · crop routing', text: 'Random Forest identifies which of the 8 crops the leaf belongs to.' },
  { icon: 'hub', title: 'Feature graph + WL relabelling', text: '38 feature nodes labelled Low/Medium/High, linked by the crop\'s selected topology, refined for h WL iterations.' },
  { icon: 'model_training', title: 'Stage 2 · WL kernel SVM', text: 'The crop-specific WL subtree kernel SVM decides stressed vs. non-stressed.' },
];

export const AnalyzePlantView: React.FC<AnalyzePlantViewProps> = ({ onOpenSample }) => {
  const [tab, setTab] = useState<'dataset' | 'upload'>('dataset');
  const [crop, setCrop] = useState('tomato');
  const [condition, setCondition] = useState('all');
  const [selected, setSelected] = useState<string | null>(null);
  const [step, setStep] = useState(-1); // -1 idle, STEPS.length done
  const [upload, setUpload] = useState<{ url: string; name: string } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const cropInfo = RESEARCH.crops.find((c) => c.id === crop)!;
  const shown = useMemo(
    () => RESEARCH.samples.filter((s) => s.crop === crop && (condition === 'all' || s.condition === condition)),
    [crop, condition],
  );
  const sample = RESEARCH.samples.find((s) => s.id === selected);

  useEffect(() => {
    if (step < 0 || step >= STEPS.length) return;
    const t = setTimeout(() => setStep((s) => s + 1), 650);
    return () => clearTimeout(t);
  }, [step]);
  useEffect(() => {
    if (step === STEPS.length && sample) {
      const t = setTimeout(() => onOpenSample(sample.id), 400);
      return () => clearTimeout(t);
    }
  }, [step, sample, onOpenSample]);
  useEffect(
    () => () => {
      if (upload) URL.revokeObjectURL(upload.url);
    },
    [upload],
  );

  const running = step >= 0 && step <= STEPS.length;

  return (
    <div className="flex flex-col gap-6 w-full max-w-[1440px] mx-auto">
      <div>
        <Chip className="bg-pc-mint text-pc-mint-ink">
          <span className="material-symbols-outlined text-[12px]">hub</span> Hierarchical WL pipeline
        </Chip>
        <h1 className="font-headline-lg text-pc-deep-ink mt-2">Analyze a Leaf</h1>
        <p className="font-body-md text-pc-ink2 mt-1 max-w-3xl">
          Pick an image from the OLID-I dataset to trace it through the pipeline. The result shown is that image's recorded
          out-of-fold prediction — produced by a model trained on the other four folds.
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <Card className="xl:col-span-2">
          <div className="flex gap-1 p-1 rounded-xl bg-pc-low border border-pc-line/60 w-fit mb-5">
            {(
              [
                ['dataset', 'photo_library', 'OLID-I dataset'],
                ['upload', 'upload', 'Upload image'],
              ] as const
            ).map(([id, icon, label]) => (
              <button
                key={id}
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
                  <SampleThumb key={s.id} sample={s} active={s.id === selected} onClick={() => (setSelected(s.id), setStep(-1))} />
                ))}
              </div>
              <p className="font-body-sm text-[11px] text-pc-muted mt-3">
                Showing {shown.length} curated examples of {num(cropInfo.images)} {cropInfo.name.toLowerCase()} images (two per class). Every
                image's prediction is listed in History &amp; Logs.
              </p>
            </>
          ) : (
            <div>
              <input
                ref={fileInput}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) setUpload({ url: URL.createObjectURL(f), name: f.name });
                }}
              />
              <button
                onClick={() => fileInput.current?.click()}
                className="w-full border-2 border-dashed border-pc-line rounded-xl p-8 flex flex-col items-center gap-2 hover:border-pc-green/60 hover:bg-pc-low transition-colors"
              >
                {upload ? (
                  <img src={upload.url} alt={upload.name} className="max-h-72 rounded-lg object-contain" />
                ) : (
                  <span className="material-symbols-outlined text-[40px] text-pc-green">add_photo_alternate</span>
                )}
                <span className="font-title-sm text-pc-ink">{upload ? upload.name : 'Choose a leaf image'}</span>
              </button>
              <div className="mt-4 p-4 rounded-xl bg-pc-amber-bg text-pc-amber flex gap-3">
                <span className="material-symbols-outlined">info</span>
                <p className="font-body-sm text-[13px] leading-relaxed">
                  Scoring a new photo needs the Python pipeline (segmentation, the 38 features and the trained WL kernel SVMs)
                  running as a backend service, which is not connected to this portal yet. Until then, use the dataset tab to
                  inspect the model's recorded predictions.
                </p>
              </div>
            </div>
          )}
        </Card>

        {/* Pipeline run */}
        <Card className="flex flex-col">
          <SectionTitle icon="play_circle" title="Pipeline" sub={sample ? `${cropName(sample.crop)} · ${sample.file}` : 'Select an image first'} />
          {sample && <img src={sample.image} alt="" className="w-full aspect-[4/3] object-cover rounded-xl mb-4" />}
          <ol className="flex flex-col gap-2 flex-1">
            {STEPS.map((s, i) => {
              const state = step > i ? 'done' : step === i ? 'active' : 'idle';
              return (
                <li
                  key={s.title}
                  className={`flex gap-3 p-3 rounded-xl border transition-colors ${
                    state === 'active' ? 'border-pc-green bg-pc-low' : 'border-pc-line/60'
                  }`}
                >
                  <span
                    className={`material-symbols-outlined text-[20px] ${
                      state === 'done' ? 'text-pc-green' : state === 'active' ? 'text-pc-green animate-pulse' : 'text-pc-muted'
                    }`}
                  >
                    {state === 'done' ? 'check_circle' : s.icon}
                  </span>
                  <div>
                    <p className="font-title-sm text-[13px] text-pc-ink">{s.title}</p>
                    <p className="font-body-sm text-[11px] text-pc-muted">{s.text}</p>
                  </div>
                </li>
              );
            })}
          </ol>
          <button
            disabled={!sample || running || tab !== 'dataset'}
            onClick={() => setStep(0)}
            className="mt-4 inline-flex items-center justify-center gap-2 bg-pc-deep text-white px-5 py-3 rounded-lg hover:bg-pc-forest transition-colors font-title-sm disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <span className="material-symbols-outlined text-[20px]">{running ? 'progress_activity' : 'play_arrow'}</span>
            {running ? 'Running…' : 'Run pipeline & open diagnostics'}
          </button>
        </Card>
      </div>
    </div>
  );
};
