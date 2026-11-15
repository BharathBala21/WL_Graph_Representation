import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { toast } from 'sonner';
import { api, fileUrl, formatRelativeTime } from '../../lib/api';
import { useAuth } from '../../lib/hooks';
import { Card, Chip, Stat } from '../lib';
import { DiagnosisRecord, DiagnosisResult } from './DiagnosisResult';

/** The signed-in user's saved diagnoses (backend database). */
export const MyDiagnosesView: React.FC = () => {
  const { status: authStatus, user } = useAuth();
  const [params, setParams] = useSearchParams();
  const openId = params.get('id');
  const [items, setItems] = useState<DiagnosisRecord[] | null>(null);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<'all' | 'stressed' | 'healthy'>('all');
  const [cropFilter, setCropFilter] = useState('');
  const [query, setQuery] = useState('');

  const load = useCallback(() => {
    setError('');
    api
      .predictions()
      .then(setItems)
      .catch((e: any) => setError(e?.message || 'Could not load your diagnoses.'));
  }, []);
  useEffect(() => {
    if (authStatus === 'signed-in') load();
  }, [authStatus, load]);

  const crops = useMemo(() => [...new Set((items ?? []).map((i) => i.result.crop.name))].sort(), [items]);
  const shown = useMemo(
    () =>
      (items ?? []).filter(
        (i) =>
          (filter === 'all' || i.result.stress.status === filter) &&
          (!cropFilter || i.result.crop.name === cropFilter) &&
          (!query || `${i.filename} ${i.note ?? ''} ${i.result.conditions[0].name}`.toLowerCase().includes(query.toLowerCase())),
      ),
    [items, filter, cropFilter, query],
  );
  const open = items?.find((i) => i.id === openId) ?? null;

  if (authStatus === 'loading') return null;
  if (authStatus !== 'signed-in') {
    return (
      <Card className="max-w-xl mx-auto mt-10 text-center">
        <span className="material-symbols-outlined text-[44px] text-pc-green">lock</span>
        <h1 className="font-headline-md text-pc-ink mt-2">Sign in to see your diagnoses</h1>
        <p className="font-body-md text-pc-ink2 mt-1">Every leaf you analyse is saved to your account.</p>
        <Link to="/login?next=/plantcare/my-diagnoses" className="mt-5 inline-flex items-center gap-1.5 bg-pc-deep text-white px-5 py-2.5 rounded-lg font-title-sm hover:bg-pc-forest">
          <span className="material-symbols-outlined text-[18px]">login</span> Sign in
        </Link>
      </Card>
    );
  }

  const remove = async (rec: DiagnosisRecord) => {
    if (!window.confirm(`Delete the diagnosis for "${rec.filename}"? This cannot be undone.`)) return;
    try {
      await api.deletePrediction(rec.id);
      setItems((xs) => (xs ?? []).filter((x) => x.id !== rec.id));
      setParams({});
      toast.success('Diagnosis deleted');
    } catch (e: any) {
      toast.error(e?.message || 'Could not delete');
    }
  };
  const saveNote = async (rec: DiagnosisRecord, note: string) => {
    try {
      const updated = await api.saveNote(rec.id, note);
      setItems((xs) => (xs ?? []).map((x) => (x.id === rec.id ? updated : x)));
      toast.success('Note saved');
    } catch (e: any) {
      toast.error(e?.message || 'Could not save the note');
    }
  };

  if (open) {
    return (
      <div className="flex flex-col gap-4 w-full max-w-[1200px] mx-auto">
        <button type="button" onClick={() => setParams({})} className="inline-flex items-center gap-1 w-fit font-title-sm text-[13px] text-pc-green hover:underline">
          <span className="material-symbols-outlined text-[18px]">arrow_back</span> All diagnoses
        </button>
        <DiagnosisResult record={open} onDelete={() => remove(open)} onSaveNote={(n) => saveNote(open, n)} />
      </div>
    );
  }

  const total = items?.length ?? 0;
  const stressed = (items ?? []).filter((i) => i.result.stress.status === 'stressed').length;

  return (
    <div className="flex flex-col gap-6 w-full max-w-[1440px] mx-auto">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Chip className="bg-pc-mint text-pc-mint-ink">
            <span className="material-symbols-outlined text-[12px]">person</span> {user?.name}
          </Chip>
          <h1 className="font-headline-lg text-pc-deep-ink mt-2">My Diagnoses</h1>
          <p className="font-body-md text-pc-ink2 mt-1">Every leaf you have analysed, newest first.</p>
        </div>
        <Link to="/plantcare/analyze" className="inline-flex items-center gap-1.5 bg-pc-deep text-white px-5 py-2.5 rounded-lg font-title-sm hover:bg-pc-forest">
          <span className="material-symbols-outlined text-[18px]">add_photo_alternate</span> New diagnosis
        </Link>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Stat label="Leaves checked" value={String(total)} icon="eco" />
        <Stat label="Stressed" value={String(stressed)} icon="warning" sub={total ? `${Math.round((100 * stressed) / total)}% of checks` : undefined} />
        <Stat label="Healthy" value={String(total - stressed)} icon="check_circle" />
        <Stat label="Crops" value={String(crops.length)} icon="grass" />
      </div>

      <Card>
        <div className="flex flex-col lg:flex-row gap-3 lg:items-center mb-5">
          <div className="flex gap-1 p-1 rounded-xl bg-pc-low border border-pc-line/60 w-fit">
            {(['all', 'stressed', 'healthy'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-4 py-1.5 rounded-lg font-title-sm text-[13px] capitalize ${filter === f ? 'bg-pc-card text-pc-ink shadow-xs' : 'text-pc-muted hover:text-pc-ink'}`}
              >
                {f}
              </button>
            ))}
          </div>
          <select
            value={cropFilter}
            onChange={(e) => setCropFilter(e.target.value)}
            className="rounded-lg border border-pc-line bg-pc-low px-3 py-2 font-body-sm text-pc-ink"
            aria-label="Filter by crop"
          >
            <option value="">All crops</option>
            {crops.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search file name, note or condition"
            className="flex-1 rounded-lg border border-pc-line bg-pc-low px-3 py-2 font-body-sm text-pc-ink placeholder:text-pc-muted"
          />
        </div>

        {error && (
          <p role="alert" className="flex gap-2 items-center rounded-lg bg-pc-err-bg text-pc-err-ink px-3 py-2 font-body-sm text-[13px] mb-4">
            <span className="material-symbols-outlined text-[18px]">error</span>
            {error}
            <button onClick={load} className="ml-auto underline">
              Retry
            </button>
          </p>
        )}

        {items === null && !error ? (
          <p className="font-body-md text-pc-muted py-10 text-center">Loading…</p>
        ) : total === 0 ? (
          <div className="py-12 text-center">
            <span className="material-symbols-outlined text-[44px] text-pc-muted">photo_camera</span>
            <p className="font-title-md text-pc-ink mt-2">No diagnoses yet</p>
            <p className="font-body-sm text-pc-ink2 mt-1">
              Go to{' '}
              <Link to="/plantcare/analyze" className="text-pc-green underline">
                Analyze Leaf
              </Link>{' '}
              and upload your first leaf photo.
            </p>
          </div>
        ) : shown.length === 0 ? (
          <p className="font-body-md text-pc-muted py-10 text-center">Nothing matches these filters.</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {shown.map((i) => {
              const s = i.result.stress.status === 'stressed';
              return (
                <button
                  key={i.id}
                  type="button"
                  onClick={() => setParams({ id: i.id })}
                  className="group text-left rounded-xl overflow-hidden border border-pc-line/70 hover:border-pc-green/60 bg-pc-card transition-colors"
                >
                  <div className="aspect-[4/3] bg-pc-cont overflow-hidden">
                    <img src={fileUrl(i.overlay_url)} alt="" loading="lazy" className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
                  </div>
                  <div className="p-3 flex flex-col gap-1">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-title-sm text-pc-ink truncate">{i.result.crop.name}</span>
                      <Chip className={s ? 'bg-pc-err-bg text-pc-err-ink' : 'bg-pc-mint text-pc-mint-ink'}>{s ? 'Stressed' : 'Healthy'}</Chip>
                    </div>
                    <p className="font-body-sm text-[12px] text-pc-ink2 truncate">Likely: {i.result.conditions[0].name}</p>
                    <p className="font-body-sm text-[11px] text-pc-muted truncate">
                      {formatRelativeTime(i.created_at)} · {i.filename}
                    </p>
                    {i.note && <p className="font-body-sm text-[11px] text-pc-ink2 truncate italic">“{i.note}”</p>}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </Card>
    </div>
  );
};
