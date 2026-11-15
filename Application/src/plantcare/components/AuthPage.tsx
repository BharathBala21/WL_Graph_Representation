import React, { useState } from 'react';
import { Navigate, useNavigate, useSearchParams } from 'react-router-dom';
import { toast } from 'sonner';
import { useAuth } from '../../lib/hooks';
import '../plantcare.css';

/** Sign in / create account. Redirects to ?next= (default: Analyze Leaf) once signed in. */
export const AuthPage: React.FC = () => {
  const { status, login, register } = useAuth();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const next = params.get('next') || '/plantcare/analyze';
  const [mode, setMode] = useState<'signin' | 'signup'>(params.get('mode') === 'signup' ? 'signup' : 'signin');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  if (status === 'signed-in') return <Navigate to={next} replace />;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setBusy(true);
    try {
      const u = mode === 'signin' ? await login(email, password) : await register(name, email, password);
      toast.success(mode === 'signin' ? `Welcome back, ${u.name}` : `Account created. Welcome, ${u.name}`);
      navigate(next, { replace: true });
    } catch (err: any) {
      setError(err?.message || 'Something went wrong.');
    } finally {
      setBusy(false);
    }
  };

  const input =
    'w-full rounded-lg border border-pc-line bg-pc-low px-3 py-2.5 font-body-md text-pc-ink placeholder:text-pc-muted focus:outline-none focus:ring-2 focus:ring-pc-green/50';

  return (
    <div className="plantcare min-h-screen bg-pc-bg pt-24 pb-16 px-4 text-pc-ink">
      <div className="mx-auto max-w-md">
        <div className="text-center mb-6">
          <span className="material-symbols-outlined text-[40px] text-pc-green">eco</span>
          <h1 className="font-headline-lg text-pc-deep-ink mt-1">PlantCare AI</h1>
          <p className="font-body-md text-pc-ink2 mt-1">
            {mode === 'signin' ? 'Sign in to diagnose leaves and see your history.' : 'Create an account to save your diagnoses.'}
          </p>
        </div>

        <section className="bg-pc-card rounded-2xl border border-pc-line/70 p-6 shadow-xs">
          <div className="flex gap-1 p-1 rounded-xl bg-pc-low border border-pc-line/60 mb-5">
            {(
              [
                ['signin', 'Sign in'],
                ['signup', 'Create account'],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => (setMode(id), setError(''))}
                className={`flex-1 px-4 py-2 rounded-lg font-title-sm transition-colors ${
                  mode === id ? 'bg-pc-card text-pc-ink shadow-xs' : 'text-pc-muted hover:text-pc-ink'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="flex flex-col gap-4">
            {mode === 'signup' && (
              <label className="flex flex-col gap-1.5">
                <span className="font-label-sm text-[12px] text-pc-ink2">Name</span>
                <input className={input} value={name} onChange={(e) => setName(e.target.value)} required maxLength={80} autoComplete="name" />
              </label>
            )}
            <label className="flex flex-col gap-1.5">
              <span className="font-label-sm text-[12px] text-pc-ink2">Email</span>
              <input className={input} type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="font-label-sm text-[12px] text-pc-ink2">Password</span>
              <input
                className={input}
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
              />
              {mode === 'signup' && <span className="font-body-sm text-[11px] text-pc-muted">At least 6 characters.</span>}
            </label>

            {error && (
              <p role="alert" className="flex gap-2 items-start rounded-lg bg-pc-err-bg text-pc-err-ink px-3 py-2 font-body-sm text-[13px]">
                <span className="material-symbols-outlined text-[18px]">error</span>
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={busy}
              className="mt-1 inline-flex items-center justify-center gap-2 bg-pc-deep text-white px-5 py-3 rounded-lg hover:bg-pc-forest transition-colors font-title-sm disabled:opacity-50"
            >
              <span className={`material-symbols-outlined text-[20px] ${busy ? 'animate-spin' : ''}`}>{busy ? 'progress_activity' : 'login'}</span>
              {mode === 'signin' ? 'Sign in' : 'Create account'}
            </button>
          </form>
        </section>
      </div>
    </div>
  );
};
