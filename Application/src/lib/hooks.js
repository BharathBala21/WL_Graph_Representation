import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { api, tokenStore } from './api'

// ── signed-in user ───────────────────────────────────────────────────────────
export const AuthContext = createContext({ user: null, status: 'loading', login: async () => {}, register: async () => {}, logout: () => {} })
export const useAuth = () => useContext(AuthContext)

export function useAuthState() {
  const [user, setUser] = useState(null)
  const [status, setStatus] = useState(() => (tokenStore.get() ? 'loading' : 'signed-out'))

  useEffect(() => {
    if (!tokenStore.get()) return
    const ctl = new AbortController()
    api
      .me(ctl.signal)
      .then((u) => (setUser(u), setStatus('signed-in')))
      .catch((e) => {
        if (e.name === 'AbortError') return
        if (e.status === 401) tokenStore.set(null) // expired or invalid token
        setStatus('signed-out')
      })
    return () => ctl.abort()
  }, [])

  const accept = useCallback(({ token, user: u }) => {
    tokenStore.set(token)
    setUser(u)
    setStatus('signed-in')
    return u
  }, [])
  const login = useCallback(async (email, password) => accept(await api.login(email, password)), [accept])
  const register = useCallback(async (name, email, password) => accept(await api.register(name, email, password)), [accept])
  const logout = useCallback(() => {
    tokenStore.set(null)
    setUser(null)
    setStatus('signed-out')
  }, [])
  return { user, status, login, register, logout }
}

// ── theme ────────────────────────────────────────────────────────────────────
export function useThemeState() {
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem('wl-theme') || 'dark'
    } catch {
      return 'dark'
    }
  })
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem('wl-theme', theme)
    } catch {
      /* private mode */
    }
  }, [theme])
  return [theme, () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))]
}

// ── backend health (polled) ──────────────────────────────────────────────────
export const BackendContext = createContext({ health: null, status: 'checking', refresh: () => {} })
export const useBackend = () => useContext(BackendContext)

export function useHealthState() {
  const [health, setHealth] = useState(null)
  const [status, setStatus] = useState('checking')
  const refresh = useCallback(async () => {
    try {
      const h = await api.health()
      if (typeof h?.models_loaded !== 'boolean') throw new Error('Invalid health response')
      setHealth(h)
      setStatus('online')
    } catch {
      setHealth(null)
      setStatus('offline')
    }
  }, [])
  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 15000)
    return () => clearInterval(id)
  }, [refresh])
  return { health, status, refresh }
}

// ── model registry ───────────────────────────────────────────────────────────
const modelCache = {}
export function useModels(stage) {
  const { status } = useBackend()
  const [state, setState] = useState(() => modelCache[stage] || { models: [], default: null, loading: true })
  useEffect(() => {
    if (status !== 'online' || modelCache[stage]) return
    const ctl = new AbortController()
    api
      .models(stage, ctl.signal)
      .then((r) => {
        modelCache[stage] = { models: r.models, default: r.default, loading: false }
        setState(modelCache[stage])
      })
      .catch((e) => e.name !== 'AbortError' && setState((s) => ({ ...s, loading: false, error: e.message })))
    return () => ctl.abort()
  }, [stage, status])
  return status === 'offline' && !modelCache[stage] ? { ...state, loading: false, error: 'offline' } : state
}

// ── elapsed-time ticker for long model calls ─────────────────────────────────
export function useElapsed(running) {
  const [ms, setMs] = useState(0)
  const start = useRef(0)
  useEffect(() => {
    if (!running) return
    start.current = performance.now()
    setMs(0)
    const id = setInterval(() => setMs(performance.now() - start.current), 100)
    return () => clearInterval(id)
  }, [running])
  return ms
}

// ── theme context + chart colours (reference dataviz palette, slots 1-4, stepped per theme) ──
export const ThemeContext = createContext('dark')
export const useTheme = () => useContext(ThemeContext)

const CHART = {
  light: { series: ['#16a34a', '#2563eb', '#d97706', '#059669'], grid: '#e2e8f0', axis: '#475569', muted: '#94a3b8', surface: '#ffffff', text: '#0b1324' },
  dark: { series: ['#22c55e', '#3b82f6', '#f59e0b', '#10b981'], grid: '#1a3324', axis: '#94a3b8', muted: '#2d4a36', surface: '#0f2317', text: '#e6edf7' },
}
export const useChartColors = () => CHART[useTheme()] || CHART.dark

// ─── Date Formatting Utilities ─────────────────────────────────
export const formatRelativeTime = (dateStr) => {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return diffMins + 'm ago';
  if (diffHours < 24) return diffHours + 'h ago';
  if (diffDays < 7) return diffDays + 'd ago';
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
};
