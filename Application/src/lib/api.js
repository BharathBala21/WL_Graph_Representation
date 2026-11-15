// Client for the PlantCare AI FastAPI backend (Application/backend).
// In development Vite proxies /api and /files to http://localhost:8000 (see vite.config.js),
// so the default base URL is the page's own origin. Set VITE_API_URL to call another host.
export const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

const TOKEN_KEY = 'plantcare-token'

export const tokenStore = {
  get() {
    try {
      return localStorage.getItem(TOKEN_KEY)
    } catch {
      return null
    }
  },
  set(token) {
    try {
      if (token) localStorage.setItem(TOKEN_KEY, token)
      else localStorage.removeItem(TOKEN_KEY)
    } catch {
      /* private mode: session lasts until reload */
    }
  },
}

/** Absolute URL for a file served by the backend, e.g. "/files/abc.jpg". */
export const fileUrl = (path) => (path ? API_URL + path : '')

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, { method = 'GET', json, body, headers = {}, signal } = {}) {
  const token = tokenStore.get()
  const h = { ...headers }
  if (token) h.Authorization = `Bearer ${token}`
  if (json !== undefined) h['Content-Type'] = 'application/json'
  let res
  try {
    res = await fetch(API_URL + path, { method, headers: h, body: json !== undefined ? JSON.stringify(json) : body, signal })
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError('Cannot reach the PlantCare AI server. Start it with "uvicorn app.main:app --port 8000" in Application/backend.', 0)
  }
  if (res.status === 204) return null
  let data = null
  try {
    data = await res.json()
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    const detail = data?.detail
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : detail || res.statusText
    throw new ApiError(msg || `Request failed (${res.status})`, res.status)
  }
  return data
}

export const api = {
  health: (signal) => request('/api/health', { signal }),
  modelInfo: (signal) => request('/api/model-info', { signal }),

  register: (name, email, password) => request('/api/auth/register', { method: 'POST', json: { name, email, password } }),
  login: (email, password) => request('/api/auth/login', { method: 'POST', json: { email, password } }),
  me: (signal) => request('/api/auth/me', { signal }),

  /** Diagnose a leaf photo (File). `crop` is an optional known crop code. */
  predict: (file, crop, signal) =>
    request(`/api/predict${crop ? `?crop=${encodeURIComponent(crop)}` : ''}`, {
      method: 'POST',
      body: file,
      headers: { 'Content-Type': file.type || 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name || 'leaf.jpg') },
      signal,
    }),
  predictions: (signal) => request('/api/predictions', { signal }),
  prediction: (id, signal) => request(`/api/predictions/${id}`, { signal }),
  stats: (signal) => request('/api/predictions/stats', { signal }),
  saveNote: (id, note) => request(`/api/predictions/${id}`, { method: 'PATCH', json: { note } }),
  deletePrediction: (id) => request(`/api/predictions/${id}`, { method: 'DELETE' }),

  // legacy name kept for lib/hooks.js useModels
  models: (_stage, signal) => request('/api/model-info', { signal }),
}

// ─── Date Formatting Utilities ─────────────────────────────────
export const formatRelativeTime = (dateStr) => {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)
  if (diffMins < 1) return 'just now'
  if (diffMins < 60) return diffMins + 'm ago'
  if (diffHours < 24) return diffHours + 'h ago'
  if (diffDays < 7) return diffDays + 'd ago'
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i]
}
