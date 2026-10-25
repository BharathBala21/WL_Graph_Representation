import { motion } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'
import { cx } from '../lib/utils'

// Cinematic intro for the WL Graph Kernel research portal.
const STEPS = [
  { label: 'Feature Extraction',   short: 'Extract' },
  { label: 'Graph Construction',   short: 'Build'   },
  { label: 'WL Kernel Computation', short: 'Kernel'  },
  { label: 'Stress Classification', short: 'Classify' },
]
const START   = 900
const STEP_MS = 650
const FINALE  = START + STEP_MS * (STEPS.length - 1) + 650
const LEAVE   = FINALE + 650
const GREEN   = '34, 197, 94'
const EMERALD = '74, 222, 128'

/** Drifting graph-like nodes joined by faint links — a WL graph backdrop. */
function GraphField({ reduced }) {
  const ref = useRef(null)
  useEffect(() => {
    const c = ref.current
    const ctx = c.getContext('2d')
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    let w, h, raf
    const resize = () => {
      w = c.width  = window.innerWidth  * dpr
      h = c.height = window.innerHeight * dpr
    }
    resize()
    window.addEventListener('resize', resize)
    const n = Math.min(100, Math.floor((window.innerWidth * window.innerHeight) / 14000))
    const pts = Array.from({ length: n }, () => ({
      x:  Math.random() * w,
      y:  Math.random() * h,
      vx: (Math.random() - 0.5) * 0.4 * dpr,
      vy: (Math.random() - 0.5) * 0.4 * dpr,
      r:  (Math.random() * 1.4 + 0.6) * dpr,
      c:  Math.random() < 0.65 ? GREEN : EMERALD,
    }))
    const D = 130 * dpr
    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      for (const p of pts) {
        p.x = (p.x + p.vx + w) % w
        p.y = (p.y + p.vy + h) % h
      }
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const dx = pts[i].x - pts[j].x
          const dy = pts[i].y - pts[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < D) {
            ctx.beginPath()
            ctx.strokeStyle = `rgba(${GREEN}, ${(1 - dist / D) * 0.25})`
            ctx.lineWidth = dpr * 0.6
            ctx.moveTo(pts[i].x, pts[i].y)
            ctx.lineTo(pts[j].x, pts[j].y)
            ctx.stroke()
          }
        }
      }
      for (const p of pts) {
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${p.c}, 0.55)`
        ctx.fill()
      }
      raf = requestAnimationFrame(draw)
    }
    if (!reduced) draw()
    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(raf)
    }
  }, [reduced])
  return <canvas ref={ref} className="absolute inset-0 h-full w-full" aria-hidden />
}

export default function Splash({ onDone }) {
  const [step, setStep]     = useState(-1)
  const [done, setDone]     = useState(false)
  const [leaving, setLeaving] = useState(false)
  const reduced = typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

  useEffect(() => {
    const timers = []
    STEPS.forEach((_, i) => {
      timers.push(setTimeout(() => setStep(i), START + STEP_MS * i))
    })
    timers.push(setTimeout(() => setDone(true),    FINALE))
    timers.push(setTimeout(() => setLeaving(true), LEAVE))
    timers.push(setTimeout(onDone,                 LEAVE + 500))
    return () => timers.forEach(clearTimeout)
  }, [onDone])

  return (
    <motion.div
      className="fixed inset-0 z-[200] flex items-center justify-center overflow-hidden bg-bg"
      animate={leaving ? { opacity: 0 } : { opacity: 1 }}
      transition={{ duration: 0.5 }}
      aria-hidden
    >
      {!reduced && <GraphField reduced={reduced} />}
      <div className="relative z-10 flex flex-col items-center gap-8 px-4 text-center">
        {/* WL ring */}
        <div className="relative">
          <svg viewBox="0 0 140 140" className="h-36 w-36" aria-hidden>
            <circle cx="70" cy="70" r="60" fill="none" stroke={`rgba(${GREEN},0.15)`} strokeWidth="14" />
            {!reduced && (
              <motion.circle
                cx="70" cy="70" r="60"
                fill="none"
                stroke={`rgba(${GREEN},1)`}
                strokeWidth="14"
                strokeLinecap="round"
                strokeDasharray={Math.PI * 2 * 60}
                initial={{ strokeDashoffset: Math.PI * 2 * 60 }}
                animate={{ strokeDashoffset: done ? 0 : Math.PI * 2 * 60 * (1 - (step + 1) / STEPS.length) }}
                transition={{ duration: 0.55, ease: 'easeOut' }}
              />
            )}
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="font-display text-2xl font-bold text-gradient">WL</span>
            <span className="text-[10px] font-medium text-muted tracking-wider">Kernel</span>
          </div>
        </div>

        {/* steps */}
        <ul className="flex flex-col gap-2">
          {STEPS.map((s, i) => (
            <li
              key={s.short}
              className={cx(
                'flex items-center gap-3 rounded-xl px-5 py-2.5 text-sm font-medium transition-all duration-300',
                i <= step ? 'bg-primary/15 text-primary ring-1 ring-primary/30' : 'text-muted',
              )}
            >
              <span
                className={cx(
                  'grid h-5 w-5 shrink-0 place-items-center rounded-full ring-1',
                  i <= step ? 'bg-primary ring-primary/50 text-white' : 'ring-line/25',
                )}
              >
                {i <= step ? (
                  <svg viewBox="0 0 10 10" className="h-3 w-3" fill="none" aria-hidden>
                    <path d="M2 5.5L4 7.5L8 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : (
                  <span className="h-1.5 w-1.5 rounded-full bg-line/40" />
                )}
              </span>
              {s.label}
            </li>
          ))}
        </ul>

        {done && !leaving && (
          <motion.p
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-sm font-medium text-primary"
          >
            Loading portal…
          </motion.p>
        )}
      </div>
    </motion.div>
  )
}
