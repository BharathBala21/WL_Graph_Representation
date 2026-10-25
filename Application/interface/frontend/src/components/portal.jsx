import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, Check, CircleDot, Github, Linkedin, Clock } from 'lucide-react'
import { Fragment, useEffect, useState } from 'react'
import { STAGES, isTodo } from '../lib/content'
import { cx } from '../lib/utils'
import { Badge } from './ui'

// ── Hero visual: animated WL pipeline steps ─────────────────────────────────
const HERO_FRAMES = [
  { label: 'Leaf image input',         tone: 'muted'   },
  { label: '38 features extracted',    tone: 'primary'  },
  { label: 'Feature graph constructed', tone: 'accent'  },
  { label: 'WL kernel computed',       tone: 'primary'  },
  { label: 'Stress class predicted',   tone: 'ok'       },
]

/** Animated representation of the WL pipeline — SVG graph visualization. */
export function HeroVisual() {
  const [i, setI] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setI((x) => (x + 1) % HERO_FRAMES.length), 2800)
    return () => clearInterval(id)
  }, [])
  const f = HERO_FRAMES[i]

  // Node positions for a small feature graph (38 nodes → schematic 8-node illustration)
  const nodes = [
    { x: 90,  y: 80,  label: 'Area',    group: 'morph'   },
    { x: 165, y: 55,  label: 'Colour',  group: 'colour'  },
    { x: 230, y: 90,  label: 'Texture', group: 'texture' },
    { x: 200, y: 155, label: 'LBP',     group: 'texture' },
    { x: 130, y: 175, label: 'Spatial', group: 'spatial' },
    { x: 60,  y: 155, label: 'Peri.',   group: 'morph'   },
    { x: 50,  y: 110, label: 'Eccen.',  group: 'morph'   },
    { x: 165, y: 125, label: 'GLCM',   group: 'texture' },
  ]
  const edges = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,0],[1,7],[7,3],[0,7],[2,7]]
  const gColor = {
    morph: 'rgb(var(--primary))',
    colour: 'rgb(var(--accent))',
    texture: 'rgb(134, 239, 172)',
    spatial: 'rgb(var(--ok))',
  }

  return (
    <div className="relative mx-auto w-full max-w-md animate-float">
      <div className="absolute -inset-6 rounded-[2.5rem] bg-gradient-to-br from-primary/30 via-accent/20 to-transparent blur-2xl" aria-hidden />
      <div className="glass relative overflow-hidden p-4">
        <div className="relative aspect-square overflow-hidden rounded-xl bg-bg/60">
          {/* animated graph */}
          <svg viewBox="0 0 280 230" className="absolute inset-0 h-full w-full" aria-hidden>
            {/* edges */}
            {edges.map(([a, b], k) => (
              <motion.line
                key={k}
                x1={nodes[a].x} y1={nodes[a].y}
                x2={nodes[b].x} y2={nodes[b].y}
                stroke={i >= 2 ? 'rgb(var(--primary))' : 'rgb(100,116,139)'}
                strokeWidth={i >= 2 ? 1.5 : 0.8}
                strokeOpacity={i >= 2 ? 0.6 : 0.25}
                animate={{ strokeOpacity: i >= 2 ? 0.6 : 0.2 }}
                transition={{ duration: 0.6 }}
              />
            ))}
            {/* WL refinement rings (shown at step 3+) */}
            {i >= 3 && nodes.map((n, k) => (
              <motion.circle
                key={`ring-${k}`}
                cx={n.x} cy={n.y} r={18}
                fill="none"
                stroke="rgb(var(--primary))"
                strokeWidth="0.8"
                strokeOpacity={0.3}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: k * 0.06 }}
              />
            ))}
            {/* nodes */}
            {nodes.map((n, k) => (
              <g key={k}>
                <motion.circle
                  cx={n.x} cy={n.y} r={i >= 1 ? 14 : 10}
                  fill={i >= 1 ? (gColor[n.group] || 'rgb(var(--primary))') : 'rgb(100,116,139)'}
                  fillOpacity={0.25}
                  stroke={i >= 1 ? (gColor[n.group] || 'rgb(var(--primary))') : 'rgb(100,116,139)'}
                  strokeWidth={i >= 2 ? 2 : 1}
                  animate={{ r: i >= 1 ? 14 : 10 }}
                  transition={{ duration: 0.5, delay: k * 0.05 }}
                />
                <text x={n.x} y={n.y + 1} textAnchor="middle" dominantBaseline="middle" fontSize="7" fill="rgb(230,237,247)" fontFamily="monospace">
                  {n.label}
                </text>
              </g>
            ))}
            {/* result label at step 4 */}
            {i >= 4 && (
              <motion.g initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <rect x="90" y="198" width="100" height="22" rx="6" fill="rgb(var(--ok))" fillOpacity="0.2" stroke="rgb(var(--ok))" strokeWidth="1" />
                <text x="140" y="210" textAnchor="middle" fontSize="9" fill="rgb(var(--ok))" fontFamily="sans-serif" fontWeight="600">
                  Stressed
                </text>
              </motion.g>
            )}
          </svg>

          {/* step label */}
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-lg bg-black/60 px-2.5 py-1.5 text-xs font-medium text-white backdrop-blur">
            <span className="h-2 w-2 rounded-full" style={{ background: `rgb(var(--${f.tone === 'muted' ? 'muted' : f.tone}))` }} />
            {f.label}
          </div>
        </div>

        {/* progress bar */}
        <div className="mt-3 flex gap-1.5" aria-hidden>
          {HERO_FRAMES.map((_, k) => (
            <span key={k} className={cx('h-1 flex-1 rounded-full transition', k <= i ? 'bg-primary' : 'bg-ink/10')} />
          ))}
        </div>
      </div>

      {/* floating chips */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.9 }}
        className="glass absolute -right-3 top-10 hidden items-center gap-2 px-3 py-2 text-xs sm:flex"
      >
        <Check size={14} className="text-ok" /> Leakage-safe CV
      </motion.div>
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 1.2 }}
        className="glass absolute -left-4 bottom-16 hidden items-center gap-2 px-3 py-2 text-xs sm:flex"
      >
        <CircleDot size={14} className="text-accent" /> 5 graph topologies
      </motion.div>
    </div>
  )
}

export function HeroBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
      <div className="grid-bg absolute inset-0 [mask-image:radial-gradient(ellipse_at_center,black_20%,transparent_75%)]" />
      <motion.div
        className="absolute -left-32 top-10 h-[28rem] w-[28rem] rounded-full bg-primary/25 blur-[110px]"
        animate={{ x: [0, 60, 0], y: [0, 40, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute -right-24 top-40 h-[26rem] w-[26rem] rounded-full bg-accent/25 blur-[110px]"
        animate={{ x: [0, -50, 0], y: [0, -30, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: 'easeInOut' }}
      />
    </div>
  )
}

// ── Interactive pipeline diagram ─────────────────────────────────────────────
const NODE_COLOR = {
  muted:   'from-slate-500/30 to-slate-500/5 text-ink',
  primary: 'from-primary/30 to-primary/5 text-primary',
  accent:  'from-accent/30 to-accent/5 text-accent',
  ok:      'from-ok/30 to-ok/5 text-ok',
}

export function PipelineDiagram() {
  const [active, setActive] = useState(null)
  const curr = STAGES.find((s) => s.id === active)

  return (
    <div className="grid gap-6 lg:grid-cols-[auto_1fr]">
      {/* pipeline nodes */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2 lg:flex-col lg:overflow-x-visible lg:pb-0">
        {STAGES.map((s, i) => (
          <Fragment key={s.id}>
            <button
              type="button"
              onClick={() => setActive((a) => (a === s.id ? null : s.id))}
              aria-pressed={active === s.id}
              className={cx(
                'relative shrink-0 rounded-2xl bg-gradient-to-br px-4 py-3 text-left transition hover:-translate-y-0.5',
                NODE_COLOR[s.color],
                active === s.id ? 'ring-2 ring-primary/50 shadow-glow' : 'ring-1 ring-line/15',
              )}
            >
              <p className="text-xs font-semibold">{s.label}</p>
              <p className="mt-0.5 text-[10px] opacity-70">{s.sub}</p>
            </button>
            {i < STAGES.length - 1 && (
              <ArrowRight size={14} className="shrink-0 text-muted lg:rotate-90" aria-hidden />
            )}
          </Fragment>
        ))}
      </div>

      {/* detail panel */}
      <AnimatePresence mode="wait">
        {curr ? (
          <motion.div
            key={curr.id}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            transition={{ duration: 0.25 }}
            className="glass rounded-2xl p-5"
          >
            <p className={cx('label mb-2', `text-${curr.color}`)}>{curr.label}</p>
            <p className="text-sm leading-relaxed text-muted">{curr.details.what}</p>
            <ul className="mt-4 space-y-2">
              {curr.details.bullets.map((b) => (
                <li key={b} className="flex gap-2 text-sm">
                  <Check size={14} className="mt-0.5 shrink-0 text-ok" />
                  <span className="text-muted">{b}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center justify-center rounded-2xl border border-dashed hairline p-8 text-sm text-muted"
          >
            Click a pipeline stage to see details
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/** Team member card — uses placeholder-safe display. */
export function TeamCard({ m, i }) {
  const hasPic    = m.photo && !isTodo(m.photo)
  const hasLi     = m.linkedin && !isTodo(m.linkedin)
  const hasGh     = m.github && !isTodo(m.github)
  const isPlaceholder = isTodo(m.name)

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: i * 0.07, duration: 0.5 }}
      className="glass group relative overflow-hidden p-5 spotlight"
    >
      {/* avatar */}
      <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center overflow-hidden rounded-2xl bg-gradient-to-br from-primary/30 to-accent/30">
        {hasPic ? (
          <img src={m.photo} alt={m.name} className="h-full w-full object-cover" loading="lazy" />
        ) : (
          <span className="font-display text-3xl font-bold text-primary/60">
            {isPlaceholder ? '?' : m.name.charAt(0)}
          </span>
        )}
      </div>
      <p className={cx('text-center font-display font-semibold', isPlaceholder && 'text-muted italic')}>
        {isPlaceholder ? 'Team member' : m.name}
      </p>
      {m.reg && !isTodo(m.reg) && (
        <p className="mt-0.5 text-center text-xs text-muted">{m.reg}</p>
      )}
      {m.role && !isTodo(m.role) && (
        <p className="mt-1 text-center text-xs text-muted">{m.role}</p>
      )}
      {(hasLi || hasGh) && (
        <div className="mt-4 flex justify-center gap-2">
          {hasLi && (
            <a href={m.linkedin} target="_blank" rel="noreferrer" className="grid h-8 w-8 place-items-center rounded-lg bg-ink/5 text-muted ring-1 ring-line/15 transition hover:text-primary" aria-label={`${m.name} LinkedIn`}>
              <Linkedin size={15} />
            </a>
          )}
          {hasGh && (
            <a href={m.github} target="_blank" rel="noreferrer" className="grid h-8 w-8 place-items-center rounded-lg bg-ink/5 text-muted ring-1 ring-line/15 transition hover:text-ink" aria-label={`${m.name} GitHub`}>
              <Github size={15} />
            </a>
          )}
        </div>
      )}
    </motion.div>
  )
}

/** Project progress timeline. */
const STATUS_STYLES = {
  done:    { dot: 'bg-ok', label: 'Completed', text: 'text-ok',    ring: 'ring-ok/30'    },
  active:  { dot: 'bg-primary animate-pulse', label: 'In progress', text: 'text-primary', ring: 'ring-primary/30' },
  planned: { dot: 'bg-muted', label: 'Planned', text: 'text-muted', ring: 'ring-line/20'  },
}

export function Timeline({ items }) {
  return (
    <ol className="relative border-l border-line/20 pl-6 space-y-8">
      {items.map((it, i) => {
        const s = STATUS_STYLES[it.status] || STATUS_STYLES.planned
        return (
          <motion.li
            key={i}
            initial={{ opacity: 0, x: -12 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.06, duration: 0.45 }}
            className="relative"
          >
            {/* dot */}
            <span className={cx('absolute -left-[1.625rem] top-1.5 h-3 w-3 rounded-full ring-2 ring-bg', s.dot)} aria-hidden />
            <div className={cx('glass rounded-2xl p-4 ring-1', s.ring)}>
              <div className="mb-1 flex flex-wrap items-center gap-2">
                <Badge tone={it.status === 'done' ? 'ok' : it.status === 'active' ? 'primary' : 'muted'}>
                  {s.label}
                </Badge>
                <span className="label">{it.phase}</span>
              </div>
              <p className="font-display font-semibold">{it.title}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{it.body}</p>
            </div>
          </motion.li>
        )
      })}
    </ol>
  )
}
