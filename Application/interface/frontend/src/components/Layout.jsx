import { AnimatePresence, motion } from 'framer-motion'
import { Activity, GitBranch, Github, HelpCircle, Home, Menu, Moon, Search, Sun, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { DISCLAIMER, LINKS, PROJECT, isTodo } from '../lib/content'
import { cx } from '../lib/utils'
import { useUi } from './extras'

/** Navigation — only the portal (home) page exists; section bar handles in-page navigation. */
export const NAV = [
  { to: '/', label: 'Portal', icon: Home },
]

/** WL-graph logo: leaf outline with a small network overlay. */
export function Logo({ className }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <defs>
        <linearGradient id="lg-wl" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="rgb(var(--primary))" />
          <stop offset="1" stopColor="rgb(var(--accent))" />
        </linearGradient>
      </defs>
      {/* background tile */}
      <rect x="1" y="1" width="30" height="30" rx="9" fill="url(#lg-wl)" />
      {/* stylised leaf with graph nodes */}
      <path
        d="M16 6 C10 6 6 11 6 16 C6 22 10 26 16 26 C22 26 26 22 26 16 C26 11 22 6 16 6Z"
        fill="none"
        stroke="white"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      {/* graph edges inside leaf */}
      <line x1="11" y1="19" x2="16" y2="13" stroke="white" strokeWidth="1.2" strokeOpacity="0.7" />
      <line x1="16" y1="13" x2="21" y2="19" stroke="white" strokeWidth="1.2" strokeOpacity="0.7" />
      <line x1="11" y1="19" x2="21" y2="19" stroke="white" strokeWidth="1.2" strokeOpacity="0.5" />
      {/* graph nodes */}
      <circle cx="16" cy="13" r="2" fill="white" />
      <circle cx="11" cy="19" r="1.8" fill="white" fillOpacity="0.85" />
      <circle cx="21" cy="19" r="1.8" fill="white" fillOpacity="0.85" />
    </svg>
  )
}

export function Navbar({ theme, toggleTheme }) {
  const [open, setOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const { openHelp, openPalette } = useUi()
  const loc = useLocation()
  useEffect(() => setOpen(false), [loc.pathname])
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8)
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])

  return (
    <nav
      aria-label="Main"
      className={cx(
        'fixed inset-x-0 top-0 z-50 transition-all duration-300',
        scrolled ? 'border-b hairline bg-bg/75 backdrop-blur-xl' : 'bg-transparent',
      )}
    >
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:rounded-lg focus:bg-elev focus:px-3 focus:py-2">
        Skip to content
      </a>
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <Link to="/" className="flex items-center gap-2.5" aria-label={`${PROJECT.short} home`}>
          <Logo className="h-8 w-8" />
          <span className="font-display text-lg font-bold">
            WL<span className="text-primary">-Leaf</span><span className="text-accent">Kernel</span>
          </span>
        </Link>

        <ul className="hidden items-center gap-1 lg:flex">
          {NAV.map((n) => (
            <li key={n.to}>
              <NavLink
                to={n.to}
                end={n.to === '/'}
                className={({ isActive }) =>
                  cx('relative flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition', isActive ? 'text-ink' : 'text-muted hover:text-ink')
                }
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <motion.span layoutId="nav-pill" className="absolute inset-0 rounded-lg bg-ink/[0.07] ring-1 ring-line/15" transition={{ type: 'spring', bounce: 0.2, duration: 0.5 }} />
                    )}
                    <n.icon size={15} className="relative" aria-hidden />
                    <span className="relative">{n.label}</span>
                  </>
                )}
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={openPalette}
            className="hidden h-10 items-center gap-2 rounded-xl px-3 text-sm text-muted ring-1 ring-line/15 transition hover:text-ink xl:flex"
            aria-label="Search pages and sections (Ctrl+K)"
          >
            <Search size={15} />
            <kbd className="rounded bg-ink/[0.07] px-1.5 font-mono text-[10px]">Ctrl K</kbd>
          </button>
          <button
            type="button"
            onClick={openHelp}
            className="flex h-10 items-center gap-1.5 rounded-xl bg-primary/12 px-3 text-sm font-semibold text-primary ring-1 ring-primary/30 transition hover:bg-primary/20"
          >
            <HelpCircle size={16} /> <span className="hidden sm:inline">How to use</span>
          </button>
          <button
            type="button"
            onClick={toggleTheme}
            className="grid h-10 w-10 place-items-center rounded-xl text-muted ring-1 ring-line/15 transition hover:text-ink"
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            <AnimatePresence mode="wait" initial={false}>
              <motion.span key={theme} initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.2 }}>
                {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
              </motion.span>
            </AnimatePresence>
          </button>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="grid h-10 w-10 place-items-center rounded-xl ring-1 ring-line/15 lg:hidden"
            aria-label="Toggle menu"
            aria-expanded={open}
          >
            {open ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-b hairline bg-bg/95 px-4 backdrop-blur-xl lg:hidden"
          >
            {NAV.map((n) => (
              <li key={n.to}>
                <NavLink
                  to={n.to}
                  end={n.to === '/'}
                  className={({ isActive }) => cx('flex items-center gap-3 rounded-lg px-3 py-3 text-sm font-medium', isActive ? 'bg-ink/5 text-primary' : 'text-muted')}
                >
                  <n.icon size={16} /> {n.label}
                </NavLink>
              </li>
            ))}
            <li className="h-3" />
          </motion.ul>
        )}
      </AnimatePresence>
    </nav>
  )
}

export function Footer() {
  return (
    <footer className="border-t hairline">
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="mb-8 flex items-start gap-3 rounded-xl bg-warn/[0.08] px-4 py-3 text-sm text-warn ring-1 ring-warn/20" role="note">
          <Activity size={16} className="mt-0.5 shrink-0" aria-hidden />
          <p>
            <strong>Research disclaimer:</strong> {DISCLAIMER}
          </p>
        </div>
        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <Logo className="h-8 w-8" />
            <div>
              <p className="font-display font-semibold">{PROJECT.short}</p>
              <p className="text-xs text-muted">
                {PROJECT.team} · {PROJECT.type} · Guide: {PROJECT.guide.name}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-4 text-sm text-muted">
            <a
              href={isTodo(LINKS.github) ? undefined : LINKS.github}
              target={isTodo(LINKS.github) ? undefined : '_blank'}
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 hover:text-ink"
            >
              <Github size={15} /> GitHub
            </a>
            <a href="#overview" className="hover:text-ink flex items-center gap-1.5">
              <GitBranch size={15} /> WL Graph Kernels
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
