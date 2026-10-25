import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  BookOpen,
  BrainCircuit,
  ChevronDown,
  GitBranch,
  Github,
  HelpCircle,
  History,
  Home,
  LayoutDashboard,
  Menu,
  Microscope,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  ScanLine,
  Search,
  FileCheck,
  Sprout,
  Sun,
  X,
} from 'lucide-react'
import { useEffect, useId, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { PLANTCARE_PAGES } from '../plantcare/PlantCareApp'
import { DISCLAIMER, HOME_SECTIONS, LINKS, PROJECT, isTodo } from '../lib/content'
import { cx } from '../lib/utils'
import { useGoToSection, useUi } from './extras'

const PLANTCARE_ICONS = {
  dashboard: LayoutDashboard,
  'analyze-plant': ScanLine,
  'results-diagnostics': Microscope,
  'plant-stress-library': BookOpen,
  crops: Sprout,
  'history-logs': History,
  'model-pipeline': BrainCircuit,
  protocol: FileCheck,
}

/** Sidebar navigation: the research portal (with its sections) and the PlantCare AI feature pages. */
export const NAV_GROUPS = [
  { title: 'Research portal', items: [{ to: '/', label: 'Portal home', icon: Home, sections: true }] },
  {
    title: 'PlantCare AI',
    items: PLANTCARE_PAGES.map((p) => ({ to: `/plantcare/${p.route}`, label: p.label, icon: PLANTCARE_ICONS[p.id], badge: p.badge })),
  },
]

/** Page title shown in the top bar for the current route. */
function routeTitle(pathname) {
  for (const g of NAV_GROUPS) {
    const hit = g.items.find((n) => (n.to === '/' ? pathname === '/' : pathname.startsWith(n.to)))
    if (hit) return [g.title, hit.label]
  }
  return ['WL-LeafKernel', '']
}

/** WL-graph logo: leaf outline with a small network overlay. */
export function Logo({ className }) {
  // unique per instance: a shared id breaks when the first copy sits in a hidden (display: none) sidebar
  const grad = `lg-wl-${useId().replace(/:/g, '')}`
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <defs>
        <linearGradient id={grad} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="rgb(var(--primary))" />
          <stop offset="1" stopColor="rgb(var(--accent))" />
        </linearGradient>
      </defs>
      {/* background tile */}
      <rect x="1" y="1" width="30" height="30" rx="9" fill={`url(#${grad})`} />
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

/** Desktop sidebar collapse state; publishes the current width as `--sb` for the offset layout. */
export function useSidebarState() {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem('wl-sidebar') === 'collapsed'
    } catch {
      return false
    }
  })
  useEffect(() => {
    document.documentElement.style.setProperty('--sb', collapsed ? '4.5rem' : '16.5rem')
    try {
      localStorage.setItem('wl-sidebar', collapsed ? 'collapsed' : 'open')
    } catch {
      /* private mode */
    }
  }, [collapsed])
  return [collapsed, () => setCollapsed((c) => !c)]
}

function SidebarBody({ collapsed, onToggle, onClose }) {
  const loc = useLocation()
  const goSection = useGoToSection()
  const [sectionsOpen, setSectionsOpen] = useState(loc.pathname === '/')
  useEffect(() => {
    if (loc.pathname === '/') setSectionsOpen(true)
  }, [loc.pathname])

  return (
    <div className="flex h-full flex-col">
      <div className={cx('flex h-16 shrink-0 items-center gap-2 border-b hairline', collapsed ? 'justify-center px-2' : 'justify-between px-4')}>
        {!collapsed && (
          <Link to="/" className="flex min-w-0 items-center gap-2.5" aria-label={`${PROJECT.short} home`}>
            <Logo className="h-8 w-8 shrink-0" />
            <span className="truncate font-display text-lg font-bold">
              WL<span className="text-primary">-Leaf</span><span className="text-accent">Kernel</span>
            </span>
          </Link>
        )}
        {onToggle && (
          <button
            type="button"
            onClick={onToggle}
            className="grid h-9 w-9 shrink-0 place-items-center rounded-lg text-muted transition hover:bg-ink/5 hover:text-ink"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          </button>
        )}
        {onClose && (
          <button type="button" onClick={onClose} className="grid h-9 w-9 place-items-center rounded-lg text-muted hover:text-ink" aria-label="Close menu">
            <X size={18} />
          </button>
        )}
      </div>

      <nav aria-label="Main" className="flex-1 overflow-y-auto overflow-x-hidden px-3 py-4 [scrollbar-width:thin]">
        {NAV_GROUPS.map((g) => (
          <div key={g.title} className="mb-5">
            {collapsed ? <div className="mx-auto mb-2 h-px w-6 bg-line/30" /> : <p className="label mb-2 px-3">{g.title}</p>}
            <ul className="flex flex-col gap-0.5">
              {g.items.map((n) => (
                <li key={n.to}>
                  <div className="flex items-center">
                    <NavLink
                      to={n.to}
                      end={n.to === '/'}
                      title={collapsed ? n.label : undefined}
                      className={({ isActive }) =>
                        cx(
                          'relative flex flex-1 items-center gap-3 rounded-lg py-2 text-sm font-medium transition',
                          collapsed ? 'justify-center px-0' : 'px-3',
                          isActive ? 'bg-primary/12 text-primary ring-1 ring-primary/25' : 'text-muted hover:bg-ink/5 hover:text-ink',
                        )
                      }
                    >
                      <n.icon size={18} className="shrink-0" aria-hidden />
                      {!collapsed && <span className="flex-1 truncate">{n.label}</span>}
                      {!collapsed && n.badge && (
                        <span className="rounded-full bg-primary/15 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-primary">{n.badge}</span>
                      )}
                      {collapsed && n.badge && <span className="absolute right-2 top-1.5 h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />}
                    </NavLink>
                    {n.sections && !collapsed && (
                      <button
                        type="button"
                        onClick={() => setSectionsOpen((o) => !o)}
                        className="ml-1 grid h-8 w-8 place-items-center rounded-lg text-muted hover:bg-ink/5 hover:text-ink"
                        aria-label={sectionsOpen ? 'Hide portal sections' : 'Show portal sections'}
                        aria-expanded={sectionsOpen}
                      >
                        <ChevronDown size={15} className={cx('transition-transform', sectionsOpen && 'rotate-180')} />
                      </button>
                    )}
                  </div>
                  <AnimatePresence initial={false}>
                    {n.sections && !collapsed && sectionsOpen && (
                      <motion.ul
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="ml-5 overflow-hidden border-l hairline pl-2"
                      >
                        {HOME_SECTIONS.map((s) => (
                          <li key={s.id}>
                            <button
                              type="button"
                              onClick={() => (goSection(s.id), onClose?.())}
                              className="w-full rounded-md px-3 py-1.5 text-left text-[13px] text-muted transition hover:bg-ink/5 hover:text-ink"
                            >
                              {s.label}
                            </button>
                          </li>
                        ))}
                      </motion.ul>
                    )}
                  </AnimatePresence>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {!collapsed && (
        <div className="m-3 rounded-xl bg-warn/[0.08] p-3 text-[11px] leading-relaxed text-muted ring-1 ring-warn/20">
          <p className="mb-0.5 flex items-center gap-1.5 font-semibold text-warn">
            <Activity size={12} /> Research prototype
          </p>
          PlantCare AI shows recorded out-of-fold results on OLID-I; new uploads are not scored yet.
        </div>
      )}
    </div>
  )
}

/** Fixed left sidebar on desktop (collapsible to an icon rail); slide-in drawer on smaller screens. */
export function Sidebar({ collapsed, onToggle, mobileOpen, onMobileClose }) {
  const loc = useLocation()
  useEffect(() => onMobileClose(), [loc.pathname]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <>
      <aside
        className="fixed inset-y-0 left-0 z-50 hidden w-[var(--sb)] border-r hairline bg-elev/80 backdrop-blur-xl transition-[width] duration-300 print:hidden lg:block"
        aria-label="Sidebar"
      >
        <SidebarBody collapsed={collapsed} onToggle={onToggle} />
      </aside>

      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              className="fixed inset-0 z-[70] bg-black/50 backdrop-blur-sm lg:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onMobileClose}
            />
            <motion.aside
              className="fixed inset-y-0 left-0 z-[71] w-72 max-w-[85vw] border-r hairline bg-elev shadow-2xl lg:hidden"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', bounce: 0, duration: 0.35 }}
              aria-label="Sidebar"
            >
              <SidebarBody collapsed={false} onClose={onMobileClose} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  )
}

/** Slim top bar to the right of the sidebar: menu button, current page, search, help and theme. */
export function TopBar({ theme, toggleTheme, onOpenMenu }) {
  const [scrolled, setScrolled] = useState(false)
  const { openHelp, openPalette } = useUi()
  const loc = useLocation()
  const [group, page] = routeTitle(loc.pathname)
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8)
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])

  return (
    <header
      className={cx(
        'fixed right-0 top-0 left-0 z-40 transition-[background-color,left] duration-300 print:hidden lg:left-[var(--sb)]',
        scrolled ? 'border-b hairline bg-bg/75 backdrop-blur-xl' : 'bg-transparent',
      )}
    >
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:rounded-lg focus:bg-elev focus:px-3 focus:py-2">
        Skip to content
      </a>
      <div className="flex h-16 items-center justify-between gap-4 px-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <button
            type="button"
            onClick={onOpenMenu}
            className="grid h-10 w-10 shrink-0 place-items-center rounded-xl ring-1 ring-line/15 lg:hidden"
            aria-label="Open menu"
          >
            <Menu size={18} />
          </button>
          <Link to="/" className="lg:hidden" aria-label={`${PROJECT.short} home`}>
            <Logo className="h-8 w-8" />
          </Link>
          <p className="hidden min-w-0 truncate text-sm sm:block">
            <span className="text-muted">{group}</span>
            {page && <span className="text-muted"> / </span>}
            <span className="font-medium text-ink">{page}</span>
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={openPalette}
            className="hidden h-10 items-center gap-2 rounded-xl px-3 text-sm text-muted ring-1 ring-line/15 transition hover:text-ink md:flex"
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
        </div>
      </div>
    </header>
  )
}

export function Footer() {
  return (
    <footer className="border-t hairline print:hidden">
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
