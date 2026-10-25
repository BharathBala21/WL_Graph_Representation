import { MotionConfig } from 'framer-motion'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link, Route, Routes, useLocation } from 'react-router-dom'
import { Toaster } from 'sonner'
import { BackToTop, CommandPalette, HelpDrawer, ScrollProgress, UiContext } from './components/extras'
import { Footer, Navbar } from './components/Layout'
import Splash from './components/Splash'
import { BackendContext, ThemeContext, useHealthState, useThemeState } from './lib/hooks'
import Home from './pages/Home'

const TITLES = { '/': 'Portal · WL-LeafKernel' }

function NotFound() {
  return (
    <div className="mx-auto grid min-h-[70vh] max-w-xl place-items-center px-4 pt-24 text-center">
      <div>
        <p className="font-display text-7xl font-bold text-gradient">404</p>
        <p className="mt-3 text-muted">This page does not exist.</p>
        <Link to="/" className="btn-primary mt-6">Back to the portal</Link>
      </div>
    </div>
  )
}

export default function App() {
  const [theme, toggleTheme] = useThemeState()
  const backend = useHealthState()
  const loc = useLocation()
  const [helpOpen, setHelpOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [splash, setSplash] = useState(true)
  const hideSplash = useCallback(() => setSplash(false), [])
  const openHelp = useCallback(() => setHelpOpen(true), [])
  const closeHelp = useCallback(() => setHelpOpen(false), [])
  const openPalette = useCallback(() => setPaletteOpen(true), [])
  const ui = useMemo(() => ({ openHelp, openPalette }), [openHelp, openPalette])

  useEffect(() => {
    document.title = TITLES[loc.pathname] || 'WL-LeafKernel'
    if (!loc.hash) window.scrollTo({ top: 0 })
  }, [loc.pathname, loc.hash])

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((o) => !o)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <ThemeContext.Provider value={theme}>
      <BackendContext.Provider value={backend}>
        <UiContext.Provider value={ui}>
          <MotionConfig reducedMotion="user">
            {splash && <Splash onDone={hideSplash} />}
            <ScrollProgress />
            <Navbar theme={theme} toggleTheme={toggleTheme} />
            <main id="main" key={loc.pathname} className="page-enter min-h-screen">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </main>
            <Footer />
            <BackToTop />
            <HelpDrawer open={helpOpen} onClose={closeHelp} />
            <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} toggleTheme={toggleTheme} openHelp={openHelp} />
            <Toaster theme={theme} position="bottom-right" richColors closeButton toastOptions={{ className: 'font-sans' }} />
          </MotionConfig>
        </UiContext.Provider>
      </BackendContext.Provider>
    </ThemeContext.Provider>
  )
}
