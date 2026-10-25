/** @type {import('tailwindcss').Config} */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`

export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: ['class', '[data-theme="dark"]'],
  // classes built from tone names at runtime (e.g. `text-${tone}`)
  safelist: [{ pattern: /^(text|bg|ring|border)-(primary|accent|ok|warn|bad|muted)$/ }],
  theme: {
    extend: {
      opacity: { 3: '0.03', 6: '0.06', 8: '0.08', 12: '0.12', 15: '0.15', 35: '0.35' },
      colors: {
        bg: token('bg'),
        elev: token('elev'),
        line: token('line'),
        ink: token('ink'),
        muted: token('muted'),
        primary: token('primary'),
        accent: token('accent'),
        ok: token('ok'),
        warn: token('warn'),
        bad: token('bad'),
        card: token('card'),
        // PlantCare AI palette (src/plantcare/plantcare.css)
        pc: {
          'bg': token('pc-bg'),
          'card': token('pc-card'),
          'low': token('pc-low'),
          'cont': token('pc-cont'),
          'high': token('pc-high'),
          'line': token('pc-line'),
          'ink': token('pc-ink'),
          'ink2': token('pc-ink2'),
          'muted': token('pc-muted'),
          'outline': token('pc-outline'),
          'green': token('pc-green'),
          'green-bg': token('pc-green-bg'),
          'deep': token('pc-deep'),
          'deep-ink': token('pc-deep-ink'),
          'forest': token('pc-forest'),
          'mint': token('pc-mint'),
          'mint-ink': token('pc-mint-ink'),
          'amber-bg': token('pc-amber-bg'),
          'amber': token('pc-amber'),
          'amber-ink': token('pc-amber-ink'),
          'err-bg': token('pc-err-bg'),
          'err-ink': token('pc-err-ink'),
          'err': token('pc-err'),
        },
        dz: { cardio: token('d-cardio'), edema: token('d-edema'), effusion: token('d-effusion') },
      },
      fontFamily: {
        sans: ['"Inter Variable"', 'Inter', 'system-ui', 'sans-serif'],
        display: ['"Space Grotesk"', '"Inter Variable"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 0 1px rgb(var(--primary) / 0.25), 0 8px 40px -8px rgb(var(--primary) / 0.35)',
        soft: '0 1px 2px rgb(0 0 0 / 0.06), 0 12px 32px -12px rgb(0 0 0 / 0.25)',
        xs: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
      },
      backdropBlur: { xs: '2px' },
      keyframes: {
        shimmer: { '100%': { transform: 'translateX(100%)' } },
        scan: { '0%': { top: '0%' }, '100%': { top: '100%' } },
        float: { '0%,100%': { transform: 'translateY(0)' }, '50%': { transform: 'translateY(-10px)' } },
      },
      animation: {
        shimmer: 'shimmer 1.6s infinite',
        scan: 'scan 2.2s ease-in-out infinite alternate',
        float: 'float 6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
