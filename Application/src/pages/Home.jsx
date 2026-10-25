import { motion } from 'framer-motion'
import {
  ArrowRight,
  BookOpen,
  Code2,
  Database,
  Download,
  ExternalLink,
  FlaskConical,
  Github,
  GitBranch,
  GraduationCap,
  HelpCircle,
  Leaf,
  Lightbulb,
  Mail,
  Network,
  PlayCircle,
  Target,
  Workflow,
} from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'
import { RecallComparisonChart, TomatoMetricsTable, TomatoResultsChart } from '../components/charts'
import { HowToSteps, SectionNav, useUi } from '../components/extras'
import { HeroBackground, HeroVisual, PipelineDiagram, TeamCard, Timeline } from '../components/portal'
import { Accordion, Badge, Card, Counter, EmptyState, Notice, Reveal, Section, Tabs } from '../components/ui'
import {
  DATASETS,
  DOCUMENTS,
  GAPS,
  HERO_STATS,
  LINKS,
  NOVELTY,
  OBJECTIVES,
  PROBLEMS,
  PROJECT,
  REFERENCES,
  STAGES,
  TEAM,
  TIMELINE,
  isTodo,
} from '../lib/content'

const PROBLEM_ICONS = { Leaf, Network, GitBranch }

function scrollTo(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
}

function Hero() {
  const github = () =>
    isTodo(LINKS.github)
      ? (scrollTo('github'), toast.info('GitHub URL not set yet', { description: 'Set LINKS.github in src/lib/content.js' }))
      : window.open(LINKS.github, '_blank', 'noopener')
  return (
    <header id="overview" className="relative scroll-mt-16 overflow-hidden pb-16 pt-28 sm:pb-24 sm:pt-36">
      <HeroBackground />
      <div className="relative mx-auto grid max-w-7xl items-center gap-14 px-4 sm:px-6 lg:grid-cols-[1.15fr_1fr] lg:px-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}>
          <div className="mb-5 flex flex-wrap gap-2">
            <Badge tone="primary">{PROJECT.team}</Badge>
            <Badge>{PROJECT.type}</Badge>
            <Badge tone="accent" icon={GraduationCap}>{PROJECT.department}</Badge>
          </div>
          <h1 className="text-4xl font-bold leading-[1.05] sm:text-5xl">
            <span className="text-gradient">WL-LeafKernel</span>
            <span className="mt-3 block text-lg font-semibold leading-tight text-muted sm:text-xl">
              How Does Graph Construction Influence the Effectiveness of Weisfeiler–Lehman Graph Kernels for Fine-Grained Plant-Stress Classification from Leaf Images?
            </span>
          </h1>
          <p className="mt-6 max-w-xl text-base leading-relaxed text-muted">
            Systematically comparing five feature-graph topologies and four WL subtree heights for plant-stress classification from handcrafted leaf features. Evaluated against Random Forest and RBF SVM baselines under a leakage-safe nested cross-validation protocol.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <button type="button" onClick={() => scrollTo('results')} className="btn-primary px-5 py-3 text-base">
              <FlaskConical size={18} /> View Results
            </button>
            <button type="button" onClick={github} className="btn-ghost px-5 py-3 text-base">
              <Github size={18} /> GitHub
            </button>
            <button type="button" onClick={() => scrollTo('video')} className="btn-ghost px-5 py-3 text-base">
              <PlayCircle size={18} /> Watch Video
            </button>
          </div>
          <dl className="mt-12 grid max-w-xl grid-cols-2 gap-6 sm:grid-cols-4">
            {HERO_STATS.map((x) => (
              <div key={x.l}>
                <dt className="sr-only">{x.l}</dt>
                <dd className="font-display text-3xl font-bold">
                  <Counter value={x.v} decimals={0} />
                </dd>
                <p className="mt-1 text-xs leading-snug text-muted">{x.l}</p>
              </div>
            ))}
          </dl>
        </motion.div>
        <motion.div initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8, delay: 0.2 }}>
          <HeroVisual />
        </motion.div>
      </div>
    </header>
  )
}

function HowToUse() {
  const { openHelp } = useUi()
  return (
    <Section
      id="how-to-use"
      eyebrow="How to use"
      title="Using the WL inference interface"
      intro="The interface is a research prototype. Backend inference is in progress. The steps below describe the intended workflow."
      className="bg-elev/40"
    >
      {/* <Reveal>
        <Notice tone="warn">
          <strong>Prototype notice:</strong> The image upload and WL inference endpoint are currently in progress. Steps are shown for planning purposes and will be labeled clearly when functional.
        </Notice>
      </Reveal> */}
      <Reveal className="mt-6">
        <HowToSteps />
      </Reveal>
      <Reveal className="mt-6 flex flex-wrap items-center gap-3">
        <button type="button" onClick={() => scrollTo('methodology')} className="btn-primary">
          <Workflow size={16} /> View Methodology
        </button>
        <button type="button" onClick={() => scrollTo('results')} className="btn-ghost">
          <FlaskConical size={16} /> View Results
        </button>
        <button type="button" onClick={openHelp} className="btn-ghost">
          <HelpCircle size={16} /> Page-by-page tips
        </button>
      </Reveal>
    </Section>
  )
}

function Results() {
  const [tab, setTab] = useState('tomato')
  return (
    <div id="results" className="mt-16 scroll-mt-32">
      <Reveal className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="label mb-2 text-primary">Experimental results</p>
          <h3 className="font-display text-2xl font-bold">What the experiments showed</h3>
        </div>
        <Tabs
          value={tab}
          onChange={setTab}
          tabs={[
            { value: 'tomato', label: 'Tomato binary' },
            { value: 'multiclass', label: '57-class task' },
            { value: 'farm', label: 'Farm evaluation' },
          ]}
        />
      </Reveal>

      {tab === 'tomato' && (
        <div className="space-y-6">
          <Reveal>
            <Notice tone="muted">
              <strong>Experiment scope:</strong> Results below are for the <strong>tomato binary stress detection task only</strong> — 580 images (184 healthy, 396 stressed). Do not generalise to all 57 classes or all crops.
            </Notice>
          </Reveal>
          <div className="grid gap-6 lg:grid-cols-2">
            <Reveal>
              <Card>
                <p className="mb-4 font-display font-semibold">Accuracy / Balanced Accuracy / Macro-F1</p>
                <TomatoResultsChart />
              </Card>
            </Reveal>
            <Reveal delay={0.07}>
              <Card>
                <p className="mb-1 font-display font-semibold">Healthy-class recall comparison</p>
                <RecallComparisonChart />
              </Card>
            </Reveal>
          </div>
          <Reveal>
            <Card>
              <p className="mb-4 font-display font-semibold">Full metrics table — mean ± std (5 outer folds)</p>
              <TomatoMetricsTable />
            </Card>
          </Reveal>
          <Reveal>
            <Notice tone="muted">
              WL kernel healthy-class recall (65.32%) exceeds Random Forest (57.67%), even though RF has higher overall accuracy (83.45% vs 78.97%). This illustrates why balanced accuracy and per-class recall are critical evaluation metrics alongside aggregate accuracy.
            </Notice>
          </Reveal>
        </div>
      )}

      {tab === 'multiclass' && (
        <Reveal>
          <EmptyState icon={Database} title="57-class results pending verification">
            Verified result files for the full 57-class plant-stress experiment have not yet been confirmed. Results will be displayed here once cross-validated and audited. Do not use unverified numbers in the presentation.
          </EmptyState>
        </Reveal>
      )}

      {tab === 'farm' && (
        <Reveal>
          <Card className="relative overflow-hidden">
            <div className="absolute -right-8 -top-8 h-48 w-48 rounded-full bg-warn/10 blur-3xl" aria-hidden />
            <div className="relative">
              <Badge tone="warn" className="mb-4">Planned — not yet completed</Badge>
              <p className="font-display text-xl font-semibold">Prospective farm-image evaluation</p>
              <p className="mt-3 text-sm leading-relaxed text-muted">
                Leaf images will be collected during a future farm visit and used as an independent external test set. This evaluation will assess how well the WL graph-kernel models generalise to real-world field conditions not represented in OLID-I.
              </p>
              <ul className="mt-4 space-y-2">
                {['Images collected in uncontrolled field conditions', 'Independent of OLID-I training and validation data', 'Same feature extraction and graph construction pipeline', 'Results to be reported here after collection and evaluation'].map((b) => (
                  <li key={b} className="flex gap-2 text-sm text-muted">
                    <ArrowRight size={14} className="mt-0.5 shrink-0 text-warn" /> {b}
                  </li>
                ))}
              </ul>
            </div>
          </Card>
        </Reveal>
      )}
    </div>
  )
}

const DOC_SECTIONS = [
  {
    title: 'Repository structure',
    icon: Code2,
    content: (
      <div className="space-y-3">
        <p>The web app lives in <code className="font-mono text-ink">Application/</code> alongside the pipeline scripts in the project root.</p>
        <pre className="overflow-x-auto rounded-xl bg-black/80 p-4 font-mono text-xs leading-relaxed text-slate-200">{`# frontend (React + Vite)
cd Application
npm install
npm run dev        # http://localhost:5173

# WL pipeline (Python)
# Full pipeline scripts are in the project root and full_pipeline/`}</pre>
      </div>
    ),
  },
  {
    title: 'WL kernel pipeline',
    icon: GitBranch,
    content: (
      <ol className="list-decimal space-y-1.5 pl-5">
        <li>Extract 38 handcrafted features from a segmented leaf image.</li>
        <li>Construct a feature graph according to the selected topology strategy.</li>
        <li>Discretise node labels using training-fold statistics only (no leakage).</li>
        <li>Compute WL subtree kernel for the selected height h.</li>
        <li>Train SVM on the normalised Gram matrix; evaluate on held-out outer fold.</li>
      </ol>
    ),
  },
  {
    title: 'Backend integration status',
    icon: Database,
    content: (
      <div className="space-y-2">
        <p>A FastAPI backend endpoint for leaf-image inference is <strong>planned</strong>. The frontend currently shows the research results and methodology. Inference integration will follow once model artifacts are finalised.</p>
        <p>The existing FastAPI architecture from the reference project (<code className="font-mono text-ink">interface/backend/</code>) will be adapted.</p>
      </div>
    ),
  },
]

function Documentation() {
  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <Card className="p-3 sm:p-4">
        <p className="label mb-3 text-primary">Research documents</p>
        <div className="space-y-3">
          {DOCUMENTS.map((d) => (
            <div key={d.title} className="flex items-start gap-3 rounded-xl bg-ink/[0.03] px-4 py-3 ring-1 ring-line/10">
              <BookOpen size={18} className="mt-0.5 shrink-0 text-primary" />
              <div>
                <p className="font-semibold text-sm">{d.title}</p>
                <p className="mt-0.5 text-xs text-muted">{d.desc}</p>
                {d.internal && (
                  <Badge tone="warn" className="mt-2">Internal research artifact</Badge>
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>
      <div className="space-y-4">
        <Accordion items={DOC_SECTIONS} />
        <Card className="p-5">
          <p className="label mb-3">References</p>
          <ol className="list-decimal space-y-1.5 pl-4 text-xs leading-relaxed text-muted">
            {REFERENCES.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ol>
        </Card>
      </div>
    </div>
  )
}

function VideoSection() {
  if (isTodo(LINKS.video))
    return (
      <div className="glass relative grid aspect-video place-items-center overflow-hidden">
        <div className="grid-bg absolute inset-0 opacity-60" aria-hidden />
        <div className="relative text-center">
          <motion.div
            animate={{ scale: [1, 1.08, 1] }}
            transition={{ duration: 2.4, repeat: Infinity }}
            className="mx-auto mb-4 grid h-20 w-20 place-items-center rounded-full bg-gradient-to-br from-primary to-accent text-white shadow-glow"
          >
            <PlayCircle size={40} />
          </motion.div>
          <p className="font-display text-lg font-semibold">Project demonstration video coming soon</p>
          <p className="mt-1 text-sm text-muted">
            Set <code className="font-mono">LINKS.video</code> in <code className="font-mono">src/lib/content.js</code>
          </p>
          <button type="button" onClick={() => scrollTo('results')} className="btn-ghost mt-5">
            View results instead <ArrowRight size={16} />
          </button>
        </div>
      </div>
    )
  const isFile = /\.(mp4|webm)$/i.test(LINKS.video)
  return (
    <div className="glass overflow-hidden p-2">
      {isFile ? (
        <video src={LINKS.video} controls className="aspect-video w-full rounded-xl" />
      ) : (
        <iframe src={LINKS.video} title="WL-LeafKernel demonstration video" className="aspect-video w-full rounded-xl" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen />
      )}
    </div>
  )
}

export default function Home() {
  return (
    <>
      <SectionNav />
      <Hero />
      <HowToUse />

      {/* Objectives */}
      <Section id="objectives" eyebrow="Project objectives" title="Six research goals" intro="Each objective is testable and tied to the verified experimental design.">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {OBJECTIVES.map((o, i) => (
            <Reveal key={o.title} delay={i * 0.07}>
              <Card hover className="h-full">
                <div className="mb-4 flex items-center justify-between">
                  <span className="grid h-10 w-10 place-items-center rounded-xl bg-primary/12 text-primary">
                    <Target size={20} />
                  </span>
                  <span className="font-display text-3xl font-bold text-ink/10">0{i + 1}</span>
                </div>
                <h3 className="font-display text-lg font-semibold">{o.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{o.body}</p>
              </Card>
            </Reveal>
          ))}
        </div>
      </Section>

      {/* Problem */}
      <Section id="problem" eyebrow="Research problem" title="Three open challenges in plant-stress classification" className="bg-elev/40">
        <div className="grid gap-4 md:grid-cols-3">
          {PROBLEMS.map((p, i) => {
            const Icon = PROBLEM_ICONS[p.icon] || Leaf
            return (
              <Reveal key={p.title} delay={i * 0.1}>
                <Card hover className="h-full border-t-2 border-t-bad/60">
                  <Icon className="mb-4 text-bad" size={28} />
                  <h3 className="font-display text-lg font-semibold">{p.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-muted">{p.body}</p>
                </Card>
              </Reveal>
            )
          })}
        </div>
        <Reveal className="mt-10">
          <p className="label mb-4">Research gaps this project addresses</p>
          <div className="grid gap-4 md:grid-cols-3">
            {GAPS.map((g) => (
              <div
                key={g.title}
                className="group relative overflow-hidden rounded-2xl bg-ok/[0.04] p-5 pl-6 ring-1 ring-ok/20 transition duration-300 hover:-translate-y-1 hover:ring-ok/50 hover:shadow-[0_8px_32px_-10px_rgb(var(--ok)/0.45)]"
              >
                <span className="absolute inset-y-0 left-0 w-1 bg-gradient-to-b from-ok via-primary to-ok/30" aria-hidden />
                <span className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-ok/70 via-primary/40 to-transparent" aria-hidden />
                <p className="flex items-center gap-2 font-semibold">
                  <Lightbulb size={16} className="shrink-0 text-ok" /> {g.title}
                </p>
                <p className="mt-2 text-sm leading-relaxed text-muted">{g.body}</p>
              </div>
            ))}
          </div>
        </Reveal>
        <Reveal className="mt-8">
          <Notice tone="muted">
            <strong>Research question vs hypothesis:</strong> This project asks <em>how</em> graph topology affects WL kernel performance — it does not claim that graph-based representations necessarily outperform conventional tabular methods. The comparison is empirical and the answer is drawn from the nested-CV results.
          </Notice>
        </Reveal>
      </Section>

      {/* Methodology */}
      <Section id="methodology" eyebrow="Methodology" title="An eight-step, leakage-safe experimental pipeline" intro="Click a stage to see implementation details. Graph nodes represent the 38 extracted features — not individual pixels or images.">
        <Reveal>
          <PipelineDiagram />
        </Reveal>
        <div className="mt-10 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <Reveal>
            {DATASETS.map((d, i) => (
              <Card key={d.name}>
                <div className="mb-3 flex items-center justify-between gap-2">
                  <p className="font-display text-lg font-semibold">{d.name}</p>
                  <Badge tone="accent">{d.use}</Badge>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {d.stats.map((s) => (
                    <div key={s.label}>
                      <p className="font-display text-xl font-bold text-primary">
                        <Counter value={s.value} decimals={s.decimals || 0} suffix={s.suffix || ''} />
                      </p>
                      <p className="text-xs text-muted">{s.label}</p>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-xs leading-relaxed text-muted">{d.body}</p>
              </Card>
            ))}
          </Reveal>
          <Reveal delay={0.1}>
            <Card>
              <p className="label mb-3 text-primary">Graph topology strategies</p>
              <ul className="space-y-3">
                {[
                  { name: 'Correlation', desc: 'Edge if |Pearson r| > threshold (training fold only)' },
                  { name: 'Feature-similarity kNN', desc: 'Connect k nearest feature channels by value similarity' },
                  { name: 'Feature-family', desc: 'Intra-group edges within morphology / colour / texture / spatial' },
                  { name: 'Spatial/Regional', desc: 'Connect spatially co-located feature descriptors' },
                  { name: 'Hybrid', desc: 'Combine feature-family and correlation edges' },
                ].map((t, i) => (
                  <li key={t.name} className="flex gap-3 text-sm">
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-lg bg-primary/12 text-xs font-bold text-primary">{i + 1}</span>
                    <div>
                      <p className="font-semibold">{t.name}</p>
                      <p className="text-xs text-muted">{t.desc}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </Card>
          </Reveal>
        </div>
        <Results />
      </Section>

      {/* Novelty */}
      <Section id="novelty" eyebrow="Novelty & key contributions" title="What this project contributes" className="bg-elev/40">
        <div className="grid gap-5 lg:grid-cols-3">
          {NOVELTY.map((n, i) => (
            <Reveal key={n.n} delay={i * 0.1}>
              <Card hover className="relative h-full overflow-hidden">
                <span className="absolute -right-3 -top-6 font-display text-[7rem] font-bold leading-none text-primary/[0.08]" aria-hidden>
                  {n.n}
                </span>
                <p className="label text-primary">Contribution {n.n}</p>
                <h3 className="mt-2 font-display text-xl font-semibold">{n.title}</h3>
                <ul className="mt-4 space-y-2.5">
                  {n.points.map((p) => (
                    <li key={p} className="flex gap-2.5 text-sm text-muted">
                      <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" /> {p}
                    </li>
                  ))}
                </ul>
              </Card>
            </Reveal>
          ))}
        </div>
        <Reveal className="mt-8">
          <Notice tone="muted">
            This project does <strong>not</strong> claim to have invented the Weisfeiler–Lehman algorithm, achieved state-of-the-art performance across all plant diseases, proved that graph representations universally outperform tabular methods, or reached 90% accuracy. The contribution is systematic empirical comparison under rigorous experimental conditions.
          </Notice>
        </Reveal>
      </Section>

      {/* Team */}
      <Section id="team" eyebrow="Team details" title={isTodo(PROJECT.team) ? 'Project team' : PROJECT.team}>
        <Reveal className="mb-8">
          <Card className="flex flex-col gap-5 p-6 md:flex-row md:items-center md:justify-between spotlight">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
              <span className="grid h-16 w-16 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-primary to-accent text-white shadow-lg shadow-primary/20">
                <GraduationCap size={30} />
              </span>
              <div>
                <p className="label text-primary font-semibold tracking-wider">Project Supervisor</p>
                <div className="flex flex-wrap items-center gap-2 mt-0.5">
                  <p className="font-display text-2xl font-bold tracking-tight text-ink">{PROJECT.guide.name}</p>
                  {PROJECT.guide.qualification && (
                    <span className="rounded-md bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary ring-1 ring-primary/25">
                      Qualification: {PROJECT.guide.qualification}
                    </span>
                  )}
                </div>
                <p className="mt-1 text-sm font-medium text-ink/90">
                  {PROJECT.guide.role}, {PROJECT.guide.department}
                </p>
                {PROJECT.guide.researchInterests && (
                  <p className="mt-1.5 text-xs text-muted">
                    <span className="text-ink/80 font-semibold">Research Interest:</span>{' '}
                    <span className="text-muted">{PROJECT.guide.researchInterests}</span>
                  </p>
                )}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 pt-4 border-t border-line/15 md:border-t-0 md:pt-0">
              {PROJECT.guide.email && (
                <a
                  href={`mailto:${PROJECT.guide.email}`}
                  className="inline-flex items-center gap-2 rounded-xl bg-elev px-3.5 py-2 text-xs font-medium text-ink ring-1 ring-line/20 transition hover:ring-primary/40 hover:text-primary"
                  title="Send email to supervisor"
                >
                  <Mail size={14} className="text-primary" />
                  <span>{PROJECT.guide.email}</span>
                </a>
              )}
              {/* {PROJECT.guide.scholar && (
                <a
                  href={PROJECT.guide.scholar}
                  target="_blank"
                  rel="noreferrer"
                  className="btn-primary text-xs flex items-center gap-2 px-4 py-2"
                  title="Open Google Scholar Profile"
                >
                  <BookOpen size={14} />
                  <span>Google Scholar Profile</span>
                  <ExternalLink size={12} className="opacity-80" />
                </a>
              )} */}
            </div>
          </Card>
        </Reveal>
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          {TEAM.map((m, i) => (
            <TeamCard key={m.reg || i} m={m} i={i} />
          ))}
        </div>
        {TEAM.some((m) => isTodo(m.name)) && (
          <Reveal className="mt-6">
            <Notice tone="warn">
              Team member details are placeholders. Update the <code className="font-mono">TEAM</code> array in <code className="font-mono">src/lib/content.js</code> with actual names, registration numbers, and social links.
            </Notice>
          </Reveal>
        )}
      </Section>

      {/* Progress */}
      <Section id="progress" eyebrow="Project progress" title="Where the project stands" intro="Milestones are based on verified completed work. Future milestones are clearly marked as Planned." className="bg-elev/40">
        <Timeline items={TIMELINE} />
      </Section>

      {/* Docs */}
      <Section id="docs" eyebrow="Documentation" title="Research documents and setup">
        <Documentation />
      </Section>

      {/* GitHub */}
      <Section id="github" eyebrow="Source code" title="GitHub repository" className="bg-elev/40">
        <Reveal>
          <Card className="relative overflow-hidden p-8 sm:p-10">
            <div className="absolute -right-10 -top-10 h-56 w-56 rounded-full bg-accent/20 blur-3xl" aria-hidden />
            <div className="relative flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
              <div className="flex items-start gap-5">
                <span className="grid h-16 w-16 shrink-0 place-items-center rounded-2xl bg-ink text-bg">
                  <Github size={32} />
                </span>
                <div>
                  <p className="font-display text-2xl font-bold">{PROJECT.short}</p>
                  <p className="mt-1 max-w-lg text-sm text-muted">
                    WL graph-kernel experiments, feature extraction pipeline, nested cross-validation scripts, and this React interface.
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {['Python', 'scikit-learn', 'grakel', 'React', 'Vite', 'Tailwind'].map((t) => (
                      <Badge key={t}>{t}</Badge>
                    ))}
                  </div>
                </div>
              </div>
              {isTodo(LINKS.github) ? (
                <div className="text-sm">
                  <button type="button" disabled className="btn-primary px-6 py-3">
                    <Github size={18} /> View on GitHub
                  </button>
                  {/* <p className="mt-2 text-xs text-warn">Set LINKS.github in src/lib/content.js</p> */}
                </div>
              ) : (
                <a href={LINKS.github} target="_blank" rel="noreferrer" className="btn-primary px-6 py-3">
                  <Github size={18} /> View on GitHub <ExternalLink size={14} />
                </a>
              )}
            </div>
          </Card>
        </Reveal>
      </Section>

      {/* Video
      <Section id="video" eyebrow="Demonstration" title="Watch WL-LeafKernel in action">
        <Reveal>
          <VideoSection />
        </Reveal>
        <Reveal className="mt-10 text-center">
          <button type="button" onClick={() => scrollTo('results')} className="btn-primary px-6 py-3 text-base">
            <FlaskConical size={18} /> Explore the results
          </button>
        </Reveal>
      </Section> */}
    </>
  )
}
