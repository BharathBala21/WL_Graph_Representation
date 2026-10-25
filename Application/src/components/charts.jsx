import { useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar,
  LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { TOMATO_RESULTS } from '../lib/content'
import { useChartColors } from '../lib/hooks'
import { cx } from '../lib/utils'
import { Tabs } from './ui'

function ChartTooltip({ active, payload, label, unit = '', digits = 2 }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg bg-elev px-3 py-2 text-xs shadow-soft ring-1 ring-line/20">
      <p className="mb-1 font-semibold text-ink">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} className="flex items-center gap-2 text-muted">
          <span className="h-2 w-2 rounded-sm" style={{ background: p.color || p.payload?.fill }} />
          {p.name}: <b className="tabular-nums text-ink">{Number(p.value).toFixed(digits)}{unit}</b>
        </p>
      ))}
    </div>
  )
}

const axisProps = (c) => ({ stroke: c.axis, tick: { fill: c.axis, fontSize: 11 }, tickLine: false, axisLine: { stroke: c.grid } })

/**
 * Tomato binary stress results — accuracy / balanced accuracy / Macro-F1 per model.
 * Displayed as grouped bars with ± std displayed in the tooltip.
 */
export function TomatoResultsChart() {
  const c = useChartColors()
  const [metric, setMetric] = useState('acc')
  const METRICS = [
    { key: 'acc',  label: 'Accuracy (%)',         sdKey: 'acc_sd'  },
    { key: 'bacc', label: 'Balanced Accuracy (%)', sdKey: 'bacc_sd' },
    { key: 'f1',   label: 'Macro-F1 (%)',          sdKey: 'f1_sd'   },
  ]
  const m = METRICS.find((x) => x.key === metric)
  const data = TOMATO_RESULTS.map((r) => ({
    model: r.model,
    value: r[metric],
    sd:    r[m.sdKey],
    ours:  r.ours,
  }))

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-muted">Tomato binary stress · nested 5-fold CV · mean ± std</p>
        <Tabs size="sm" value={metric} onChange={setMetric} tabs={METRICS.map((x) => ({ value: x.key, label: x.label.replace(' (%)', '') }))} />
      </div>
      <div className="h-72" role="img" aria-label={`${m.label} for WL, RF and RBF SVM on tomato binary stress task`}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 56, top: 4, bottom: 4 }} barCategoryGap={10}>
            <CartesianGrid horizontal={false} stroke={c.grid} />
            <XAxis type="number" {...axisProps(c)} domain={[55, 92]} unit="%" />
            <YAxis type="category" dataKey="model" width={140} {...axisProps(c)} />
            <Tooltip
              cursor={{ fill: c.grid, opacity: 0.4 }}
              content={({ active, payload, label }) => {
                if (!active || !payload?.length) return null
                const r = payload[0]?.payload
                return (
                  <div className="rounded-lg bg-elev px-3 py-2 text-xs shadow-soft ring-1 ring-line/20">
                    <p className="mb-1 font-semibold text-ink">{label}</p>
                    <p className="text-muted">
                      {m.label}: <b className="tabular-nums text-ink">{r.value?.toFixed(2)}%</b>
                    </p>
                    {r.sd != null && (
                      <p className="text-muted">± std: <b className="tabular-nums text-ink">{r.sd?.toFixed(2)}%</b></p>
                    )}
                    {r.ours && <p className="mt-1 text-primary font-semibold">WL kernel (our method)</p>}
                  </div>
                )
              }}
            />
            <Bar dataKey="value" name={m.label} radius={[0, 4, 4, 0]} isAnimationActive>
              {data.map((r) => (
                <Cell key={r.model} fill={r.ours ? c.series[0] : c.muted} />
              ))}
              <LabelList dataKey="value" position="right" fill={c.axis} fontSize={11} formatter={(v) => `${Number(v).toFixed(2)}%`} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-xs text-muted">
        Source: verified nested-CV audit (5 outer folds, 5 inner folds for configuration selection). Tomato binary task only — 580 images, 184 healthy, 396 stressed.
      </p>
    </div>
  )
}

/** Healthy-class vs stressed-class recall comparison for models with verified recall data. */
export function RecallComparisonChart() {
  const c = useChartColors()
  // Only WL and RF have healthy-class recall in the verified source
  const data = [
    { model: 'Nested WL Kernel', healthy: 65.32, stressed: null, ours: true  },
    { model: 'Random Forest',    healthy: 57.67, stressed: null, ours: false },
  ]
  return (
    <div>
      <p className="mb-1 text-xs text-muted">
        Healthy-class recall — the harder class (minority). Stressed-class recall for RBF SVM not in verified source.
      </p>
      <div className="h-56" role="img" aria-label="Healthy-class recall comparison for WL kernel and Random Forest">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 56, top: 4, bottom: 4 }} barCategoryGap={10}>
            <CartesianGrid horizontal={false} stroke={c.grid} />
            <XAxis type="number" {...axisProps(c)} domain={[40, 80]} unit="%" />
            <YAxis type="category" dataKey="model" width={150} {...axisProps(c)} />
            <Tooltip cursor={{ fill: c.grid, opacity: 0.4 }} content={<ChartTooltip unit="%" />} />
            <Bar dataKey="healthy" name="Healthy-class recall" radius={[0, 4, 4, 0]}>
              {data.map((r) => (
                <Cell key={r.model} fill={r.ours ? c.series[0] : c.series[1]} />
              ))}
              <LabelList dataKey="healthy" position="right" fill={c.axis} fontSize={11} formatter={(v) => `${Number(v).toFixed(2)}%`} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-1 text-xs text-muted">
        WL Kernel healthy recall (65.32%) exceeds Random Forest (57.67%), despite lower overall accuracy — highlighting the importance of class-wise evaluation.
      </p>
    </div>
  )
}

/** Summary metric table for all three tomato models. */
export function TomatoMetricsTable() {
  return (
    <div className="overflow-x-auto rounded-xl ring-1 ring-line/15">
      <table className="w-full min-w-[580px] text-sm">
        <thead className="bg-ink/[0.04] text-left text-[11px] uppercase tracking-wider text-muted">
          <tr>
            {['Model', 'Accuracy', 'Balanced Acc.', 'Macro-F1', 'Healthy Recall'].map((h) => (
              <th key={h} className="px-3 py-2 font-semibold">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line/10 tabular-nums">
          {TOMATO_RESULTS.map((r) => (
            <tr key={r.model} className={cx(r.ours && 'bg-primary/[0.06] font-semibold')}>
              <td className="px-3 py-2 flex items-center gap-2">
                {r.ours && <span className="h-1.5 w-1.5 rounded-full bg-primary shrink-0" />}
                {r.model}
              </td>
              <td className="px-3 py-2">{r.acc.toFixed(2)} ± {r.acc_sd.toFixed(2)}%</td>
              <td className="px-3 py-2">{r.bacc.toFixed(2)} ± {r.bacc_sd.toFixed(2)}%</td>
              <td className="px-3 py-2">{r.f1.toFixed(2)} ± {r.f1_sd.toFixed(2)}%</td>
              <td className="px-3 py-2">
                {r.recall_healthy != null ? `${r.recall_healthy.toFixed(2)}%` : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="px-3 py-2 text-[11px] text-muted">
        Mean ± std across 5 outer folds. Source: verified nested-CV audit. Tomato binary task only.
      </p>
    </div>
  )
}

/** Generic two-series grouped bar — re-exported from reference for compatibility. */
export function PairChart({ data, a, b, unit = '', digits = 3, domain, label }) {
  const c = useChartColors()
  return (
    <div className="h-64" role="img" aria-label={label}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ left: -8, right: 8, top: 16 }} barGap={2}>
          <CartesianGrid vertical={false} stroke={c.grid} />
          <XAxis dataKey="metric" {...axisProps(c)} tick={{ fill: c.axis, fontSize: 10 }} interval={0} />
          <YAxis domain={domain} {...axisProps(c)} />
          <Tooltip cursor={{ fill: c.grid, opacity: 0.4 }} content={<ChartTooltip unit={unit} digits={digits} />} />
          <Legend wrapperStyle={{ fontSize: 12, color: c.axis }} iconType="square" iconSize={10} />
          <Bar dataKey={a.key} name={a.name} fill={c.series[0]} radius={[4, 4, 0, 0]}>
            <LabelList dataKey={a.key} position="top" fill={c.axis} fontSize={10} formatter={(v) => Number(v).toFixed(unit === '%' ? 1 : 2)} />
          </Bar>
          <Bar dataKey={b.key} name={b.name} fill={c.series[1]} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
