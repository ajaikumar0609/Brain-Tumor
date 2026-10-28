import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ClfRow, GanRow } from '@/lib/api'

const NAVY = '#1f3a5f', TEAL = '#2a9d8f', GREY = '#9aa7b4', AMBER = '#c98a1a'
const SHORT: Record<string, string> = { A: 'Real only', B: 'Real + synthetic', C: 'Real, matched size' }
const GAN_COLORS: Record<string, string> = { 'Real-vs-real floor': GREY, DCGAN: NAVY, 'WGAN-GP': AMBER, 'cWGAN-GP': TEAL }

export function GanBar({ rows, metric }: { rows: GanRow[]; metric: 'fid' | 'kid' }) {
  const data = rows.map((r) => ({
    name: r.model,
    value: metric === 'fid' ? r.fid : r.kid_x1000,
    err: metric === 'fid' ? r.fid_std : r.kid_std_x1000,
  }))
  const label = metric === 'fid' ? 'FID (lower is better)' : 'KID × 1000 (lower is better)'
  return (
    <figure aria-label={label}>
      <figcaption className="mb-2 text-sm font-medium">{label}</figcaption>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 16, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(1) : String(v))} />
          <Bar dataKey="value" name={label} radius={[4, 4, 0, 0]}>
            {data.map((d) => <Cell key={d.name} fill={GAN_COLORS[d.name] ?? NAVY} />)}
            <ErrorBar dataKey="err" width={6} stroke="#14213d" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </figure>
  )
}

export function ClfBar({ rows }: { rows: ClfRow[] }) {
  const data = rows.map((r) => ({ name: `${r.id}: ${SHORT[r.id] ?? r.name}`, accuracy: r.accuracy, accErr: r.accuracy_std, macroF1: r.macro_f1, f1Err: r.macro_f1_std }))
  return (
    <figure aria-label="Test accuracy and macro-F1 for conditions A, B and C">
      <figcaption className="mb-2 text-sm font-medium">Test accuracy and macro-F1 (mean ± sd over 3 seeds)</figcaption>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 16, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis domain={[0.6, 1]} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))} />
          <Legend />
          <Bar dataKey="accuracy" name="Test accuracy" fill={NAVY} radius={[4, 4, 0, 0]}><ErrorBar dataKey="accErr" width={6} stroke="#14213d" /></Bar>
          <Bar dataKey="macroF1" name="Macro-F1" fill={TEAL} radius={[4, 4, 0, 0]}><ErrorBar dataKey="f1Err" width={6} stroke="#14213d" /></Bar>
        </BarChart>
      </ResponsiveContainer>
    </figure>
  )
}
