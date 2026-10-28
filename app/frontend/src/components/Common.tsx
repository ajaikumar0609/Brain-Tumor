import type { ReactNode } from 'react'
import { AlertTriangle, Loader2 } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'

export function Disclaimer({ compact = false }: { compact?: boolean }) {
  return (
    <div role="note" className="border-warning-border/60 bg-warning flex gap-3 rounded-lg border p-3 text-sm text-[#5a3d00]">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <p>
        <strong>Academic research prototype, not a medical device.</strong>{' '}
        {compact ? 'Outputs are not a diagnosis.' : 'Nothing shown here is a medical diagnosis or has been clinically validated.'}
      </p>
    </div>
  )
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="text-muted-foreground flex items-center gap-2 py-6 text-sm" role="status">
      <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> {label}…
    </div>
  )
}

export function ErrorBox({ error }: { error: Error }) {
  const detail = (error as { detail?: string }).detail
  return (
    <Alert variant="destructive">
      <AlertTitle>Something went wrong</AlertTitle>
      <AlertDescription>{error.message}{detail ? ` (${detail})` : ''}</AlertDescription>
    </Alert>
  )
}

export function Caveats({ items, title = 'Read before interpreting' }: { items: string[]; title?: string }) {
  return (
    <div className="bg-muted rounded-lg border p-4 text-sm">
      <p className="mb-2 font-semibold">{title}</p>
      <ul className="text-muted-foreground list-disc space-y-1.5 pl-5">{items.map((c) => <li key={c}>{c}</li>)}</ul>
    </div>
  )
}

export function PageHeader({ title, subtitle, children }: { title: string; subtitle?: string; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-primary text-2xl font-bold tracking-tight">{title}</h1>
        {subtitle && <p className="text-muted-foreground mt-1 max-w-3xl text-sm">{subtitle}</p>}
      </div>
      {children}
    </div>
  )
}

export const pct = (x: number, d = 1) => `${(x * 100).toFixed(d)}%`
