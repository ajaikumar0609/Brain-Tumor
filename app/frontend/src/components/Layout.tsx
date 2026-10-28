import { NavLink, Outlet } from 'react-router-dom'
import { Brain } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Disclaimer } from '@/components/Common'

const links = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/classify', label: 'Classify MRI' },
  { to: '/generate', label: 'Generate' },
  { to: '/gallery', label: 'Gallery' },
  { to: '/results', label: 'Experiment results' },
]

export default function Layout() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="bg-primary text-primary-foreground sticky top-0 z-10 shadow">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-8 gap-y-2 px-4 py-3">
          <div className="flex items-center gap-2 font-semibold">
            <Brain className="text-accent h-6 w-6" aria-hidden />
            <span className="text-lg tracking-tight">NeuroVision AI</span>
            <span className="rounded bg-white/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider">research prototype</span>
          </div>
          <nav aria-label="Main" className="flex flex-wrap gap-1 text-sm">
            {links.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end}
                className={({ isActive }) => cn('rounded-md px-3 py-1.5 transition-colors hover:bg-white/10', isActive && 'bg-white/20 font-medium')}>
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        <div className="mb-6"><Disclaimer /></div>
        <Outlet />
      </main>
      <footer className="text-muted-foreground border-t py-4 text-center text-xs">
        Brain Tumor MRI IA3 project · academic prototype · synthetic images are not real scans · not for clinical use
      </footer>
    </div>
  )
}
