import { useState } from 'react'
import { Download, Loader2, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Caveats, ErrorBox, PageHeader } from '@/components/Common'
import { api, CLASS_KEYS, CLASS_LABELS, type ClassKey, type GenerateResult } from '@/lib/api'
import { useAsync } from '@/lib/useAsync'

const COUNTS = [4, 8, 12, 16]

function downloadUrl(url: string, name: string) {
  const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove()
}

async function downloadGrid(r: GenerateResult) {
  const cols = Math.min(4, r.images.length), rows = Math.ceil(r.images.length / cols), s = r.image_size * 2
  const c = document.createElement('canvas'); c.width = cols * s; c.height = rows * s
  const ctx = c.getContext('2d'); if (!ctx) return
  ctx.imageSmoothingEnabled = false
  await Promise.all(r.images.map((src, i) => new Promise<void>((res) => {
    const im = new Image(); im.onload = () => { ctx.drawImage(im, (i % cols) * s, Math.floor(i / cols) * s, s, s); res() }; im.src = src
  })))
  downloadUrl(c.toDataURL('image/png'), `synthetic_${r.class}_seed${r.seed}_grid.png`)
}

export default function Generate() {
  const status = useAsync(api.status)
  const [cls, setCls] = useState<ClassKey>('glioma')
  const [count, setCount] = useState(8)
  const [seed, setSeed] = useState('')
  const [busy, setBusy] = useState(false)
  const [res, setRes] = useState<GenerateResult | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const unavailable = status.data && !status.data.generator.available
  const seedNum = seed.trim() === '' ? undefined : Number(seed)
  const seedBad = seedNum !== undefined && (!Number.isInteger(seedNum) || seedNum < 0 || seedNum > 2147483647)

  const run = async () => {
    setBusy(true); setError(null)
    try { setRes(await api.generate(cls, count, seedNum)) } catch (e) { setError(e as Error) } finally { setBusy(false) }
  }

  return (
    <>
      <PageHeader title="Synthetic MRI generator" subtitle="Sample images from the saved conditional WGAN-GP (64×64, grayscale). The class is a requested label, not a verified diagnosis." />
      {unavailable && <div className="mb-4"><ErrorBox error={new Error(`Generator unavailable: ${status.data?.generator.error ?? 'unknown reason'}`)} /></div>}
      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        <Card>
          <CardHeader><CardTitle>Settings</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <fieldset><legend className="mb-1 text-sm font-medium">Requested class</legend>
              <div className="grid grid-cols-2 gap-2">
                {CLASS_KEYS.map((k) => (
                  <Button key={k} variant={cls === k ? 'default' : 'outline'} size="sm" aria-pressed={cls === k} onClick={() => setCls(k)}>{CLASS_LABELS[k]}</Button>))}
              </div></fieldset>
            <div><label htmlFor="count" className="mb-1 block text-sm font-medium">Number of images</label>
              <select id="count" value={count} onChange={(e) => setCount(Number(e.target.value))} className="bg-card w-full rounded-md border px-3 py-2 text-sm">
                {COUNTS.map((c) => <option key={c} value={c}>{c}</option>)}</select>
              <p className="text-muted-foreground mt-1 text-xs">Limit: 16 per request.</p></div>
            <div><label htmlFor="seed" className="mb-1 block text-sm font-medium">Seed (optional)</label>
              <input id="seed" inputMode="numeric" placeholder="random" value={seed} onChange={(e) => setSeed(e.target.value)} aria-invalid={seedBad}
                className="bg-card w-full rounded-md border px-3 py-2 text-sm" />
              <p className={`mt-1 text-xs ${seedBad ? 'text-destructive' : 'text-muted-foreground'}`}>{seedBad ? 'Seed must be an integer from 0 to 2147483647.' : 'The same seed and class reproduce the same images.'}</p></div>
            <Button className="w-full" onClick={run} disabled={busy || !!unavailable || seedBad}>
              {busy ? <><Loader2 className="animate-spin" aria-hidden /> Generating…</> : <><Sparkles aria-hidden /> Generate</>}</Button>
            {error && <ErrorBox error={error} />}
          </CardContent>
        </Card>
        <Card aria-live="polite">
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div><CardTitle>Generated images</CardTitle>
                <CardDescription>{res ? `${res.count} × ${res.label} · seed ${res.seed} · ${res.device} · ${res.generation_ms} ms` : 'Nothing generated yet.'}</CardDescription></div>
              {res && <Button variant="outline" size="sm" onClick={() => downloadGrid(res)}><Download aria-hidden /> Download grid (PNG)</Button>}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {busy && !res && <p className="text-muted-foreground flex items-center gap-2 text-sm"><Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Running the generator…</p>}
            {res && (<>
              <div className="relative">
                <span className="bg-warning text-[#5a3d00] border-warning-border/60 absolute left-2 top-2 z-[1] rounded border px-2 py-0.5 text-[11px] font-semibold">SYNTHETIC</span>
                <ul className="grid grid-cols-2 gap-3 pt-8 sm:grid-cols-4" aria-label="Generated images">
                  {res.images.map((src, i) => (
                    <li key={i} className="group relative overflow-hidden rounded-lg border bg-black">
                      <img src={src} alt={`Synthetic ${res.label} sample ${i + 1}`} className="pixelated aspect-square w-full" />
                      <button type="button" aria-label={`Download image ${i + 1}`} onClick={() => downloadUrl(src, `synthetic_${res.class}_seed${res.seed}_${i + 1}.png`)}
                        className="absolute right-1 bottom-1 rounded bg-white/90 p-1 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"><Download className="h-4 w-4" /></button>
                    </li>))}
                </ul>
              </div>
              <Caveats items={res.caveats} />
            </>)}
          </CardContent>
        </Card>
      </div>
    </>
  )
}
