import { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, Trash2, Upload } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Caveats, ErrorBox, PageHeader, pct } from '@/components/Common'
import { api, type ClassifyResult } from '@/lib/api'
import { useAsync } from '@/lib/useAsync'

const OK_TYPES = ['image/png', 'image/jpeg']
const MAX_MB = 10

export default function Classify() {
  const status = useAsync(api.status)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<ClassifyResult | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [drag, setDrag] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  const pick = useCallback((f: File | undefined) => {
    setResult(null); setError(null)
    if (!f) return
    if (!OK_TYPES.includes(f.type)) { setError(new Error('Please choose a PNG or JPEG image.')); return }
    if (f.size > MAX_MB * 1024 * 1024) { setError(new Error(`Image is larger than ${MAX_MB} MB.`)); return }
    setFile(f); setPreview(URL.createObjectURL(f))
  }, [])

  const clear = () => { setFile(null); setPreview(null); setResult(null); setError(null); if (input.current) input.current.value = '' }
  const run = async () => {
    if (!file) return
    setBusy(true); setError(null); setResult(null)
    try { setResult(await api.classify(file)) } catch (e) { setError(e as Error) } finally { setBusy(false) }
  }
  const unavailable = status.data && !status.data.classifier.available

  return (
    <>
      <PageHeader title="MRI classification" subtitle="Upload a brain MRI slice (PNG or JPEG) to see which of the four classes a pretrained model assigns. This is a demonstration, not a diagnosis." />
      {unavailable && <div className="mb-4"><ErrorBox error={new Error(`Classifier unavailable: ${status.data?.classifier.error ?? 'unknown reason'}`)} /></div>}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>1. Choose an image</CardTitle><CardDescription>PNG or JPEG, up to {MAX_MB} MB. It is sent to the local backend only.</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <div role="button" tabIndex={0} aria-label="Upload MRI image: drop a file or press Enter to browse"
              onClick={() => input.current?.click()} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
              onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]) }}
              className={`flex min-h-56 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-4 text-center transition-colors ${drag ? 'border-accent bg-secondary' : 'hover:bg-muted'}`}>
              {preview ? <img src={preview} alt="Uploaded MRI preview" className="max-h-64 rounded object-contain" /> : (<>
                <Upload className="text-muted-foreground h-8 w-8" aria-hidden /><p className="text-sm font-medium">Drag and drop an MRI image here</p><p className="text-muted-foreground text-xs">or click to browse</p></>)}
              <input ref={input} type="file" accept="image/png,image/jpeg" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
            </div>
            {file && <p className="text-muted-foreground truncate text-xs">{file.name} · {(file.size / 1024).toFixed(0)} KB</p>}
            <div className="flex gap-2">
              <Button onClick={run} disabled={!file || busy || !!unavailable}>{busy ? <><Loader2 className="animate-spin" aria-hidden /> Classifying…</> : 'Classify'}</Button>
              <Button variant="outline" onClick={clear} disabled={!file || busy}><Trash2 aria-hidden /> Clear</Button>
            </div>
            {error && <ErrorBox error={error} />}
          </CardContent>
        </Card>
        <Card aria-live="polite">
          <CardHeader><CardTitle>2. Model output</CardTitle><CardDescription>Shown only when the real model returns a result.</CardDescription></CardHeader>
          <CardContent>
            {!result && !busy && <p className="text-muted-foreground text-sm">No result yet.</p>}
            {busy && <p className="text-muted-foreground flex items-center gap-2 text-sm"><Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Running the model…</p>}
            {result && (
              <div className="space-y-4">
                <div className="bg-secondary rounded-lg p-4">
                  <p className="text-muted-foreground text-xs uppercase tracking-wide">Predicted class (not a diagnosis)</p>
                  <p className="text-primary text-2xl font-bold">{result.predicted_label}</p>
                  <p className="text-sm">Softmax confidence {pct(result.confidence)}</p>
                </div>
                <ul className="space-y-2" aria-label="Class probabilities">
                  {result.probabilities.map((p) => (
                    <li key={p.class} className="text-sm">
                      <div className="mb-0.5 flex justify-between"><span>{p.label}</span><span>{pct(p.probability)}</span></div>
                      <div className="bg-muted h-2 rounded"><div className={`h-2 rounded ${p.class === result.predicted_class ? 'bg-accent' : 'bg-primary/40'}`} style={{ width: `${p.probability * 100}%` }} /></div>
                    </li>))}
                </ul>
                <p className="text-muted-foreground text-xs">{result.model} · {result.device} · {result.inference_ms} ms · input {result.input_size[0]}×{result.input_size[1]}</p>
                <Caveats items={result.caveats} />
              </div>)}
          </CardContent>
        </Card>
      </div>
    </>
  )
}
