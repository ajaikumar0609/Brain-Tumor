import { Link } from 'react-router-dom'
import { CheckCircle2, ImageIcon, Microscope, Sparkles, BarChart3, XCircle } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Caveats, ErrorBox, Loading, PageHeader, pct } from '@/components/Common'
import { ClfBar, GanBar } from '@/components/Charts'
import { api } from '@/lib/api'
import { useAsync } from '@/lib/useAsync'

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <Card><CardContent className="pt-6">
      <p className="text-muted-foreground text-xs font-medium uppercase tracking-wide">{label}</p>
      <p className="text-primary mt-1 text-3xl font-bold">{value}</p>
      <p className="text-muted-foreground mt-1 text-xs">{sub}</p>
    </CardContent></Card>
  )
}

function Ready({ ok, name, detail }: { ok: boolean; name: string; detail: string }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border p-3">
      {ok ? <CheckCircle2 className="text-accent mt-0.5 h-5 w-5 shrink-0" aria-label="available" /> : <XCircle className="text-destructive mt-0.5 h-5 w-5 shrink-0" aria-label="unavailable" />}
      <div><p className="text-sm font-medium">{name}</p><p className="text-muted-foreground text-xs">{detail}</p></div>
    </div>
  )
}

export default function Dashboard() {
  const m = useAsync(api.metrics)
  const s = useAsync(api.status)
  const e = useAsync(api.experiment)
  const gan = m.data?.gan.filter((g) => !g.is_reference) ?? []
  const bestGan = gan.length ? gan.reduce((a, b) => (a.fid < b.fid ? a : b)) : undefined
  const bestClf = m.data?.classifiers.length ? m.data.classifiers.reduce((a, b) => (a.accuracy > b.accuracy ? a : b)) : undefined
  return (
    <>
      <PageHeader title="Research dashboard" subtitle="Brain tumour MRI generation and classification: DCGAN vs WGAN-GP vs the proposed conditional WGAN-GP (cWGAN-GP), evaluated on a de-duplicated, cluster-disjoint split. Every number below is read from the saved experiment files." />
      {m.loading && <Loading label="Loading saved results" />}
      {m.error && <ErrorBox error={m.error} />}
      {m.data && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Best FID / KID" value={bestGan ? bestGan.model : '–'} sub={bestGan ? `FID ${bestGan.fid.toFixed(1)} · KID×1000 ${bestGan.kid_x1000.toFixed(1)}` : ''} />
            <Stat label="Best mean test accuracy" value={bestClf ? `${bestClf.id}: ${pct(bestClf.accuracy)}` : '–'} sub={bestClf ? bestClf.name : ''} />
            <Stat label="Test set" value={`${m.data.n_test} images`} sub={`${e.data?.dataset.partition_counts.test?.notumor ?? 46} no-tumour; cluster-disjoint`} />
            <Stat label="Synthetic benefit" value="Not shown" sub="B (real + synthetic) did not beat matched control C" />
          </div>
          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <Card><CardHeader><CardTitle>GAN quality (FID)</CardTitle><CardDescription>3 generation seeds vs the 492 test images; bars = mean, whiskers = sd across seeds.</CardDescription></CardHeader>
              <CardContent><GanBar rows={m.data.gan} metric="fid" /></CardContent></Card>
            <Card><CardHeader><CardTitle>Downstream classifier (A / B / C)</CardTitle><CardDescription>Same small CNN: real only (A), real + synthetic (B), real resampled to B's size (C).</CardDescription></CardHeader>
              <CardContent><ClfBar rows={m.data.classifiers} /></CardContent></Card>
          </div>
        </>
      )}
      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card><CardHeader><CardTitle>Model readiness</CardTitle><CardDescription>Which features have a real, loadable model behind them.</CardDescription></CardHeader>
          <CardContent className="space-y-2">
            {s.loading && <Loading />}
            {s.error && <ErrorBox error={s.error} />}
            {s.data && (<>
              <Ready ok={s.data.classifier.available} name="MRI classification" detail={s.data.classifier.available ? `${s.data.classifier.name} (${s.data.classifier.license}) on ${s.data.device}` : (s.data.classifier.error ?? 'unavailable')} />
              <Ready ok={s.data.generator.available} name="Synthetic MRI generation" detail={s.data.generator.available ? `cWGAN-GP, ${s.data.generator.generator_updates} updates, ${s.data.generator.image_size}×${s.data.generator.image_size}, ${s.data.device}` : (s.data.generator.error ?? 'unavailable')} />
              <Ready ok={!!m.data} name="Saved experiment results" detail="outputs/deduplicated_experiment/" />
              <Ready ok={s.data.project_classifiers_abc.available} name="Project's own A/B/C classifiers" detail={s.data.project_classifiers_abc.reason} />
            </>)}
          </CardContent></Card>
        <Card><CardHeader><CardTitle>Explore</CardTitle></CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2">
            {[{ to: '/classify', icon: Microscope, t: 'Classify an MRI', d: 'Upload an image and see a prediction.' },
              { to: '/generate', icon: Sparkles, t: 'Generate samples', d: 'Conditional GAN images by class.' },
              { to: '/gallery', icon: ImageIcon, t: 'Gallery', d: 'Saved real and synthetic samples.' },
              { to: '/results', icon: BarChart3, t: 'Experiment results', d: 'Full metrics and limitations.' }].map(({ to, icon: I, t, d }) => (
              <Link key={to} to={to} className="hover:border-accent hover:bg-secondary rounded-lg border p-3 transition-colors">
                <I className="text-accent mb-1 h-5 w-5" aria-hidden /><p className="text-sm font-semibold">{t}</p><p className="text-muted-foreground text-xs">{d}</p>
              </Link>))}
          </CardContent></Card>
      </div>
      {e.data && <div className="mt-6"><Caveats title="Key findings and limits" items={[...e.data.key_findings, ...e.data.limitations.slice(0, 4)]} /></div>}
      <p className="text-muted-foreground mt-4 text-xs"><Badge variant="secondary">sources</Badge> {m.data?.sources.join(' · ')}</p>
    </>
  )
}
