import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Caveats, ErrorBox, Loading, PageHeader, pct } from '@/components/Common'
import { ClfBar, GanBar } from '@/components/Charts'
import { api, CLASS_KEYS, CLASS_LABELS } from '@/lib/api'
import { useAsync } from '@/lib/useAsync'

const th = 'px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide'
const td = 'px-3 py-2 text-sm'

export default function Results() {
  const m = useAsync(api.metrics)
  const e = useAsync(api.experiment)
  return (
    <>
      <PageHeader title="Experiment results" subtitle="Final de-duplicated experiment: cluster-disjoint 70/15/15 split, one training run per GAN, 3 classifier seeds. Read from the saved result files." />
      {(m.loading || e.loading) && <Loading />}
      {m.error && <ErrorBox error={m.error} />}
      {m.data && (<div className="space-y-6">
        <Card><CardHeader><CardTitle>GAN comparison: FID and KID</CardTitle><CardDescription>Generated images vs the {m.data.n_test}-image test partition; 3 generation seeds. The real-vs-real row is the floor a perfect generator would approach. DCGAN scored best; the proposed cWGAN-GP was second.</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2"><GanBar rows={m.data.gan} metric="fid" /><GanBar rows={m.data.gan} metric="kid" /></div>
            <div className="overflow-x-auto"><table className="w-full">
              <thead className="bg-primary text-primary-foreground"><tr><th className={th}>Model</th><th className={th}>FID</th><th className={th}>KID ×1000</th></tr></thead>
              <tbody>{m.data.gan.map((r, i) => (<tr key={r.model} className={i % 2 ? 'bg-card' : 'bg-muted'}>
                <td className={td}>{r.model}{r.is_reference ? ' (reference)' : ''}</td><td className={td}>{r.fid.toFixed(1)} ± {r.fid_std.toFixed(1)}</td><td className={td}>{r.kid_x1000.toFixed(1)} ± {r.kid_std_x1000.toFixed(1)}</td></tr>))}</tbody></table></div>
          </CardContent></Card>

        <Card><CardHeader><CardTitle>Downstream classification: A, B, C</CardTitle>
          <CardDescription>{e.data?.matched_control ?? 'C trains on real images only, matched to B in size and class mix.'}</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <ClfBar rows={m.data.classifiers} />
            <div className="overflow-x-auto"><table className="w-full">
              <thead className="bg-primary text-primary-foreground"><tr><th className={th}>Condition</th><th className={th}>Train n</th><th className={th}>Test accuracy</th><th className={th}>Macro-F1</th><th className={th}>Val. accuracy</th></tr></thead>
              <tbody>{m.data.classifiers.map((r, i) => (<tr key={r.id} className={i % 2 ? 'bg-card' : 'bg-muted'}>
                <td className={td}>{r.id}: {r.name}</td><td className={td}>{r.n_train}</td><td className={td}>{r.accuracy.toFixed(3)} ± {r.accuracy_std.toFixed(3)}</td><td className={td}>{r.macro_f1.toFixed(3)} ± {r.macro_f1_std.toFixed(3)}</td><td className={td}>{r.val_accuracy.toFixed(3)}</td></tr>))}</tbody></table></div>
            <p className="text-sm">The matched-size real-only control (C) had the highest mean accuracy. Adding the hybrid's synthetic images (B) did not beat simply resampling real images to the same size, so no benefit of the synthetic data is demonstrated. Validation accuracy was used to select the epoch only.</p>
          </CardContent></Card>

        <Card><CardHeader><CardTitle>Class-wise F1 (mean over seeds)</CardTitle><CardDescription>No-tumour has only {e.data?.dataset.partition_counts.test?.notumor ?? 46} test images, so its estimates are very uncertain.</CardDescription></CardHeader>
          <CardContent className="overflow-x-auto"><table className="w-full">
            <thead className="bg-primary text-primary-foreground"><tr><th className={th}>Class</th>{m.data.classifiers.map((c) => <th key={c.id} className={th}>{c.id}: F1 / recall</th>)}</tr></thead>
            <tbody>{CLASS_KEYS.map((k, i) => (<tr key={k} className={i % 2 ? 'bg-card' : 'bg-muted'}><td className={td}>{CLASS_LABELS[k]}</td>
              {m.data!.classifiers.map((c) => <td key={c.id} className={td}>{c.per_class_f1[k].toFixed(2)} / {pct(c.per_class_recall[k], 0)}</td>)}</tr>))}</tbody></table></CardContent></Card>
      </div>)}

      {e.error && <ErrorBox error={e.error} />}
      {e.data && (<div className="mt-6 space-y-6">
        <Card><CardHeader><CardTitle>Dataset and methodology</CardTitle></CardHeader><CardContent className="space-y-2 text-sm">
          <p>{e.data.dataset.description}</p><p>{e.data.dataset.split}</p><p><strong>Provenance:</strong> {e.data.dataset.provenance}</p>
          <ul className="list-disc space-y-1 pl-5">{Object.entries(e.data.models).map(([k, v]) => <li key={k}><strong>{k}:</strong> {v}</li>)}</ul></CardContent></Card>
        <Caveats title="Limitations and caveats" items={e.data.limitations} />
      </div>)}
    </>
  )
}
