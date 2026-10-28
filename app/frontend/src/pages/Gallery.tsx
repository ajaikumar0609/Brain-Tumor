import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ErrorBox, Loading, PageHeader } from '@/components/Common'
import { api, type GalleryItem } from '@/lib/api'
import { useAsync } from '@/lib/useAsync'

const KIND: Record<GalleryItem['kind'], { label: string; cls: string }> = {
  real: { label: 'REAL (dataset images)', cls: 'bg-secondary text-secondary-foreground border-accent' },
  synthetic: { label: 'SYNTHETIC (GAN output)', cls: 'bg-warning text-[#5a3d00] border-warning-border' },
  mixed: { label: 'MIXED (labelled panels)', cls: 'bg-muted text-foreground border-border' },
}

export default function Gallery() {
  const g = useAsync(api.gallery)
  return (
    <>
      <PageHeader title="Image gallery" subtitle="Saved reference and generated image sheets. Only images whose provenance is recorded in the project are listed." />
      {g.loading && <Loading />}
      {g.error && <ErrorBox error={g.error} />}
      {g.data && g.data.items.length === 0 && <p className="text-muted-foreground text-sm">No saved images were found.</p>}
      <div className="grid gap-4 md:grid-cols-2">
        {g.data?.items.map((it) => (
          <Card key={it.id} className={`border-2 ${KIND[it.kind].cls.split(' ').find((c) => c.startsWith('border-')) ?? ''}`}>
            <CardHeader>
              <Badge className={`w-fit border ${KIND[it.kind].cls}`} variant="outline">{KIND[it.kind].label}</Badge>
              <CardTitle className="text-base">{it.title}</CardTitle>
              <CardDescription>{it.note}</CardDescription>
            </CardHeader>
            <CardContent><a href={it.url} target="_blank" rel="noreferrer"><img src={it.url} alt={it.title} loading="lazy" className="w-full rounded border bg-white" /></a></CardContent>
          </Card>))}
      </div>
      {g.data && <p className="text-muted-foreground mt-4 text-xs">{g.data.note}</p>}
    </>
  )
}
