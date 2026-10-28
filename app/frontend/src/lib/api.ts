// Typed client for the FastAPI backend. In dev, /api is proxied to http://127.0.0.1:8000 (see vite.config.ts).
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? ''

export class ApiError extends Error {
  status: number
  code: string
  detail?: string
  constructor(status: number, code: string, message: string, detail?: string) {
    super(message)
    this.status = status
    this.code = code
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, init)
  } catch {
    throw new ApiError(0, 'network', 'Cannot reach the backend. Is it running on port 8000?')
  }
  const text = await res.text()
  let body: unknown = null
  try { body = text ? JSON.parse(text) : null } catch { /* non-JSON error body */ }
  if (!res.ok) {
    const e = (body as { error?: { code?: string; message?: string; detail?: string } } | null)?.error
    throw new ApiError(res.status, e?.code ?? 'http_error', e?.message ?? `Request failed (${res.status})`, e?.detail)
  }
  return body as T
}

export type ClassKey = 'glioma' | 'meningioma' | 'notumor' | 'pituitary'
export const CLASS_LABELS: Record<ClassKey, string> = { glioma: 'Glioma', meningioma: 'Meningioma', notumor: 'No tumour', pituitary: 'Pituitary tumour' }
export const CLASS_KEYS = Object.keys(CLASS_LABELS) as ClassKey[]

export interface ModelStatus {
  device: string
  disclaimer: string
  generator: { name: string; available: boolean; error: string | null; checkpoint?: string; generator_updates?: number; trained_on_images?: number; image_size?: number }
  classifier: { name: string; available: boolean; error: string | null; source: string; license: string; weights_sha256_verified: boolean; input: string }
  project_classifiers_abc: { available: boolean; reason: string }
}
export interface GanRow { model: string; is_reference: boolean; fid: number; fid_std: number; kid_x1000: number; kid_std_x1000: number; n_generation_seeds: number; n_generated_per_seed: number; n_real: number }
export interface ClfRow {
  id: string; name: string; n_train: number; n_runs: number; accuracy: number; accuracy_std: number; accuracy_values: number[]
  macro_f1: number; macro_f1_std: number; balanced_accuracy: number; val_accuracy: number
  per_class_f1: Record<ClassKey, number>; per_class_recall: Record<ClassKey, number>; mean_confusion_matrix: number[][]
}
export interface Metrics { n_test: number; classes: ClassKey[]; gan: GanRow[]; classifiers: ClfRow[]; sources: string[]; notes: string }
export interface Experiment {
  title: string; status: string
  dataset: { description: string; provenance: string; split: string; partition_counts: Record<string, Record<string, number>> }
  models: Record<string, string>; key_findings: string[]; matched_control: string; limitations: string[]; source_files: string[]
}
export interface ClassifyResult {
  predicted_class: ClassKey; predicted_label: string; confidence: number
  probabilities: { class: ClassKey; label: string; probability: number }[]
  model: string; device: string; input_size: [number, number]; inference_ms: number; disclaimer: string; caveats: string[]
}
export interface GenerateResult {
  class: ClassKey; label: string; count: number; seed: number; image_size: number; images: string[]
  model: string; checkpoint: string; generator_updates: number; device: string; generation_ms: number; disclaimer: string; caveats: string[]
}
export interface GalleryItem { id: string; title: string; kind: 'real' | 'synthetic' | 'mixed'; note: string; url: string }

export const api = {
  health: () => request<{ status: string }>('/api/health'),
  status: () => request<ModelStatus>('/api/models/status'),
  metrics: () => request<Metrics>('/api/metrics'),
  experiment: () => request<Experiment>('/api/experiment'),
  gallery: () => request<{ items: GalleryItem[]; note: string }>('/api/gallery'),
  classify: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return request<ClassifyResult>('/api/classify', { method: 'POST', body: fd })
  },
  generate: (cls: ClassKey, count: number, seed?: number) =>
    request<GenerateResult>('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ class: cls, count, ...(seed !== undefined ? { seed } : {}) }),
    }),
}
