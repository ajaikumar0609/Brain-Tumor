import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev server on 5173; /api is proxied to the FastAPI backend on 8000 (override with VITE_API_PROXY).
const apiTarget = process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  server: { port: 5173, strictPort: true, proxy: { '/api': { target: apiTarget, changeOrigin: true } } },
  preview: { port: 5173, strictPort: true, proxy: { '/api': { target: apiTarget, changeOrigin: true } } },
})
