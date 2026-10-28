import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from '@/components/Layout'
import Dashboard from '@/pages/Dashboard'
import Classify from '@/pages/Classify'
import Generate from '@/pages/Generate'
import Gallery from '@/pages/Gallery'
import Results from '@/pages/Results'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="classify" element={<Classify />} />
          <Route path="generate" element={<Generate />} />
          <Route path="gallery" element={<Gallery />} />
          <Route path="results" element={<Results />} />
          <Route path="*" element={<p className="py-10 text-center">Page not found.</p>} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
