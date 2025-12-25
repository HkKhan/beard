import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ScanView } from './components/ScanView'
import { ProjectView } from './components/ProjectView'
import { useBeardStore } from './store/beardStore'

type AppView = 'home' | 'scan' | 'project'

function App() {
  const [view, setView] = useState<AppView>('home')
  const { savedTemplates } = useBeardStore()
  const hasTemplates = savedTemplates.length > 0

  return (
    <div className="h-full w-full flex flex-col" style={{ background: 'var(--bg-primary)' }}>
      <AnimatePresence mode="wait">
        {view === 'home' && (
          <HomeView 
            key="home"
            onScan={() => setView('scan')}
            onProject={() => setView('project')}
            hasTemplates={hasTemplates}
          />
        )}
        {view === 'scan' && (
          <ScanView 
            key="scan"
            onComplete={() => setView('home')}
            onCancel={() => setView('home')}
          />
        )}
        {view === 'project' && (
          <ProjectView 
            key="project"
            onBack={() => setView('home')}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

function HomeView({ 
  onScan, 
  onProject,
  hasTemplates 
}: { 
  onScan: () => void
  onProject: () => void
  hasTemplates: boolean
}) {
  const { savedTemplates } = useBeardStore()
  const [savedScans, setSavedScans] = useState<string[]>([])
  const [loadingScan, setLoadingScan] = useState(false)
  
  useEffect(() => {
    // Load list of saved scans
    import('./utils/api').then(({ listSavedScans }) => {
      listSavedScans().then(setSavedScans).catch(console.error)
    })
  }, [])
  
  const handleLoadScan = async (templateId: string) => {
    setLoadingScan(true)
    try {
      const { loadScanFrames } = await import('./utils/api')
      await loadScanFrames(templateId)
      // After loading, finalize the template
      const response = await fetch(`/api/template/finalize?template_id=${templateId}&threshold=0.4`, {
        method: 'POST',
      })
      if (response.ok) {
        const result = await response.json()
        const { addTemplate } = await import('./store/beardStore')
        const template = {
          id: templateId,
          name: `Loaded Scan ${new Date().toLocaleDateString()}`,
          createdAt: new Date().toISOString(),
          beardVertexIndices: [],
          boundaryVertexIndices: [],
          calibrationViews: ['loaded'],
          templateData: result.template_data,
        }
        addTemplate(template)
        alert('Scan loaded and processed successfully!')
      } else {
        throw new Error('Failed to finalize loaded scan')
      }
    } catch (err) {
      console.error('Load scan error:', err)
      alert(`Failed to load scan: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setLoadingScan(false)
    }
  }
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col"
    >
      {/* Header */}
      <header className="px-6 py-4 flex items-center justify-between border-b" style={{ borderColor: 'var(--border)' }}>
        <h1 className="text-xl font-semibold">BeardAR</h1>
      </header>

      {/* Content */}
      <main className="flex-1 flex flex-col items-center justify-center p-8">
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="text-center max-w-md"
        >
          {/* Icon */}
          <div 
            className="w-20 h-20 mx-auto mb-6 rounded-full flex items-center justify-center"
            style={{ background: 'var(--bg-tertiary)' }}
          >
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="8" r="5" />
              <path d="M3 21c0-4.5 4-8 9-8s9 3.5 9 8" />
              <path d="M8 14c0 2 1.5 4 4 4s4-2 4-4" strokeLinecap="round" />
            </svg>
          </div>

          <h2 className="text-3xl font-semibold mb-3">
            Perfect Lineup, Every Time
          </h2>
          <p className="mb-8" style={{ color: 'var(--text-secondary)' }}>
            Scan your face to capture your ideal beard shape. 
            Project it later when you need to trim.
          </p>

          {/* Actions */}
          <div className="space-y-3">
            <button
              onClick={onScan}
              className="w-full py-3.5 px-6 rounded-xl font-medium text-white transition-smooth"
              style={{ background: 'var(--accent)' }}
              onMouseOver={(e) => e.currentTarget.style.background = 'var(--accent-hover)'}
              onMouseOut={(e) => e.currentTarget.style.background = 'var(--accent)'}
            >
              Scan New Template
            </button>

            {hasTemplates && (
              <button
                onClick={onProject}
                className="w-full py-3.5 px-6 rounded-xl font-medium border transition-smooth"
                style={{ 
                  borderColor: 'var(--border)',
                  background: 'var(--bg-secondary)'
                }}
              >
                Project Saved Template
              </button>
            )}
            
            {/* Load Saved Scan */}
            {savedScans.length > 0 && (
              <div className="mt-6 pt-6 border-t" style={{ borderColor: 'var(--border)' }}>
                <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
                  Saved Scans (Load & Process)
                </h3>
                <div className="space-y-2">
                  {savedScans.map((scanId) => (
                    <button
                      key={scanId}
                      onClick={() => handleLoadScan(scanId)}
                      disabled={loadingScan}
                      className="w-full py-2 px-4 rounded-lg text-sm font-medium border transition-smooth disabled:opacity-50"
                      style={{ 
                        borderColor: 'var(--border)',
                        background: 'var(--bg-secondary)'
                      }}
                    >
                      {loadingScan ? 'Processing...' : `Load: ${scanId.slice(0, 20)}...`}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* Saved Templates Preview */}
        {hasTemplates && (
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="mt-12 w-full max-w-md"
          >
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>
              Saved Templates
            </h3>
            <div className="space-y-2">
              {savedTemplates.slice(0, 3).map((template) => (
                <div
                  key={template.id}
                  className="p-4 rounded-xl border flex items-center justify-between"
                  style={{ 
                    background: 'var(--bg-secondary)',
                    borderColor: 'var(--border)'
                  }}
                >
                  <div>
                    <p className="font-medium">{template.name}</p>
                    <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      {new Date(template.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                  <span 
                    className="text-sm px-2 py-1 rounded-md"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-tertiary)' }}
                  >
                    {template.beardVertexIndices.length} points
                  </span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </main>

      {/* Footer */}
      <footer className="px-6 py-4 text-center text-sm" style={{ color: 'var(--text-secondary)' }}>
        Uses AI segmentation and face tracking
      </footer>
    </motion.div>
  )
}

export default App
