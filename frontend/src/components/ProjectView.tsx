import { useCallback, useRef, useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FaceTracker } from './FaceTracker'
import { useBeardStore } from '../store/beardStore'

interface ProjectViewProps {
  onBack: () => void
}

interface ContourPoint {
  x: number
  y: number
}

export function ProjectView({ onBack }: ProjectViewProps) {
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const [lineOpacity, setLineOpacity] = useState(0.8)
  const [lineWidth, setLineWidth] = useState(2.5)
  const [fillOpacity, setFillOpacity] = useState(0.15)
  const [showSettings, setShowSettings] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Cached contour from live segmentation
  const lastContourRef = useRef<ContourPoint[]>([])
  
  // Throttle requests
  const lastRequestTime = useRef(0)
  const REQUEST_INTERVAL = 100  // Live segmentation is more expensive (~100ms)
  
  const { savedTemplates, activeTemplateId, setActiveTemplate } = useBeardStore()
  const activeTemplate = savedTemplates.find((t) => t.id === activeTemplateId)
  
  useEffect(() => {
    if (!activeTemplateId && savedTemplates.length > 0) {
      setActiveTemplate(savedTemplates[0].id)
    }
  }, [activeTemplateId, savedTemplates, setActiveTemplate])

  const segmentLive = useCallback(async (
    landmarks: number[][],
    width: number,
    height: number,
    videoElement: HTMLVideoElement
  ) => {
    if (!activeTemplate) return

    const now = Date.now()
    if (now - lastRequestTime.current < REQUEST_INTERVAL) {
      return
    }
    lastRequestTime.current = now

    try {
      // Capture current video frame
      const canvas = document.createElement('canvas')
      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(videoElement, 0, 0, width, height)
      const imageData = canvas.toDataURL('image/jpeg', 0.8)

      // Normalize landmarks
      const normalizedLandmarks = landmarks.map(([x, y]) => [x / width, y / height])

      const response = await fetch('/api/segment/live', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageData,
          face_mesh_landmarks: normalizedLandmarks,
          user_prompts: [],
          return_boundary: true,
        }),
      })

      if (!response.ok) {
        throw new Error(`Live segmentation failed: ${response.status}`)
      }

      const result = await response.json()

      if (result.contour_points && result.contour_points.length > 0) {
        lastContourRef.current = result.contour_points.map(([x, y]: [number, number]) => ({ x, y }))
        setError(null)
      }

    } catch (err) {
      console.error('Live segmentation error:', err)
      // Don't set error for every failed request - too noisy
    }
  }, [activeTemplate])

  const handleLandmarksUpdate = useCallback((
    landmarks: number[][],
    width: number,
    height: number,
    videoElement?: HTMLVideoElement
  ) => {
    if (!overlayRef.current) return

    const canvas = overlayRef.current
    const ctx = canvas.getContext('2d')!

    canvas.width = width
    canvas.height = height
    ctx.clearRect(0, 0, width, height)

    // Request live segmentation
    if (landmarks.length >= 468 && videoElement) {
      segmentLive(landmarks, width, height, videoElement)
    }

    // Draw current contour
    const contour = lastContourRef.current
    if (contour.length > 2) {
      // Smooth the contour using quadratic curves
      ctx.beginPath()
      ctx.moveTo(contour[0].x, contour[0].y)

      for (let i = 1; i < contour.length - 1; i++) {
        const xc = (contour[i].x + contour[i + 1].x) / 2
        const yc = (contour[i].y + contour[i + 1].y) / 2
        ctx.quadraticCurveTo(contour[i].x, contour[i].y, xc, yc)
      }
      ctx.lineTo(contour[contour.length - 1].x, contour[contour.length - 1].y)
      ctx.closePath()

      // Fill
      ctx.fillStyle = `rgba(255, 255, 255, ${fillOpacity})`
      ctx.fill()

      // Stroke
      ctx.strokeStyle = `rgba(255, 255, 255, ${lineOpacity})`
      ctx.lineWidth = lineWidth
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.shadowColor = 'rgba(255, 255, 255, 0.5)'
      ctx.shadowBlur = 8
      ctx.stroke()
    }

  }, [lineOpacity, lineWidth, fillOpacity, segmentLive])

  if (savedTemplates.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="h-full flex flex-col items-center justify-center p-8"
      >
        <div className="w-20 h-20 mb-6 rounded-full border-2 border-white/20 flex items-center justify-center">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" opacity="0.5">
            <path d="M3 3l18 18M10.5 10.5a3 3 0 0 0 4.24 4.24" />
            <path d="M13.5 13.5L21 21M3 3l7.5 7.5" />
          </svg>
        </div>
        <p className="text-white/80 text-lg mb-2">No Templates Yet</p>
        <p className="text-white/40 text-sm text-center max-w-xs mb-6">
          Scan your face to create a beard template first
        </p>
        <button
          onClick={onBack}
          className="px-6 py-2 rounded-full bg-white text-black font-medium hover:bg-white/90 transition-smooth"
        >
          Go Back
        </button>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col"
      style={{ background: '#000' }}
    >
      {/* Header */}
      <header className="absolute top-0 left-0 right-0 z-10 px-6 py-4 flex items-center justify-between">
        <button
          onClick={onBack}
          className="text-white/80 hover:text-white transition-smooth flex items-center gap-2"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M15 18l-6-6 6-6" />
          </svg>
          Back
        </button>
        
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-2 rounded-lg text-white/80 hover:text-white hover:bg-white/10 transition-smooth"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
          </svg>
        </button>
      </header>

      {/* Camera + Overlay */}
      <main className="flex-1 relative">
        <FaceTracker
          onLandmarksUpdate={handleLandmarksUpdate}
          showMesh={false}
        >
          <canvas
            ref={overlayRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
          />
        </FaceTracker>
        
        {error && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 px-4 py-2 rounded-lg bg-red-500/80 text-white text-sm">
            {error}
          </div>
        )}
      </main>

      {/* Template selector */}
      {savedTemplates.length > 1 && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
          {savedTemplates.map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTemplate(t.id)}
              className={`px-4 py-2 rounded-full text-sm transition-smooth ${
                t.id === activeTemplateId
                  ? 'bg-white text-black'
                  : 'bg-white/10 text-white/80 hover:bg-white/20'
              }`}
            >
              {t.name}
            </button>
          ))}
        </div>
      )}

      {/* Settings panel */}
      {showSettings && (
        <motion.div
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          className="absolute top-0 right-0 bottom-0 w-72 bg-black/90 backdrop-blur-md p-6 pt-20"
        >
          <h3 className="text-white font-medium mb-6">Display Settings</h3>
          
          <div className="space-y-6">
            <div>
              <label className="block text-white/60 text-sm mb-2">
                Line Opacity: {Math.round(lineOpacity * 100)}%
              </label>
              <input
                type="range"
                min="0.2"
                max="1"
                step="0.1"
                value={lineOpacity}
                onChange={(e) => setLineOpacity(Number(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-white/60 text-sm mb-2">
                Line Width: {lineWidth}px
              </label>
              <input
                type="range"
                min="1"
                max="5"
                step="0.5"
                value={lineWidth}
                onChange={(e) => setLineWidth(Number(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-white/60 text-sm mb-2">
                Fill Opacity: {Math.round(fillOpacity * 100)}%
              </label>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.05"
                value={fillOpacity}
                onChange={(e) => setFillOpacity(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}

export default ProjectView
