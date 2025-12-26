import { useCallback, useRef, useState } from 'react'
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

  const { savedTemplates, activeTemplateId, setActiveTemplate } = useBeardStore()

  // Cached contour from live segmentation
  const lastContourRef = useRef<ContourPoint[]>([])

  // Rate limiting for template projection requests
  const lastRequestTime = useRef<number>(0)
  const REQUEST_INTERVAL = 100 // ms between requests
  

  const projectTemplate = useCallback(async (
    landmarks: number[][],
    width: number,
    height: number,
    videoElement: HTMLVideoElement,
    faceBox?: {x: number, y: number, width: number, height: number},
    croppedFace?: HTMLCanvasElement
  ) => {
    if (!activeTemplateId || !faceBox) {
      console.log('No active template or face box')
      return
    }

    const activeTemplate = savedTemplates.find((t) => t.id === activeTemplateId)
    if (!activeTemplate) {
      console.error('Active template not found in frontend store:', activeTemplateId, 'Available:', savedTemplates.map(t => t.id))
      return
    }

    const now = Date.now()
    if (now - lastRequestTime.current < REQUEST_INTERVAL) {
      return
    }
    lastRequestTime.current = now

    try {
      // Adjust landmarks relative to cropped face
      const adjustedLandmarks = landmarks.map(([x, y]) => [
        x - faceBox.x,
        y - faceBox.y
      ])

      console.log('Projecting template:', {
        template_id: activeTemplateId,
        hasActiveTemplate: !!activeTemplate,
        templateIds: savedTemplates.map(t => t.id),
        landmarksCount: adjustedLandmarks.length,
        imageSize: { width: faceBox.width, height: faceBox.height },
        faceBox: faceBox,
        originalLandmarks: landmarks.slice(0, 5), // First 5 landmarks
        adjustedLandmarks: adjustedLandmarks.slice(0, 5) // First 5 adjusted
      })

      // Check what templates are available on backend
      try {
        const checkResponse = await fetch('/api/template/list')
        if (checkResponse.ok) {
          const checkData = await checkResponse.json()
          console.log('Backend templates available:', checkData)
        } else {
          console.log('Backend template check failed:', checkResponse.status, checkResponse.statusText)
        }
      } catch (checkErr) {
        console.log('Could not check backend templates:', checkErr.message)
      }

      console.log('Sending to backend:', {
        template_id: activeTemplateId,
        landmarksSample: adjustedLandmarks.slice(0, 5), // First 5 landmarks
        image_size: { width: Math.round(faceBox.width), height: Math.round(faceBox.height) },
        is_mirrored: true
      })

      // Use backend template projection
      const response = await fetch(`/api/template/project?template_id=${encodeURIComponent(activeTemplateId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          landmarks: adjustedLandmarks,
          image_width: Math.round(faceBox.width),
          image_height: Math.round(faceBox.height),
          is_mirrored: true, // Webcam feed is mirrored
        }),
      })

      console.log('Backend response status:', response.status)
      console.log('Response headers:', Object.fromEntries(response.headers.entries()))

      if (!response.ok) {
        const errorText = await response.text()
        console.error('Template projection failed:', response.status, errorText)
        throw new Error(`Template projection failed: ${response.status} - ${errorText}`)
      }

      let result;
      try {
        result = await response.json()
        console.log('Backend projection result:', result)

        if (!result || typeof result !== 'object') {
          console.error('Invalid response format:', result)
          return
        }
      } catch (jsonError) {
        console.error('Failed to parse JSON response:', jsonError)
        const textResponse = await response.text()
        console.error('Raw response text:', textResponse)
        throw jsonError
      }

      console.log('Processing contours:', result.contour_points?.length || 0, 'points')

      if (result.contour_points && result.contour_points.length > 0) {
        // Contour points are already in cropped face coordinates, transform to display coordinates
        // faceBox is already mirrored to match display coordinates
        const transformedContour = result.contour_points.map(([x, y]: [number, number]) => ({
          x: x + faceBox.x,
          y: y + faceBox.y
        }))

        console.log('Projection: Received contour points', {
          originalCount: result.contour_points.length,
          transformedCount: transformedContour.length,
          samplePoints: result.contour_points.slice(0, 3),
          faceBox: faceBox,
          transformedSample: transformedContour.slice(0, 3)
        })

        lastContourRef.current = transformedContour
        setError(null)
      }

    } catch (err) {
      console.error('Template projection error:', err)
      // Don't set error for every failed request - too noisy
    }
  }, [activeTemplateId, savedTemplates])

  const handleLandmarksUpdate = useCallback((
    landmarks: number[][],
    width: number,
    height: number,
    videoElement?: HTMLVideoElement,
    faceBox?: {x: number, y: number, width: number, height: number},
    croppedFace?: HTMLCanvasElement
  ) => {
    console.log('Projection: Face detected', {
      hasVideo: !!videoElement,
      hasFaceBox: !!faceBox,
      faceBox: faceBox,
      landmarksCount: landmarks?.length,
      canvasSize: { width, height }
    })

    if (!overlayRef.current) return

    const canvas = overlayRef.current
    const ctx = canvas.getContext('2d')!

    canvas.width = width
    canvas.height = height
    ctx.clearRect(0, 0, width, height)

    // Project selected template using face data
    if (videoElement && faceBox) {
      console.log('Projection: Calling projectTemplate with face data')
      projectTemplate(landmarks, width, height, videoElement, faceBox, croppedFace)
    } else {
      console.log('Projection: Missing video or faceBox', { videoElement: !!videoElement, faceBox: !!faceBox })
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

  }, [lineOpacity, lineWidth, fillOpacity, projectTemplate])


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

      {/* Status and template reference */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 max-w-[calc(100vw-2rem)]">
        <div className="bg-black/60 backdrop-blur-sm rounded-lg p-3">
          <div className="text-green-400 text-xs text-center mb-2 flex items-center justify-center gap-1">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            Template Projection
          </div>
          {savedTemplates.length > 0 && (
            <>
              <div className="text-white/60 text-xs text-center mb-2">Your Templates ({savedTemplates.length})</div>
              <div className="flex gap-2 overflow-x-auto max-w-[300px] custom-scrollbar">
                {savedTemplates.map((t, index) => (
                  <button
                    key={`${t.id}-${index}`}
                    onClick={() => setActiveTemplate(t.id)}
                    className={`flex-shrink-0 px-3 py-2 rounded-md text-sm whitespace-nowrap transition-colors ${
                      t.id === activeTemplateId
                        ? 'bg-blue-500 text-white'
                        : 'bg-white/10 text-white/80 hover:bg-white/20'
                    }`}
                  >
                    {t.name}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

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
