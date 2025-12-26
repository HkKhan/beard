import { useState, useCallback, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { FaceTracker } from './FaceTracker'
import { useBeardStore, BeardTemplate } from '../store/beardStore'

type ScanPhase = 'intro' | 'scanning' | 'processing' | 'naming' | 'complete' | 'error'

interface ScanViewProps {
  onComplete: () => void
  onCancel: () => void
}

const TOTAL_FRAMES = 60
const CAPTURE_INTERVAL = 100 // ms between captures

export function ScanView({ onComplete, onCancel }: ScanViewProps) {
  const [phase, setPhase] = useState<ScanPhase>('intro')
  const [progress, setProgress] = useState(0)
  const [frameCount, setFrameCount] = useState(0)
  const [scanName, setScanName] = useState('')
  const [templateData, setTemplateData] = useState<any>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [instruction, setInstruction] = useState('Look straight ahead')
  
  const templateIdRef = useRef(`template_${Date.now()}`)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const landmarksRef = useRef<number[][] | null>(null)
  const faceBoxRef = useRef<{x: number, y: number, width: number, height: number} | null>(null)
  const croppedFaceRef = useRef<HTMLCanvasElement | null>(null)
  const frameCountRef = useRef(0)
  const isCapturingRef = useRef(false)
  const captureIntervalRef = useRef<number | null>(null)
  
  const addTemplate = useBeardStore((s) => s.addTemplate)

  const createNamedTemplate = () => {
    if (!templateData) return

    const finalName = scanName.trim() || `Scan ${new Date().toLocaleDateString()}`
    const template: BeardTemplate = {
      id: templateIdRef.current,
      name: finalName,
      createdAt: new Date().toISOString(),
      beardVertexIndices: [],
      boundaryVertexIndices: [],
      calibrationViews: ['multi-angle'],
      templateData: templateData,
    }

    addTemplate(template)
    setPhase('complete')
    setTimeout(onComplete, 1500)
  }

  const instructions = [
    'Look straight ahead',
    'Slowly turn left',
    'Back to center',
    'Slowly turn right', 
    'Back to center',
    'Tilt chin up slightly',
    'Back to center',
    'Tilt chin down slightly',
    'Almost done...',
  ]

  // Capture a single frame - synchronously capture, async API call
  const captureFrame = useCallback(async (): Promise<boolean> => {
    const video = videoRef.current
    const landmarks = landmarksRef.current
    const faceBox = faceBoxRef.current
    const croppedFace = croppedFaceRef.current

    if (!video || !landmarks || landmarks.length < 468 || !faceBox || !croppedFace) {
      console.log('Waiting for face detection...', { video: !!video, landmarks: landmarks?.length, faceBox: !!faceBox, croppedFace: !!croppedFace })
      return false
    }

    try {
      // Use the cropped face canvas from FaceTracker
      const { canvasToBase64 } = await import('../utils/faceDetection')
      const croppedImageBase64 = canvasToBase64(croppedFace)

      // Adjust landmarks relative to cropped face
      const adjustedLandmarks = landmarks.map(([x, y]) => [
        x - faceBox.x,
        y - faceBox.y
      ])

      // Normalize adjusted landmarks to 0-1 range relative to cropped face
      const normalizedLandmarks = adjustedLandmarks.map(([x, y]) => [
        x / faceBox.width,
        y / faceBox.height
      ])

      // Send cropped face to backend
      const response = await fetch('/api/template/add-frame?template_id=' + templateIdRef.current, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: croppedImageBase64,
          face_mesh_landmarks: normalizedLandmarks,
          user_prompts: [],
        }),
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Request failed' }))
        console.error('Frame capture failed:', err.detail)
        return false
      }

      return true

    } catch (err) {
      console.error('Frame capture error:', err)
      return false
    }
  }, [])

  // Start capturing when phase changes to scanning
  useEffect(() => {
    if (phase !== 'scanning') return
    
    isCapturingRef.current = true
    frameCountRef.current = 0
    setFrameCount(0)
    setProgress(0)
    
    const runCapture = async () => {
      if (!isCapturingRef.current) return
      
      const success = await captureFrame()
      
      if (success) {
        frameCountRef.current++
        const count = frameCountRef.current
        setFrameCount(count)
        setProgress((count / TOTAL_FRAMES) * 100)
        
        // Update instruction based on progress
        const idx = Math.floor((count / TOTAL_FRAMES) * instructions.length)
        setInstruction(instructions[Math.min(idx, instructions.length - 1)])
        
        if (count >= TOTAL_FRAMES) {
          isCapturingRef.current = false
          if (captureIntervalRef.current) {
            clearInterval(captureIntervalRef.current)
            captureIntervalRef.current = null
          }
          finalizeTemplate()
          return
        }
      }
    }
    
    // Start capture loop
    captureIntervalRef.current = setInterval(runCapture, CAPTURE_INTERVAL)
    
    // Also run immediately
    runCapture()
    
    return () => {
      isCapturingRef.current = false
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current)
        captureIntervalRef.current = null
      }
    }
  }, [phase, captureFrame])

  const finalizeTemplate = async (useSavedScan: boolean = false) => {
    setPhase('processing')

    try {
      if (!useSavedScan) {
        // First, process all stored frames with SAM
        console.log('[SCANVIEW] Processing stored frames with SAM...')
        const processResponse = await fetch(`/api/template/process-frames?template_id=${templateIdRef.current}&threshold=0.4`, {
          method: 'POST',
        })

        if (!processResponse.ok) {
          const errorText = await processResponse.text()
          console.error('[SCANVIEW] Frame processing error:', errorText)
          throw new Error(`Failed to process frames: ${processResponse.status} ${processResponse.statusText}`)
        }

        const processResult = await processResponse.json()
        console.log('[SCANVIEW] Frame processing complete:', processResult)
      }

      // Now finalize the template
      const endpoint = useSavedScan
        ? `/api/scans/${templateIdRef.current}/finalize?threshold=0.4`
        : `/api/template/finalize?template_id=${templateIdRef.current}&threshold=0.4`

      console.log('[SCANVIEW] Starting finalize, template_id:', templateIdRef.current, 'useSavedScan:', useSavedScan)
      const response = await fetch(endpoint, {
        method: 'POST',
      })

      console.log('[SCANVIEW] Finalize response status:', response.status, response.statusText)

      if (!response.ok) {
        const errorText = await response.text()
        console.error('[SCANVIEW] Finalize error response:', errorText)
        throw new Error(`Failed to finalize template: ${response.status} ${response.statusText}`)
      }

      const result = await response.json()
      console.log('[SCANVIEW] Finalize success, result:', result)

      // Store the template data for naming phase
      setTemplateData(result.template_data)
      setPhase('naming')

    } catch (err) {
      console.error('Finalize error:', err)
      setErrorMsg(err instanceof Error ? err.message : 'Failed to create template')
      setPhase('error')
    }
  }

  // Handle landmarks from FaceTracker - just store them
  const handleLandmarksUpdate = useCallback((
    landmarks: number[][],
    _width: number,
    _height: number,
    video?: HTMLVideoElement,
    faceBox?: {x: number, y: number, width: number, height: number},
    croppedFace?: HTMLCanvasElement
  ) => {
    landmarksRef.current = landmarks
    faceBoxRef.current = faceBox || null
    croppedFaceRef.current = croppedFace || null
    if (video) videoRef.current = video
  }, [])

  const startScan = () => {
    setPhase('scanning')
    setErrorMsg(null)
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col"
      style={{ background: '#000' }}
    >
      {/* Header - always visible */}
      <header className="absolute top-0 left-0 right-0 z-20 px-6 py-4 flex items-center justify-between bg-gradient-to-b from-black/80 to-transparent">
        <button
          onClick={onCancel}
          className="text-white/80 hover:text-white flex items-center gap-2"
        >
          ← Cancel
        </button>
        
        {phase === 'scanning' && (
          <span className="text-white/60 text-sm">
            Frame {frameCount}/{TOTAL_FRAMES}
          </span>
        )}
      </header>

      {/* Main content */}
      <main className="flex-1 relative">
        {/* Intro view */}
        {phase === 'intro' && (
          <div className="h-full flex flex-col items-center justify-center p-8 text-center">
            <div className="w-24 h-24 mb-8 rounded-full border-2 border-white/30 flex items-center justify-center">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5">
                <circle cx="12" cy="8" r="5" />
                <path d="M20 21a8 8 0 1 0-16 0" />
              </svg>
            </div>
            
            <h2 className="text-2xl font-medium text-white mb-4">Scan Your Beard</h2>
            <p className="text-white/60 max-w-md mb-8">
              We'll capture your beard from multiple angles. Slowly move your head as instructed.
            </p>
            
            <button
              onClick={startScan}
              className="px-8 py-3 rounded-full bg-white text-black font-medium hover:bg-white/90"
            >
              Start Scanning
            </button>
          </div>
        )}
        
        {/* Scanning view - FaceTracker is the main content */}
        {phase === 'scanning' && (
          <>
            <FaceTracker
              onLandmarksUpdate={handleLandmarksUpdate}
              showMesh={true}
              meshColor="rgba(255, 255, 255, 0.3)"
            />
            
            {/* Overlay UI */}
            <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center">
              {/* Progress ring */}
              <div className="relative">
                <svg width="200" height="200" className="transform -rotate-90">
                  <circle 
                    cx="100" cy="100" r="90" 
                    fill="none" 
                    stroke="rgba(255,255,255,0.1)" 
                    strokeWidth="4" 
                  />
                  <circle
                    cx="100" cy="100" r="90" 
                    fill="none"
                    stroke="rgba(255,255,255,0.9)" 
                    strokeWidth="4"
                    strokeDasharray={`${(progress / 100) * 565} 565`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-4xl font-light text-white">{Math.round(progress)}%</span>
                </div>
              </div>

              {/* Instruction */}
              <div className="absolute bottom-24 left-0 right-0 text-center">
                <div className="inline-block px-6 py-3 rounded-full bg-black/60 backdrop-blur-sm">
                  <span className="text-white text-lg">{instruction}</span>
                </div>
              </div>
            </div>
          </>
        )}
        
        {/* Processing overlay */}
        {phase === 'processing' && (
          <div className="h-full flex flex-col items-center justify-center bg-black">
            <div className="w-16 h-16 border-4 border-white/20 border-t-white rounded-full animate-spin mb-6" />
            <h3 className="text-xl font-medium text-white mb-2">Creating Template</h3>
            <p className="text-white/60">Combining {frameCount} frames...</p>
          </div>
        )}

        {/* Naming overlay */}
        {phase === 'naming' && (
          <div className="h-full flex flex-col items-center justify-center bg-black p-8">
            <div className="w-20 h-20 mb-6 rounded-full bg-blue-500 flex items-center justify-center">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
              </svg>
            </div>
            <h3 className="text-2xl font-medium text-white mb-4">Name Your Template</h3>
            <p className="text-white/60 mb-6 text-center max-w-md">
              Give your beard template a name so you can easily identify it later
            </p>

            <div className="w-full max-w-sm">
              <input
                type="text"
                value={scanName}
                onChange={(e) => setScanName(e.target.value)}
                placeholder="My Beard Template"
                className="w-full px-4 py-3 rounded-lg bg-white/10 border border-white/20 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-4"
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    createNamedTemplate()
                  }
                }}
              />
              <button
                onClick={createNamedTemplate}
                className="w-full py-3 px-6 rounded-lg bg-blue-500 text-white font-medium hover:bg-blue-600 transition-smooth"
              >
                Save Template
              </button>
            </div>
          </div>
        )}

        {/* Complete overlay */}
        {phase === 'complete' && (
          <div className="h-full flex flex-col items-center justify-center bg-black">
            <div className="w-20 h-20 mb-6 rounded-full bg-green-500 flex items-center justify-center">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                <path d="M20 6L9 17l-5-5" />
              </svg>
            </div>
            <h3 className="text-2xl font-medium text-white mb-2">Template Created!</h3>
            <p className="text-white/60">Your beard map is ready</p>
          </div>
        )}
        
        {/* Error view */}
        {phase === 'error' && (
          <div className="h-full flex flex-col items-center justify-center bg-black p-8 text-center">
            <div className="w-20 h-20 mb-6 rounded-full bg-red-500/20 flex items-center justify-center">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <path d="M15 9l-6 6M9 9l6 6" />
              </svg>
            </div>
            <h3 className="text-xl font-medium text-white mb-2">
              {errorMsg?.includes('finalize') ? 'Template Creation Failed' : 'Something went wrong'}
            </h3>
            <p className="text-white/60 mb-6">
              {errorMsg?.includes('finalize')
                ? `Frames were saved (${frameCount} captured). You can retry creating the template or scan again.`
                : errorMsg
              }
            </p>
            <div className="flex gap-4">
              {errorMsg?.includes('finalize') && (
                <button
                  onClick={() => finalizeTemplate(true)}
                  className="px-6 py-2 rounded-full bg-blue-500 text-white font-medium hover:bg-blue-600"
                >
                  Retry from Saved Frames
                </button>
              )}
              <button
                onClick={() => setPhase('intro')}
                className="px-6 py-2 rounded-full bg-white text-black font-medium"
              >
                {errorMsg?.includes('finalize') ? 'New Scan' : 'Try Again'}
              </button>
            </div>
          </div>
        )}
      </main>
    </motion.div>
  )
}

export default ScanView
