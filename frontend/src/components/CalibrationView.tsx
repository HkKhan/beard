import { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FaceTracker } from './FaceTracker'
import { useBeardStore, fuseCalibrationCaptures, BeardTemplate } from '../store/beardStore'
import { segmentBeard, videoFrameToBase64 } from '../utils/api'

const CALIBRATION_STEPS = [
  { id: 'center', label: 'CENTER', instruction: 'Look straight at the camera' },
  { id: 'left', label: 'LEFT', instruction: 'Turn your head slightly left' },
  { id: 'right', label: 'RIGHT', instruction: 'Turn your head slightly right' },
]

interface CalibrationViewProps {
  onComplete: () => void
}

export function CalibrationView({ onComplete }: CalibrationViewProps) {
  const [step, setStep] = useState(0)
  const [isCapturing, setIsCapturing] = useState(false)
  const [capturedSteps, setCapturedSteps] = useState<number[]>([])
  const [showPromptMode, setShowPromptMode] = useState(false)
  const [prompts, setPrompts] = useState<{ x: number; y: number }[]>([])
  const [error, setError] = useState<string | null>(null)
  
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const currentLandmarks = useBeardStore((s) => s.currentLandmarks)
  const { addCalibrationCapture, calibrationCaptures, addTemplate } = useBeardStore()
  
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!showPromptMode) return
    
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    // Scale to actual canvas size
    const canvas = e.currentTarget.querySelector('canvas')
    if (canvas) {
      const scaleX = canvas.width / rect.width
      const scaleY = canvas.height / rect.height
      setPrompts((prev) => [...prev, { x: x * scaleX, y: y * scaleY }])
    }
  }, [showPromptMode])
  
  const handleCapture = async () => {
    if (prompts.length === 0) {
      setError('Click on your beard to add at least one point prompt')
      return
    }
    
    if (!currentLandmarks) {
      setError('No face detected. Please face the camera.')
      return
    }
    
    setIsCapturing(true)
    setError(null)
    
    try {
      // Get video frame
      const canvas = canvasRef.current || document.querySelector('canvas')
      if (!canvas) throw new Error('Canvas not found')
      
      const imageData = canvas.toDataURL('image/jpeg', 0.9).split(',')[1]
      
      // Call segmentation API
      const response = await segmentBeard({
        image: imageData,
        user_prompts: prompts.map((p) => ({ x: p.x, y: p.y, label: 1 })),
        face_mesh_landmarks: currentLandmarks,
        return_boundary: true,
      })
      
      if (!response.success || !response.vertex_containment) {
        throw new Error('Segmentation failed')
      }
      
      // Save calibration capture
      addCalibrationCapture({
        step: CALIBRATION_STEPS[step].id as 'center' | 'left' | 'right',
        beardIndices: response.vertex_containment.beard_vertex_indices,
        boundaryIndices: response.vertex_containment.boundary_vertex_indices,
        timestamp: Date.now(),
      })
      
      // Mark step as captured
      setCapturedSteps((prev) => [...prev, step])
      
      // Move to next step or finish
      if (step < 2) {
        setStep(step + 1)
        setPrompts([])
        setShowPromptMode(false)
      } else {
        // All steps complete - fuse and save
        await finishCalibration()
      }
      
    } catch (err) {
      console.error('Capture error:', err)
      setError(err instanceof Error ? err.message : 'Capture failed')
    } finally {
      setIsCapturing(false)
    }
  }
  
  const finishCalibration = async () => {
    const allCaptures = [...calibrationCaptures]
    
    // Fuse captures
    const { beardIndices, boundaryIndices } = fuseCalibrationCaptures(allCaptures, 1)
    
    // Create template
    const template: BeardTemplate = {
      id: `template_${Date.now()}`,
      name: 'Perfect Lineup',
      createdAt: new Date().toISOString(),
      beardVertexIndices: beardIndices,
      boundaryVertexIndices: boundaryIndices,
      calibrationViews: ['center', 'left', 'right'],
    }
    
    addTemplate(template)
    onComplete()
  }
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col"
    >
      {/* Progress bar */}
      <div className="flex gap-2 p-4 bg-dark-800">
        {CALIBRATION_STEPS.map((s, i) => (
          <div
            key={s.id}
            className={`
              flex-1 h-1 rounded-full transition-all
              ${capturedSteps.includes(i)
                ? 'bg-neon-green'
                : i === step
                  ? 'bg-neon-green/50'
                  : 'bg-dark-600'
              }
            `}
          />
        ))}
      </div>
      
      {/* Main camera view */}
      <div 
        className="flex-1 relative cursor-crosshair"
        onClick={handleCanvasClick}
      >
        <FaceTracker
          showMesh={true}
          meshColor="rgba(57, 255, 20, 0.2)"
        >
          {/* Prompt points overlay */}
          {prompts.map((p, i) => (
            <motion.div
              key={i}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="absolute w-4 h-4 -ml-2 -mt-2 rounded-full bg-neon-green neon-border"
              style={{ left: `${(p.x / 1280) * 100}%`, top: `${(p.y / 720) * 100}%` }}
            />
          ))}
          
          {/* Capture indicator */}
          {isCapturing && (
            <div className="absolute inset-0 bg-neon-green/10 flex items-center justify-center">
              <div className="w-12 h-12 border-4 border-neon-green border-t-transparent rounded-full animate-spin" />
            </div>
          )}
        </FaceTracker>
      </div>
      
      {/* Controls */}
      <div className="p-6 bg-dark-800 border-t border-dark-600">
        <div className="max-w-xl mx-auto">
          {/* Step info */}
          <div className="text-center mb-6">
            <h3 className="font-display text-2xl text-neon-green mb-2">
              STEP {step + 1}: {CALIBRATION_STEPS[step].label}
            </h3>
            <p className="text-gray-400">
              {showPromptMode
                ? 'Click on your beard area to mark it, then capture'
                : CALIBRATION_STEPS[step].instruction
              }
            </p>
          </div>
          
          {/* Error message */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="text-red-500 text-center mb-4 text-sm"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>
          
          {/* Buttons */}
          <div className="flex gap-4 justify-center">
            {!showPromptMode ? (
              <motion.button
                onClick={() => setShowPromptMode(true)}
                className="
                  px-8 py-3 font-display tracking-wider
                  bg-neon-green/10 border-2 border-neon-green text-neon-green
                  hover:bg-neon-green hover:text-dark-900
                  transition-all duration-300
                "
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                MARK BEARD
              </motion.button>
            ) : (
              <>
                <motion.button
                  onClick={() => {
                    setPrompts([])
                    setShowPromptMode(false)
                  }}
                  className="
                    px-6 py-3 font-display tracking-wider
                    border border-gray-600 text-gray-400
                    hover:border-gray-400 hover:text-gray-200
                    transition-all duration-300
                  "
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  RESET
                </motion.button>
                
                <motion.button
                  onClick={handleCapture}
                  disabled={isCapturing || prompts.length === 0}
                  className={`
                    px-8 py-3 font-display tracking-wider
                    transition-all duration-300
                    ${prompts.length > 0
                      ? 'bg-neon-green text-dark-900 hover:shadow-lg hover:shadow-neon-green/30'
                      : 'bg-dark-600 text-gray-500 cursor-not-allowed'
                    }
                  `}
                  whileHover={prompts.length > 0 ? { scale: 1.02 } : {}}
                  whileTap={prompts.length > 0 ? { scale: 0.98 } : {}}
                >
                  {isCapturing ? 'PROCESSING...' : 'CAPTURE'}
                </motion.button>
              </>
            )}
          </div>
          
          {/* Prompts count */}
          {showPromptMode && (
            <p className="text-center text-gray-500 text-xs mt-4">
              {prompts.length} point{prompts.length !== 1 ? 's' : ''} marked
            </p>
          )}
        </div>
      </div>
    </motion.div>
  )
}

export default CalibrationView



