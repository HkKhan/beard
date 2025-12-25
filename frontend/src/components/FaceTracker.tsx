import { useEffect, useRef, useState } from 'react'
import { useBeardStore } from '../store/beardStore'

// MediaPipe types
interface FaceMeshResults {
  multiFaceLandmarks?: {
    x: number
    y: number
    z: number
  }[][]
}

// Declare global types for MediaPipe loaded via CDN
declare global {
  interface Window {
    FaceMesh: new (config: { locateFile: (file: string) => string }) => {
      setOptions: (options: Record<string, unknown>) => void
      onResults: (callback: (results: FaceMeshResults) => void) => void
      initialize: () => Promise<void>
      send: (inputs: { image: HTMLVideoElement }) => Promise<void>
      close: () => void
    }
  }
}

interface FaceTrackerProps {
  onLandmarksUpdate?: (landmarks: number[][], width: number, height: number, video?: HTMLVideoElement) => void
  showMesh?: boolean
  meshColor?: string
  children?: React.ReactNode
}

// Load script helper
function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve()
      return
    }
    const script = document.createElement('script')
    script.src = src
    script.crossOrigin = 'anonymous'
    script.onload = () => resolve()
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`))
    document.head.appendChild(script)
  })
}

export function FaceTracker({
  onLandmarksUpdate,
  showMesh = true,
  meshColor = 'rgba(255, 255, 255, 0.25)',
  children,
}: FaceTrackerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const faceMeshRef = useRef<ReturnType<typeof window.FaceMesh> | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const setCurrentLandmarks = useBeardStore((s) => s.setCurrentLandmarks)
  
  // Use refs for callback props to avoid re-running effect
  const onLandmarksUpdateRef = useRef(onLandmarksUpdate)
  const showMeshRef = useRef(showMesh)
  const meshColorRef = useRef(meshColor)
  
  // Keep refs updated
  useEffect(() => {
    onLandmarksUpdateRef.current = onLandmarksUpdate
    showMeshRef.current = showMesh
    meshColorRef.current = meshColor
  })
  
  useEffect(() => {
    let mounted = true
    
    const initFaceMesh = async () => {
      try {
        setIsLoading(true)
        
        // Load MediaPipe scripts from CDN
        await loadScript('https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js')
        
        if (!mounted) return
        
        // Wait for FaceMesh to be available
        await new Promise<void>((resolve) => {
          const check = () => {
            if (window.FaceMesh) resolve()
            else setTimeout(check, 100)
          }
          check()
        })
        
        if (!mounted || !videoRef.current) return
        
        // Initialize FaceMesh
        const faceMesh = new window.FaceMesh({
          locateFile: (file: string) => 
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
        })
        
        faceMesh.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        })
        
        faceMesh.onResults((results: FaceMeshResults) => {
          if (!canvasRef.current || !videoRef.current) return
          
          const canvas = canvasRef.current
          const ctx = canvas.getContext('2d')!
          const video = videoRef.current
          
          // Set canvas size to match video
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
          }
          
          // Clear and draw video frame (mirrored)
          ctx.save()
          ctx.scale(-1, 1)
          ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height)
          ctx.restore()
          
          if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0]
            
            // Convert to array format and mirror x coordinates
            const landmarkArray = landmarks.map((lm) => [
              (1 - lm.x) * canvas.width,  // Mirror x
              lm.y * canvas.height,
            ])
            
            // Update store
            setCurrentLandmarks(landmarkArray)
            
            // Callback with video element for frame capture
            if (onLandmarksUpdateRef.current) {
              onLandmarksUpdateRef.current(landmarkArray, canvas.width, canvas.height, video)
            }
            
            // Draw mesh if enabled
            if (showMeshRef.current) {
              drawFaceMesh(ctx, landmarks, canvas.width, canvas.height, meshColorRef.current)
            }
          } else {
            setCurrentLandmarks(null)
          }
        })
        
        await faceMesh.initialize()
        
        if (!mounted) {
          faceMesh.close()
          return
        }
        
        faceMeshRef.current = faceMesh
        
        // Get camera stream
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user',
          },
        })
        
        if (!mounted) {
          stream.getTracks().forEach(t => t.stop())
          return
        }
        
        streamRef.current = stream
        videoRef.current.srcObject = stream
        
        // Wait for video to be ready
        await new Promise<void>((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = () => resolve()
          }
        })
        
        await videoRef.current?.play()
        
        if (!mounted) return
        
        setIsLoading(false)
        
        // Start processing frames
        const processFrame = async () => {
          if (!mounted || !faceMeshRef.current || !videoRef.current) return
          
          if (videoRef.current.readyState >= 2) {
            await faceMeshRef.current.send({ image: videoRef.current })
          }
          
          animationFrameRef.current = requestAnimationFrame(processFrame)
        }
        
        processFrame()
        
      } catch (err) {
        console.error('FaceMesh init error:', err)
        if (mounted) {
          setError('Failed to initialize face tracking. Please allow camera access.')
          setIsLoading(false)
        }
      }
    }
    
    initFaceMesh()
    
    return () => {
      mounted = false
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
      }
      
      if (faceMeshRef.current) {
        faceMeshRef.current.close()
      }
    }
  }, [setCurrentLandmarks]) // Only depend on stable store setter
  
  if (error) {
    return (
      <div className="h-full flex items-center justify-center bg-black">
        <div className="text-center p-8">
          <div className="text-xl mb-4">⚠️</div>
          <p className="text-white/60">{error}</p>
        </div>
      </div>
    )
  }
  
  return (
    <div className="relative h-full w-full bg-black overflow-hidden">
      {/* Hidden video element */}
      <video
        ref={videoRef}
        className="hidden"
        playsInline
        autoPlay
        muted
      />
      
      {/* Canvas with face mesh overlay */}
      <canvas
        ref={canvasRef}
        className="h-full w-full object-contain"
      />
      
      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/80">
          <div className="text-center">
            <div className="w-8 h-8 border-2 border-white/30 border-t-white rounded-full animate-spin mx-auto mb-4" />
            <p className="text-white/60 text-sm">Initializing camera...</p>
          </div>
        </div>
      )}
      
      {/* Children (overlays, buttons, etc.) */}
      {children}
    </div>
  )
}

/**
 * Draw face mesh on canvas
 */
function drawFaceMesh(
  ctx: CanvasRenderingContext2D,
  landmarks: { x: number; y: number; z: number }[],
  width: number,
  height: number,
  color: string
) {
  // Face mesh connections (simplified for performance)
  const FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 
    162, 21, 54, 103, 67, 109, 10
  ]
  
  // Draw face oval
  ctx.beginPath()
  ctx.strokeStyle = color
  ctx.lineWidth = 1
  
  for (let i = 0; i < FACE_OVAL.length; i++) {
    const lm = landmarks[FACE_OVAL[i]]
    const x = (1 - lm.x) * width  // Mirror
    const y = lm.y * height
    
    if (i === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  }
  
  ctx.stroke()
  
  // Draw key landmarks as dots
  const KEY_POINTS = [
    1,    // Nose tip
    33, 263,  // Eyes
    61, 291,  // Mouth corners
    152,  // Chin
  ]
  
  ctx.fillStyle = color.replace('0.3', '0.8')
  for (const idx of KEY_POINTS) {
    const lm = landmarks[idx]
    const x = (1 - lm.x) * width
    const y = lm.y * height
    ctx.beginPath()
    ctx.arc(x, y, 3, 0, Math.PI * 2)
    ctx.fill()
  }
}

export default FaceTracker
