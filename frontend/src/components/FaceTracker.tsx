import { useEffect, useRef, useState } from 'react'
import { useBeardStore } from '../store/beardStore'
import * as faceapi from 'face-api.js'

interface FaceTrackerProps {
  onLandmarksUpdate?: (landmarks: number[][], width: number, height: number, video?: HTMLVideoElement, faceBox?: {x: number, y: number, width: number, height: number}, croppedFace?: HTMLCanvasElement) => void
  showMesh?: boolean
  meshColor?: string
  children?: React.ReactNode
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
  const [cameraPermission, setCameraPermission] = useState<'checking' | 'granted' | 'denied' | 'unknown'>('checking')
  const [retryKey, setRetryKey] = useState(0) // Used to trigger re-initialization
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

  const checkCameraPermission = async (): Promise<boolean> => {
    try {
      // First check if we're in a secure context
      if (!window.isSecureContext && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        const errorMsg = 'Camera access requires HTTPS or localhost. Please access this app via https:// or localhost.'
        setError(errorMsg)
        setCameraPermission('denied')
        setIsLoading(false)
        console.error('Not in secure context:', {
          isSecureContext: window.isSecureContext,
          hostname: window.location.hostname,
          protocol: window.location.protocol
        })
        return false
      }

      // Check if getUserMedia is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const errorMsg = 'Camera API not available in this browser. Please use a modern browser like Chrome, Firefox, or Edge.'
        setError(errorMsg)
        setCameraPermission('denied')
        setIsLoading(false)
        console.error('getUserMedia not available')
        return false
      }

      // Check if we have camera permission (optional - not all browsers support this)
      if (navigator.permissions && navigator.permissions.query) {
        try {
          const result = await navigator.permissions.query({ name: 'camera' as PermissionName })
          setCameraPermission(result.state as typeof cameraPermission)

          if (result.state === 'denied') {
            setError('Camera access denied. Please enable camera access in your browser settings (click the camera icon 🔒 in the address bar) and refresh the page.')
            setIsLoading(false)
            return false
          }
        } catch (permErr) {
          // Permissions API might not support 'camera' query, continue anyway
          console.log('Permissions API query not supported, continuing...', permErr)
          setCameraPermission('unknown')
        }
      } else {
        setCameraPermission('unknown')
      }
      return true
    } catch (err) {
      // Permissions API not supported, continue with getUserMedia
      console.log('Permissions check failed, continuing...', err)
      setCameraPermission('unknown')
      return true
    }
  }

  const requestCameraAccess = async () => {
    try {
      setCameraPermission('checking')

      // Check if getUserMedia is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const errorMessage = 'Camera API not available. This app requires HTTPS (or localhost) to access the camera.'
        setError(errorMessage)
        setCameraPermission('denied')
        setIsLoading(false)
        console.error('getUserMedia not available:', {
          hasMediaDevices: !!navigator.mediaDevices,
          hasGetUserMedia: !!(navigator.mediaDevices?.getUserMedia),
          isSecureContext: window.isSecureContext,
          location: window.location.href
        })
        return null
      }

      // Get camera stream with FULL resolution for user experience
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 }, // Full resolution for user
          height: { ideal: 720 },
          facingMode: 'user',
          frameRate: { ideal: 60 } // Smooth 60 FPS for user
        },
      })

      setCameraPermission('granted')
      return stream
    } catch (err: any) {
      console.error('Camera access error:', err)
      setCameraPermission('denied')

      let errorMessage = 'Camera access failed. '
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMessage += 'Please click "Allow" when prompted for camera access, or enable camera access in your browser settings (click the camera icon 🔒 in the address bar).'
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMessage += 'No camera found on this device. Please connect a camera and try again.'
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMessage += 'Camera is being used by another application. Please close other apps using the camera and try again.'
      } else if (err.name === 'OverconstrainedError' || err.name === 'ConstraintNotSatisfiedError') {
        errorMessage += 'Camera does not support the required settings. Trying with default settings...'
        // Try again with minimal constraints
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ video: true })
          setCameraPermission('granted')
          return stream
        } catch (retryErr: any) {
          errorMessage = 'Camera access failed. Please check your camera settings and try again.'
        }
      } else {
        errorMessage += `Error: ${err.message || err.name || 'Unknown error'}. Please check your camera settings and try again.`
      }

      setError(errorMessage)
      setIsLoading(false)
      return null
    }
  }

  useEffect(() => {
    let mounted = true

    const initFaceDetection = async () => {
      try {
        setIsLoading(true)
        setError(null)

        // Check camera permission first (before loading models)
        const hasPermission = await checkCameraPermission()
        if (!hasPermission || !mounted) return

        // Load only the tiny face detector for maximum speed
        try {
          // Use CDN first as it's more reliable for binary files
          // Fallback to local if CDN fails (for offline development)
          let loaded = false
          try {
            console.log('Attempting to load models from CDN...')
            await faceapi.nets.tinyFaceDetector.loadFromUri('https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights/')
            console.log('Face detection models loaded successfully from CDN')
            loaded = true
          } catch (cdnErr: any) {
            console.warn('CDN load failed, trying local files:', cdnErr.message)
            try {
              await faceapi.nets.tinyFaceDetector.loadFromUri('/models')
              console.log('Face detection models loaded successfully from local files')
              loaded = true
            } catch (localErr: any) {
              console.error('Both CDN and local loading failed')
              throw localErr // Throw the local error as it's likely more specific
            }
          }
          
          if (!loaded) {
            throw new Error('Failed to load models from any source')
          }
        } catch (modelErr: any) {
          console.error('Failed to load face detection models:', modelErr)
          console.error('Error details:', {
            name: modelErr.name,
            message: modelErr.message,
            stack: modelErr.stack
          })
          
          if (mounted) {
            let errorMsg = 'Failed to load face detection models. '
            
            if (modelErr.message?.includes('byte length') || modelErr.message?.includes('Float32Array')) {
              errorMsg += 'Model weights file appears to be corrupted or incomplete. Please try:\n' +
                '1. Hard refresh the page (Ctrl+Shift+R)\n' +
                '2. Clear browser cache\n' +
                '3. Check that the model files downloaded correctly'
            } else if (modelErr.message?.includes('WebGL') || modelErr.message?.includes('backend')) {
              errorMsg += 'TensorFlow.js initialization issue. This might be a browser compatibility problem. Try refreshing the page.'
            } else if (modelErr.message?.includes('404') || modelErr.message?.includes('not found') || modelErr.message?.includes('Failed to fetch')) {
              errorMsg += 'Model files not found. Please check that model files exist in /public/models/ and are accessible.'
            } else {
              errorMsg += `${modelErr.message || 'Unknown error'}. Please check the browser console for more details.`
            }
            
            setError(errorMsg)
            setIsLoading(false)
          }
          return
        }

        if (!mounted) return

        // Request camera access
        const stream = await requestCameraAccess()
        if (!stream) {
          if (!mounted) return
          // Error already set by requestCameraAccess
          return
        }

        if (!mounted) {
          stream.getTracks().forEach(t => t.stop())
          return
        }

        if (!videoRef.current) {
          console.error('Video element not available')
          stream.getTracks().forEach(t => t.stop())
          if (mounted) {
            setError('Video element not found. Please refresh the page.')
            setIsLoading(false)
          }
          return
        }

        streamRef.current = stream
        videoRef.current.srcObject = stream

        // Wait for video to be ready
        await new Promise<void>((resolve, reject) => {
          if (!videoRef.current) {
            reject(new Error('Video element not available'))
            return
          }
          
          const timeout = setTimeout(() => {
            reject(new Error('Video metadata loading timeout'))
          }, 5000)

          videoRef.current!.onloadedmetadata = () => {
            clearTimeout(timeout)
            resolve()
          }

          videoRef.current!.onerror = (e) => {
            clearTimeout(timeout)
            reject(new Error('Video loading error'))
          }
        })

        try {
          await videoRef.current.play()
          console.log('Video started playing')
        } catch (playErr: any) {
          console.error('Failed to play video:', playErr)
          stream.getTracks().forEach(t => t.stop())
          if (mounted) {
            setError(`Failed to start video: ${playErr.message || 'Please check your camera and try again'}`)
            setIsLoading(false)
          }
          return
        }

        if (!mounted) return

        setIsLoading(false)

        // Video rendering loop - runs at 60 FPS for smooth display
        const renderVideo = () => {
          if (!mounted || !videoRef.current || !canvasRef.current) return

          const canvas = canvasRef.current
          const ctx = canvas.getContext('2d')!
          const video = videoRef.current

          // Set canvas size to match video
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            console.log('Canvas resized to match video:', { width: canvas.width, height: canvas.height })
          }

          // Clear and draw video frame (mirrored for selfie view)
          ctx.save()
          ctx.scale(-1, 1)
          ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height)
          ctx.restore()

          // Draw face box if we have detection results (from face detection loop)
          if (showMeshRef.current && currentFaceBoxRef.current) {
            // Account for mirroring when drawing face box
            const mirroredBox = {
              x: canvas.width - currentFaceBoxRef.current.x - currentFaceBoxRef.current.width,
              y: currentFaceBoxRef.current.y,
              width: currentFaceBoxRef.current.width,
              height: currentFaceBoxRef.current.height
            }
            drawSimpleFaceBox(ctx, mirroredBox, meshColorRef.current)
          }

          animationFrameRef.current = requestAnimationFrame(renderVideo)
        }

        // Face detection loop - runs at ~12 FPS for performance
        let lastDetectionTime = 0
        const currentFaceBoxRef = { current: null as any }

        const detectFaces = async () => {
          if (!mounted || !videoRef.current) return

          const now = Date.now()
          if (now - lastDetectionTime < 80) { // ~12 FPS
            setTimeout(detectFaces, 10)
            return
          }
          lastDetectionTime = now

          try {
            const video = videoRef.current

            // Create a smaller detection canvas for speed
            const detectionCanvas = document.createElement('canvas')
            const detectionSize = 256 // Very small for speed
            detectionCanvas.width = detectionSize
            detectionCanvas.height = detectionSize
            const detCtx = detectionCanvas.getContext('2d')!

            // Scale down video for detection (much faster)
            detCtx.drawImage(video, 0, 0, detectionSize, detectionSize)

            // Detect faces on the small canvas
            const detections = await faceapi.detectAllFaces(
              detectionCanvas,
              new faceapi.TinyFaceDetectorOptions({
                inputSize: 256, // Much smaller input size for speed
                scoreThreshold: 0.3 // Lower threshold for more detections
              })
            )

            if (detections.length > 0) {
              const detection = detections[0]

              // Scale detection results back to full video size
              const scaleX = video.videoWidth / detectionSize
              const scaleY = video.videoHeight / detectionSize

              const box = detection.box
              const scaledBox = {
                x: box.x * scaleX,
                y: box.y * scaleY,
                width: box.width * scaleX,
                height: box.height * scaleY
              }

              console.log('Face detection results:', {
                detectionBox: box,
                scaleFactors: { scaleX, scaleY },
                scaledBox: scaledBox,
                videoSize: { width: video.videoWidth, height: video.videoHeight },
                detectionSize: detectionSize
              })

              // Mirror coordinates to match display (selfie view)
              const mirroredBox = {
                x: video.videoWidth - scaledBox.x - scaledBox.width,
                y: scaledBox.y,
                width: scaledBox.width,
                height: scaledBox.height
              }

              // Store mirrored face box for rendering loop
              currentFaceBoxRef.current = mirroredBox

              // Create simple landmarks from bounding box (approximation for speed)
              const landmarkArray = createSimpleLandmarks(mirroredBox, video.videoWidth, video.videoHeight)

              // Update store
              setCurrentLandmarks(landmarkArray)

              // Create cropped face canvas for SAM (use original non-mirrored coordinates)
              const croppedCanvas = document.createElement('canvas')
              croppedCanvas.width = scaledBox.width
              croppedCanvas.height = scaledBox.height
              const croppedCtx = croppedCanvas.getContext('2d')!

              // Draw cropped face (use original coordinates, not mirrored)
              croppedCtx.drawImage(
                video,
                scaledBox.x, scaledBox.y, scaledBox.width, scaledBox.height,
                0, 0, scaledBox.width, scaledBox.height
              )

              // Callback with mirrored face box for display consistency
              if (onLandmarksUpdateRef.current) {
                onLandmarksUpdateRef.current(
                  landmarkArray,
                  video.videoWidth,
                  video.videoHeight,
                  video,
                  mirroredBox,
                  croppedCanvas
                )
              }
            } else {
              currentFaceBoxRef.current = null
              setCurrentLandmarks(null)
            }
          } catch (err) {
            console.error('Face detection error:', err)
          }

          setTimeout(detectFaces, 10)
        }

        // Start both loops
        renderVideo()
        detectFaces()

      } catch (err) {
        console.error('Face detection init error:', err)
        if (mounted) {
          let errorMessage = 'Failed to initialize face tracking. '
          if (err instanceof Error) {
            console.error('Error details:', {
              name: err.name,
              message: err.message,
              stack: err.stack
            })
            
            if (err.message?.includes('model') || err.message?.includes('Model')) {
              errorMessage = `Failed to load face detection models: ${err.message}. Please ensure model files are available.`
            } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
              errorMessage += 'Please click "Allow" when prompted for camera access, or enable camera access in your browser settings.'
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
              errorMessage += 'No camera found on this device.'
            } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
              errorMessage += 'Camera is being used by another application.'
            } else if (err.message?.includes('timeout')) {
              errorMessage += 'Camera initialization timed out. Please try again.'
            } else {
              errorMessage += `${err.message || err.name || 'Unknown error'}. Please check your camera settings and try again.`
            }
          } else {
            errorMessage += 'Please allow camera access and try again.'
          }
          setError(errorMessage)
          setIsLoading(false)
        }
      }
    }

    initFaceDetection()

    return () => {
      mounted = false

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
      }
    }
  }, [setCurrentLandmarks, retryKey]) // Add retryKey as dependency
  
  if (error) {
    return (
      <div className="h-full flex items-center justify-center bg-black">
        <div className="text-center p-8 max-w-md">
          <div className="text-4xl mb-4">📷</div>
          <h3 className="text-xl font-semibold text-white mb-4">Camera Access Required</h3>
          <p className="text-white/70 mb-6 leading-relaxed whitespace-pre-line">{error}</p>

          {/* Diagnostic info (collapsible) */}
          <details className="mb-6 text-left">
            <summary className="text-white/60 text-sm cursor-pointer hover:text-white/80 mb-2">
              Show diagnostic information
            </summary>
            <div className="bg-black/40 p-4 rounded text-xs text-white/70 font-mono space-y-1 mt-2">
              <div>Secure Context: {window.isSecureContext ? '✓ Yes' : '✗ No'}</div>
              <div>Protocol: {window.location.protocol}</div>
              <div>Hostname: {window.location.hostname}</div>
              <div>MediaDevices API: {navigator.mediaDevices ? '✓ Available' : '✗ Not available'}</div>
              <div>getUserMedia: {navigator.mediaDevices?.getUserMedia ? '✓ Available' : '✗ Not available'}</div>
              <div>Permission Status: {cameraPermission}</div>
            </div>
          </details>

          {cameraPermission === 'denied' && (
            <div className="space-y-4">
              <button
                onClick={() => {
                  // Stop any existing stream
                  if (streamRef.current) {
                    streamRef.current.getTracks().forEach(t => t.stop())
                    streamRef.current = null
                  }
                  // Clear error and trigger re-initialization
                  setError(null)
                  setIsLoading(true)
                  setCameraPermission('checking')
                  setRetryKey(prev => prev + 1) // This will trigger the useEffect to re-run
                }}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
              >
                Request Camera Access
              </button>

              <button
                onClick={() => window.location.reload()}
                className="bg-gray-600 hover:bg-gray-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
              >
                Refresh Page
              </button>

              <div className="text-sm text-white/50 text-left">
                <p className="mb-2 font-medium">Manual steps to enable camera:</p>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Look for the camera icon 🔒 or 🔐 in your browser's address bar</li>
                  <li>Click it and select "Allow" or "Always allow" for camera access</li>
                  <li>If no icon appears, go to browser settings → Privacy → Site Settings → Camera</li>
                  <li>Make sure this site is allowed to use the camera</li>
                  <li>Refresh this page after changing settings</li>
                </ol>
              </div>
            </div>
          )}

          {cameraPermission !== 'denied' && (
            <button
              onClick={() => {
                // Stop any existing stream
                if (streamRef.current) {
                  streamRef.current.getTracks().forEach(t => t.stop())
                  streamRef.current = null
                }
                // Clear error and trigger re-initialization
                setError(null)
                setIsLoading(true)
                setCameraPermission('checking')
                setRetryKey(prev => prev + 1) // This will trigger the useEffect to re-run
              }}
              className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg font-medium transition-colors"
            >
              Try Again
            </button>
          )}
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
 * Create 468 landmarks from bounding box for backend compatibility
 * Maps key landmarks to expected MediaPipe indices
 */
function createSimpleLandmarks(faceBox: {x: number, y: number, width: number, height: number}, canvasWidth: number, canvasHeight: number): number[][] {
  const landmarks: number[][] = []

  // Create 468 landmarks (most will be dummy/placeholder positions)
  // Focus on the key indices that the backend uses for beard prompts
  const { x, y, width, height } = faceBox

  // Initialize all landmarks with dummy positions
  for (let i = 0; i < 468; i++) {
    landmarks.push([x + width * 0.5, y + height * 0.5])
  }

  // Key landmark indices used by backend (from live_segmentation.py)
  const CHIN = 152
  const JAW_LEFT = 234
  const JAW_RIGHT = 454
  const FOREHEAD = 10
  const LEFT_EYE = 33
  const RIGHT_EYE = 263
  const NOSE_TIP = 4
  const UPPER_LIP = 13

  // Set key landmarks to realistic positions
  // Chin (bottom center of face)
  landmarks[CHIN] = [x + width * 0.5, y + height * 0.85]

  // Jaw line points
  landmarks[JAW_LEFT] = [x + width * 0.1, y + height * 0.8]
  landmarks[JAW_RIGHT] = [x + width * 0.9, y + height * 0.8]

  // Forehead (top center)
  landmarks[FOREHEAD] = [x + width * 0.5, y + height * 0.15]

  // Eyes
  landmarks[LEFT_EYE] = [x + width * 0.25, y + height * 0.35]
  landmarks[RIGHT_EYE] = [x + width * 0.75, y + height * 0.35]

  // Nose tip
  landmarks[NOSE_TIP] = [x + width * 0.5, y + height * 0.45]

  // Upper lip
  landmarks[UPPER_LIP] = [x + width * 0.5, y + height * 0.65]

  // Fill in some additional points for realism around the face
  for (let i = 0; i < 468; i++) {
    if (i !== CHIN && i !== JAW_LEFT && i !== JAW_RIGHT && i !== FOREHEAD &&
        i !== LEFT_EYE && i !== RIGHT_EYE && i !== NOSE_TIP && i !== UPPER_LIP) {
      // Add some variation around the face outline
      const angle = (i / 468) * Math.PI * 2
      const radius = 0.45 + (Math.random() - 0.5) * 0.1
      landmarks[i] = [
        x + width * (0.5 + Math.cos(angle) * radius),
        y + height * (0.5 + Math.sin(angle) * radius)
      ]
    }
  }

  return landmarks
}

/**
 * Draw simple face box
 */
function drawSimpleFaceBox(
  ctx: CanvasRenderingContext2D,
  faceBox: {x: number, y: number, width: number, height: number},
  color: string
) {
  // Draw face bounding box
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.strokeRect(faceBox.x, faceBox.y, faceBox.width, faceBox.height)
}

export default FaceTracker
