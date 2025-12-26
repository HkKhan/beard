import * as faceapi from 'face-api.js'

// Model loading state
let modelsLoaded = false

export async function loadFaceModels(): Promise<void> {
  if (modelsLoaded) return

  try {
    // Load models from public/models directory
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models')
    modelsLoaded = true
    console.log('Face detection models loaded')
  } catch (error) {
    console.error('Failed to load face detection models:', error)
    throw error
  }
}

export interface FaceDetection {
  box: {
    x: number
    y: number
    width: number
    height: number
  }
  landmarks: number[][]
}

export async function detectFace(
  videoElement: HTMLVideoElement,
  width: number,
  height: number
): Promise<FaceDetection | null> {
  if (!modelsLoaded) {
    await loadFaceModels()
  }

  try {
    // Create a smaller detection canvas for speed
    const detectionCanvas = document.createElement('canvas')
    const detectionSize = 256
    detectionCanvas.width = detectionSize
    detectionCanvas.height = detectionSize
    const detCtx = detectionCanvas.getContext('2d')!

    // Scale down video for detection (much faster)
    detCtx.drawImage(videoElement, 0, 0, detectionSize, detectionSize)

    // Detect faces on the small canvas
    const detections = await faceapi.detectAllFaces(
      detectionCanvas,
      new faceapi.TinyFaceDetectorOptions({
        inputSize: 256,
        scoreThreshold: 0.3
      })
    )

    if (detections.length === 0) return null

    const detection = detections[0]

    // Scale detection results back to full video size
    const scaleX = width / detectionSize
    const scaleY = height / detectionSize

    const box = detection.box
    const scaledBox = {
      x: box.x * scaleX,
      y: box.y * scaleY,
      width: box.width * scaleX,
      height: box.height * scaleY
    }

    // Add padding
    const padding = Math.max(scaledBox.width, scaledBox.height) * 0.2

    const faceBox = {
      x: Math.max(0, scaledBox.x - padding),
      y: Math.max(0, scaledBox.y - padding),
      width: Math.min(width - (scaledBox.x - padding), scaledBox.width + 2 * padding),
      height: Math.min(height - (scaledBox.y - padding), scaledBox.height + 2 * padding)
    }

    // Create simple landmarks from bounding box
    const landmarks = createSimpleLandmarks(faceBox, width, height)

    return {
      box: faceBox,
      landmarks
    }
  } catch (error) {
    console.error('Face detection failed:', error)
    return null
  }
}

export function cropFaceFromCanvas(
  sourceCanvas: HTMLCanvasElement,
  faceBox: { x: number; y: number; width: number; height: number }
): HTMLCanvasElement {
  const croppedCanvas = document.createElement('canvas')
  const croppedCtx = croppedCanvas.getContext('2d')!

  // Set cropped canvas size to face bounding box
  croppedCanvas.width = faceBox.width
  croppedCanvas.height = faceBox.height

  // Draw the cropped face region
  croppedCtx.drawImage(
    sourceCanvas,
    faceBox.x, faceBox.y, faceBox.width, faceBox.height, // source
    0, 0, faceBox.width, faceBox.height // destination
  )

  return croppedCanvas
}

export function canvasToBase64(canvas: HTMLCanvasElement, quality = 0.9): string {
  return canvas.toDataURL('image/jpeg', quality).split(',')[1]
}

/**
 * Create 468 landmarks from bounding box for backend compatibility
 */
function createSimpleLandmarks(faceBox: {x: number, y: number, width: number, height: number}, canvasWidth: number, canvasHeight: number): number[][] {
  const landmarks: number[][] = []

  // Create 468 landmarks for backend compatibility
  const { x, y, width, height } = faceBox

  // Initialize all landmarks with dummy positions
  for (let i = 0; i < 468; i++) {
    landmarks.push([x + width * 0.5, y + height * 0.5])
  }

  // Key landmark indices used by backend
  const CHIN = 152
  const JAW_LEFT = 234
  const JAW_RIGHT = 454
  const FOREHEAD = 10
  const LEFT_EYE = 33
  const RIGHT_EYE = 263
  const NOSE_TIP = 4
  const UPPER_LIP = 13

  // Set key landmarks to realistic positions
  landmarks[CHIN] = [x + width * 0.5, y + height * 0.85]
  landmarks[JAW_LEFT] = [x + width * 0.1, y + height * 0.8]
  landmarks[JAW_RIGHT] = [x + width * 0.9, y + height * 0.8]
  landmarks[FOREHEAD] = [x + width * 0.5, y + height * 0.15]
  landmarks[LEFT_EYE] = [x + width * 0.25, y + height * 0.35]
  landmarks[RIGHT_EYE] = [x + width * 0.75, y + height * 0.35]
  landmarks[NOSE_TIP] = [x + width * 0.5, y + height * 0.45]
  landmarks[UPPER_LIP] = [x + width * 0.5, y + height * 0.65]

  // Fill in remaining points with face outline variation
  for (let i = 0; i < 468; i++) {
    if (i !== CHIN && i !== JAW_LEFT && i !== JAW_RIGHT && i !== FOREHEAD &&
        i !== LEFT_EYE && i !== RIGHT_EYE && i !== NOSE_TIP && i !== UPPER_LIP) {
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
