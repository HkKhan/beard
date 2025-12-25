/**
 * API client for BeardAR backend
 */

const API_BASE = '/api'

export interface PointPrompt {
  x: number
  y: number
  label: number // 1 = foreground, 0 = background
}

export interface SegmentationRequest {
  image: string // Base64 encoded
  user_prompts: PointPrompt[]
  face_mesh_landmarks?: number[][]
  return_boundary?: boolean
}

export interface MaskResponse {
  mask_base64: string
  confidence: number
  width: number
  height: number
}

export interface VertexContainmentResult {
  beard_vertex_indices: number[]
  boundary_vertex_indices: number[]
  total_vertices_checked: number
}

export interface SegmentationResponse {
  success: boolean
  mask: MaskResponse
  vertex_containment?: VertexContainmentResult
  processing_time_ms: number
}

export interface CalibrationStep {
  step: number
  beard_indices: number[]
  boundary_indices: number[]
}

export interface FusionRequest {
  calibration_steps: CalibrationStep[]
  voting_threshold?: number
}

export interface FusionResponse {
  final_beard_indices: number[]
  final_boundary_indices: number[]
  vertex_vote_counts: Record<string, number>
}

export interface BeardTemplate {
  user_id: string
  template_name: string
  created_at: string
  beard_vertex_indices: number[]
  boundary_vertex_indices: number[]
  calibration_views: string[]
}

/**
 * Check API health
 */
export async function checkHealth(): Promise<{
  status: string
  model_loaded: boolean
  device: string
}> {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) throw new Error('Health check failed')
  return response.json()
}

/**
 * Run segmentation on image with point prompts
 */
export async function segmentBeard(
  request: SegmentationRequest
): Promise<SegmentationResponse> {
  const response = await fetch(`${API_BASE}/segment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Segmentation failed')
  }
  
  return response.json()
}

/**
 * Fuse multiple calibration captures
 */
export async function fuseCalibrations(
  request: FusionRequest
): Promise<FusionResponse> {
  const response = await fetch(`${API_BASE}/calibrate/fuse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Fusion failed')
  }
  
  return response.json()
}

/**
 * Save beard template
 */
export async function saveTemplate(
  template: BeardTemplate
): Promise<BeardTemplate> {
  const response = await fetch(`${API_BASE}/templates`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(template),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Save failed')
  }
  
  return response.json()
}

/**
 * Get saved template
 */
export async function getTemplate(
  userId: string,
  templateName: string
): Promise<BeardTemplate> {
  const response = await fetch(`${API_BASE}/templates/${userId}/${templateName}`)
  
  if (!response.ok) {
    if (response.status === 404) throw new Error('Template not found')
    throw new Error('Failed to get template')
  }
  
  return response.json()
}

/**
 * Convert canvas to base64
 */
export function canvasToBase64(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/jpeg', 0.9).split(',')[1]
}

/**
 * Convert video frame to base64
 */
export function videoFrameToBase64(video: HTMLVideoElement): string {
  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(video, 0, 0)
  return canvasToBase64(canvas)
}

/**
 * List all saved scans
 */
export async function listSavedScans(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/scans`)
  if (!response.ok) throw new Error('Failed to list scans')
  const data = await response.json()
  return data.scans || []
}

/**
 * Load and reprocess a saved scan
 */
export async function loadScanFrames(templateId: string): Promise<{ success: boolean; frame_count: number; message: string }> {
  const response = await fetch(`${API_BASE}/scans/${templateId}/load`, {
    method: 'POST',
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to load scan' }))
    throw new Error(error.detail || 'Failed to load scan')
  }
  return response.json()
}

