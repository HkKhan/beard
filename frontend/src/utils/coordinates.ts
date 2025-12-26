/**
 * Coordinate conversion utilities for MediaPipe Face Mesh
 * MediaPipe returns normalized coordinates (0.0 to 1.0)
 */

export interface Point2D {
  x: number
  y: number
}

export interface Point3D extends Point2D {
  z: number
}

/**
 * Convert normalized coordinates to pixel coordinates
 */
export function normalizedToPixel(
  x: number,
  y: number,
  width: number,
  height: number
): Point2D {
  return {
    x: x * width,
    y: y * height,
  }
}

/**
 * Convert pixel coordinates to normalized coordinates
 */
export function pixelToNormalized(
  x: number,
  y: number,
  width: number,
  height: number
): Point2D {
  return {
    x: x / width,
    y: y / height,
  }
}

/**
 * Convert MediaPipe landmarks to pixel coordinates
 */
export function landmarksToPixels(
  landmarks: { x: number; y: number; z?: number }[],
  width: number,
  height: number
): Point2D[] {
  return landmarks.map((lm) => normalizedToPixel(lm.x, lm.y, width, height))
}

/**
 * Get bounding box from landmarks
 */
export function getLandmarksBoundingBox(
  landmarks: Point2D[]
): { x: number; y: number; width: number; height: number } {
  if (landmarks.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 }
  }
  
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  
  for (const lm of landmarks) {
    minX = Math.min(minX, lm.x)
    minY = Math.min(minY, lm.y)
    maxX = Math.max(maxX, lm.x)
    maxY = Math.max(maxY, lm.y)
  }
  
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  }
}

/**
 * Face mesh vertex indices for beard region (approximate)
 * These are the vertices typically in the lower face area
 */
export const BEARD_REGION_INDICES = [
  // Chin area
  152, 377, 400, 378, 379, 365, 397, 288, 361, 323,
  // Jawline left
  234, 127, 162, 21, 54, 103, 67, 109, 10,
  // Jawline right
  454, 356, 389, 251, 284, 332, 297, 338, 10,
  // Lower cheeks
  132, 93, 234, 454, 323, 361,
  // Under nose
  164, 165, 166, 167, 168, 169, 170, 171,
  // Mouth area (for lineup reference)
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
]

/**
 * Catmull-Rom spline interpolation for smooth curves
 */
export function catmullRomSpline(
  points: Point2D[],
  segments: number = 10,
  closed: boolean = false
): Point2D[] {
  if (points.length < 2) return points
  
  const result: Point2D[] = []
  const n = points.length
  
  for (let i = 0; i < n - (closed ? 0 : 1); i++) {
    const p0 = points[(i - 1 + n) % n]
    const p1 = points[i]
    const p2 = points[(i + 1) % n]
    const p3 = points[(i + 2) % n]
    
    for (let j = 0; j < segments; j++) {
      const t = j / segments
      const t2 = t * t
      const t3 = t2 * t
      
      const x = 0.5 * (
        (2 * p1.x) +
        (-p0.x + p2.x) * t +
        (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
        (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3
      )
      
      const y = 0.5 * (
        (2 * p1.y) +
        (-p0.y + p2.y) * t +
        (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
        (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3
      )
      
      result.push({ x, y })
    }
  }
  
  return result
}

/**
 * Order boundary points for smooth path drawing
 * Uses nearest-neighbor algorithm
 */
export function orderBoundaryPoints(points: Point2D[]): Point2D[] {
  if (points.length <= 2) return points
  
  const ordered: Point2D[] = []
  const remaining = [...points]
  
  // Start with the leftmost point
  let current = remaining.reduce((min, p) => 
    p.x < min.x ? p : min, remaining[0]
  )
  ordered.push(current)
  remaining.splice(remaining.indexOf(current), 1)
  
  // Connect nearest neighbors
  while (remaining.length > 0) {
    let nearest = remaining[0]
    let minDist = Infinity
    
    for (const p of remaining) {
      const dist = Math.hypot(p.x - current.x, p.y - current.y)
      if (dist < minDist) {
        minDist = dist
        nearest = p
      }
    }
    
    ordered.push(nearest)
    remaining.splice(remaining.indexOf(nearest), 1)
    current = nearest
  }
  
  return ordered
}


