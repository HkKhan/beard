import { useCallback, useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { FaceTracker } from './FaceTracker'
import { useBeardStore, getActiveTemplate } from '../store/beardStore'
import { catmullRomSpline, orderBoundaryPoints, Point2D } from '../utils/coordinates'

export function ProjectionView() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [lineColor, setLineColor] = useState('#39ff14')
  const [lineWidth, setLineWidth] = useState(3)
  const [showFill, setShowFill] = useState(false)
  const [smoothing, setSmoothing] = useState(true)
  
  const { savedTemplates, activeTemplateId, setActiveTemplate } = useBeardStore()
  const activeTemplate = savedTemplates.find((t) => t.id === activeTemplateId)
  
  const handleLandmarksUpdate = useCallback((
    landmarks: number[][],
    width: number,
    height: number
  ) => {
    if (!overlayRef.current || !activeTemplate) return
    
    const canvas = overlayRef.current
    const ctx = canvas.getContext('2d')!
    
    // Match canvas size
    canvas.width = width
    canvas.height = height
    ctx.clearRect(0, 0, width, height)
    
    const { boundaryVertexIndices, beardVertexIndices } = activeTemplate
    
    // Get boundary points from current landmarks
    let boundaryPoints: Point2D[] = boundaryVertexIndices
      .filter((idx) => idx < landmarks.length)
      .map((idx) => ({ x: landmarks[idx][0], y: landmarks[idx][1] }))
    
    if (boundaryPoints.length < 3) return
    
    // Order points for smooth path
    boundaryPoints = orderBoundaryPoints(boundaryPoints)
    
    // Apply smoothing
    if (smoothing && boundaryPoints.length >= 4) {
      boundaryPoints = catmullRomSpline(boundaryPoints, 8, true)
    }
    
    // Draw fill if enabled
    if (showFill) {
      const fillPoints: Point2D[] = beardVertexIndices
        .filter((idx) => idx < landmarks.length)
        .map((idx) => ({ x: landmarks[idx][0], y: landmarks[idx][1] }))
      
      ctx.beginPath()
      ctx.fillStyle = hexToRgba(lineColor, 0.1)
      for (let i = 0; i < fillPoints.length; i++) {
        const p = fillPoints[i]
        if (i === 0) ctx.moveTo(p.x, p.y)
        else ctx.lineTo(p.x, p.y)
      }
      ctx.closePath()
      ctx.fill()
    }
    
    // Draw boundary line with glow effect
    drawGlowingPath(ctx, boundaryPoints, lineColor, lineWidth)
    
  }, [activeTemplate, lineColor, lineWidth, showFill, smoothing])
  
  // Set first template as active if none selected
  useEffect(() => {
    if (!activeTemplateId && savedTemplates.length > 0) {
      setActiveTemplate(savedTemplates[0].id)
    }
  }, [activeTemplateId, savedTemplates, setActiveTemplate])
  
  if (savedTemplates.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="h-full flex items-center justify-center"
      >
        <div className="text-center">
          <p className="text-gray-400 text-lg mb-4">No saved templates</p>
          <p className="text-gray-600">Complete calibration first to create a beard template</p>
        </div>
      </motion.div>
    )
  }
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex"
    >
      {/* Main camera view */}
      <div className="flex-1 relative">
        <FaceTracker
          onLandmarksUpdate={handleLandmarksUpdate}
          showMesh={false}
        >
          {/* Projection overlay canvas */}
          <canvas
            ref={overlayRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ mixBlendMode: 'screen' }}
          />
          
          {/* Template name badge */}
          {activeTemplate && (
            <div className="absolute top-4 left-4 glass px-4 py-2 rounded">
              <span className="text-neon-green font-display text-sm">
                {activeTemplate.name}
              </span>
            </div>
          )}
          
          {/* Settings toggle */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="absolute top-4 right-4 glass px-3 py-2 rounded hover:bg-dark-700 transition-colors"
          >
            <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" 
              />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </FaceTracker>
      </div>
      
      {/* Settings panel */}
      {showSettings && (
        <motion.div
          initial={{ x: 300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: 300, opacity: 0 }}
          className="w-72 bg-dark-800 border-l border-dark-600 p-6 overflow-y-auto"
        >
          <h3 className="font-display text-lg text-neon-green mb-6">SETTINGS</h3>
          
          {/* Template selector */}
          <div className="mb-6">
            <label className="block text-gray-400 text-sm mb-2">Template</label>
            <select
              value={activeTemplateId || ''}
              onChange={(e) => setActiveTemplate(e.target.value)}
              className="w-full bg-dark-700 border border-dark-600 text-white px-3 py-2 rounded focus:border-neon-green outline-none"
            >
              {savedTemplates.map((t) => (
                <option key={t.id} value={t.id}>{t.name}</option>
              ))}
            </select>
          </div>
          
          {/* Line color */}
          <div className="mb-6">
            <label className="block text-gray-400 text-sm mb-2">Line Color</label>
            <div className="flex gap-2">
              {['#39ff14', '#00f0ff', '#ff00ff', '#ff6600', '#ffffff'].map((color) => (
                <button
                  key={color}
                  onClick={() => setLineColor(color)}
                  className={`
                    w-8 h-8 rounded-full border-2 transition-transform
                    ${lineColor === color ? 'border-white scale-110' : 'border-transparent'}
                  `}
                  style={{ backgroundColor: color }}
                />
              ))}
            </div>
          </div>
          
          {/* Line width */}
          <div className="mb-6">
            <label className="block text-gray-400 text-sm mb-2">
              Line Width: {lineWidth}px
            </label>
            <input
              type="range"
              min="1"
              max="8"
              value={lineWidth}
              onChange={(e) => setLineWidth(Number(e.target.value))}
              className="w-full accent-neon-green"
            />
          </div>
          
          {/* Smoothing toggle */}
          <div className="mb-6">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={smoothing}
                onChange={(e) => setSmoothing(e.target.checked)}
                className="w-4 h-4 accent-neon-green"
              />
              <span className="text-gray-400">Smooth lines</span>
            </label>
          </div>
          
          {/* Fill toggle */}
          <div className="mb-6">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={showFill}
                onChange={(e) => setShowFill(e.target.checked)}
                className="w-4 h-4 accent-neon-green"
              />
              <span className="text-gray-400">Show fill</span>
            </label>
          </div>
          
          {/* Template info */}
          {activeTemplate && (
            <div className="pt-6 border-t border-dark-600">
              <h4 className="text-gray-500 text-xs uppercase mb-3">Template Info</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Vertices</span>
                  <span className="text-gray-300">{activeTemplate.beardVertexIndices.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Boundary</span>
                  <span className="text-gray-300">{activeTemplate.boundaryVertexIndices.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Views</span>
                  <span className="text-gray-300">{activeTemplate.calibrationViews.length}</span>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  )
}

/**
 * Draw path with glow effect
 */
function drawGlowingPath(
  ctx: CanvasRenderingContext2D,
  points: Point2D[],
  color: string,
  width: number
) {
  if (points.length < 2) return
  
  // Outer glow
  ctx.shadowColor = color
  ctx.shadowBlur = 15
  ctx.strokeStyle = color
  ctx.lineWidth = width
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  
  ctx.beginPath()
  ctx.moveTo(points[0].x, points[0].y)
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i].x, points[i].y)
  }
  ctx.closePath()
  ctx.stroke()
  
  // Inner bright line
  ctx.shadowBlur = 0
  ctx.strokeStyle = 'white'
  ctx.lineWidth = width * 0.3
  ctx.stroke()
}

/**
 * Convert hex to rgba
 */
function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

export default ProjectionView

