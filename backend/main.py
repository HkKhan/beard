"""
BeardAR Backend API
FastAPI server for beard segmentation and AR overlay projection.
"""

import os
import base64
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.schemas import (
    SegmentationRequest,
    SegmentationResponse,
    FusionRequest,
    FusionResponse,
    CaptureObject,
    HealthResponse,
)
from api.segmentation import (
    process_segmentation,
    fuse_calibration_captures,
)
from api.live_segmentation import process_live_segmentation
from api.beard_template import (
    BeardTemplateBuilder,
    get_or_create_template,
    get_template as get_beard_template,
    save_template,
    delete_template,
)
from api.frame_storage import save_frames, load_frames, list_saved_scans, delete_scan
from core.model_loader import get_model_loader


# Configuration
SAM_CHECKPOINT = os.getenv(
    "SAM_CHECKPOINT",
    str(Path(__file__).parent.parent / "checkpoints" / "sam_vit_b_01ec64.pth")
)
# Fine-tuned checkpoint is in the project root checkpoints folder
FINE_TUNED_CHECKPOINT = os.getenv(
    "FINE_TUNED_CHECKPOINT",
    str(Path(__file__).parent.parent / "checkpoints" / "sam_beard_best.pth")
)
MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - load model on startup."""
    # Write startup log
    from pathlib import Path
    import time
    log_file = Path(__file__).parent.parent / "backend.log"
    with open(log_file, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [STARTUP] BeardAR Backend Starting...\n")
    
    # Startup
    print("\n" + "="*50)
    print("BeardAR Backend Starting...")
    print("="*50)
    
    model_loader = get_model_loader()
    
    # Check for checkpoints
    checkpoint = SAM_CHECKPOINT if os.path.exists(SAM_CHECKPOINT) else None
    fine_tuned = FINE_TUNED_CHECKPOINT if os.path.exists(FINE_TUNED_CHECKPOINT) else None
    
    if not checkpoint:
        print("\nWARNING: SAM checkpoint not found!")
        print(f"  Expected: {SAM_CHECKPOINT}")
        print("  Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        print("  Place sam_vit_b_01ec64.pth in the checkpoints/ directory")
    
    try:
        model_loader.load_model(
            checkpoint_path=checkpoint,
            model_type=MODEL_TYPE,
            fine_tuned_path=fine_tuned,
        )
    except Exception as e:
        print(f"\nWARNING: Failed to load model: {e}")
        print("  Server will start but segmentation will not work")
    
    print("\n" + "="*50)
    print("BeardAR Backend Ready!")
    print("="*50 + "\n")
    
    yield
    
    # Shutdown
    print("\nShutting down BeardAR Backend...")


# Create FastAPI app
app = FastAPI(
    title="BeardAR API",
    description="AI-powered beard segmentation and AR overlay API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Endpoints ====================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    model_loader = get_model_loader()
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader.is_loaded(),
        device=str(model_loader.device) if model_loader.device else "not initialized",
        model_type=model_loader.model_type,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    model_loader = get_model_loader()
    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "degraded",
        model_loaded=model_loader.is_loaded(),
        device=str(model_loader.device) if model_loader.device else "not initialized",
        model_type=model_loader.model_type,
    )


# ==================== Segmentation Endpoints ====================

@app.post("/segment", response_model=SegmentationResponse)
async def segment_beard(request: SegmentationRequest):
    """
    Segment beard from image using SAM with point prompts.
    
    **Input:**
    - `image`: Base64-encoded image
    - `user_prompts`: List of {x, y, label} point prompts (label: 1=beard, 0=not beard)
    - `face_mesh_landmarks`: Optional 468 face mesh coordinates for vertex containment
    - `return_boundary`: Whether to extract boundary vertices
    
    **Output:**
    - `mask`: Base64 PNG and RLE encoded mask
    - `vertex_containment`: Which face mesh vertices are inside the beard
    - `processing_time_ms`: Processing duration
    """
    model_loader = get_model_loader()
    
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details."
        )
    
    try:
        response = await process_segmentation(request, model_loader)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(e)}"
        )


@app.post("/segment/live", response_model=SegmentationResponse)
async def segment_beard_live(request: SegmentationRequest):
    """
    Live beard segmentation using direct SAM.
    
    This runs SAM directly on the provided image frame using
    prompts derived from face mesh landmarks. This gives much
    better quality than projecting stored vertex maps.
    
    **Use this for real-time projection in the web app.**
    
    **Input:**
    - `image`: Base64-encoded image frame
    - `face_mesh_landmarks`: 468 face mesh coordinates (normalized or pixel)
    - `user_prompts`: Optional additional point prompts
    
    **Output:**
    - `mask`: Base64 PNG mask
    - `contour_points`: Simplified outline for drawing
    - `processing_time_ms`: Processing duration
    """
    model_loader = get_model_loader()
    
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details."
        )
    
    try:
        response = await process_live_segmentation(request, model_loader)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Live segmentation failed: {str(e)}"
        )


# ==================== Calibration Endpoints ====================

@app.post("/calibrate/fuse", response_model=FusionResponse)
async def fuse_calibrations(request: FusionRequest):
    """
    Fuse multiple calibration captures into a single beard map.
    
    Use after capturing center, left, and right views during Face ID-style setup.
    
    **Input:**
    - `calibration_steps`: List of captures with beard_indices and boundary_indices
    - `voting_threshold`: Minimum votes for a vertex to be included (default: 1)
    
    **Output:**
    - `final_beard_indices`: Union of all beard vertices meeting threshold
    - `final_boundary_indices`: Union of all boundary vertices meeting threshold
    - `vertex_vote_counts`: How many times each vertex was detected
    """
    try:
        final_beard, final_boundary, votes = fuse_calibration_captures(
            [step.model_dump() for step in request.calibration_steps],
            request.voting_threshold,
        )
        
        return FusionResponse(
            final_beard_indices=final_beard,
            final_boundary_indices=final_boundary,
            vertex_vote_counts=votes,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fusion failed: {str(e)}"
        )


# ==================== Beard Template Building Endpoints ====================

@app.post("/template/add-frame")
async def add_frame_to_template(
    template_id: str,
    request: SegmentationRequest
):
    """
    Add a single frame to the beard template being built.

    Call this many times during the scan phase (e.g., 50-100 frames).
    Frames are stored for later batch SAM processing.

    **Input:**
    - `template_id`: Unique ID for this template (e.g., user ID + timestamp)
    - `request`: Image + face mesh landmarks

    **Output:**
    - Frame count so far
    """
    # Log to file
    import time
    from pathlib import Path
    log_file = Path(__file__).parent.parent / "backend.log"
    
    def log(msg):
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
                f.flush()
            print(f"[ADD_FRAME] {msg}")
        except:
            print(f"[ADD_FRAME] {msg}")
    
    log(f"[ADD_FRAME] ===== STORING FRAME =====")
    log(f"[ADD_FRAME] template_id={template_id}")

    try:
        # Validate input
        if not request.face_mesh_landmarks or len(request.face_mesh_landmarks) < 468:
            raise HTTPException(status_code=400, detail="Face mesh landmarks required")

        # Store frame data for later batch processing
        frame_data = {
            'image': request.image,
            'face_mesh_landmarks': request.face_mesh_landmarks,
            'timestamp': time.time()
        }

        # Store in memory cache for this session
        if not hasattr(add_frame_to_template, '_frame_cache'):
            add_frame_to_template._frame_cache = {}
        if template_id not in add_frame_to_template._frame_cache:
            add_frame_to_template._frame_cache[template_id] = []
        add_frame_to_template._frame_cache[template_id].append(frame_data)

        frame_count = len(add_frame_to_template._frame_cache[template_id])
        log(f"[ADD_FRAME] Frame stored, total frames: {frame_count}")

        return {
            "frame_count": frame_count,
            "message": "Frame stored for processing"
        }
        try:
            from api.frame_storage import save_frames
            saved_path = save_frames(template_id, add_frame_to_template._frame_cache[template_id])
            log(f"[ADD_FRAME] SUCCESS: Saved {builder.frame_count} frames to {saved_path}")
            print(f"[SAVE] Saved {builder.frame_count} frames to disk for {template_id}")
        except Exception as save_err:
            import traceback
            # Sanitize error message to avoid Unicode issues
            error_str = str(save_err).encode('ascii', 'ignore').decode('ascii')
            log(f"[ADD_FRAME] SAVE ERROR: {error_str}")
            print(f"[SAVE ERROR] Failed to save frames: {error_str}")
            print("Full traceback available in backend.log")
        
        log(f"[ADD_FRAME] Returning success, frame_count={builder.frame_count}")
        return {
            "success": True,
            "frame_count": builder.frame_count,
            "confidence": confidence,
        }
        
    except Exception as e:
        import traceback
        log(f"[ADD_FRAME] EXCEPTION: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/template/finalize")
async def finalize_template(template_id: str, threshold: float = 0.4):
    """
    Finalize the beard template after collecting all frames.
    
    This averages all accumulated masks into a smooth, interpolated template.
    
    **Input:**
    - `template_id`: ID of the template to finalize
    - `threshold`: Probability threshold for binary mask (default 0.4)
    
    **Output:**
    - Template data including contour points
    """
    # Write directly to log file first - do this BEFORE anything else
    import time
    from pathlib import Path
    import traceback
    
    # Calculate log file path - backend/main.py -> backend/ -> root/backend.log
    log_file = Path(__file__).parent.parent / "backend.log"
    
    def log(msg):
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
                f.flush()  # Force write
            print(f"[LOG] {msg}")
        except Exception as e:
            print(f"[LOG ERROR] Failed to write log: {e}")
    
    try:
        log(f"[FINALIZE_ENDPOINT] ===== STARTING FINALIZE =====")
        log(f"[FINALIZE_ENDPOINT] template_id={template_id}, threshold={threshold}")
        log(f"[FINALIZE_ENDPOINT] log_file path: {log_file.absolute()}")
    
        builder = get_beard_template(template_id)
        log(f"[FINALIZE_ENDPOINT] Got builder: {builder is not None}")
        
        if not builder:
            log("[FINALIZE_ENDPOINT] ERROR: Builder not found")
            raise HTTPException(status_code=404, detail="Template not found")
        
        log(f"[FINALIZE_ENDPOINT] Builder frame_count={builder.frame_count}")
        log(f"[FINALIZE_ENDPOINT] Builder type: {type(builder)}")
        log(f"[FINALIZE_ENDPOINT] Builder has mask_sum: {hasattr(builder, 'mask_sum')}")
        if hasattr(builder, 'mask_sum'):
            log(f"[FINALIZE_ENDPOINT] mask_sum shape: {builder.mask_sum.shape if hasattr(builder.mask_sum, 'shape') else 'N/A'}")
        
        if builder.frame_count == 0:
            log("[FINALIZE_ENDPOINT] ERROR: frame_count is 0")
            raise HTTPException(status_code=400, detail="No frames added to template")
        
        from api.beard_template import log_to_file
        log_to_file(f"[FINALIZE_ENDPOINT] Starting finalize for template_id={template_id}, threshold={threshold}")
        log_to_file(f"[FINALIZE_ENDPOINT] Builder frame_count={builder.frame_count}")
        
        # Save frames before finalizing (if we have them cached) - FINAL SAVE
        if hasattr(add_frame_to_template, '_frame_cache') and template_id in add_frame_to_template._frame_cache:
            frames = add_frame_to_template._frame_cache[template_id]
            log(f"[FINALIZE_ENDPOINT] Final save: {len(frames)} cached frames")
            log_to_file(f"[FINALIZE_ENDPOINT] Saving {len(frames)} cached frames")
            save_frames(template_id, frames)
            # Don't clear cache - keep it in case we need to retry
        else:
            log(f"[FINALIZE_ENDPOINT] No cached frames in memory, checking disk...")
            log_to_file(f"[FINALIZE_ENDPOINT] No cached frames found")
            # Try to load from disk if available
            from api.frame_storage import load_frames
            saved_frames = load_frames(template_id)
            if saved_frames:
                log(f"[FINALIZE_ENDPOINT] Found {len(saved_frames)} frames on disk")
                log_to_file(f"[FINALIZE_ENDPOINT] Loaded {len(saved_frames)} frames from disk")
        
        log_to_file(f"[FINALIZE_ENDPOINT] Calling builder.finalize()")
        try:
            binary_mask, contour = builder.finalize(threshold)
            log_to_file(f"[FINALIZE_ENDPOINT] finalize() returned: binary_mask shape={binary_mask.shape}, contour length={len(contour)}")

            # Save the finalized template for projection
            from api.beard_template import save_template
            save_template(template_id, builder)
            log_to_file(f"[FINALIZE_ENDPOINT] Template saved for projection")
        except Exception as finalize_error:
            log_to_file(f"[FINALIZE_ENDPOINT] finalize() failed: {finalize_error}")
            # Still try to create template data if possible
            if hasattr(builder, 'mask_sum') and builder.mask_sum.sum() > 0:
                log_to_file(f"[FINALIZE_ENDPOINT] Attempting to create template from partial data")
                try:
                    # Try with a more lenient threshold
                    binary_mask, contour = builder.finalize(min(threshold + 0.2, 0.8))
                    log_to_file(f"[FINALIZE_ENDPOINT] Fallback finalize succeeded")

                    # Save the finalized template for projection
                    from api.beard_template import save_template
                    save_template(template_id, builder)
                    log_to_file(f"[FINALIZE_ENDPOINT] Template saved for projection (fallback)")
                except Exception as fallback_error:
                    log_to_file(f"[FINALIZE_ENDPOINT] Fallback finalize also failed: {fallback_error}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Template finalization failed. Frames are saved and can be reprocessed later. Error: {str(finalize_error)}"
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"No valid frame data accumulated. Please scan again. Error: {str(finalize_error)}"
                )

        log_to_file(f"[FINALIZE_ENDPOINT] Calling builder.to_dict()")
        try:
            template_data = builder.to_dict()
            log_to_file(f"[FINALIZE_ENDPOINT] to_dict() returned with keys: {list(template_data.keys())}")
        except Exception as to_dict_error:
            log_to_file(f"[FINALIZE_ENDPOINT] to_dict() failed: {to_dict_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Template serialization failed. Frames are saved and can be reprocessed later. Error: {str(to_dict_error)}"
            )
        
        # Ensure all values are JSON serializable
        response_data = {
            "success": True,
            "frame_count": int(builder.frame_count),
            "contour_points": [[float(x), float(y)] for x, y in contour] if contour else [],
            "template_data": template_data,
        }
        
        log_to_file(f"[FINALIZE_ENDPOINT] Success! Returning response")
        log(f"[FINALIZE_ENDPOINT] ===== SUCCESS =====")
        return response_data
        
    except HTTPException as e:
        log(f"[FINALIZE_ENDPOINT] HTTPException: {e.status_code} - {e.detail}")
        raise
    except Exception as e:
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        log(f"[FINALIZE_ENDPOINT] ===== EXCEPTION CAUGHT =====")
        log(f"[FINALIZE_ENDPOINT] Exception type: {type(e).__name__}")
        log(f"[FINALIZE_ENDPOINT] Exception message: {str(e)}")
        log(f"[FINALIZE_ENDPOINT] Full traceback:\n{error_detail}")
        try:
            from api.beard_template import log_to_file
            log_to_file(f"[FINALIZE_ENDPOINT] ERROR: {error_detail}")
        except Exception as log_err:
            log(f"[FINALIZE_ENDPOINT] Failed to use log_to_file: {log_err}")
        raise HTTPException(status_code=500, detail=f"Finalization failed: {str(e)}")


@app.post("/template/process-frames")
async def process_stored_frames(
    template_id: str,
    threshold: float = 0.4
):
    """
    Process all stored frames with SAM to build comprehensive beard model.

    Call this after scanning is complete, before finalization.
    Runs SAM on all collected frames to create the full beard model.

    **Input:**
    - `template_id`: Template ID with stored frames
    - `threshold`: SAM confidence threshold

    **Output:**
    - Processing status and frame count
    """
    import time
    from pathlib import Path
    log_file = Path(__file__).parent.parent / "backend.log"

    def log(msg):
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
                f.flush()
            print(f"[PROCESS_FRAMES] {msg}")
        except:
            print(f"[PROCESS_FRAMES] {msg}")

    log(f"[PROCESS_FRAMES] ===== PROCESSING STORED FRAMES =====")
    log(f"[PROCESS_FRAMES] template_id={template_id}, threshold={threshold}")

    # Check if frames are stored
    if not hasattr(add_frame_to_template, '_frame_cache') or template_id not in add_frame_to_template._frame_cache:
        log("[PROCESS_FRAMES] ERROR: No stored frames found")
        raise HTTPException(status_code=404, detail="No stored frames found for this template")

    stored_frames = add_frame_to_template._frame_cache[template_id]
    frame_count = len(stored_frames)

    if frame_count == 0:
        log("[PROCESS_FRAMES] ERROR: No frames to process")
        raise HTTPException(status_code=400, detail="No frames to process")

    log(f"[PROCESS_FRAMES] Processing {frame_count} stored frames")

    try:
        # Load SAM model
        model_loader = get_model_loader()
        if not model_loader.is_loaded():
            log("[PROCESS_FRAMES] ERROR: Model not loaded")
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Get or create template builder
        builder = get_or_create_template(template_id)

        # Process each stored frame with SAM
        processed_count = 0
        total_confidence = 0.0

        for i, frame_data in enumerate(stored_frames):
            try:
                log(f"[PROCESS_FRAMES] Processing frame {i+1}/{frame_count}")

                # Decode image
                from api.segmentation import decode_base64_image
                from api.live_segmentation import generate_beard_prompts_from_landmarks
                import numpy as np

                image = decode_base64_image(frame_data['image'])
                h, w = image.shape[:2]

                # Get landmarks
                landmarks = np.array(frame_data['face_mesh_landmarks'])
                if landmarks.max() <= 1.0:
                    landmarks[:, 0] *= w
                    landmarks[:, 1] *= h

                # Generate prompts and run SAM
                pos_prompts, neg_prompts = generate_beard_prompts_from_landmarks(
                    frame_data['face_mesh_landmarks'], w, h
                )

                all_points = np.array([list(p) for p in pos_prompts] + [list(p) for p in neg_prompts])
                all_labels = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))

                masks, scores, _ = model_loader.predict(
                    image=image,
                    point_coords=all_points,
                    point_labels=all_labels,
                    multimask_output=True,
                )

                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                confidence = float(scores[best_idx])

                # Add to template builder
                builder.add_frame(mask, landmarks, confidence, is_mirrored=True)

                processed_count += 1
                total_confidence += confidence

                log(f"[PROCESS_FRAMES] Frame {i+1} processed, confidence: {confidence:.3f}")

            except Exception as frame_error:
                log(f"[PROCESS_FRAMES] ERROR processing frame {i+1}: {str(frame_error)}")
                continue

        avg_confidence = total_confidence / processed_count if processed_count > 0 else 0

        log(f"[PROCESS_FRAMES] Completed: {processed_count}/{frame_count} frames processed")
        log(f"[PROCESS_FRAMES] Average confidence: {avg_confidence:.3f}")

        # Save the processed template
        from api.beard_template import save_template
        save_template(template_id, builder)

        # Clear the stored frames (no longer needed)
        del add_frame_to_template._frame_cache[template_id]

        return {
            "success": True,
            "frames_processed": processed_count,
            "total_frames": frame_count,
            "average_confidence": avg_confidence,
            "template_saved": True
        }

    except Exception as e:
        log(f"[PROCESS_FRAMES] ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Frame processing failed: {str(e)}")


@app.get("/template/list")
async def list_templates():
    """List all available templates."""
    from api.beard_template import _templates
    return {"templates": list(_templates.keys())}


@app.post("/template/project")
async def project_template(
    template_id: str,
    landmarks: List[List[float]] = Body(...),
    image_width: int = Body(...),
    image_height: int = Body(...),
    is_mirrored: bool = Body(True)
):
    """
    Project a finalized template onto a face.

    This uses pose-specific SAM masks for accurate projection!
    Very fast (~1ms) compared to live SAM (~100ms).

    **Input:**
    - `template_id`: ID of the finalized template
    - `landmarks`: Current face mesh landmarks (468 points)
    - `image_width`, `image_height`: Size of target image
    - `is_mirrored`: Whether the webcam feed is mirrored

    **Output:**
    - Contour points in image coordinates for drawing
    """
    # Enhanced logging
    import time
    from pathlib import Path
    log_file = Path(__file__).parent.parent / "backend.log"

    def log(msg):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} - {msg}\n")
                f.flush()
        except:
            pass  # Ignore file write errors
        print(f"[{timestamp}] [PROJECT] {msg}")

    log(f"===== STARTING PROJECTION =====")
    log(f"template_id={template_id}, image_size={image_width}x{image_height}")
    log(f"landmarks received: {len(landmarks)} points")

    try:
        builder = get_beard_template(template_id)
        log(f"[PROJECT] Template found: {builder is not None}")

        if not builder:
            log("[PROJECT] ERROR: Template not found")
            raise HTTPException(status_code=404, detail="Template not found")

        if not hasattr(builder, 'final_template') or builder.final_template is None:
            log("[PROJECT] ERROR: Template not finalized")
            raise HTTPException(status_code=400, detail="Template not finalized")

        import numpy as np
        landmarks_array = np.array(landmarks)
        log(f"[PROJECT] Landmarks array shape: {landmarks_array.shape}")

        # Convert normalized to pixel if needed
        if landmarks_array.max() <= 1.0:
            landmarks_array[:, 0] *= image_width
            landmarks_array[:, 1] *= image_height
            log("[PROJECT] Converted normalized landmarks to pixels")

        log(f"[PROJECT] Calling project_to_image with is_mirrored={is_mirrored}")
        projected_mask, contour = builder.project_to_image(
            landmarks_array,
            (image_height, image_width),
            is_mirrored=is_mirrored
        )

        log(f"[PROJECT] Projection complete - mask shape: {projected_mask.shape}, contour points: {len(contour)}")

        result = {
            "success": True,
            "contour_points": contour,
            "debug_info": {
                "mask_shape": projected_mask.shape,
                "contour_count": len(contour),
                "template_type": "pose_specific" if hasattr(builder, 'pose_masks') and len(builder.pose_masks) > 0 else "legacy"
            }
        }

        log(f"[PROJECT] Returning result: {len(contour)} contour points")
        return result

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        log(f"[PROJECT] ERROR: {str(e)}")
        log(f"[PROJECT] Traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"Projection failed: {str(e)}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/template/{template_id}")
async def remove_template(template_id: str):
    """Delete a template."""
    delete_template(template_id)
    delete_scan(template_id)  # Also delete saved frames
    return {"success": True}


# ==================== Frame Storage Endpoints ====================

@app.get("/scans")
async def list_scans():
    """List all saved scans."""
    scans = list_saved_scans()
    return {"scans": scans}


@app.post("/scans/{template_id}/load")
async def load_scan_frames(template_id: str):
    """
    Load saved frames and reprocess them into a template.

    This allows you to skip the scan phase and just process saved frames.
    """
    frames = load_frames(template_id)
    if not frames:
        raise HTTPException(status_code=404, detail="Scan not found")

    model_loader = get_model_loader()
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from api.segmentation import decode_base64_image
        from api.live_segmentation import generate_beard_prompts_from_landmarks
        import numpy as np
        import cv2

        # Create new template builder
        builder = get_or_create_template(template_id)
        builder.frame_count = 0  # Reset
        builder.mask_sum.fill(0)
        builder.weight_sum.fill(0)

        # Process each saved frame
        for i, frame_data in enumerate(frames):
            image = decode_base64_image(frame_data["image"])
            h, w = image.shape[:2]

            landmarks = np.array(frame_data["face_mesh_landmarks"])
            if landmarks.max() <= 1.0:
                landmarks[:, 0] *= w
                landmarks[:, 1] *= h

            # If we have a saved mask, use it; otherwise run SAM
            if "sam_mask_base64" in frame_data and frame_data["sam_mask_base64"]:
                # Decode saved mask
                mask_bytes = base64.b64decode(frame_data["sam_mask_base64"])
                mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
                mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE) / 255.0
                confidence = frame_data.get("sam_confidence", 0.8)
            else:
                # Run SAM
                pos_prompts, neg_prompts = generate_beard_prompts_from_landmarks(
                    frame_data["face_mesh_landmarks"], w, h
                )
                all_points = np.array([list(p) for p in pos_prompts] + [list(p) for p in neg_prompts])
                all_labels = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))

                masks, scores, _ = model_loader.predict(
                    image=image,
                    point_coords=all_points,
                    point_labels=all_labels,
                    multimask_output=True,
                )
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                confidence = float(scores[best_idx])

            builder.add_frame(mask, landmarks, confidence, is_mirrored=True)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(frames)} frames")

        return {
            "success": True,
            "frame_count": builder.frame_count,
            "message": f"Loaded and processed {len(frames)} frames"
        }

    except Exception as e:
        import traceback
        print(f"Load scan error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scans/{template_id}/finalize")
async def finalize_saved_scan(template_id: str, threshold: float = 0.4):
    """
    Load saved frames and finalize them into a complete template.

    This combines loading and finalizing in one step for convenience.
    """
    # First load the frames
    frames = load_frames(template_id)
    if not frames:
        raise HTTPException(status_code=404, detail="Scan not found")

    model_loader = get_model_loader()
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from api.segmentation import decode_base64_image
        from api.live_segmentation import generate_beard_prompts_from_landmarks
        import numpy as np
        import cv2

        # Create new template builder
        builder = get_or_create_template(template_id)
        builder.frame_count = 0  # Reset
        builder.mask_sum.fill(0)
        builder.weight_sum.fill(0)

        # Process each saved frame
        for i, frame_data in enumerate(frames):
            image = decode_base64_image(frame_data["image"])
            h, w = image.shape[:2]

            landmarks = np.array(frame_data["face_mesh_landmarks"])
            if landmarks.max() <= 1.0:
                landmarks[:, 0] *= w
                landmarks[:, 1] *= h

            # If we have a saved mask, use it; otherwise run SAM
            if "sam_mask_base64" in frame_data and frame_data["sam_mask_base64"]:
                # Decode saved mask
                mask_bytes = base64.b64decode(frame_data["sam_mask_base64"])
                mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
                mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE) / 255.0
                confidence = frame_data.get("sam_confidence", 0.8)
            else:
                # Run SAM
                pos_prompts, neg_prompts = generate_beard_prompts_from_landmarks(
                    frame_data["face_mesh_landmarks"], w, h
                )
                all_points = np.array([list(p) for p in pos_prompts] + [list(p) for p in neg_prompts])
                all_labels = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))

                masks, scores, _ = model_loader.predict(
                    image=image,
                    point_coords=all_points,
                    point_labels=all_labels,
                    multimask_output=True,
                )
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                confidence = float(scores[best_idx])

            builder.add_frame(mask, landmarks, confidence, is_mirrored=True)

        # Now finalize
        from api.beard_template import log_to_file
        log_to_file(f"[FINALIZE_SAVED_SCAN] Starting finalize for saved scan {template_id}")

        try:
            binary_mask, contour = builder.finalize(threshold)
            template_data = builder.to_dict()

            response_data = {
                "success": True,
                "frame_count": int(builder.frame_count),
                "contour_points": [[float(x), float(y)] for x, y in contour] if contour else [],
                "template_data": template_data,
                "message": f"Successfully finalized saved scan with {len(frames)} frames"
            }

            log_to_file(f"[FINALIZE_SAVED_SCAN] Success for {template_id}")
            return response_data

        except Exception as finalize_error:
            log_to_file(f"[FINALIZE_SAVED_SCAN] Finalize failed: {finalize_error}")
            # Try fallback
            if hasattr(builder, 'mask_sum') and builder.mask_sum.sum() > 0:
                try:
                    binary_mask, contour = builder.finalize(min(threshold + 0.2, 0.8))
                    template_data = builder.to_dict()
                    return {
                        "success": True,
                        "frame_count": int(builder.frame_count),
                        "contour_points": [[float(x), float(y)] for x, y in contour] if contour else [],
                        "template_data": template_data,
                        "message": f"Finalized with fallback threshold after initial failure"
                    }
                except Exception as fallback_error:
                    log_to_file(f"[FINALIZE_SAVED_SCAN] Fallback also failed: {fallback_error}")

            raise HTTPException(
                status_code=500,
                detail=f"Failed to finalize saved scan. Original error: {str(finalize_error)}"
            )

    except Exception as e:
        import traceback
        print(f"Finalize saved scan error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/scans/{template_id}")
async def delete_saved_scan(template_id: str):
    """Delete a saved scan."""
    deleted = delete_scan(template_id)
    return {"success": deleted, "message": "Scan deleted" if deleted else "Scan not found"}


# ==================== Template Storage Endpoints ====================

# In-memory storage for demo (use database in production)
_templates = {}


@app.post("/templates", response_model=CaptureObject)
async def save_template(template: CaptureObject):
    """Save a beard template for later projection."""
    key = f"{template.user_id}:{template.template_name}"
    template.created_at = datetime.utcnow().isoformat() + "Z"
    _templates[key] = template
    return template


@app.get("/templates/{user_id}/{template_name}", response_model=CaptureObject)
async def get_template(user_id: str, template_name: str):
    """Retrieve a saved beard template."""
    key = f"{user_id}:{template_name}"
    if key not in _templates:
        raise HTTPException(status_code=404, detail="Template not found")
    return _templates[key]


@app.get("/templates/{user_id}")
async def list_templates(user_id: str):
    """List all templates for a user."""
    user_templates = [
        t for k, t in _templates.items() 
        if k.startswith(f"{user_id}:")
    ]
    return {"templates": user_templates}


@app.delete("/templates/{user_id}/{template_name}")
async def delete_template(user_id: str, template_name: str):
    """Delete a beard template."""
    key = f"{user_id}:{template_name}"
    if key not in _templates:
        raise HTTPException(status_code=404, detail="Template not found")
    del _templates[key]
    return {"status": "deleted"}


# ==================== Run Server ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

