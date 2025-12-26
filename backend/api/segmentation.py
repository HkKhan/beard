"""
Segmentation API endpoints.
Handles beard segmentation and vertex containment analysis.
"""

import base64
import io
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .schemas import (
    SegmentationRequest,
    SegmentationResponse,
    MaskResponse,
    VertexContainmentResult,
    FusionRequest,
    FusionResponse,
)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array (RGB)."""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return np.array(image)


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG."""
    # Ensure mask is uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Encode to PNG
    success, buffer = cv2.imencode('.png', mask_uint8)
    if not success:
        raise ValueError("Failed to encode mask to PNG")
    
    return base64.b64encode(buffer).decode('utf-8')


def mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to run-length encoding."""
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return {
        'counts': runs.tolist(),
        'size': list(mask.shape)
    }


def extract_boundary(mask: np.ndarray, thickness: int = 3) -> np.ndarray:
    """
    Extract boundary of mask using morphological operations.
    Returns binary boundary mask.
    """
    # Dilate and erode to get boundary
    kernel = np.ones((thickness, thickness), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = dilated - eroded
    
    return boundary


def extract_boundary_canny(mask: np.ndarray) -> np.ndarray:
    """Extract boundary using Canny edge detection."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_uint8, 50, 150)
    return edges


def check_vertex_containment(
    mask: np.ndarray,
    landmarks: List[List[float]],
    boundary_mask: Optional[np.ndarray] = None,
    boundary_threshold: float = 10.0,
) -> VertexContainmentResult:
    """
    Check which face mesh vertices are inside the mask.
    
    Args:
        mask: Binary mask [H, W]
        landmarks: List of [x, y] coordinates for 468 face mesh vertices
        boundary_mask: Optional boundary mask for identifying edge vertices
        boundary_threshold: Distance threshold for boundary vertices
    
    Returns:
        VertexContainmentResult with beard and boundary indices
    """
    h, w = mask.shape
    beard_indices = []
    boundary_indices = []
    
    for idx, (x, y) in enumerate(landmarks):
        # Convert normalized coords if necessary
        px = int(x) if x > 1 else int(x * w)
        py = int(y) if y > 1 else int(y * h)
        
        # Clamp to image bounds
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        
        # Check if vertex is inside beard mask
        if mask[py, px] > 0:
            beard_indices.append(idx)
        
        # Check if vertex is near boundary
        if boundary_mask is not None and boundary_mask[py, px] > 0:
            boundary_indices.append(idx)
    
    # If no explicit boundary vertices found, find closest to edge
    if len(boundary_indices) == 0 and len(beard_indices) > 0:
        # Use distance transform to find vertices closest to edge
        boundary = extract_boundary(mask)
        boundary_coords = np.argwhere(boundary > 0)
        
        if len(boundary_coords) > 0:
            for idx in beard_indices:
                x, y = landmarks[idx]
                px = int(x) if x > 1 else int(x * w)
                py = int(y) if y > 1 else int(y * h)
                
                # Calculate min distance to boundary
                distances = np.sqrt(
                    (boundary_coords[:, 0] - py) ** 2 + 
                    (boundary_coords[:, 1] - px) ** 2
                )
                min_dist = distances.min()
                
                if min_dist < boundary_threshold:
                    boundary_indices.append(idx)
    
    return VertexContainmentResult(
        beard_vertex_indices=beard_indices,
        boundary_vertex_indices=list(set(boundary_indices)),
        total_vertices_checked=len(landmarks),
    )


def fuse_calibration_captures(
    calibration_steps: List[dict],
    voting_threshold: int = 1,
) -> Tuple[List[int], List[int], dict]:
    """
    Fuse multiple calibration captures into final beard map.
    Uses voting system to reduce noise.
    
    Args:
        calibration_steps: List of {beard_indices, boundary_indices}
        voting_threshold: Minimum votes for inclusion
    
    Returns:
        final_beard_indices, final_boundary_indices, vote_counts
    """
    # Vote counting arrays
    beard_votes = {}
    boundary_votes = {}
    
    for step in calibration_steps:
        for idx in step['beard_indices']:
            beard_votes[idx] = beard_votes.get(idx, 0) + 1
        
        for idx in step['boundary_indices']:
            boundary_votes[idx] = boundary_votes.get(idx, 0) + 1
    
    # Filter by threshold
    final_beard = [
        idx for idx, votes in beard_votes.items() 
        if votes >= voting_threshold
    ]
    final_boundary = [
        idx for idx, votes in boundary_votes.items() 
        if votes >= voting_threshold
    ]
    
    return final_beard, final_boundary, beard_votes


async def process_segmentation(
    request: SegmentationRequest,
    model_loader,
) -> SegmentationResponse:
    """
    Process a segmentation request.
    
    Args:
        request: The segmentation request
        model_loader: The ModelLoader instance
    
    Returns:
        SegmentationResponse with mask and vertex data
    """
    start_time = time.time()
    
    # Decode image
    image = decode_base64_image(request.image)
    h, w = image.shape[:2]
    
    # Prepare point prompts
    point_coords = np.array([[p.x, p.y] for p in request.user_prompts])
    point_labels = np.array([p.label for p in request.user_prompts])
    
    # Run SAM prediction
    masks, scores, _ = model_loader.predict(
        image=image,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    
    # Get best mask
    best_mask = masks[0]
    confidence = float(scores[0])
    
    # Encode mask
    mask_base64 = encode_mask_to_base64(best_mask)
    mask_rle = mask_to_rle(best_mask)
    
    # Vertex containment analysis
    vertex_result = None
    if request.face_mesh_landmarks:
        boundary_mask = None
        if request.return_boundary:
            boundary_mask = extract_boundary_canny(best_mask)
        
        vertex_result = check_vertex_containment(
            mask=best_mask,
            landmarks=request.face_mesh_landmarks,
            boundary_mask=boundary_mask,
        )
    
    processing_time = (time.time() - start_time) * 1000
    
    return SegmentationResponse(
        success=True,
        mask=MaskResponse(
            mask_base64=mask_base64,
            mask_rle=mask_rle,
            confidence=confidence,
            width=w,
            height=h,
        ),
        vertex_containment=vertex_result,
        processing_time_ms=processing_time,
    )


