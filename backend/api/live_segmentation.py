"""
Live SAM Segmentation API

This endpoint runs SAM directly on the incoming frame,
using learned prompt positions from the user's template.
This gives much better quality than projecting stored vertices.
"""

import base64
import io
import time
from typing import List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from .schemas import (
    SegmentationRequest,
    SegmentationResponse,
    MaskResponse,
    VertexContainmentResult,
)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array (RGB)."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return np.array(image)


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    success, buffer = cv2.imencode('.png', mask_uint8)
    return base64.b64encode(buffer).decode('utf-8')


def generate_beard_prompts_from_landmarks(
    landmarks: List[List[float]], 
    width: int, 
    height: int
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Generate SAM prompts from face mesh landmarks.
    Returns (positive_prompts, negative_prompts)
    """
    # Key landmark indices
    CHIN = 152
    JAW_LEFT = 234
    JAW_RIGHT = 454
    FOREHEAD = 10
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE_TIP = 4
    UPPER_LIP = 13
    
    # Convert normalized to pixel if needed
    def to_pixel(lm):
        x, y = lm
        if x <= 1.0 and y <= 1.0:
            return (x * width, y * height)
        return (x, y)
    
    chin = np.array(to_pixel(landmarks[CHIN]))
    jaw_left = np.array(to_pixel(landmarks[JAW_LEFT]))
    jaw_right = np.array(to_pixel(landmarks[JAW_RIGHT]))
    forehead = np.array(to_pixel(landmarks[FOREHEAD]))
    left_eye = np.array(to_pixel(landmarks[LEFT_EYE]))
    right_eye = np.array(to_pixel(landmarks[RIGHT_EYE]))
    nose = np.array(to_pixel(landmarks[NOSE_TIP]))
    upper_lip = np.array(to_pixel(landmarks[UPPER_LIP]))
    
    # Positive prompts (beard area)
    positive = [
        tuple(chin),
        tuple((chin + jaw_left) / 2),
        tuple((chin + jaw_right) / 2),
        tuple((chin + upper_lip) / 2),  # Mustache area
        tuple(chin + np.array([0, -20])),  # Just above chin
    ]
    
    # Negative prompts (not beard)
    negative = [
        tuple(forehead),
        tuple(left_eye),
        tuple(right_eye),
        tuple(nose),  # Nose itself is not beard
    ]
    
    return positive, negative


async def process_live_segmentation(
    request: SegmentationRequest,
    model_loader,
) -> SegmentationResponse:
    """
    Process a live segmentation request using direct SAM.
    
    This runs SAM on the actual image frame using prompts
    derived from face mesh landmarks - giving much better
    quality than projecting stored vertex maps.
    """
    start_time = time.time()
    
    # Decode image
    image = decode_base64_image(request.image)
    h, w = image.shape[:2]
    
    # If landmarks provided, use them to generate smart prompts
    if request.face_mesh_landmarks and len(request.face_mesh_landmarks) >= 468:
        pos_prompts, neg_prompts = generate_beard_prompts_from_landmarks(
            request.face_mesh_landmarks, w, h
        )
        
        # Combine with any user-provided prompts
        point_coords = []
        point_labels = []
        
        for pt in pos_prompts:
            point_coords.append([pt[0], pt[1]])
            point_labels.append(1)
        
        for pt in neg_prompts:
            point_coords.append([pt[0], pt[1]])
            point_labels.append(0)
        
        # Add user prompts if any
        for p in request.user_prompts:
            point_coords.append([p.x, p.y])
            point_labels.append(p.label)
        
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
    else:
        # Use only user-provided prompts
        point_coords = np.array([[p.x, p.y] for p in request.user_prompts])
        point_labels = np.array([p.label for p in request.user_prompts])
    
    # Run SAM
    masks, scores, _ = model_loader.predict(
        image=image,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    
    # Get best mask
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    confidence = float(scores[best_idx])
    
    # Encode mask
    mask_base64 = encode_mask_to_base64(best_mask)
    
    # Get mask contours for drawing
    contours, _ = cv2.findContours(
        best_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Convert contours to list of points
    contour_points = []
    if contours:
        # Get the largest contour
        largest = max(contours, key=cv2.contourArea)
        # Simplify contour
        epsilon = 0.005 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        contour_points = approx.reshape(-1, 2).tolist()
    
    processing_time = (time.time() - start_time) * 1000
    
    return SegmentationResponse(
        success=True,
        mask=MaskResponse(
            mask_base64=mask_base64,
            mask_rle=None,
            confidence=confidence,
            width=w,
            height=h,
        ),
        vertex_containment=None,  # Not using vertices anymore
        processing_time_ms=processing_time,
        contour_points=contour_points,  # New field for drawing
    )

