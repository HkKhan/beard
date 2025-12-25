"""
Beard Template Creation and Projection

This module handles:
1. Accumulating SAM masks from multiple frames during scan
2. Creating a smoothed, interpolated beard map in canonical face space
3. Projecting the template onto new faces via affine warping (no live SAM)
"""

import base64
import io
import time
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass, field
from collections import defaultdict

# Log file path
LOG_FILE = Path(__file__).parent.parent.parent / "backend.log"

def log_to_file(message: str):
    """Write log message to file and print to console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

# Canonical face size for normalization
CANONICAL_SIZE = 512

# Key landmark indices
LANDMARKS = {
    'left_eye': 33,
    'right_eye': 263,
    'nose_tip': 4,
    'chin': 152,
    'jaw_left': 234,
    'jaw_right': 454,
    'forehead': 10,
    'upper_lip': 13,
}


@dataclass
class BeardTemplateBuilder:
    """Accumulates SAM masks to build a smoothed beard template."""
    
    # Accumulated mask in canonical space
    mask_sum: np.ndarray = field(default_factory=lambda: np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.float64))
    weight_sum: np.ndarray = field(default_factory=lambda: np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.float64))
    
    # Frame count
    frame_count: int = 0
    
    # Store the final template
    final_template: Optional[np.ndarray] = None
    final_contour: Optional[List[List[float]]] = None
    
    def compute_alignment_transform(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute affine transform to align face to canonical position."""
        left_eye = landmarks[LANDMARKS['left_eye']]
        right_eye = landmarks[LANDMARKS['right_eye']]
        
        # Eye center
        eye_center = (left_eye + right_eye) / 2
        
        # Rotation angle from eye line
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Scale based on eye distance
        eye_dist = np.linalg.norm(right_eye - left_eye)
        if eye_dist < 10:
            eye_dist = 100  # Fallback
        scale = CANONICAL_SIZE * 0.3 / eye_dist
        
        # Transform matrix
        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)
        
        # Adjust translation to center face
        M[0, 2] += CANONICAL_SIZE / 2 - eye_center[0]
        M[1, 2] += CANONICAL_SIZE / 3 - eye_center[1]
        
        return M
    
    def add_frame(
        self,
        mask: np.ndarray,
        landmarks: np.ndarray,
        confidence: float,
        is_mirrored: bool = True
    ):
        """
        Add a SAM mask from one frame to the accumulator.
        
        Args:
            mask: Binary mask from SAM (H x W)
            landmarks: 468 face landmarks in pixel coordinates
            confidence: SAM confidence score
            is_mirrored: Whether the input is from mirrored webcam
        """
        h, w = mask.shape
        
        # If mirrored, flip the mask and landmarks
        if is_mirrored:
            mask = np.fliplr(mask)
            landmarks = landmarks.copy()
            landmarks[:, 0] = w - landmarks[:, 0]
        
        # Compute alignment transform
        M = self.compute_alignment_transform(landmarks)
        
        # Warp mask to canonical space
        aligned_mask = cv2.warpAffine(
            mask.astype(np.float32), 
            M, 
            (CANONICAL_SIZE, CANONICAL_SIZE),
            flags=cv2.INTER_LINEAR
        )
        
        # Create weight mask (center has more weight, edges fade)
        weight = np.ones_like(aligned_mask) * confidence
        # Fade edges to reduce artifacts
        fade = 20
        weight[:fade, :] *= np.linspace(0, 1, fade)[:, None]
        weight[-fade:, :] *= np.linspace(1, 0, fade)[:, None]
        weight[:, :fade] *= np.linspace(0, 1, fade)[None, :]
        weight[:, -fade:] *= np.linspace(1, 0, fade)[None, :]
        
        # Accumulate
        self.mask_sum += aligned_mask * weight
        self.weight_sum += weight
        self.frame_count += 1
    
    def finalize(self, threshold: float = 0.4) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Finalize the template by averaging all accumulated masks.
        
        Returns:
            (binary_mask, contour_points)
        """
        log_to_file(f"[FINALIZE] Starting finalize with frame_count={self.frame_count}, threshold={threshold}")
        
        if self.frame_count == 0:
            log_to_file("[FINALIZE] ERROR: frame_count is 0")
            return np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8), []
        
        # Check mask_sum and weight_sum
        log_to_file(f"[FINALIZE] mask_sum shape={self.mask_sum.shape}, dtype={self.mask_sum.dtype}, min={self.mask_sum.min()}, max={self.mask_sum.max()}, sum={self.mask_sum.sum()}")
        log_to_file(f"[FINALIZE] weight_sum shape={self.weight_sum.shape}, dtype={self.weight_sum.dtype}, min={self.weight_sum.min()}, max={self.weight_sum.max()}, sum={self.weight_sum.sum()}, non-zero count={(self.weight_sum > 0.01).sum()}")
        
        if self.weight_sum.sum() == 0:
            log_to_file("[FINALIZE] ERROR: weight_sum is all zeros!")
            raise ValueError("No valid weights accumulated - all frames may have failed")
        
        # Compute probability map
        try:
            prob_map = np.divide(
                self.mask_sum,
                self.weight_sum,
                out=np.zeros_like(self.mask_sum),
                where=self.weight_sum > 0.01
            )
            log_to_file(f"[FINALIZE] prob_map computed: min={prob_map.min()}, max={prob_map.max()}, mean={prob_map.mean()}, non-zero={(prob_map > 0).sum()}")
        except Exception as e:
            log_to_file(f"[FINALIZE] ERROR in prob_map computation: {e}")
            import traceback
            log_to_file(traceback.format_exc())
            raise
        
        # Apply morphological smoothing
        try:
            prob_map_uint8 = (prob_map * 255).astype(np.uint8)
            log_to_file(f"[FINALIZE] prob_map_uint8: min={prob_map_uint8.min()}, max={prob_map_uint8.max()}, dtype={prob_map_uint8.dtype}")
        except Exception as e:
            log_to_file(f"[FINALIZE] ERROR converting to uint8: {e}")
            import traceback
            log_to_file(traceback.format_exc())
            raise
        
        # Gaussian blur for smoothing
        try:
            prob_map_uint8 = cv2.GaussianBlur(prob_map_uint8, (15, 15), 0)
            log_to_file(f"[FINALIZE] After blur: min={prob_map_uint8.min()}, max={prob_map_uint8.max()}")
        except Exception as e:
            log_to_file(f"[FINALIZE] ERROR in GaussianBlur: {e}")
            import traceback
            log_to_file(traceback.format_exc())
            raise
        
        # Threshold to binary
        try:
            threshold_value = int(threshold * 255)
            log_to_file(f"[FINALIZE] Thresholding with value={threshold_value}")
            _, binary = cv2.threshold(prob_map_uint8, threshold_value, 255, cv2.THRESH_BINARY)
            log_to_file(f"[FINALIZE] Binary mask: shape={binary.shape}, dtype={binary.dtype}, white_pixels={(binary > 0).sum()}, total_pixels={binary.size}")
        except Exception as e:
            log_to_file(f"[FINALIZE] ERROR in threshold: {e}")
            import traceback
            log_to_file(traceback.format_exc())
            raise
        
        if (binary > 0).sum() == 0:
            log_to_file("[FINALIZE] WARNING: Binary mask is all zeros - no beard detected!")
        
        # Morphological close to fill gaps
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            log_to_file(f"[FINALIZE] After morphology: white_pixels={(binary > 0).sum()}")
        except Exception as e:
            log_to_file(f"[FINALIZE] ERROR in morphology: {e}")
            import traceback
            log_to_file(traceback.format_exc())
            raise
        
        # Find contours
        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            log_to_file(f"[FINALIZE] Found {len(contours)} contours")
        except Exception as e:
            log_to_file(f"[FINALIZE] ERROR in findContours: {e}")
            import traceback
            log_to_file(traceback.format_exc())
            raise
        
        contour_points = []
        if contours and len(contours) > 0:
            try:
                # Get largest contour
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                log_to_file(f"[FINALIZE] Largest contour area={area}, points={len(largest)}")
                
                # Only process if contour has area
                if area > 10:
                    # Simplify
                    epsilon = 0.01 * cv2.arcLength(largest, True)
                    log_to_file(f"[FINALIZE] Simplifying with epsilon={epsilon}")
                    approx = cv2.approxPolyDP(largest, epsilon, True)
                    log_to_file(f"[FINALIZE] Simplified to {len(approx)} points")
                    if len(approx) > 0:
                        contour_points = approx.reshape(-1, 2).tolist()
                        # Ensure all values are native Python types
                        contour_points = [[float(x), float(y)] for x, y in contour_points]
                        log_to_file(f"[FINALIZE] Final contour_points: {len(contour_points)} points")
                else:
                    log_to_file(f"[FINALIZE] WARNING: Contour area {area} too small, skipping")
            except Exception as e:
                log_to_file(f"[FINALIZE] ERROR in contour processing: {e}")
                import traceback
                log_to_file(traceback.format_exc())
                contour_points = []
        else:
            log_to_file("[FINALIZE] WARNING: No contours found")
        
        self.final_template = binary
        self.final_contour = contour_points
        
        log_to_file(f"[FINALIZE] Success! Returning binary mask and {len(contour_points)} contour points")
        return binary, contour_points
    
    def project_to_image(
        self,
        landmarks: np.ndarray,
        image_size: Tuple[int, int],
        is_mirrored: bool = True
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Project the template onto a new image.
        
        Args:
            landmarks: Face landmarks in the target image (pixel coords)
            image_size: (height, width) of target image
            is_mirrored: Whether the target is mirrored webcam
            
        Returns:
            (projected_mask, contour_points)
        """
        if self.final_template is None:
            return np.zeros(image_size, dtype=np.uint8), []
        
        h, w = image_size
        landmarks_adj = landmarks.copy()
        
        # If mirrored, flip landmarks for alignment computation
        if is_mirrored:
            landmarks_adj[:, 0] = w - landmarks_adj[:, 0]
        
        # Compute transform from canonical to image space
        M = self.compute_alignment_transform(landmarks_adj)
        M_inv = cv2.invertAffineTransform(M)
        
        # Warp template to image space
        projected = cv2.warpAffine(
            self.final_template,
            M_inv,
            (w, h),
            flags=cv2.INTER_LINEAR
        )
        
        # If mirrored, flip back
        if is_mirrored:
            projected = np.fliplr(projected)
        
        # Get contour in image space
        contour_points = []
        if self.final_contour:
            for pt in self.final_contour:
                # Transform point from canonical to image
                pt_canonical = np.array([pt[0], pt[1], 1])
                pt_image = M_inv @ pt_canonical
                
                if is_mirrored:
                    pt_image[0] = w - pt_image[0]
                
                contour_points.append([float(pt_image[0]), float(pt_image[1])])
        
        return projected, contour_points
    
    def to_dict(self) -> dict:
        """Serialize template to dict for storage."""
        log_to_file(f"[TO_DICT] Starting to_dict, final_template is None: {self.final_template is None}")
        
        if self.final_template is None:
            log_to_file("[TO_DICT] final_template is None, calling finalize()")
            self.finalize()
        
        log_to_file(f"[TO_DICT] final_template shape={self.final_template.shape}, dtype={self.final_template.dtype}")
        
        # Encode template as base64 PNG
        try:
            success, buffer = cv2.imencode('.png', self.final_template)
            log_to_file(f"[TO_DICT] imencode success={success}, buffer type={type(buffer)}, buffer shape={buffer.shape if hasattr(buffer, 'shape') else 'N/A'}")
            if not success:
                raise ValueError("Failed to encode template to PNG")
            template_b64 = base64.b64encode(buffer).decode('utf-8')
            log_to_file(f"[TO_DICT] Encoded to base64, length={len(template_b64)}")
        except Exception as e:
            log_to_file(f"[TO_DICT] ERROR encoding template: {e}")
            import traceback
            log_to_file(traceback.format_exc())
            raise
        
        # Ensure contour is serializable
        contour = self.final_contour or []
        log_to_file(f"[TO_DICT] Contour points: {len(contour)}, type check: {type(contour)}")
        
        # Validate contour points are all native Python types
        try:
            validated_contour = []
            for point in contour:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    validated_contour.append([float(point[0]), float(point[1])])
                else:
                    log_to_file(f"[TO_DICT] WARNING: Invalid contour point: {point}")
            contour = validated_contour
        except Exception as e:
            log_to_file(f"[TO_DICT] ERROR validating contour: {e}")
            import traceback
            log_to_file(traceback.format_exc())
            contour = []
        
        result = {
            'template_base64': template_b64,
            'contour': contour,
            'frame_count': int(self.frame_count),  # Ensure int
            'canonical_size': int(CANONICAL_SIZE),  # Ensure int
        }
        
        log_to_file(f"[TO_DICT] Success! Returning dict with keys: {list(result.keys())}")
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BeardTemplateBuilder':
        """Load template from dict."""
        builder = cls()
        
        # Decode template
        template_bytes = base64.b64decode(data['template_base64'])
        template_array = np.frombuffer(template_bytes, dtype=np.uint8)
        builder.final_template = cv2.imdecode(template_array, cv2.IMREAD_GRAYSCALE)
        builder.final_contour = data.get('contour', [])
        builder.frame_count = data.get('frame_count', 0)
        
        return builder


# Global template storage (in production, use a database)
_templates: Dict[str, BeardTemplateBuilder] = {}


def get_or_create_template(template_id: str) -> BeardTemplateBuilder:
    """Get existing template or create new one."""
    if template_id not in _templates:
        _templates[template_id] = BeardTemplateBuilder()
    return _templates[template_id]


def get_template(template_id: str) -> Optional[BeardTemplateBuilder]:
    """Get existing template or None."""
    return _templates.get(template_id)


def save_template(template_id: str, builder: BeardTemplateBuilder):
    """Save a template."""
    _templates[template_id] = builder


def delete_template(template_id: str):
    """Delete a template."""
    if template_id in _templates:
        del _templates[template_id]

