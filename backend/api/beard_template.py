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
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass, field
from collections import defaultdict

# Log file path
LOG_FILE = Path(__file__).parent.parent.parent / "backend.log"

# Template storage directory
TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True)

def log_to_file(message: str):
    """Write log message to file and print to console."""
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    except:
        pass  # Ignore file write errors
    print(f"[TEMPLATE] {message}")  # Always print to console

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
class PoseSpecificMask:
    """Stores a SAM mask for a specific face pose."""
    mask: np.ndarray  # Binary mask in original image coordinates
    landmarks: np.ndarray  # Face landmarks used for this mask
    confidence: float  # SAM confidence score
    timestamp: float  # When this mask was captured

@dataclass
class BeardTemplateBuilder:
    """Accumulates pose-specific SAM masks for true 3D beard modeling."""

    # Store individual pose-specific masks instead of accumulating
    pose_masks: List[PoseSpecificMask] = field(default_factory=list)

    # Frame count
    frame_count: int = 0

    # Store the final template (for backward compatibility)
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
        Store a pose-specific SAM mask.

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

        # Store the pose-specific mask
        pose_mask = PoseSpecificMask(
            mask=mask.astype(np.uint8),
            landmarks=landmarks.copy(),
            confidence=confidence,
            timestamp=time.time()
        )

        self.pose_masks.append(pose_mask)
        self.frame_count += 1

        log_to_file(f"Added pose-specific mask #{self.frame_count}, confidence: {confidence:.3f}")
    
    def finalize(self, threshold: float = 0.4) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Finalize the template - handle both new pose masks and backward compatibility.

        Returns:
            (binary_mask, contour_points)
        """
        if hasattr(self, 'pose_masks') and len(self.pose_masks) > 0:
            # New structure - pose-specific masks
            log_to_file(f"[FINALIZE] Finalizing with {len(self.pose_masks)} pose-specific masks")

            # Use the highest confidence mask as the representative template
            best_mask = max(self.pose_masks, key=lambda m: m.confidence).mask

            # Store as final template for any legacy code
            self.final_template = best_mask
            self.final_contour = []  # Will be computed dynamically during projection

            log_to_file(f"[FINALIZE] Template finalized with {len(self.pose_masks)} pose masks, best confidence: {max(self.pose_masks, key=lambda m: m.confidence).confidence:.3f}")
            return best_mask, []

        elif hasattr(self, 'mask_sum') and hasattr(self, 'weight_sum'):
            # Backward compatibility - old mask_sum/weight_sum structure
            log_to_file("[FINALIZE] Using backward compatibility for old template structure")

            if self.weight_sum.sum() == 0:
                log_to_file("[FINALIZE] ERROR: Old template has no valid weights")
                return np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8), []

            # Compute the old-style averaged template
            prob_map = np.divide(
                self.mask_sum,
                self.weight_sum,
                out=np.zeros_like(self.mask_sum),
                where=self.weight_sum > 0.01
            )

            prob_map_uint8 = (prob_map * 255).astype(np.uint8)
            _, binary = cv2.threshold(prob_map_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            self.final_template = binary
            self.final_contour = []

            log_to_file("[FINALIZE] Old template finalized using backward compatibility")
            return binary, []

        else:
            log_to_file("[FINALIZE] ERROR: Template has neither pose_masks nor old structure")
            return np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.uint8), []
        
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
    
    def find_best_pose_match(self, target_landmarks: np.ndarray) -> Optional[PoseSpecificMask]:
        """
        Find the best matching pose from stored masks based on landmark similarity.

        Args:
            target_landmarks: Current face landmarks

        Returns:
            Best matching PoseSpecificMask or None
        """
        log_to_file(f"Finding pose match for template with pose_masks: {len(self.pose_masks) if hasattr(self, 'pose_masks') else 'N/A'}")

        # Check if we have pose_masks (new structure) or need backward compatibility
        if hasattr(self, 'pose_masks') and len(self.pose_masks) > 0:
            # New structure - use pose matching
            log_to_file(f"Using pose matching with {len(self.pose_masks)} stored masks")
            best_match = None
            best_similarity = -1

            for i, pose_mask in enumerate(self.pose_masks):
                # Compute pose similarity based on key landmark distances
                similarity = self.compute_pose_similarity(target_landmarks, pose_mask.landmarks)
                log_to_file(f"Pose {i}: similarity {similarity:.3f}, confidence {pose_mask.confidence:.3f}")
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = pose_mask

            if best_match:
                log_to_file(f"Best pose match found with similarity: {best_similarity:.3f}")
            else:
                log_to_file("No suitable pose match found among stored masks")
            return best_match

        elif hasattr(self, 'final_template') and self.final_template is not None:
            # Backward compatibility - old template structure
            log_to_file("Using backward compatibility mode for old template structure")
            # Create a dummy PoseSpecificMask from the final template
            dummy_mask = PoseSpecificMask(
                mask=self.final_template,
                landmarks=np.zeros(468),  # Dummy landmarks
                confidence=0.5,
                timestamp=time.time()
            )
            return dummy_mask

        else:
            log_to_file(f"No pose masks ({len(self.pose_masks) if hasattr(self, 'pose_masks') else 0}) or final template available")
            return None

    def compute_pose_similarity(self, landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
        """
        Compute similarity between two sets of landmarks.
        Uses relative distances between key facial features.
        """
        try:
            # Key landmark indices
            LEFT_EYE = LANDMARKS['left_eye']
            RIGHT_EYE = LANDMARKS['right_eye']
            NOSE_TIP = LANDMARKS['nose_tip']
            CHIN = LANDMARKS['chin']

            # Extract key points
            def get_key_points(lm):
                return {
                    'left_eye': lm[LEFT_EYE],
                    'right_eye': lm[RIGHT_EYE],
                    'nose': lm[NOSE_TIP],
                    'chin': lm[CHIN]
                }

            points1 = get_key_points(landmarks1)
            points2 = get_key_points(landmarks2)

            # Compute relative distances (normalized by inter-eye distance)
            eye_dist1 = np.linalg.norm(points1['left_eye'] - points1['right_eye'])
            eye_dist2 = np.linalg.norm(points2['left_eye'] - points2['right_eye'])

            if eye_dist1 < 1 or eye_dist2 < 1:
                return 0

            # Normalized distances
            distances1 = {
                'eye_to_nose': np.linalg.norm(points1['left_eye'] - points1['nose']) / eye_dist1,
                'nose_to_chin': np.linalg.norm(points1['nose'] - points1['chin']) / eye_dist1,
                'eye_to_chin': np.linalg.norm(points1['left_eye'] - points1['chin']) / eye_dist1,
            }

            distances2 = {
                'eye_to_nose': np.linalg.norm(points2['left_eye'] - points2['nose']) / eye_dist2,
                'nose_to_chin': np.linalg.norm(points2['nose'] - points2['chin']) / eye_dist2,
                'eye_to_chin': np.linalg.norm(points2['left_eye'] - points2['chin']) / eye_dist2,
            }

            # Compute similarity (inverse of distance between normalized feature vectors)
            diff = np.array([
                distances1['eye_to_nose'] - distances2['eye_to_nose'],
                distances1['nose_to_chin'] - distances2['nose_to_chin'],
                distances1['eye_to_chin'] - distances2['eye_to_chin']
            ])

            similarity = 1.0 / (1.0 + np.linalg.norm(diff))
            return similarity

        except (IndexError, KeyError):
            return 0

    def project_to_image(
        self,
        landmarks: np.ndarray,
        image_size: Tuple[int, int],
        is_mirrored: bool = True
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """
        Project the best matching pose-specific SAM mask to the target image.

        Args:
            landmarks: Face landmarks in the target image (pixel coords)
            image_size: (height, width) of target image
            is_mirrored: Whether the target is mirrored webcam

        Returns:
            (projected_mask, contour_points)
        """
        h, w = image_size
        landmarks_adj = landmarks.copy()

        # If mirrored, flip landmarks for pose matching
        if is_mirrored:
            landmarks_adj[:, 0] = w - landmarks_adj[:, 0]

        # Find best matching pose
        best_match = self.find_best_pose_match(landmarks_adj)

        if best_match is None:
            log_to_file(f"No pose matches found for template with {len(self.pose_masks)} pose masks")
            return np.zeros((h, w), dtype=np.uint8), []

        log_to_file(f"Using pose match with confidence: {best_match.confidence:.3f}")

        # Compute transform from stored pose to current pose
        stored_landmarks = best_match.landmarks
        current_landmarks = landmarks_adj

        # Use key points for alignment
        stored_key_points = np.array([
            stored_landmarks[LANDMARKS['left_eye']],
            stored_landmarks[LANDMARKS['right_eye']],
            stored_landmarks[LANDMARKS['nose_tip']],
            stored_landmarks[LANDMARKS['chin']]
        ])

        current_key_points = np.array([
            current_landmarks[LANDMARKS['left_eye']],
            current_landmarks[LANDMARKS['right_eye']],
            current_landmarks[LANDMARKS['nose_tip']],
            current_landmarks[LANDMARKS['chin']]
        ])

        # Compute affine transform
        transform_result = cv2.estimateAffinePartial2D(stored_key_points, current_key_points)
        if transform_result[1] is None:
            log_to_file("Failed to compute alignment transform, using identity")
            M = np.eye(2, 3, dtype=np.float32)
        else:
            M = transform_result[0]

        # Warp the stored mask to current pose
        projected = cv2.warpAffine(
            best_match.mask.astype(np.float32),
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # If input was mirrored, flip the result back for display
        if is_mirrored:
            projected = np.fliplr(projected)

        # Threshold to binary
        _, binary_mask = cv2.threshold(projected, 127, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8)

        # Extract contours from projected mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        log_to_file(f"Found {len(contours)} contours in projected mask")

        contour_points = []
        if contours and len(contours) > 0:
            # Sort by area and take the largest
            contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
            largest = contours_sorted[0]
            area = cv2.contourArea(largest)
            log_to_file(f"Largest contour area: {area}")

            if area > 10:  # Minimum area threshold
                epsilon = 0.01 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)
                log_to_file(f"Approximated contour has {len(approx)} points")

                if len(approx) > 0:
                    contour_points = approx.reshape(-1, 2).tolist()
                    log_to_file(f"Final contour: {len(contour_points)} points")
                else:
                    log_to_file("Approximated contour is empty")
            else:
                log_to_file(f"Contour area {area} too small, skipping")
        else:
            log_to_file("No contours found in projected mask")
            # Debug: check if mask has any white pixels
            white_pixels = np.sum(binary_mask > 0)
            total_pixels = binary_mask.size
            log_to_file(f"Mask stats: {white_pixels}/{total_pixels} white pixels ({white_pixels/total_pixels*100:.1f}%)")

        log_to_file(f"Returning projected mask with {len(contour_points)} contour points")
        return binary_mask, contour_points
    
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
    """Get existing template from memory or disk."""
    # First check memory
    if template_id in _templates:
        return _templates[template_id]

    # If not in memory, try to load from disk
    try:
        template_path = TEMPLATE_DIR / f"{template_id}.pkl"
        if template_path.exists():
            with open(template_path, 'rb') as f:
                builder = pickle.load(f)

                # Ensure the loaded template has the required attributes for backward compatibility
                if not hasattr(builder, 'pose_masks'):
                    builder.pose_masks = []  # Initialize empty pose masks for old templates
                    log_to_file(f"Loaded old-style template {template_id}, added pose_masks compatibility")

                # Restore to memory
                _templates[template_id] = builder
                log_to_file(f"Loaded template {template_id} from disk")
                return builder
    except Exception as e:
        log_to_file(f"Failed to load template {template_id} from disk: {e}")

    return None


def save_template(template_id: str, builder: BeardTemplateBuilder):
    """Save a template to memory and disk."""
    _templates[template_id] = builder

    # Also save to disk for persistence
    try:
        template_path = TEMPLATE_DIR / f"{template_id}.pkl"
        with open(template_path, 'wb') as f:
            pickle.dump(builder, f)
        log_to_file(f"Saved template {template_id} to disk")
    except Exception as e:
        log_to_file(f"Failed to save template {template_id} to disk: {e}")


def delete_template(template_id: str):
    """Delete a template from memory and disk."""
    if template_id in _templates:
        del _templates[template_id]

    # Also delete from disk
    try:
        template_path = TEMPLATE_DIR / f"{template_id}.pkl"
        if template_path.exists():
            template_path.unlink()
        log_to_file(f"Deleted template {template_id} from disk")
    except Exception as e:
        log_to_file(f"Failed to delete template {template_id} from disk: {e}")

