"""
Beard Map V2 - High Quality Mask Aggregation

This version:
1. Captures many frames continuously as user moves head
2. Aligns each face to a canonical view
3. Runs SAM on each frame
4. Aggregates masks in normalized face space (not vertex space)
5. For projection: runs SAM directly with learned prompts

Key insight: Store the MASK in normalized face coordinates, not just vertices.
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "segment-anything"))

import torch
from segment_anything import sam_model_registry, SamPredictor
import mediapipe as mp

# Paths
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "sam_vit_b_01ec64.pth"
FINE_TUNED_PATH = Path(__file__).parent.parent / "checkpoints" / "sam_beard_best.pth"
MY_IMAGES_DIR = Path(__file__).parent.parent / "my"
TEST_IMAGE = Path(__file__).parent.parent / "my" / "my_test" / "test.png"
OUTPUT_DIR = Path(__file__).parent.parent / "beard_map_v2_results"

# Canonical face size for normalization
CANONICAL_SIZE = 512


class FaceAligner:
    """Aligns faces to a canonical frontal view."""
    
    # Key landmark indices for alignment
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE_TIP = 4
    CHIN = 152
    
    def __init__(self):
        self.face_mesh = None
        self._init_face_mesh()
    
    def _init_face_mesh(self):
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True
            )
            self._use_new_api = False
        except AttributeError:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            model_path = Path(__file__).parent / "face_landmarker.task"
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
            self.face_mesh = vision.FaceLandmarker.create_from_options(options)
            self._use_new_api = True
    
    def get_landmarks(self, image_rgb):
        """Get 468 face landmarks."""
        h, w = image_rgb.shape[:2]
        
        if self._use_new_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.face_mesh.detect(mp_image)
            if not results.face_landmarks:
                return None
            return np.array([[lm.x * w, lm.y * h] for lm in results.face_landmarks[0]])
        else:
            results = self.face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return None
            return np.array([[lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark])
    
    def compute_alignment_transform(self, landmarks):
        """Compute affine transform to align face to canonical position."""
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        nose = landmarks[self.NOSE_TIP]
        chin = landmarks[self.CHIN]
        
        # Eye center
        eye_center = (left_eye + right_eye) / 2
        
        # Rotation angle from eye line
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Scale based on eye distance
        eye_dist = np.linalg.norm(right_eye - left_eye)
        scale = CANONICAL_SIZE * 0.3 / eye_dist  # Eyes should be 30% of image width
        
        # Transform matrix
        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)
        
        # Adjust translation to center face
        M[0, 2] += CANONICAL_SIZE / 2 - eye_center[0]
        M[1, 2] += CANONICAL_SIZE / 3 - eye_center[1]  # Eyes at 1/3 from top
        
        return M
    
    def align_image(self, image, landmarks):
        """Warp image to canonical view."""
        M = self.compute_alignment_transform(landmarks)
        aligned = cv2.warpAffine(image, M, (CANONICAL_SIZE, CANONICAL_SIZE))
        return aligned, M
    
    def align_mask(self, mask, landmarks):
        """Warp mask to canonical view."""
        M = self.compute_alignment_transform(landmarks)
        aligned = cv2.warpAffine(mask.astype(np.uint8), M, (CANONICAL_SIZE, CANONICAL_SIZE))
        return aligned > 0.5, M
    
    def unalign_mask(self, aligned_mask, M, original_size):
        """Warp mask back to original image space."""
        M_inv = cv2.invertAffineTransform(M)
        h, w = original_size
        unaligned = cv2.warpAffine(aligned_mask.astype(np.uint8), M_inv, (w, h))
        return unaligned > 0.5


class BeardMapperV2:
    def __init__(self):
        self.aligner = FaceAligner()
        self.sam_predictor = None
        
        # Aggregated mask in canonical space
        self.mask_accumulator = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.float32)
        self.mask_count = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=np.float32)
        
        # Store prompt positions in canonical space
        self.positive_prompts = []  # List of (x, y) in canonical coords
        self.negative_prompts = []
    
    def load_sam(self):
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_PATH))
        if FINE_TUNED_PATH.exists():
            print("Loading fine-tuned SAM...")
            checkpoint = torch.load(FINE_TUNED_PATH, map_location=device)
            sam.load_state_dict(checkpoint['model_state_dict'])
        sam.to(device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)
    
    def generate_prompts(self, landmarks, image_shape):
        """Generate SAM prompts from face landmarks."""
        h, w = image_shape[:2]
        
        chin = landmarks[152]
        jaw_left = landmarks[234]
        jaw_right = landmarks[454]
        forehead = landmarks[10]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Positive prompts in beard area
        positive = [
            chin,
            (chin + jaw_left) / 2,
            (chin + jaw_right) / 2,
            chin + np.array([0, -30]),  # Above chin
        ]
        
        # Negative prompts
        negative = [
            forehead,
            left_eye,
            right_eye,
        ]
        
        return positive, negative
    
    def run_sam(self, image_rgb, positive_pts, negative_pts):
        """Run SAM segmentation."""
        self.sam_predictor.set_image(image_rgb)
        
        all_points = np.array(positive_pts + negative_pts)
        all_labels = np.array([1] * len(positive_pts) + [0] * len(negative_pts))
        
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=all_points,
            point_labels=all_labels,
            multimask_output=True,
        )
        
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx], logits[best_idx]
    
    def process_image(self, image_path):
        """Process a single image and add to aggregated map."""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get landmarks
        landmarks = self.aligner.get_landmarks(image_rgb)
        if landmarks is None:
            return None
        
        # Generate prompts
        pos_pts, neg_pts = self.generate_prompts(landmarks, image_rgb.shape)
        
        # Run SAM
        mask, score, logits = self.run_sam(image_rgb, pos_pts, neg_pts)
        
        # Align mask to canonical space
        aligned_mask, M = self.aligner.align_mask(mask, landmarks)
        
        # Also align prompts for storage
        M_pts = self.aligner.compute_alignment_transform(landmarks)
        for pt in pos_pts:
            pt_aligned = M_pts @ np.array([pt[0], pt[1], 1])
            self.positive_prompts.append(pt_aligned[:2])
        for pt in neg_pts:
            pt_aligned = M_pts @ np.array([pt[0], pt[1], 1])
            self.negative_prompts.append(pt_aligned[:2])
        
        # Accumulate
        self.mask_accumulator += aligned_mask.astype(np.float32) * score
        self.mask_count += score  # Weight by confidence
        
        return {
            'score': score,
            'mask_coverage': mask.sum() / mask.size,
            'aligned_mask': aligned_mask,
        }
    
    def get_aggregated_mask(self, threshold=0.5):
        """Get the aggregated probability mask."""
        prob_mask = np.divide(
            self.mask_accumulator, 
            self.mask_count, 
            out=np.zeros_like(self.mask_accumulator),
            where=self.mask_count > 0
        )
        return prob_mask, prob_mask > threshold
    
    def get_average_prompts(self):
        """Get averaged prompt positions in canonical space."""
        if not self.positive_prompts:
            return None, None
        
        # Cluster and average prompts
        pos_array = np.array(self.positive_prompts)
        neg_array = np.array(self.negative_prompts) if self.negative_prompts else np.array([])
        
        # Simple averaging by quadrant
        avg_positive = []
        # Center
        center_pts = pos_array[(pos_array[:, 0] > CANONICAL_SIZE * 0.4) & (pos_array[:, 0] < CANONICAL_SIZE * 0.6)]
        if len(center_pts) > 0:
            avg_positive.append(center_pts.mean(axis=0))
        # Left
        left_pts = pos_array[pos_array[:, 0] <= CANONICAL_SIZE * 0.4]
        if len(left_pts) > 0:
            avg_positive.append(left_pts.mean(axis=0))
        # Right
        right_pts = pos_array[pos_array[:, 0] >= CANONICAL_SIZE * 0.6]
        if len(right_pts) > 0:
            avg_positive.append(right_pts.mean(axis=0))
        
        avg_negative = []
        if len(neg_array) > 0:
            # Upper half (forehead/eyes)
            upper_pts = neg_array[neg_array[:, 1] < CANONICAL_SIZE * 0.5]
            if len(upper_pts) > 0:
                avg_negative.append(upper_pts.mean(axis=0))
        
        return np.array(avg_positive), np.array(avg_negative) if avg_negative else np.array([])
    
    def project_direct_sam(self, image_path):
        """
        Project onto new image using DIRECT SAM segmentation.
        Uses learned prompt positions transformed to the new image.
        """
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Get landmarks
        landmarks = self.aligner.get_landmarks(image_rgb)
        if landmarks is None:
            print("No face detected!")
            return None, None
        
        # Get alignment transform
        M = self.aligner.compute_alignment_transform(landmarks)
        M_inv = cv2.invertAffineTransform(M)
        
        # Transform averaged prompts to image space
        avg_pos, avg_neg = self.get_average_prompts()
        
        if avg_pos is None or len(avg_pos) == 0:
            # Fallback: use landmark-based prompts
            pos_pts, neg_pts = self.generate_prompts(landmarks, image_rgb.shape)
        else:
            # Transform canonical prompts to image space
            pos_pts = []
            for pt in avg_pos:
                pt_img = M_inv @ np.array([pt[0], pt[1], 1])
                pos_pts.append(pt_img[:2])
            
            neg_pts = []
            for pt in avg_neg:
                pt_img = M_inv @ np.array([pt[0], pt[1], 1])
                neg_pts.append(pt_img[:2])
            
            # Add landmark-based prompts for robustness
            extra_pos, extra_neg = self.generate_prompts(landmarks, image_rgb.shape)
            pos_pts.extend(extra_pos[:2])  # Add chin prompts
            neg_pts.extend(extra_neg)
        
        # Run SAM directly on the image
        mask, score, _ = self.run_sam(image_rgb, pos_pts, neg_pts)
        
        return mask, {
            'score': score,
            'positive_prompts': pos_pts,
            'negative_prompts': neg_pts,
        }
    
    def project_template_mask(self, image_path):
        """
        Project the aggregated template mask onto new image.
        (This is the old approach - for comparison)
        """
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        landmarks = self.aligner.get_landmarks(image_rgb)
        if landmarks is None:
            return None
        
        # Get aggregated mask
        prob_mask, binary_mask = self.get_aggregated_mask()
        
        # Transform back to image space
        M = self.aligner.compute_alignment_transform(landmarks)
        projected_mask = self.aligner.unalign_mask(binary_mask, M, image_rgb.shape[:2])
        
        return projected_mask


def visualize_comparison(image_rgb, direct_mask, template_mask, prompts_info, save_path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Original
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # Direct SAM (the good one)
    axes[0, 1].imshow(image_rgb)
    if direct_mask is not None:
        overlay = np.zeros((*direct_mask.shape, 4))
        overlay[direct_mask > 0] = [0, 1, 0, 0.5]
        axes[0, 1].imshow(overlay)
        # Draw prompts
        if prompts_info:
            for pt in prompts_info['positive_prompts']:
                axes[0, 1].scatter(pt[0], pt[1], c='lime', s=100, marker='o', edgecolors='white', linewidths=2)
            for pt in prompts_info['negative_prompts']:
                axes[0, 1].scatter(pt[0], pt[1], c='red', s=100, marker='x', linewidths=2)
    axes[0, 1].set_title(f'Direct SAM Segmentation\n(score: {prompts_info["score"]:.3f})', fontsize=14)
    axes[0, 1].axis('off')
    
    # Template projection (the bad one)
    axes[1, 0].imshow(image_rgb)
    if template_mask is not None:
        overlay = np.zeros((*template_mask.shape, 4))
        overlay[template_mask > 0] = [0, 0, 1, 0.5]
        axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Template Projection\n(Vertex-based - OLD)', fontsize=14)
    axes[1, 0].axis('off')
    
    # Difference
    axes[1, 1].imshow(image_rgb)
    if direct_mask is not None and template_mask is not None:
        diff = np.zeros((*direct_mask.shape, 3))
        diff[direct_mask & ~template_mask] = [0, 1, 0]  # Green: in direct, not template
        diff[~direct_mask & template_mask] = [1, 0, 0]  # Red: in template, not direct
        diff[direct_mask & template_mask] = [1, 1, 0]   # Yellow: both
        axes[1, 1].imshow(diff, alpha=0.6)
    axes[1, 1].set_title('Difference\n(Green=Direct only, Red=Template only)', fontsize=14)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Beard Map V2 - Direct SAM Projection")
    print("=" * 70)
    
    mapper = BeardMapperV2()
    mapper.load_sam()
    
    # Get images
    image_files = list(MY_IMAGES_DIR.glob("*.png")) + list(MY_IMAGES_DIR.glob("*.jpg"))
    print(f"\nFound {len(image_files)} images")
    
    # Process each image
    print("\n--- Building Template ---")
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] {img_path.name}")
        result = mapper.process_image(img_path)
        if result:
            print(f"  Score: {result['score']:.3f}, Coverage: {result['mask_coverage']*100:.1f}%")
    
    # Get aggregated mask
    prob_mask, binary_mask = mapper.get_aggregated_mask()
    print(f"\nAggregated mask coverage: {binary_mask.sum() / binary_mask.size * 100:.1f}%")
    
    # Save aggregated mask visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(prob_mask, cmap='hot')
    axes[0].set_title('Probability Mask (Canonical Space)')
    axes[0].axis('off')
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title('Binary Mask (threshold=0.5)')
    axes[1].axis('off')
    plt.savefig(OUTPUT_DIR / "aggregated_mask.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test on new image
    print("\n--- Testing on New Image ---")
    if TEST_IMAGE.exists():
        print(f"Test image: {TEST_IMAGE}")
        
        # Direct SAM (the good approach)
        direct_mask, prompts_info = mapper.project_direct_sam(TEST_IMAGE)
        
        # Template projection (the bad approach, for comparison)
        template_mask = mapper.project_template_mask(TEST_IMAGE)
        
        # Load image for visualization
        test_img = cv2.imread(str(TEST_IMAGE))
        test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        
        # Create comparison
        visualize_comparison(
            test_img_rgb, direct_mask, template_mask, prompts_info,
            OUTPUT_DIR / "comparison.png"
        )
        
        print(f"Direct SAM score: {prompts_info['score']:.3f}")
        print(f"Direct SAM coverage: {direct_mask.sum() / direct_mask.size * 100:.1f}%")
        
        # Save the direct SAM result as the recommended output
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(test_img_rgb)
        overlay = np.zeros((*direct_mask.shape, 4))
        overlay[direct_mask > 0] = [0, 1, 0, 0.5]
        ax.imshow(overlay)
        ax.set_title(f'Recommended: Direct SAM Projection\n(Score: {prompts_info["score"]:.3f})', fontsize=14)
        ax.axis('off')
        plt.savefig(OUTPUT_DIR / "final_result.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to: {OUTPUT_DIR}")
    else:
        print(f"Test image not found: {TEST_IMAGE}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("The 'Direct SAM' approach runs SAM on the actual test image")
    print("using prompts learned from the template. This gives much better")
    print("quality than projecting a stored mask through face mesh vertices.")
    print("=" * 70)


if __name__ == "__main__":
    main()


