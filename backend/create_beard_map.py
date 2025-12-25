"""
Create a 2.5D Beard Map from multiple angle captures.

This script:
1. Runs face mesh detection on each image
2. Runs SAM segmentation on each image
3. Maps segmentation masks to face mesh vertices
4. Combines all views into a unified beard map with confidence scores
5. Tests the map on a new image
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Add segment-anything to path
sys.path.insert(0, str(Path(__file__).parent.parent / "segment-anything"))

import torch
from segment_anything import sam_model_registry, SamPredictor

# MediaPipe for face mesh
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    MEDIAPIPE_LEGACY = True
except:
    import mediapipe as mp
    MEDIAPIPE_LEGACY = False

# Paths
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "sam_vit_b_01ec64.pth"
FINE_TUNED_PATH = Path(__file__).parent.parent / "checkpoints" / "sam_beard_best.pth"
MY_IMAGES_DIR = Path(__file__).parent.parent / "my"
TEST_IMAGE = Path(__file__).parent.parent / "my" / "my_test" / "test.png"
OUTPUT_DIR = Path(__file__).parent.parent / "beard_map_results"

# Face mesh landmarks for beard region (lower face)
LOWER_FACE_INDICES = list(range(0, 17)) + list(range(17, 68)) + [
    # Jawline and chin
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    # Cheeks
    116, 117, 118, 119, 120, 121, 128, 129, 130, 131,
    345, 346, 347, 348, 349, 350, 357, 358, 359, 360,
    # Under nose / mustache
    164, 165, 166, 167, 168, 169, 170, 171, 175,
    # Mouth area
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    # Lower cheek
    187, 207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194,
    411, 427, 426, 425, 424, 423, 422, 421, 420, 419, 418, 417, 416, 415, 414,
]


class BeardMapper:
    def __init__(self):
        self.sam_predictor = None
        self.face_mesh = None
        self.vertex_scores = defaultdict(list)  # vertex_idx -> list of (is_beard, confidence)
        
    def load_models(self):
        """Load SAM and MediaPipe Face Mesh."""
        # Load SAM
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_PATH))
        if FINE_TUNED_PATH.exists():
            print(f"Loading fine-tuned SAM...")
            checkpoint = torch.load(FINE_TUNED_PATH, map_location=device)
            sam.load_state_dict(checkpoint['model_state_dict'])
        sam.to(device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)
        
        # Load Face Mesh
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except AttributeError:
            # Newer mediapipe version
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download model if needed
            import urllib.request
            import os
            model_path = Path(__file__).parent / "face_landmarker.task"
            if not model_path.exists():
                print("Downloading face landmarker model...")
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, str(model_path))
            
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.face_mesh = vision.FaceLandmarker.create_from_options(options)
            self._use_new_api = True
        else:
            self._use_new_api = False
        print("Models loaded!")
    
    def process_image(self, image_path: Path) -> dict:
        """Process a single image: detect face mesh + run SAM."""
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Detect face mesh
        if hasattr(self, '_use_new_api') and self._use_new_api:
            # New MediaPipe API
            from mediapipe.tasks.python import vision
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.face_mesh.detect(mp_image)
            if not results.face_landmarks:
                print(f"  No face detected in {image_path.name}")
                return None
            landmarks_list = results.face_landmarks[0]
            landmark_pixels = np.array([[lm.x * w, lm.y * h] for lm in landmarks_list])
        else:
            # Legacy API
            results = self.face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                print(f"  No face detected in {image_path.name}")
                return None
            landmarks = results.multi_face_landmarks[0]
            landmark_pixels = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
        
        # Generate SAM prompts from face mesh (chin, jaw area)
        chin_idx = 152
        jaw_left = 234
        jaw_right = 454
        forehead = 10
        
        chin = landmark_pixels[chin_idx]
        jaw_l = landmark_pixels[jaw_left]
        jaw_r = landmark_pixels[jaw_right]
        fhead = landmark_pixels[forehead]
        
        prompts = [
            (chin, 1),
            ((chin + jaw_l) / 2, 1),
            ((chin + jaw_r) / 2, 1),
            (fhead, 0),  # Negative: forehead
            (landmark_pixels[33], 0),  # Negative: left eye
            (landmark_pixels[263], 0),  # Negative: right eye
        ]
        
        # Run SAM
        self.sam_predictor.set_image(image_rgb)
        point_coords = np.array([p[0] for p in prompts])
        point_labels = np.array([p[1] for p in prompts])
        
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = scores[best_idx]
        logit = logits[best_idx]  # Raw logits for probability
        
        # Convert logits to probabilities
        prob_map = 1 / (1 + np.exp(-logit))  # Sigmoid
        # Resize prob_map to image size
        prob_map_resized = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return {
            'landmarks': landmark_pixels,
            'mask': mask,
            'score': score,
            'prob_map': prob_map_resized,
            'image': image_rgb,
            'size': (h, w),
        }
    
    def map_to_vertices(self, result: dict) -> dict:
        """Map segmentation mask to face mesh vertices."""
        landmarks = result['landmarks']
        prob_map = result['prob_map']
        h, w = result['size']
        
        vertex_data = {}
        for idx in range(468):
            x, y = landmarks[idx]
            px, py = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
            
            # Get probability at this vertex location
            prob = prob_map[py, px]
            is_beard = prob > 0.5
            
            vertex_data[idx] = {
                'probability': float(prob),
                'is_beard': is_beard,
                'pixel': (px, py),
            }
        
        return vertex_data
    
    def add_to_map(self, vertex_data: dict, weight: float = 1.0):
        """Add vertex data to the cumulative beard map."""
        for idx, data in vertex_data.items():
            self.vertex_scores[idx].append({
                'prob': data['probability'],
                'is_beard': data['is_beard'],
                'weight': weight,
            })
    
    def compute_final_map(self, threshold: float = 0.5) -> dict:
        """Compute final beard map by averaging across all views."""
        final_map = {}
        
        for idx in range(468):
            if idx in self.vertex_scores:
                scores = self.vertex_scores[idx]
                # Weighted average probability
                total_weight = sum(s['weight'] for s in scores)
                avg_prob = sum(s['prob'] * s['weight'] for s in scores) / total_weight
                # Vote count
                vote_count = sum(1 for s in scores if s['is_beard'])
                
                final_map[idx] = {
                    'avg_probability': avg_prob,
                    'is_beard': avg_prob >= threshold,
                    'confidence': avg_prob if avg_prob >= threshold else 1 - avg_prob,
                    'vote_count': vote_count,
                    'total_views': len(scores),
                }
            else:
                final_map[idx] = {
                    'avg_probability': 0.0,
                    'is_beard': False,
                    'confidence': 1.0,
                    'vote_count': 0,
                    'total_views': 0,
                }
        
        return final_map
    
    def get_beard_vertices(self, final_map: dict, min_prob: float = 0.5) -> list:
        """Get list of vertex indices that are part of the beard."""
        return [idx for idx, data in final_map.items() if data['avg_probability'] >= min_prob]
    
    def project_on_image(self, image_path: Path, beard_vertices: list, final_map: dict) -> np.ndarray:
        """Project the beard map onto a new image."""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Detect face mesh
        if hasattr(self, '_use_new_api') and self._use_new_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.face_mesh.detect(mp_image)
            if not results.face_landmarks:
                print("No face detected in test image!")
                return image_rgb
            landmark_pixels = np.array([[lm.x * w, lm.y * h] for lm in results.face_landmarks[0]])
        else:
            results = self.face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                print("No face detected in test image!")
                return image_rgb
            landmarks = results.multi_face_landmarks[0]
            landmark_pixels = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
        
        # Create overlay
        overlay = image_rgb.copy()
        
        # Draw beard vertices with color based on confidence
        for idx in beard_vertices:
            x, y = landmark_pixels[idx]
            prob = final_map[idx]['avg_probability']
            
            # Color: green intensity based on probability
            color = (0, int(255 * prob), 0)
            cv2.circle(overlay, (int(x), int(y)), 4, color, -1)
        
        # Draw convex hull of beard vertices
        if len(beard_vertices) > 3:
            beard_points = landmark_pixels[beard_vertices].astype(np.int32)
            hull = cv2.convexHull(beard_points)
            
            # Semi-transparent fill
            mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            overlay[mask > 0] = overlay[mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
            
            # Outline
            cv2.polylines(overlay, [hull], True, (0, 255, 0), 2)
        
        return overlay


def visualize_beard_map(final_map: dict, save_path: Path):
    """Visualize the beard probability map."""
    # Create a visualization of vertex probabilities
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get probabilities for all vertices
    probs = [final_map[i]['avg_probability'] for i in range(468)]
    votes = [final_map[i]['vote_count'] for i in range(468)]
    
    # Histogram of probabilities
    axes[0].hist(probs, bins=50, color='steelblue', edgecolor='white')
    axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Average Probability')
    axes[0].set_ylabel('Number of Vertices')
    axes[0].set_title('Beard Probability Distribution')
    axes[0].legend()
    
    # Vertices by vote count
    beard_count = sum(1 for p in probs if p >= 0.5)
    non_beard_count = 468 - beard_count
    axes[1].bar(['Beard', 'Non-Beard'], [beard_count, non_beard_count], color=['green', 'gray'])
    axes[1].set_ylabel('Number of Vertices')
    axes[1].set_title(f'Vertex Classification\n({beard_count} beard vertices)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Creating 2.5D Beard Map from Multiple Angles")
    print("=" * 70)
    
    # Initialize mapper
    mapper = BeardMapper()
    mapper.load_models()
    
    # Get images
    image_files = list(MY_IMAGES_DIR.glob("*.png")) + list(MY_IMAGES_DIR.glob("*.jpg"))
    print(f"\nFound {len(image_files)} images in {MY_IMAGES_DIR}")
    
    # Process each image
    print("\n--- Processing Images ---")
    processed_results = []
    
    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {img_path.name}")
        
        result = mapper.process_image(img_path)
        if result is None:
            continue
        
        print(f"  SAM score: {result['score']:.3f}")
        print(f"  Mask coverage: {result['mask'].sum() / result['mask'].size * 100:.1f}%")
        
        # Map to vertices
        vertex_data = mapper.map_to_vertices(result)
        beard_count = sum(1 for v in vertex_data.values() if v['is_beard'])
        print(f"  Beard vertices: {beard_count}/468")
        
        # Add to cumulative map
        mapper.add_to_map(vertex_data, weight=result['score'])
        processed_results.append((img_path, result, vertex_data))
    
    # Compute final map
    print("\n--- Computing Final Beard Map ---")
    final_map = mapper.compute_final_map(threshold=0.5)
    beard_vertices = mapper.get_beard_vertices(final_map, min_prob=0.5)
    print(f"Final beard vertices: {len(beard_vertices)}/468")
    
    # High confidence vertices (>0.7)
    high_conf = mapper.get_beard_vertices(final_map, min_prob=0.7)
    print(f"High confidence (>70%): {len(high_conf)} vertices")
    
    # Save beard map (convert numpy types to native Python)
    def convert_types(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj
    
    map_data = {
        'beard_vertices': beard_vertices,
        'high_confidence_vertices': high_conf,
        'vertex_probabilities': {str(k): convert_types(v) for k, v in final_map.items()},
        'num_source_images': len(processed_results),
    }
    
    map_path = OUTPUT_DIR / "beard_map.json"
    with open(map_path, 'w') as f:
        json.dump(map_data, f, indent=2)
    print(f"\nSaved beard map to: {map_path}")
    
    # Visualize map
    viz_path = OUTPUT_DIR / "beard_map_visualization.png"
    visualize_beard_map(final_map, viz_path)
    print(f"Saved visualization to: {viz_path}")
    
    # Test on new image
    print("\n--- Testing on New Image ---")
    if TEST_IMAGE.exists():
        print(f"Test image: {TEST_IMAGE}")
        
        # Project beard map
        projection = mapper.project_on_image(TEST_IMAGE, beard_vertices, final_map)
        
        # Also run SAM on test image for comparison
        test_result = mapper.process_image(TEST_IMAGE)
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original
        test_img = cv2.imread(str(TEST_IMAGE))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(test_img)
        axes[0].set_title('Original Test Image')
        axes[0].axis('off')
        
        # Projected beard map
        axes[1].imshow(projection)
        axes[1].set_title(f'Projected Beard Map\n({len(beard_vertices)} vertices)')
        axes[1].axis('off')
        
        # SAM direct segmentation
        if test_result:
            axes[2].imshow(test_result['image'])
            mask_overlay = np.zeros((*test_result['mask'].shape, 4))
            mask_overlay[test_result['mask'] > 0] = [0, 0, 1, 0.5]
            axes[2].imshow(mask_overlay)
            axes[2].set_title(f'Direct SAM Segmentation\n(score: {test_result["score"]:.3f})')
        else:
            axes[2].text(0.5, 0.5, 'No face detected', ha='center', va='center')
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = OUTPUT_DIR / "test_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison to: {comparison_path}")
    else:
        print(f"Test image not found: {TEST_IMAGE}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

