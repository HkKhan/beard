"""
Test trained SAM model on saved frame data.
Visualizes segmentation results on captured frames to evaluate model performance.
"""

import sys
import json
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add segment-anything to path
sys.path.insert(0, str(Path(__file__).parent / "segment-anything"))

import torch
from segment_anything import sam_model_registry, SamPredictor

# Paths
CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "sam_vit_b_01ec64.pth"
FINE_TUNED_PATH = Path(__file__).parent / "checkpoints" / "sam_beard_best.pth"
SCANS_DIR = Path(__file__).parent / "scans"
OUTPUT_DIR = Path(__file__).parent / "sam_test_results"

def load_model(use_fine_tuned: bool = True):
    """Load SAM model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load base model
    sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_PATH))

    # Load fine-tuned weights if available
    if use_fine_tuned and FINE_TUNED_PATH.exists():
        print(f"Loading fine-tuned weights from: {FINE_TUNED_PATH}")
        checkpoint = torch.load(FINE_TUNED_PATH, map_location=device)
        sam.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Val IoU: {checkpoint.get('val_iou', 'unknown'):.4f}")
    else:
        print("Using base SAM model (no fine-tuning)")

    sam.to(device)
    sam.eval()

    return SamPredictor(sam)

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array (RGB)."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return np.array(image)

def generate_beard_prompts_from_landmarks(landmarks: list, width: int, height: int):
    """
    Generate SAM prompts from face mesh landmarks.
    Returns (positive_prompts, negative_prompts)
    """
    # Convert normalized to pixel if needed
    def to_pixel(lm):
        x, y = lm
        if x <= 1.0 and y <= 1.0:
            return (x * width, y * height)
        return (x, y)

    # Key landmark indices used by backend
    CHIN = 152
    JAW_LEFT = 234
    JAW_RIGHT = 454
    FOREHEAD = 10
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE_TIP = 4
    UPPER_LIP = 13

    try:
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
            tuple(forehead + np.array([0, 30])),  # Higher forehead
        ]

        return positive, negative
    except (IndexError, TypeError):
        print(f"Warning: Invalid landmarks data, using fallback prompts")
        # Fallback: use center-bottom area as positive, top as negative
        center_x, center_y = width / 2, height * 0.75
        positive = [(center_x, center_y), (center_x - 50, center_y), (center_x + 50, center_y)]
        negative = [(width / 2, height * 0.25)]
        return positive, negative

def segment_frame(predictor, image: np.ndarray, landmarks: list):
    """Run segmentation on a frame."""
    height, width = image.shape[:2]

    # Generate prompts from landmarks
    pos_prompts, neg_prompts = generate_beard_prompts_from_landmarks(landmarks, width, height)

    # Combine prompts
    all_points = np.array(pos_prompts + neg_prompts)
    all_labels = np.array([1] * len(pos_prompts) + [0] * len(neg_prompts))

    # Set image in predictor
    predictor.set_image(image)

    # Run prediction
    masks, scores, _ = predictor.predict(
        point_coords=all_points,
        point_labels=all_labels,
        multimask_output=True,
    )

    # Get best mask
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    return best_mask, best_score, all_points, all_labels

def visualize_frame_result(image, mask, score, points, labels, frame_idx, save_path):
    """Visualize segmentation result for a frame."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original image with prompts
    axes[0].imshow(image)
    axes[0].set_title(f'Frame {frame_idx} - Input + Prompts')
    for i, (pt, lbl) in enumerate(zip(points, labels)):
        color = 'green' if lbl == 1 else 'red'
        marker = '*' if lbl == 1 else 'x'
        axes[0].scatter(pt[0], pt[1], c=color, s=100, marker=marker, alpha=0.8)
    axes[0].axis('off')

    # Predicted mask overlay
    axes[1].imshow(image)
    axes[1].imshow(mask, alpha=0.6, cmap='Reds')
    axes[1].set_title(f'Frame {frame_idx} - SAM Prediction\nScore: {score:.3f}')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()

def load_latest_scan():
    """Load the most recent scan file."""
    scan_files = list(SCANS_DIR.glob("*_frames.json"))
    if not scan_files:
        print(f"No scan files found in {SCANS_DIR}")
        return None

    # Get most recent by timestamp in filename
    latest_scan = max(scan_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading scan: {latest_scan}")

    with open(latest_scan, 'r') as f:
        data = json.load(f)

    return data

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Testing Trained SAM Model on Saved Frames")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    predictor = load_model(use_fine_tuned=True)

    # Load latest scan
    print("\nLoading scan data...")
    scan_data = load_latest_scan()
    if not scan_data:
        return

    template_id = scan_data['template_id']
    frames = scan_data['frames']
    print(f"Template: {template_id}")
    print(f"Total frames: {len(frames)}")

    # Process sample frames (first, middle, last)
    sample_indices = [0, len(frames) // 2, len(frames) - 1]
    results = []

    print("\nProcessing sample frames...")
    for i, frame_idx in enumerate(sample_indices):
        if frame_idx >= len(frames):
            continue

        frame = frames[frame_idx]
        print(f"\nFrame {frame_idx + 1}/{len(frames)} (Sample {i+1})")

        try:
            # Decode image
            image = decode_base64_image(frame['image'])
            print(f"  Image size: {image.shape}")

            # Get landmarks
            landmarks = frame['face_mesh_landmarks']
            print(f"  Landmarks: {len(landmarks)} points")

            # Run segmentation
            mask, score, points, labels = segment_frame(predictor, image, landmarks)

            print(f"  Score: {score:.3f}")
            print(f"  Mask coverage: {mask.sum() / mask.size * 100:.1f}%")
            # Save visualization
            save_path = OUTPUT_DIR / "03d"
            visualize_frame_result(image, mask, score, points, labels, frame_idx + 1, save_path)

            results.append({
                'frame': frame_idx + 1,
                'score': float(score),
                'coverage': float(mask.sum() / mask.size * 100)
            })

        except Exception as e:
            print(f"  Error processing frame {frame_idx + 1}: {e}")
            continue

    # Summary
    print("\n" + "=" * 60)
    print("SAM Test Results Summary")
    print("=" * 60)

    if results:
        avg_score = np.mean([r['score'] for r in results])
        avg_coverage = np.mean([r['coverage'] for r in results])

        print(f"Average confidence: {avg_score:.3f}")
        print(f"Average coverage: {avg_coverage:.1f}%")
        print(f"\nSample frames processed: {len(results)}")
        print(f"Results saved to: {OUTPUT_DIR}")

        for result in results:
            print(f"  Frame {result['frame']:2d}: Score={result['score']:.3f}, Coverage={result['coverage']:.1f}%")

    print("\n" + "=" * 60)
    print("Analysis:")
    print("- Green stars (*): Positive prompts (beard areas)")
    print("- Red X's (x): Negative prompts (non-beard areas)")
    print("- Red overlay: Predicted beard segmentation")
    print("- Higher scores = more confident predictions")
    print("- Coverage shows % of frame predicted as beard")
    print("=" * 60)

if __name__ == "__main__":
    main()
