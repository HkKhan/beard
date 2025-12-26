"""
Test SAM segmentation on user's own images.
Auto-detects face and generates beard prompts.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add segment-anything to path
sys.path.insert(0, str(Path(__file__).parent.parent / "segment-anything"))

import torch
from segment_anything import sam_model_registry, SamPredictor

# Paths
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "sam_vit_b_01ec64.pth"
FINE_TUNED_PATH = Path(__file__).parent.parent / "checkpoints" / "sam_beard_best.pth"
MY_IMAGES_DIR = Path(__file__).parent.parent / "my"
OUTPUT_DIR = Path(__file__).parent.parent / "my_results"

def load_model(use_fine_tuned: bool = True):
    """Load SAM model."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT_PATH))
    
    if use_fine_tuned and FINE_TUNED_PATH.exists():
        print(f"Loading fine-tuned weights: {FINE_TUNED_PATH.name}")
        checkpoint = torch.load(FINE_TUNED_PATH, map_location=device)
        sam.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Val IoU: {checkpoint.get('val_iou', 0):.4f}")
    
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)

def detect_face(image):
    """Detect face using OpenCV's Haar cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Try to load face cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
    
    if len(faces) > 0:
        # Return the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces[0]  # (x, y, w, h)
    
    return None

def generate_beard_prompts(image, face_box=None):
    """Generate point prompts for beard region."""
    h, w = image.shape[:2]
    
    if face_box is not None:
        fx, fy, fw, fh = face_box
        # Beard region is lower half of face
        beard_center_x = fx + fw / 2
        beard_center_y = fy + fh * 0.75  # Lower portion
        chin_y = fy + fh * 0.95
        
        prompts = [
            # Positive prompts (beard area)
            ([beard_center_x, beard_center_y], 1),
            ([beard_center_x - fw * 0.2, beard_center_y], 1),
            ([beard_center_x + fw * 0.2, beard_center_y], 1),
            ([beard_center_x, chin_y], 1),
            # Negative prompts (eyes, forehead)
            ([beard_center_x, fy + fh * 0.3], 0),  # Eyes
            ([beard_center_x, fy + fh * 0.1], 0),  # Forehead
        ]
    else:
        # Default: assume face is centered
        prompts = [
            ([w * 0.5, h * 0.7], 1),   # Center lower
            ([w * 0.4, h * 0.7], 1),   # Left lower
            ([w * 0.6, h * 0.7], 1),   # Right lower
            ([w * 0.5, h * 0.8], 1),   # Chin
            ([w * 0.5, h * 0.3], 0),   # Upper face (negative)
        ]
    
    return prompts

def segment_with_prompts(predictor, image, prompts):
    """Run SAM segmentation."""
    predictor.set_image(image)
    
    point_coords = np.array([[p[0][0], p[0][1]] for p in prompts])
    point_labels = np.array([p[1] for p in prompts])
    
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    
    best_idx = np.argmax(scores)
    return masks[best_idx], scores[best_idx], point_coords, point_labels

def visualize_and_save(image, mask, score, points, labels, face_box, save_path):
    """Create visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original with face detection and prompts
    axes[0].imshow(image)
    axes[0].set_title('Input + Face Detection + Prompts')
    if face_box is not None:
        x, y, w, h = face_box
        rect = plt.Rectangle((x, y), w, h, fill=False, color='cyan', linewidth=2)
        axes[0].add_patch(rect)
    for pt, lbl in zip(points, labels):
        color = 'lime' if lbl == 1 else 'red'
        marker = 'o' if lbl == 1 else 'x'
        axes[0].scatter(pt[0], pt[1], c=color, s=150, marker=marker, edgecolors='white', linewidths=2)
    axes[0].axis('off')
    
    # Mask overlay
    axes[1].imshow(image)
    colored_mask = np.zeros((*mask.shape, 4))
    colored_mask[mask > 0] = [0, 1, 0, 0.5]  # Green with alpha
    axes[1].imshow(colored_mask)
    axes[1].set_title(f'SAM Segmentation (score: {score:.3f})')
    axes[1].axis('off')
    
    # Mask only
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f'Binary Mask ({mask.sum()} pixels)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("SAM Beard Segmentation - Your Images")
    print("=" * 60)
    
    # Get images
    image_files = list(MY_IMAGES_DIR.glob("*.png")) + list(MY_IMAGES_DIR.glob("*.jpg")) + list(MY_IMAGES_DIR.glob("*.jpeg"))
    print(f"\nFound {len(image_files)} images in {MY_IMAGES_DIR}")
    
    if not image_files:
        print("No images found!")
        return
    
    # Load model
    print("\nLoading SAM model...")
    predictor = load_model(use_fine_tuned=True)
    
    # Process each image
    print("\nProcessing images...")
    
    for i, img_path in enumerate(image_files):
        print(f"\n[{i+1}/{len(image_files)}] {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print("  Failed to load image")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"  Size: {image.shape[1]}x{image.shape[0]}")
        
        # Detect face
        face_box = detect_face(image)
        if face_box is not None:
            print(f"  Face detected: {face_box}")
        else:
            print("  No face detected, using default prompts")
        
        # Generate prompts
        prompts = generate_beard_prompts(image, face_box)
        print(f"  Prompts: {len([p for p in prompts if p[1]==1])} positive, {len([p for p in prompts if p[1]==0])} negative")
        
        # Run segmentation
        mask, score, points, labels = segment_with_prompts(predictor, image, prompts)
        print(f"  Score: {score:.3f}")
        print(f"  Mask coverage: {mask.sum() / mask.size * 100:.2f}%")
        
        # Save result
        save_path = OUTPUT_DIR / f"result_{i+1:02d}_{img_path.stem}.png"
        visualize_and_save(image, mask, score, points, labels, face_box, save_path)
    
    print("\n" + "=" * 60)
    print(f"Done! Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()


