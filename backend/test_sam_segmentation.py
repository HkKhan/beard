"""
Test SAM segmentation on beard images.
Visualizes the segmentation results to evaluate model performance.
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Add segment-anything to path
sys.path.insert(0, str(Path(__file__).parent.parent / "segment-anything"))

import torch
from segment_anything import sam_model_registry, SamPredictor

# Paths
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "sam_vit_b_01ec64.pth"
FINE_TUNED_PATH = Path(__file__).parent.parent / "checkpoints" / "sam_beard_best.pth"
TEST_IMAGES_DIR = Path(__file__).parent.parent / "beard-dataset.v46i.coco-segmentation" / "test"
OUTPUT_DIR = Path(__file__).parent.parent / "segmentation_results"

def load_model(use_fine_tuned: bool = True):
    """Load SAM model."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
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

def get_test_images(limit: int = 10):
    """Get test images and their annotations."""
    annotations_path = TEST_IMAGES_DIR / "_annotations.coco.json"
    
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    images = []
    for img_info in coco_data['images'][:limit]:
        img_path = TEST_IMAGES_DIR / img_info['file_name']
        if img_path.exists():
            # Get annotations for this image
            img_anns = [a for a in coco_data['annotations'] if a['image_id'] == img_info['id']]
            images.append({
                'path': img_path,
                'info': img_info,
                'annotations': img_anns,
            })
    
    return images

def generate_prompts_from_annotation(annotation, img_width, img_height):
    """Generate point prompts from COCO annotation."""
    # Get bounding box center as positive prompt
    if 'bbox' in annotation:
        x, y, w, h = annotation['bbox']
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Generate multiple prompts within the bbox
        prompts = [
            ([center_x, center_y], 1),  # Center
            ([center_x - w/4, center_y], 1),  # Left
            ([center_x + w/4, center_y], 1),  # Right
            ([center_x, center_y + h/4], 1),  # Lower
        ]
        
        # Add negative prompt above the beard (forehead area)
        prompts.append(([center_x, max(0, center_y - h)], 0))
        
        return prompts
    
    return []

def segment_image(predictor, image_path, prompts):
    """Run segmentation on an image."""
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image
    predictor.set_image(image)
    
    # Prepare prompts
    point_coords = np.array([[p[0][0], p[0][1]] for p in prompts])
    point_labels = np.array([p[1] for p in prompts])
    
    # Run prediction
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    
    # Get best mask
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]
    
    return image, best_mask, best_score, point_coords, point_labels

def visualize_result(image, mask, score, points, labels, gt_mask=None, save_path=None):
    """Visualize segmentation result."""
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(15, 5))
    
    # Original image with prompts
    axes[0].imshow(image)
    axes[0].set_title('Input + Prompts')
    for i, (pt, lbl) in enumerate(zip(points, labels)):
        color = 'green' if lbl == 1 else 'red'
        axes[0].scatter(pt[0], pt[1], c=color, s=100, marker='*')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(image)
    axes[1].imshow(mask, alpha=0.5, cmap='Greens')
    axes[1].set_title(f'SAM Prediction (score: {score:.3f})')
    axes[1].axis('off')
    
    # Ground truth if available
    if gt_mask is not None:
        axes[2].imshow(image)
        axes[2].imshow(gt_mask, alpha=0.5, cmap='Blues')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()

def decode_coco_mask(annotation, height, width):
    """Decode COCO segmentation to binary mask."""
    from pycocotools import mask as mask_utils
    
    if 'segmentation' not in annotation:
        return None
    
    seg = annotation['segmentation']
    
    if isinstance(seg, list):
        # Polygon format
        rle = mask_utils.frPyObjects(seg, height, width)
        mask = mask_utils.decode(rle)
        if len(mask.shape) == 3:
            mask = mask.sum(axis=2) > 0
    else:
        # RLE format
        mask = mask_utils.decode(seg)
    
    return mask.astype(np.uint8)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("SAM Beard Segmentation Test")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    predictor = load_model(use_fine_tuned=True)
    
    # Get test images
    print("\nLoading test images...")
    test_images = get_test_images(limit=10)
    print(f"Found {len(test_images)} test images")
    
    # Process each image
    print("\nProcessing images...")
    results = []
    
    for i, img_data in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] {img_data['path'].name}")
        
        # Get prompts from annotation
        if img_data['annotations']:
            prompts = generate_prompts_from_annotation(
                img_data['annotations'][0],
                img_data['info']['width'],
                img_data['info']['height']
            )
        else:
            # Default prompts in lower face region
            h, w = img_data['info']['height'], img_data['info']['width']
            prompts = [
                ([w/2, h*0.7], 1),  # Center lower face
                ([w/2, h*0.3], 0),  # Forehead (negative)
            ]
        
        if not prompts:
            print("  No prompts generated, skipping")
            continue
        
        # Run segmentation
        image, mask, score, points, labels = segment_image(
            predictor, img_data['path'], prompts
        )
        
        print(f"  Score: {score:.3f}")
        print(f"  Mask coverage: {mask.sum() / mask.size * 100:.1f}%")
        
        # Get ground truth mask
        gt_mask = None
        if img_data['annotations']:
            gt_mask = decode_coco_mask(
                img_data['annotations'][0],
                img_data['info']['height'],
                img_data['info']['width']
            )
            if gt_mask is not None:
                # Calculate IoU
                intersection = (mask & gt_mask).sum()
                union = (mask | gt_mask).sum()
                iou = intersection / union if union > 0 else 0
                print(f"  IoU with GT: {iou:.3f}")
                results.append({'image': img_data['path'].name, 'score': score, 'iou': iou})
        
        # Save visualization
        save_path = OUTPUT_DIR / f"result_{i+1:02d}_{img_data['path'].stem}.png"
        visualize_result(image, mask, score, points, labels, gt_mask, save_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if results:
        avg_score = np.mean([r['score'] for r in results])
        avg_iou = np.mean([r['iou'] for r in results])
        print(f"Average confidence: {avg_score:.3f}")
        print(f"Average IoU: {avg_iou:.3f}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()



