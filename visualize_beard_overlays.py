import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def load_coco_annotations(annotation_file):
    """Load COCO format annotations"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data

def get_image_annotations(coco_data, image_id):
    """Get all annotations for a specific image"""
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    return annotations

def get_image_info(coco_data, image_id):
    """Get image information by image_id"""
    image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
    return image_info

def create_segmentation_mask(segmentation, image_shape):
    """Create a binary mask from COCO segmentation polygon"""
    if not segmentation:
        return None

    # For COCO segmentation, we take the first polygon
    polygon = segmentation[0]

    # Create mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Convert polygon to numpy array and reshape
    poly = np.array(polygon).reshape(-1, 2)

    # Create path for polygon
    path = Path(poly)

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    points = np.vstack((x.flatten(), y.flatten())).T

    # Check which points are inside the polygon
    mask_flat = path.contains_points(points)
    mask = mask_flat.reshape(image_shape[:2])

    return mask.astype(np.uint8)

def visualize_beard_overlay(image_path, annotations, coco_data):
    """Visualize beard segmentation overlays on an image"""

    # Load image
    image = Image.open(image_path)
    img_array = np.array(image)

    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_array)

    # Colors for different beard categories (only actual beard, not chin-to-beard)
    colors = {
        0: ('red', 0.5),      # beard (id 0)
        1: ('blue', 0.5),     # beard (id 1)
    }

    category_names = {
        0: 'beard',
        1: 'beard',
    }

    legend_patches = []

    # Plot each annotation (only actual beard categories, not chin-to-beard, landmarks, or lips)
    for ann in annotations:
        category_id = ann['category_id']
        segmentation = ann.get('segmentation', [])

        # Skip non-beard categories (chin-to-beard, landmarks, lips)
        if category_id not in [0, 1]:
            continue

        if not segmentation:
            continue

        color, alpha = colors.get(category_id, ('gray', 0.3))
        category_name = category_names.get(category_id, f'category_{category_id}')

        # Create mask from segmentation
        mask = create_segmentation_mask(segmentation, img_array.shape)

        if mask is not None:
            # Create colored overlay
            colored_mask = np.zeros_like(img_array)
            if len(img_array.shape) == 3:
                # RGB image
                if color == 'red':
                    colored_mask[mask > 0] = [255, 0, 0]
                elif color == 'blue':
                    colored_mask[mask > 0] = [0, 0, 255]
                elif color == 'green':
                    colored_mask[mask > 0] = [0, 255, 0]
                elif color == 'yellow':
                    colored_mask[mask > 0] = [255, 255, 0]
                elif color == 'purple':
                    colored_mask[mask > 0] = [255, 0, 255]
                else:
                    colored_mask[mask > 0] = [128, 128, 128]
            else:
                # Grayscale image
                colored_mask[mask > 0] = 255

            # Overlay on original image
            overlay = img_array.copy().astype(np.float32)
            mask_3d = np.stack([mask] * 3, axis=-1) if len(img_array.shape) == 3 else mask
            overlay[mask_3d > 0] = overlay[mask_3d > 0] * (1 - alpha) + colored_mask[mask_3d > 0] * alpha

            ax.imshow(overlay.astype(np.uint8), alpha=0.7)

        # Add legend entry (only once per category)
        if category_id not in [p.get_label() for p in legend_patches]:
            legend_patches.append(patches.Patch(color=color, alpha=alpha, label=category_name))

    # Add legend
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=12, framealpha=0.8)

    ax.set_title(f'Beard Segmentation Overlay\n{os.path.basename(image_path)}', fontsize=14)
    ax.axis('off')

    plt.show()

def main():
    # Paths
    annotation_file = 'beard-dataset.v46i.coco-segmentation/test/_annotations.coco.json'
    image_dir = 'beard-dataset.v46i.coco-segmentation/test'

    # Load annotations
    print("Loading COCO annotations...")
    coco_data = load_coco_annotations(annotation_file)

    # Get all image IDs that have annotations
    image_ids_with_annotations = set(ann['image_id'] for ann in coco_data['annotations'])

    # Randomly select 5 images with beard annotations
    selected_image_ids = random.sample(list(image_ids_with_annotations), min(5, len(image_ids_with_annotations)))

    print(f"Selected {len(selected_image_ids)} images for visualization")
    print("Close each plot window to continue to the next image...")

    # Process each selected image
    for i, image_id in enumerate(selected_image_ids):
        # Get image info and annotations
        image_info = get_image_info(coco_data, image_id)
        if not image_info:
            continue

        image_filename = image_info['file_name']
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        annotations = get_image_annotations(coco_data, image_id)

        print(f"\nShowing image {i+1}/{len(selected_image_ids)}: {image_filename}")

        # Visualize directly in matplotlib
        visualize_beard_overlay(image_path, annotations, coco_data)

    print("\nAll visualizations complete!")

if __name__ == "__main__":
    main()
