#!/usr/bin/env python3
"""
Script to visualize beard mask overlays on original images.
Shows the original image with beard mask overlaid with transparency.
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from glob import glob

def create_overlay(image_path, mask_path, alpha=0.5):
    """
    Create an overlay of beard mask on original image.

    Args:
        image_path: Path to original image
        mask_path: Path to beard mask
        alpha: Transparency level for overlay (0-1)

    Returns:
        PIL Image with overlay
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Load beard mask (grayscale)
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask)

    # Create colored overlay (red for beard regions)
    overlay = np.zeros_like(image_array)
    overlay[mask_array > 0] = [255, 0, 0]  # Red color for beard

    # Blend image with overlay
    blended = image_array * (1 - alpha) + overlay * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)

def visualize_examples(image_dir, mask_dir, num_examples=5, save_dir=None):
    """
    Create visualizations for multiple examples.

    Args:
        image_dir: Directory containing original images
        mask_dir: Directory containing beard masks
        num_examples: Number of examples to show
        save_dir: Directory to save overlay images (optional)
    """
    # Get list of image files
    image_files = sorted(glob(os.path.join(image_dir, '*.jpg')))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Creating overlays for {min(num_examples, len(image_files))} examples...")

    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)

    for i, image_path in enumerate(image_files[:num_examples]):
        # Get corresponding mask path
        filename = os.path.basename(image_path).replace('.jpg', '.png')
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {filename}")
            continue

        # Create overlay
        overlay_img = create_overlay(image_path, mask_path)

        # Load original image and mask for display
        orig_img = Image.open(image_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')

        # Plot
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_img, cmap='gray')
        axes[i, 1].set_title('Beard Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(overlay_img)
        axes[i, 2].set_title('Overlay (Red = Beard)')
        axes[i, 2].axis('off')

        if save_dir:
            overlay_img.save(os.path.join(save_dir, f'overlay_{filename}'))

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize beard mask overlays on images')
    parser.add_argument('--image_dir', default='images/val',
                       help='Directory containing original images (default: images/val)')
    parser.add_argument('--mask_dir', default='beard_masks',
                       help='Directory containing beard masks (default: beard_masks)')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of examples to visualize (default: 5)')
    parser.add_argument('--save_dir', default=None,
                       help='Directory to save overlay images (optional)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Transparency level for overlay (0-1, default: 0.5)')

    args = parser.parse_args()

    # Convert to absolute paths
    image_dir = os.path.abspath(args.image_dir)
    mask_dir = os.path.abspath(args.mask_dir)
    save_dir = os.path.abspath(args.save_dir) if args.save_dir else None

    visualize_examples(image_dir, mask_dir, args.num_examples, save_dir)

if __name__ == '__main__':
    main()
