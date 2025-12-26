#!/usr/bin/env python3
"""
Simple script to create overlay images of beard masks on original images.
Saves individual overlay images that can be easily viewed.
"""

import os
import numpy as np
from PIL import Image
import argparse
from glob import glob

def create_overlay_image(image_path, mask_path, output_path, alpha=0.4):
    """
    Create and save an overlay image of beard mask on original image.

    Args:
        image_path: Path to original image
        mask_path: Path to beard mask
        output_path: Path to save overlay image
        alpha: Transparency level for overlay (0-1)
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

    # Save the result
    result_img = Image.fromarray(blended)
    result_img.save(output_path)

def create_multiple_overlays(image_dir, mask_dir, output_dir, num_examples=10, alpha=0.4):
    """
    Create overlay images for multiple examples.

    Args:
        image_dir: Directory containing original images
        mask_dir: Directory containing beard masks
        output_dir: Directory to save overlay images
        num_examples: Number of examples to create
        alpha: Transparency level for overlay
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = sorted(glob(os.path.join(image_dir, '*.jpg')))

    print(f"Creating {min(num_examples, len(image_files))} overlay images...")

    created_count = 0
    for image_path in image_files[:num_examples]:
        # Get corresponding mask path
        filename = os.path.basename(image_path).replace('.jpg', '.png')
        mask_path = os.path.join(mask_dir, filename)
        output_path = os.path.join(output_dir, f'overlay_{filename}')

        if os.path.exists(mask_path):
            try:
                create_overlay_image(image_path, mask_path, output_path, alpha)
                created_count += 1
                print(f"Created overlay for {filename}")
            except Exception as e:
                print(f"Error creating overlay for {filename}: {e}")
        else:
            print(f"Warning: Mask not found for {filename}")

    print(f"Successfully created {created_count} overlay images in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create beard mask overlay images')
    parser.add_argument('--image_dir', default='images/val',
                       help='Directory containing original images (default: images/val)')
    parser.add_argument('--mask_dir', default='beard_masks',
                       help='Directory containing beard masks (default: beard_masks)')
    parser.add_argument('--output_dir', default='beard_overlay_images',
                       help='Directory to save overlay images (default: beard_overlay_images)')
    parser.add_argument('--num_examples', type=int, default=10,
                       help='Number of overlay images to create (default: 10)')
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='Transparency level for overlay (0-1, default: 0.4)')

    args = parser.parse_args()

    # Convert to absolute paths
    image_dir = os.path.abspath(args.image_dir)
    mask_dir = os.path.abspath(args.mask_dir)
    output_dir = os.path.abspath(args.output_dir)

    create_multiple_overlays(image_dir, mask_dir, output_dir, args.num_examples, args.alpha)

if __name__ == '__main__':
    main()

