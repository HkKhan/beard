#!/usr/bin/env python3
"""
Script to extract beard regions from face segmentation annotations.

Beard is assumed to be the PERSON class (1), which includes hair and beard regions.
All non-beard pixels are set to 0, beard pixels are set to 1.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def extract_beard(annotation_path, output_path):
    """
    Extract beard regions from a single annotation.

    Args:
        annotation_path: Path to input annotation PNG
        output_path: Path to save beard mask PNG
    """
    # Load annotation
    img = Image.open(annotation_path)
    annotation = np.array(img)

    # Create beard mask: 1 where PERSON class (1), 0 elsewhere
    beard_mask = (annotation == 1).astype(np.uint8)

    # Save as PNG
    mask_img = Image.fromarray(beard_mask * 255)  # Convert to 0-255 range for visibility
    mask_img.save(output_path)

def process_all_annotations(input_dir, output_dir):
    """
    Process all annotation files in input_dir and save beard masks to output_dir.

    Args:
        input_dir: Directory containing annotation PNGs
        output_dir: Directory to save beard mask PNGs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all PNG files
    annotation_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    print(f"Processing {len(annotation_files)} annotation files...")

    for filename in tqdm(annotation_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            extract_beard(input_path, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processed {len(annotation_files)} files. Beard masks saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract beard regions from face segmentation annotations')
    parser.add_argument('--input_dir', default='annotations/val',
                       help='Directory containing annotation PNGs (default: annotations/val)')
    parser.add_argument('--output_dir', default='beard_masks',
                       help='Directory to save beard mask PNGs (default: beard_masks)')

    args = parser.parse_args()

    # Convert to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    process_all_annotations(input_dir, output_dir)

if __name__ == '__main__':
    main()
