"""
Beard Dataset Loader for SAM Fine-tuning
Loads COCO-format segmentation annotations for beard segmentation training.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "segment-anything"))

from segment_anything.utils.transforms import ResizeLongestSide


class BeardCOCODataset(Dataset):
    """
    Dataset for loading beard segmentation data in COCO format.
    Generates point prompts from mask regions for SAM training.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 1024,
        num_points: int = 3,
        include_negative_points: bool = True,
        transform: Optional[ResizeLongestSide] = None,
    ):
        """
        Args:
            data_dir: Path to the beard-dataset.v46i.coco-segmentation directory
            split: One of 'train', 'valid', 'test'
            img_size: Target image size for SAM (default 1024)
            num_points: Number of positive point prompts to sample per mask
            include_negative_points: Whether to include negative point prompts
            transform: Optional transform (uses ResizeLongestSide if None)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.num_points = num_points
        self.include_negative_points = include_negative_points
        
        # Load annotations
        split_dir = self.data_dir / split
        annotations_path = split_dir / "_annotations.coco.json"
        
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image and annotation mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Filter to only beard annotations (category id 0 or 1 based on dataset)
        self.beard_category_ids = [
            cat['id'] for cat in self.coco_data['categories'] 
            if 'beard' in cat['name'].lower()
        ]
        
        # Group annotations by image
        self.image_annotations: Dict[int, List] = {}
        for ann in self.coco_data['annotations']:
            if ann['category_id'] in self.beard_category_ids:
                img_id = ann['image_id']
                if img_id not in self.image_annotations:
                    self.image_annotations[img_id] = []
                self.image_annotations[img_id].append(ann)
        
        # Only keep images that have beard annotations
        self.valid_image_ids = list(self.image_annotations.keys())
        
        # Setup transform
        self.transform = transform or ResizeLongestSide(img_size)
        
        print(f"Loaded {len(self.valid_image_ids)} images with beard annotations from {split}")
        print(f"Beard category IDs: {self.beard_category_ids}")
    
    def __len__(self) -> int:
        return len(self.valid_image_ids)
    
    def _decode_segmentation(self, annotation: dict, height: int, width: int) -> np.ndarray:
        """Decode COCO segmentation to binary mask."""
        segmentation = annotation['segmentation']
        
        if isinstance(segmentation, dict):
            # RLE format
            if 'counts' in segmentation:
                rle = segmentation
                mask = mask_utils.decode(rle)
            else:
                raise ValueError(f"Unknown segmentation format: {segmentation.keys()}")
        elif isinstance(segmentation, list):
            # Polygon format - most common in COCO
            mask = np.zeros((height, width), dtype=np.uint8)
            for polygon in segmentation:
                if len(polygon) >= 6:  # At least 3 points
                    pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
        else:
            raise ValueError(f"Unknown segmentation type: {type(segmentation)}")
        
        return mask
    
    def _sample_points_from_mask(
        self, 
        mask: np.ndarray, 
        num_positive: int = 3,
        num_negative: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample point prompts from mask.
        
        Returns:
            point_coords: Nx2 array of (x, y) coordinates
            point_labels: N array of labels (1 = foreground, 0 = background)
        """
        # Get positive points (inside mask)
        positive_coords = np.argwhere(mask > 0)  # Returns (y, x) format
        
        if len(positive_coords) == 0:
            # Fallback: use center of image
            h, w = mask.shape
            positive_points = np.array([[w // 2, h // 2]])
            positive_labels = np.array([1])
        else:
            # Randomly sample positive points
            num_positive = min(num_positive, len(positive_coords))
            indices = random.sample(range(len(positive_coords)), num_positive)
            positive_points = positive_coords[indices][:, ::-1]  # Convert to (x, y)
            positive_labels = np.ones(num_positive, dtype=np.int64)
        
        points = [positive_points]
        labels = [positive_labels]
        
        # Get negative points (outside mask)
        if self.include_negative_points and num_negative > 0:
            negative_coords = np.argwhere(mask == 0)
            if len(negative_coords) > 0:
                num_negative = min(num_negative, len(negative_coords))
                indices = random.sample(range(len(negative_coords)), num_negative)
                negative_points = negative_coords[indices][:, ::-1]  # Convert to (x, y)
                negative_labels = np.zeros(num_negative, dtype=np.int64)
                points.append(negative_points)
                labels.append(negative_labels)
        
        return np.vstack(points), np.concatenate(labels)
    
    def _get_bounding_box(self, mask: np.ndarray) -> np.ndarray:
        """Get bounding box from mask in XYXY format."""
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            h, w = mask.shape
            return np.array([0, 0, w, h], dtype=np.float32)
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add small padding
        pad = 5
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(mask.shape[1], x_max + pad)
        y_max = min(mask.shape[0], y_max + pad)
        
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            image: Transformed image tensor [3, H, W]
            original_size: Original image size (H, W)
            point_coords: Point prompts [N, 2] in (x, y) format
            point_labels: Point labels [N] (1=foreground, 0=background)
            boxes: Bounding box [4] in XYXY format
            gt_mask: Ground truth mask [H, W]
        """
        img_id = self.valid_image_ids[idx]
        img_info = self.images[img_id]
        annotations = self.image_annotations[img_id]
        
        # Load image
        img_path = self.data_dir / self.split / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_size = (img_info['height'], img_info['width'])
        
        # Combine all beard annotations into single mask
        combined_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in annotations:
            try:
                mask = self._decode_segmentation(ann, img_info['height'], img_info['width'])
                combined_mask = np.maximum(combined_mask, mask)
            except Exception as e:
                print(f"Warning: Failed to decode annotation {ann['id']}: {e}")
                continue
        
        # Sample point prompts
        point_coords, point_labels = self._sample_points_from_mask(
            combined_mask, 
            num_positive=self.num_points,
            num_negative=1 if self.include_negative_points else 0
        )
        
        # Get bounding box
        box = self._get_bounding_box(combined_mask)
        
        # Apply transforms
        input_image = self.transform.apply_image(image)
        input_size = input_image.shape[:2]
        
        # Transform coordinates
        point_coords_transformed = self.transform.apply_coords(point_coords, original_size)
        box_transformed = self.transform.apply_boxes(box[None, :], original_size)[0]
        
        # Resize mask to input size
        gt_mask_resized = cv2.resize(
            combined_mask, 
            (input_size[1], input_size[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Pad image and mask to fixed size (1024x1024) for batching
        h, w = input_size
        pad_h = self.img_size - h
        pad_w = self.img_size - w
        
        # Pad image (H, W, C) -> add padding on right and bottom
        input_image_padded = np.pad(
            input_image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=0
        )
        
        # Pad mask (H, W) -> add padding on right and bottom
        gt_mask_padded = np.pad(
            gt_mask_resized,
            ((0, pad_h), (0, pad_w)),
            mode='constant',
            constant_values=0
        )
        
        # Convert to tensors
        image_tensor = torch.as_tensor(input_image_padded).permute(2, 0, 1).contiguous()
        
        return {
            'image': image_tensor.float(),
            'original_size': torch.tensor(original_size),
            'input_size': torch.tensor(input_size),
            'point_coords': torch.as_tensor(point_coords_transformed, dtype=torch.float32),
            'point_labels': torch.as_tensor(point_labels, dtype=torch.int64),
            'boxes': torch.as_tensor(box_transformed, dtype=torch.float32),
            'gt_mask': torch.as_tensor(gt_mask_padded, dtype=torch.float32),
            'image_id': img_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-size data."""
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'original_size': torch.stack([item['original_size'] for item in batch]),
        'input_size': torch.stack([item['input_size'] for item in batch]),
        'point_coords': [item['point_coords'] for item in batch],  # Keep as list (variable size)
        'point_labels': [item['point_labels'] for item in batch],
        'boxes': torch.stack([item['boxes'] for item in batch]),
        'gt_mask': torch.stack([item['gt_mask'] for item in batch]),
        'image_id': [item['image_id'] for item in batch],
    }


if __name__ == "__main__":
    # Test dataset loading
    dataset_path = Path(__file__).parent.parent.parent / "beard-dataset.v46i.coco-segmentation"
    
    dataset = BeardCOCODataset(str(dataset_path), split="train")
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Original size: {sample['original_size']}")
    print(f"Point coords shape: {sample['point_coords'].shape}")
    print(f"Point labels: {sample['point_labels']}")
    print(f"Box: {sample['boxes']}")
    print(f"GT mask shape: {sample['gt_mask'].shape}")
    print(f"GT mask unique values: {torch.unique(sample['gt_mask'])}")

