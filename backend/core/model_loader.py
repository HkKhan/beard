"""
ModelLoader Singleton for SAM
Loads SAM model onto MPS device and provides thread-safe inference.
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add segment-anything to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "segment-anything"))

from segment_anything import sam_model_registry, SamPredictor


class ModelLoader:
    """
    Singleton class for loading and managing SAM model.
    Thread-safe for use with FastAPI.
    """
    
    _instance: Optional['ModelLoader'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ModelLoader':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.predictor = None
        self.device = None
        self.model_type = None
        self._inference_lock = threading.Lock()
        self._initialized = True
    
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.backends.mps.is_available():
            print("✓ Using MPS (Metal Performance Shaders)")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("✓ Using CUDA")
            return torch.device("cuda")
        else:
            print("⚠ Using CPU")
            return torch.device("cpu")
    
    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_b",
        fine_tuned_path: Optional[str] = None,
    ) -> None:
        """
        Load SAM model.
        
        Args:
            checkpoint_path: Path to original SAM checkpoint
            model_type: One of 'vit_b', 'vit_l', 'vit_h'
            fine_tuned_path: Path to fine-tuned weights (optional, overrides checkpoint)
        """
        if self.model is not None:
            print("Model already loaded")
            return
        
        self.device = self._get_device()
        self.model_type = model_type
        
        print(f"Loading SAM model ({model_type})...")
        
        # Load base model
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        else:
            self.model = sam_model_registry[model_type]()
        
        # Load fine-tuned weights if available
        if fine_tuned_path and os.path.exists(fine_tuned_path):
            print(f"Loading fine-tuned weights from: {fine_tuned_path}")
            checkpoint = torch.load(fine_tuned_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Validation IoU: {checkpoint.get('val_iou', 'unknown')}")
        
        # Move to device and set eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Create predictor
        self.predictor = SamPredictor(self.model)
        
        print(f"✓ Model loaded successfully on {self.device}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.predictor is not None
    
    def predict(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ) -> tuple:
        """
        Run SAM prediction on an image.
        
        Args:
            image: Input image (RGB, HWC, uint8)
            point_coords: Nx2 array of (x, y) point prompts
            point_labels: N array of labels (1=foreground, 0=background)
            box: Optional bounding box [x1, y1, x2, y2]
            multimask_output: Whether to return multiple masks
            
        Returns:
            masks: CxHxW binary masks
            scores: C confidence scores
            logits: CxHxW low-res logits
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with self._inference_lock:
            # Set image
            self.predictor.set_image(image)
            
            # Run prediction
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
            )
            
            return masks, scores, logits
    
    def predict_batch(
        self,
        image: np.ndarray,
        point_coords_batch: list,
        point_labels_batch: list,
    ) -> list:
        """
        Run batch predictions with multiple prompt sets on same image.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with self._inference_lock:
            self.predictor.set_image(image)
            
            results = []
            for coords, labels in zip(point_coords_batch, point_labels_batch):
                masks, scores, logits = self.predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )
                results.append({
                    'mask': masks[0],
                    'score': scores[0],
                })
            
            return results


# Global singleton instance
model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Get the global ModelLoader instance."""
    return model_loader

