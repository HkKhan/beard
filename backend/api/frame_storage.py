"""
Frame Storage for Beard Template Building

Allows saving and loading captured frames to avoid rescanning.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import base64

# Storage directory
STORAGE_DIR = Path(__file__).parent.parent.parent / "scans"
STORAGE_DIR.mkdir(exist_ok=True)


def save_frames(template_id: str, frames: List[Dict]) -> str:
    """
    Save captured frames to disk.
    
    Args:
        template_id: Unique template ID
        frames: List of frame data dicts with keys:
            - image: base64 encoded image
            - face_mesh_landmarks: normalized landmarks
            - sam_mask_base64: optional SAM mask
            - sam_confidence: optional confidence score
    
    Returns:
        Path to saved file
    """
    file_path = STORAGE_DIR / f"{template_id}_frames.json"
    
    data = {
        "template_id": template_id,
        "frame_count": len(frames),
        "saved_at": time.strftime('%Y-%m-%d %H:%M:%S'),
        "frames": frames,
    }
    
    # Write atomically - write to temp file then rename
    import shutil
    temp_path = file_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        shutil.move(str(temp_path), str(file_path))  # Atomic rename
        print(f"[SAVE] Saved {len(frames)} frames to {file_path}")
    except Exception as e:
        print(f"[SAVE ERROR] Failed to save frames: {e}")
        # Try direct write as fallback
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"[SAVE] Fallback: Saved {len(frames)} frames to {file_path}")
        except Exception as e2:
            print(f"[SAVE ERROR] Fallback also failed: {e2}")
            raise
    
    return str(file_path)


def load_frames(template_id: str) -> Optional[List[Dict]]:
    """
    Load saved frames from disk.
    
    Args:
        template_id: Template ID to load
    
    Returns:
        List of frame data dicts or None if not found
    """
    file_path = STORAGE_DIR / f"{template_id}_frames.json"
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {data['frame_count']} frames from {file_path}")
    return data.get("frames", [])


def list_saved_scans() -> List[str]:
    """List all saved scan template IDs."""
    if not STORAGE_DIR.exists():
        return []
    
    scans = []
    for file in STORAGE_DIR.glob("*_frames.json"):
        template_id = file.stem.replace("_frames", "")
        scans.append(template_id)
    
    return sorted(scans)


def delete_scan(template_id: str) -> bool:
    """Delete a saved scan."""
    file_path = STORAGE_DIR / f"{template_id}_frames.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False

