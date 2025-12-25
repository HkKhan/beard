"""
Tests for segmentation API endpoints.
"""

import base64
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.segmentation import (
    check_vertex_containment,
    extract_boundary,
    extract_boundary_canny,
    fuse_calibration_captures,
)


class TestVertexContainment:
    """Tests for vertex containment checking."""
    
    def test_vertex_inside_mask(self):
        """Test that vertex inside mask is correctly identified."""
        # Create a 10x10 mask with center region filled
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1  # Center 4x4 region
        
        # Create landmarks - one inside, one outside
        landmarks = [
            [5, 5],  # Inside (center)
            [1, 1],  # Outside (corner)
        ]
        
        result = check_vertex_containment(mask, landmarks)
        
        assert 0 in result.beard_vertex_indices  # Vertex 0 should be inside
        assert 1 not in result.beard_vertex_indices  # Vertex 1 should be outside
    
    def test_empty_mask(self):
        """Test with empty mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        landmarks = [[5, 5], [1, 1]]
        
        result = check_vertex_containment(mask, landmarks)
        
        assert len(result.beard_vertex_indices) == 0
    
    def test_normalized_coordinates(self):
        """Test with normalized (0-1) coordinates."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1  # Center region
        
        # Normalized coordinates (should be converted)
        landmarks = [
            [0.5, 0.5],  # Center - should be inside
            [0.1, 0.1],  # Corner - should be outside
        ]
        
        result = check_vertex_containment(mask, landmarks)
        
        assert 0 in result.beard_vertex_indices
        assert 1 not in result.beard_vertex_indices


class TestBoundaryExtraction:
    """Tests for boundary extraction."""
    
    def test_extract_boundary_morphological(self):
        """Test morphological boundary extraction."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1  # 10x10 filled square
        
        # Use larger thickness for small mask
        boundary = extract_boundary(mask, thickness=3)
        
        # Boundary should be non-empty
        assert boundary.sum() > 0
        
        # Interior should be smaller than original (some boundary extracted)
        interior = (boundary == 0) & (mask == 1)
        assert interior.sum() < mask.sum()
    
    def test_extract_boundary_canny(self):
        """Test Canny edge boundary extraction."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        
        boundary = extract_boundary_canny(mask)
        
        # Boundary should be non-empty
        assert boundary.sum() > 0


class TestCalibrationFusion:
    """Tests for calibration capture fusion."""
    
    def test_union_of_indices(self):
        """Test that fusion creates union of all indices."""
        captures = [
            {'beard_indices': [1, 2], 'boundary_indices': [1]},
            {'beard_indices': [2, 3], 'boundary_indices': [3]},
        ]
        
        final_beard, final_boundary, votes = fuse_calibration_captures(captures, voting_threshold=1)
        
        assert set(final_beard) == {1, 2, 3}
        assert set(final_boundary) == {1, 3}
    
    def test_voting_threshold(self):
        """Test that voting threshold filters correctly."""
        captures = [
            {'beard_indices': [1, 2, 3], 'boundary_indices': []},
            {'beard_indices': [2, 3, 4], 'boundary_indices': []},
            {'beard_indices': [3, 4, 5], 'boundary_indices': []},
        ]
        
        # With threshold=2, only indices appearing 2+ times should be included
        final_beard, _, votes = fuse_calibration_captures(captures, voting_threshold=2)
        
        assert 1 not in final_beard  # Only appears once
        assert 2 in final_beard       # Appears twice
        assert 3 in final_beard       # Appears three times
        assert 4 in final_beard       # Appears twice
        assert 5 not in final_beard   # Only appears once
    
    def test_vote_counts(self):
        """Test that vote counts are accurate."""
        captures = [
            {'beard_indices': [1, 2], 'boundary_indices': []},
            {'beard_indices': [1, 2], 'boundary_indices': []},
            {'beard_indices': [1], 'boundary_indices': []},
        ]
        
        _, _, votes = fuse_calibration_captures(captures, voting_threshold=1)
        
        assert votes[1] == 3
        assert votes[2] == 2


class TestCoordinateConversion:
    """Tests for coordinate utilities."""
    
    def test_pixel_coord_conversion(self):
        """Test coordinate conversion from normalized to pixel."""
        # Simple inline implementation for testing
        def normalized_to_pixel(x, y, width, height):
            return {'x': x * width, 'y': y * height}
        
        result = normalized_to_pixel(0.5, 0.5, 100, 100)
        
        assert result['x'] == 50
        assert result['y'] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

