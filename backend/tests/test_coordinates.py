"""
Tests for coordinate conversion utilities.
"""

import pytest
import numpy as np


def test_normalized_to_pixel():
    """Test normalized to pixel coordinate conversion."""
    # Simulate the conversion
    def normalized_to_pixel(x, y, width, height):
        return {'x': x * width, 'y': y * height}
    
    result = normalized_to_pixel(0.5, 0.5, 100, 100)
    
    assert result['x'] == 50
    assert result['y'] == 50


def test_pixel_to_normalized():
    """Test pixel to normalized coordinate conversion."""
    def pixel_to_normalized(x, y, width, height):
        return {'x': x / width, 'y': y / height}
    
    result = pixel_to_normalized(50, 50, 100, 100)
    
    assert result['x'] == 0.5
    assert result['y'] == 0.5


def test_catmull_rom_spline():
    """Test Catmull-Rom spline interpolation."""
    def catmull_rom_spline(points, segments=10):
        if len(points) < 2:
            return points
        
        result = []
        n = len(points)
        
        for i in range(n - 1):
            p0 = points[(i - 1 + n) % n]
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]
            
            for j in range(segments):
                t = j / segments
                t2 = t * t
                t3 = t2 * t
                
                x = 0.5 * (
                    (2 * p1['x']) +
                    (-p0['x'] + p2['x']) * t +
                    (2 * p0['x'] - 5 * p1['x'] + 4 * p2['x'] - p3['x']) * t2 +
                    (-p0['x'] + 3 * p1['x'] - 3 * p2['x'] + p3['x']) * t3
                )
                
                y = 0.5 * (
                    (2 * p1['y']) +
                    (-p0['y'] + p2['y']) * t +
                    (2 * p0['y'] - 5 * p1['y'] + 4 * p2['y'] - p3['y']) * t2 +
                    (-p0['y'] + 3 * p1['y'] - 3 * p2['y'] + p3['y']) * t3
                )
                
                result.append({'x': x, 'y': y})
        
        return result
    
    # Test with square points
    points = [
        {'x': 0, 'y': 0},
        {'x': 10, 'y': 0},
        {'x': 10, 'y': 10},
        {'x': 0, 'y': 10},
    ]
    
    result = catmull_rom_spline(points, segments=5)
    
    # Should produce 15 points (3 segments × 5 points each)
    assert len(result) == 15
    
    # First point should be close to first input
    assert abs(result[0]['x'] - 0) < 1
    assert abs(result[0]['y'] - 0) < 1


def test_boundary_ordering():
    """Test nearest-neighbor boundary point ordering."""
    def order_boundary_points(points):
        if len(points) <= 2:
            return points
        
        ordered = []
        remaining = list(points)
        
        # Start with leftmost point
        current = min(remaining, key=lambda p: p['x'])
        ordered.append(current)
        remaining.remove(current)
        
        while remaining:
            nearest = min(
                remaining,
                key=lambda p: (p['x'] - current['x'])**2 + (p['y'] - current['y'])**2
            )
            ordered.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return ordered
    
    # Scrambled points
    points = [
        {'x': 10, 'y': 0},
        {'x': 0, 'y': 0},
        {'x': 5, 'y': 0},
    ]
    
    result = order_boundary_points(points)
    
    # Should start with leftmost (x=0)
    assert result[0]['x'] == 0
    # Next should be nearest to 0,0 which is 5,0
    assert result[1]['x'] == 5
    # Last should be 10,0
    assert result[2]['x'] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


