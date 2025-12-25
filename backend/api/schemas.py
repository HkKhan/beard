"""
Pydantic schemas for API request/response models.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class PointPrompt(BaseModel):
    """A single point prompt."""
    x: float = Field(..., description="X coordinate in pixels")
    y: float = Field(..., description="Y coordinate in pixels")
    label: int = Field(1, description="1 for foreground, 0 for background")


class FaceMeshLandmark(BaseModel):
    """A single face mesh landmark."""
    x: float
    y: float
    z: Optional[float] = None


class SegmentationRequest(BaseModel):
    """Request body for the /segment endpoint."""
    image: str = Field(..., description="Base64-encoded image")
    user_prompts: List[PointPrompt] = Field(
        ...,
        description="List of point prompts indicating beard region"
    )
    face_mesh_landmarks: Optional[List[List[float]]] = Field(
        None,
        description="Optional 468 face mesh landmarks [[x1,y1], [x2,y2], ...]"
    )
    return_boundary: bool = Field(
        True,
        description="Whether to extract and return boundary vertices"
    )


class MaskResponse(BaseModel):
    """Response containing mask data."""
    mask_base64: str = Field(..., description="Base64-encoded PNG mask")
    mask_rle: Optional[dict] = Field(None, description="Run-length encoded mask")
    confidence: float = Field(..., description="Model confidence score")
    width: int
    height: int


class VertexContainmentResult(BaseModel):
    """Result of vertex containment analysis."""
    beard_vertex_indices: List[int] = Field(
        ...,
        description="Indices of face mesh vertices inside beard mask"
    )
    boundary_vertex_indices: List[int] = Field(
        default_factory=list,
        description="Indices of vertices near the beard boundary"
    )
    total_vertices_checked: int = Field(468)


class SegmentationResponse(BaseModel):
    """Response from the /segment endpoint."""
    success: bool
    mask: MaskResponse
    vertex_containment: Optional[VertexContainmentResult] = None
    processing_time_ms: float
    contour_points: Optional[List[List[float]]] = Field(
        None,
        description="Simplified contour points for drawing the beard outline"
    )


class CaptureObject(BaseModel):
    """
    The saved beard template object.
    This is what gets persisted for later projection.
    """
    user_id: str
    template_name: str
    created_at: str
    beard_vertex_indices: List[int] = Field(
        ...,
        description="List of mesh vertex IDs that make up the beard"
    )
    boundary_vertex_indices: List[int] = Field(
        ...,
        description="Subset of vertices that form the edge/lineup"
    )
    calibration_views: List[str] = Field(
        default_factory=list,
        description="Views used in calibration (center, left, right)"
    )


class CalibrationStep(BaseModel):
    """Single step in calibration process."""
    step: int = Field(..., description="0=center, 1=left, 2=right")
    beard_indices: List[int]
    boundary_indices: List[int]


class FusionRequest(BaseModel):
    """Request to fuse multiple calibration captures."""
    calibration_steps: List[CalibrationStep]
    voting_threshold: int = Field(
        1,
        description="Minimum votes for vertex to be included"
    )


class FusionResponse(BaseModel):
    """Response with fused beard map."""
    final_beard_indices: List[int]
    final_boundary_indices: List[int]
    vertex_vote_counts: dict = Field(
        default_factory=dict,
        description="Vote count for each vertex"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_type: Optional[str] = None

