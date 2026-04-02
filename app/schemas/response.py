from pydantic import BaseModel


class HoldSegment(BaseModel):
    label: str
    confidence: float
    mask: str  # e.g. RLE or polygon


class SegmentationResponse(BaseModel):
    image_id: str
    holds: list[HoldSegment]
