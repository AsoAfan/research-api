from typing import Dict, List

from pydantic import BaseModel, Field


class DiscPrediction(BaseModel):
    level: str
    prediction: str
    probabilities: Dict[str, float]


class ModelResult(BaseModel):
    model_name: str
    overlay_b64: str
    overlay_labeled_b64: str
    discs: List[DiscPrediction]
    inference_ms: float


class DiscRegion(BaseModel):
    level: str
    x: float
    y: float
    w: float
    h: float


class StepTiming(BaseModel):
    step: str         # step id, e.g. "segmentation"
    elapsed_ms: float
    cached: bool = False


class PatientInfo(BaseModel):
    name: str = "—"
    age: str = "—"
    sex: str = "—"


class AnalyzeResponse(BaseModel):
    case_id: str
    original_image_b64: str
    disc_crops_b64: Dict[str, str]
    disc_regions: List[DiscRegion]
    timings: List[StepTiming] = Field(default_factory=list)
    patient: PatientInfo = Field(default_factory=PatientInfo)
    models: List[ModelResult]


class CaseSummary(BaseModel):
    case_id: str
    filename: str
    uploaded_at: str          # ISO 8601 UTC
    thumbnail_b64: str        # small JPEG (~5KB) of the central sagittal slice
    patient: PatientInfo = Field(default_factory=PatientInfo)


class CasesListResponse(BaseModel):
    cases: List[CaseSummary]
    total: int                # total number of analyzed cases on disk


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: List[str]
    totalspineseg_available: bool
