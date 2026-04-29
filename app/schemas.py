from pydantic import BaseModel


class PredictionResponse(BaseModel):
    diagnosis: str
    confidence: float
    probabilities: dict[str, float]
    image_size: tuple[int, int]


class ReportRequest(BaseModel):
    diagnosis: str
    confidence: float
    probabilities: dict[str, float]
