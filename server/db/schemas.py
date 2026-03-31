from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class WeightPayload(BaseModel):
    c: list[float] = Field(min_length=10, max_length=10)
    p: list[float] = Field(min_length=50, max_length=50)
    s: list[float] = Field(min_length=50, max_length=50)
    q: list[float] = Field(min_length=50, max_length=50)
    cluster_aggressive: list[float] = Field(min_length=15, max_length=15)
    cluster_normal: list[float] = Field(min_length=15, max_length=15)
    cluster_calm: list[float] = Field(min_length=15, max_length=15)


class ClientRegisterRequest(BaseModel):
    device_identifier: str = Field(min_length=1, max_length=255)
    description: str | None = None


class ClientResponse(BaseModel):
    id: UUID
    device_identifier: str
    description: str | None
    registered_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}


class LatestModelRequest(BaseModel):
    device_identifier: str = Field(min_length=1, max_length=255)
    client_version: int = Field(ge=0)


class LatestModelResponse(BaseModel):
    has_update: bool
    current_version: int
    model: WeightPayload | None = None


class SubmitWeightsRequest(BaseModel):
    device_identifier: str = Field(min_length=1, max_length=255)
    version: int = Field(ge=0)
    weights: WeightPayload


class SubmitWeightsResponse(BaseModel):
    status: str
    detail: str
    current_version: int
    latest_model: WeightPayload | None = None
    aggregation_triggered: bool = False


class AggregateResponse(BaseModel):
    status: str
    detail: str
    new_version: int | None = None