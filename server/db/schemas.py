from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

min_length_weights = os.getenv("MIN_LENGTH_WEIGHTS")
max_length_weights = os.getenv("MAX_LENGTH_WEIGHTS")
length_centroids = os.getenv("LENGTH_CENTROIDS")

class WeightPayload(BaseModel):
    c: list[float] = Field(min_length=max_length_weights, max_length=max_length_weights)
    p: list[float] = Field(min_length=max_length_weights, max_length=max_length_weights)
    s: list[float] = Field(min_length=max_length_weights, max_length=max_length_weights)
    q: list[float] = Field(min_length=min_length_weights, max_length=min_length_weights)
    cluster_aggressive: list[float] = Field(min_length=length_centroids, max_length=length_centroids)
    cluster_normal: list[float] = Field(min_length=length_centroids, max_length=length_centroids)
    cluster_calm: list[float] = Field(min_length=length_centroids, max_length=length_centroids)


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