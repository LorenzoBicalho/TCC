from datetime import datetime
from types import NoneType
from typing import Self
from uuid import UUID
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()

NUM_FEATURES = int(os.getenv("NUM_FEATURES", "5"))
NUM_RULES = int(os.getenv("NUM_RULES", "5"))
LENGTH_CENTROIDS = int(os.getenv("LENGTH_CENTROIDS", "5"))


class WeightPayload(BaseModel):
    c: list[list[float]]
    p: list[list[float]]
    s: list[list[float]]
    q: list[float]
    accuracy: float | NoneType
    mean_percentage_error: float | NoneType
    cluster_aggressive: list[float]
    cluster_normal: list[float]
    cluster_calm: list[float]
    
    @model_validator(mode="after")
    def _check_shapes(self) -> Self:
        for name in ("c", "p", "s"):
            m = getattr(self, name)
            if len(m) != NUM_FEATURES:
                raise ValueError(f"{name} must have {NUM_FEATURES} rows (NUM_FEATURES).")
            for i, row in enumerate(m):
                if len(row) != NUM_RULES:
                    raise ValueError(
                        f"{name} row {i} must have length {NUM_RULES} (NUM_RULES)."
                    )
        if len(self.q) != NUM_RULES:
            raise ValueError(f"q must have length {NUM_RULES} (NUM_RULES).")
        for name in ("cluster_aggressive", "cluster_normal", "cluster_calm"):
            v = getattr(self, name)
            if len(v) != LENGTH_CENTROIDS:
                raise ValueError(
                    f"{name} must have length {LENGTH_CENTROIDS} (LENGTH_CENTROIDS)."
                )
        return self


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
    client_version: int


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