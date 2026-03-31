import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    min_clients_ratio_for_aggregation: float
    min_submission_lead: int
    max_submissions_per_client_per_version: int


settings = Settings(
    database_url=os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/federated_learning",
    ),
    min_clients_ratio_for_aggregation=float(os.getenv("MIN_CLIENTS_RATIO_FOR_AGGREGATION", "0.75")),
    min_submission_lead=int(os.getenv("MIN_SUBMISSION_LEAD", "4")),
    max_submissions_per_client_per_version=int(os.getenv("MAX_SUBMISSIONS_PER_CLIENT_PER_VERSION", "5")),
)