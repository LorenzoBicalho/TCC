import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME")

min_clients_ratio_for_aggregation = float(
    os.getenv("MIN_CLIENTS_RATIO_FOR_AGGREGATION", "0.75")
)

min_submission_lead = int(
    os.getenv("MIN_SUBMISSION_LEAD", "4")
)

max_submissions_per_client_per_version = int(
    os.getenv("MAX_SUBMISSIONS_PER_CLIENT_PER_VERSION", "5")
)

if not db_password or not db_name:
    raise ValueError("DB_PASSWORD e DB_NAME devem estar definidos no .env")


@dataclass(frozen=True)
class Settings:
    database_url: str
    min_clients_ratio_for_aggregation: float
    min_submission_lead: int
    max_submissions_per_client_per_version: int
    
settings = Settings(
    database_url=(
        f"postgresql://postgres:{db_password}"
        f"@{db_host}:{db_port}/{db_name}"
    ),
    min_clients_ratio_for_aggregation=min_clients_ratio_for_aggregation,
    min_submission_lead=min_submission_lead,
    max_submissions_per_client_per_version=max_submissions_per_client_per_version,
)