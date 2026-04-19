import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

DATABASE_PATH = os.getenv("DB_PATH", "db/local.db")

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

@dataclass(frozen=True)
class Settings:
    database_url: str


settings = Settings(
    database_url=DATABASE_URL,
)