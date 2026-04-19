import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

db_password = os.getenv("DB_PASSWORD", "password")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "db")
db_user = os.getenv("POSTGRES_USER", "postgres")


DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


@dataclass(frozen=True)
class Settings:
    database_url: str
    
settings = Settings(
    database_url= DATABASE_URL,
)