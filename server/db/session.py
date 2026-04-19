from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from db.config import settings


# Intentionally fixed; must match across all app processes (multi-worker Uvicorn).
_SCHEMA_INIT_ADVISORY_LOCK = 5_428_713

engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def ensure_schema() -> None:
    """Create tables if missing. Serialized via PG advisory lock so parallel workers do not race on DDL."""
    from db.schemas import Base

    with engine.connect() as conn:
        conn.execute(
            text("SELECT pg_advisory_lock(:key)"),
            {"key": _SCHEMA_INIT_ADVISORY_LOCK},
        )
        try:
            Base.metadata.create_all(bind=conn)
            conn.commit()
        finally:
            conn.execute(
                text("SELECT pg_advisory_unlock(:key)"),
                {"key": _SCHEMA_INIT_ADVISORY_LOCK},
            )
            conn.commit()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()