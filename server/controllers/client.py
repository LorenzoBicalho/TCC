from sqlalchemy.orm import Session

from db.models import Client
from services import client_service


def register_client(db: Session, device_identifier: str, description: str | None) -> Client:
    existing = client_service.get_by_device_identifier(db, device_identifier)
    if existing:
        return client_service.reactivate_and_update_description(db, existing, description)
    return client_service.create_client(db, device_identifier, description)
