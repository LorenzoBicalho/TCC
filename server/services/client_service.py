from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Client


def get_by_device_identifier(db: Session, device_identifier: str) -> Client | None:
    return db.scalar(select(Client).where(Client.device_identifier == device_identifier))


def get_active_by_device_identifier(db: Session, device_identifier: str) -> Client | None:
    return db.scalar(
        select(Client).where(
            Client.device_identifier == device_identifier,
            Client.is_active.is_(True),
        )
    )


def create_client(db: Session, device_identifier: str, description: str | None) -> Client:
    client = Client(device_identifier=device_identifier, description=description)
    db.add(client)
    db.commit()
    db.refresh(client)
    return client


def reactivate_and_update_description(
    db: Session, client: Client, description: str | None
) -> Client:
    if description is not None:
        client.description = description
    client.is_active = True
    db.commit()
    db.refresh(client)
    return client
