from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session
from db.schemas import Client, ClientSubmission
from db.validators import SubmitWeightsRequest

def get_by_device_identifier(db: Session, device_identifier: str) -> Client | None:
    return db.scalar(select(Client).where(Client.device_identifier == device_identifier))


def get_active_by_device_identifier(db: Session, device_identifier: str) -> Client | None:
    return db.scalar(
        select(Client).where(
            Client.device_identifier == device_identifier,
            Client.is_active.is_(True),
        )
    )

def count_active_clients(db: Session) -> int:
    return int(db.scalar(select(func.count(Client.id)).where(Client.is_active.is_(True))) or 0)

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



def count_submissions_for_client_version(db: Session, client_id, version: int) -> int:
    q = select(func.count(ClientSubmission.id)).where(
        and_(ClientSubmission.client_id == client_id, ClientSubmission.version == version)
    )
    return int(db.scalar(q) or 0)


def insert_client_submission(db: Session, client: Client, payload: SubmitWeightsRequest) -> None:
    # print(f"payload: {payload}")
    params = payload.weights
    metrics = payload.metrics
    submission = ClientSubmission(
        client_id = client.id,
        round_id = None,
        version = payload.version,
        num_samples = payload.num_samples,
        c = params.c,
        p = params.p,
        s = params.s,
        q = params.q,
        accuracy = metrics.accuracy,
        mean_percentage_error = metrics.mean_percentage_error,
        cluster_aggressive = params.cluster_aggressive,
        cluster_normal = params.cluster_normal,
        cluster_calm = params.cluster_calm,
    )
    db.add(submission)
    db.commit()


def submission_counts_by_client_for_version(db: Session, version: int) -> list[tuple]:
    return (
        db.query(ClientSubmission.client_id, func.count(ClientSubmission.id))
        .join(Client, Client.id == ClientSubmission.client_id)
        .filter(ClientSubmission.version == version, Client.is_active.is_(True))
        .group_by(ClientSubmission.client_id)
        .all()
    )
