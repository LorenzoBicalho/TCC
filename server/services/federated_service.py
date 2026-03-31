from datetime import datetime

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from db.config import settings
from db.models import Client, ClientSubmission, FederationRound, GlobalModelVersion, RoundClientAggregate
from db.schemas import LatestModelResponse, SubmitWeightsRequest, SubmitWeightsResponse, WeightPayload
from utils import average_vectors


WEIGHT_FIELDS = (
    "c",
    "p",
    "s",
    "q",
    "cluster_aggressive",
    "cluster_normal",
    "cluster_calm",
)


def _payload_from_model(model: GlobalModelVersion | RoundClientAggregate) -> WeightPayload:
    return WeightPayload(**{field: getattr(model, field) for field in WEIGHT_FIELDS})


def register_client(db: Session, device_identifier: str, description: str | None) -> Client:
    client = db.scalar(select(Client).where(Client.device_identifier == device_identifier))
    if client:
        if description is not None:
            client.description = description
        client.is_active = True
        db.commit()
        db.refresh(client)
        return client

    client = Client(device_identifier=device_identifier, description=description)
    db.add(client)
    db.commit()
    db.refresh(client)
    return client


def _get_current_model(db: Session) -> GlobalModelVersion | None:
    return db.scalar(select(GlobalModelVersion).where(GlobalModelVersion.is_current.is_(True)))


def get_latest_model_for_client(db: Session, device_identifier: str, client_version: int) -> LatestModelResponse:
    client = db.scalar(select(Client).where(Client.device_identifier == device_identifier, Client.is_active.is_(True)))
    if not client:
        raise ValueError("Client is not registered or is inactive.")

    current_model = _get_current_model(db)
    if not current_model:
        return LatestModelResponse(has_update=False, current_version=0, model=None)

    if client_version < current_model.version:
        return LatestModelResponse(
            has_update=True,
            current_version=current_model.version,
            model=_payload_from_model(current_model),
        )

    return LatestModelResponse(has_update=False, current_version=current_model.version, model=None)


def _client_submission_count_for_version(db: Session, client_id, version: int) -> int:
    count_query = select(func.count(ClientSubmission.id)).where(
        and_(ClientSubmission.client_id == client_id, ClientSubmission.version == version)
    )
    return int(db.scalar(count_query) or 0)


def submit_client_weights(db: Session, payload: SubmitWeightsRequest) -> SubmitWeightsResponse:
    client = db.scalar(select(Client).where(Client.device_identifier == payload.device_identifier, Client.is_active.is_(True)))
    if not client:
        raise ValueError("Client is not registered or is inactive.")

    current_model = _get_current_model(db)
    current_version = current_model.version if current_model else 0

    if payload.version != current_version:
        return SubmitWeightsResponse(
            status="outdated",
            detail="Submission discarded because client version is outdated.",
            current_version=current_version,
            latest_model=_payload_from_model(current_model) if current_model else None,
        )

    current_count = _client_submission_count_for_version(db, client.id, payload.version)
    if current_count >= settings.max_submissions_per_client_per_version:
        return SubmitWeightsResponse(
            status="ignored",
            detail="Submission ignored because this client reached the submission limit for this version.",
            current_version=current_version,
            latest_model=None,
        )

    submission = ClientSubmission(
        client_id=client.id,
        round_id=None,
        version=payload.version,
        c=payload.weights.c,
        p=payload.weights.p,
        s=payload.weights.s,
        q=payload.weights.q,
        cluster_aggressive=payload.weights.cluster_aggressive,
        cluster_normal=payload.weights.cluster_normal,
        cluster_calm=payload.weights.cluster_calm,
    )
    db.add(submission)
    db.commit()

    triggered = maybe_run_aggregation(db, current_version=current_version)
    return SubmitWeightsResponse(
        status="success",
        detail="Submission stored successfully.",
        current_version=_get_current_model(db).version if _get_current_model(db) else current_version,
        latest_model=None,
        aggregation_triggered=triggered,
    )


def _aggregation_condition_met(db: Session, current_version: int) -> bool:
    active_clients_count = int(
        db.scalar(select(func.count(Client.id)).where(Client.is_active.is_(True)))
        or 0
    )
    if active_clients_count == 0:
        return False

    per_client_count = (
        db.query(ClientSubmission.client_id, func.count(ClientSubmission.id))
        .join(Client, Client.id == ClientSubmission.client_id)
        .filter(ClientSubmission.version == current_version, Client.is_active.is_(True))
        .group_by(ClientSubmission.client_id)
        .all()
    )
    if not per_client_count:
        return False

    submitted_clients = len(per_client_count)
    if submitted_clients == active_clients_count:
        return True

    counts = sorted((int(count) for _, count in per_client_count), reverse=True)
    most_submissions = counts[0]
    least_submissions = counts[-1]
    lead = most_submissions - least_submissions
    ratio = submitted_clients / active_clients_count
    return lead >= settings.min_submission_lead and ratio >= settings.min_clients_ratio_for_aggregation


def _get_or_create_round(db: Session, round_number: int) -> FederationRound:
    round_obj = db.scalar(select(FederationRound).where(FederationRound.round_number == round_number))
    if round_obj:
        if round_obj.status != "in_progress":
            round_obj.status = "in_progress"
            round_obj.started_at = round_obj.started_at or datetime.utcnow()
            round_obj.finished_at = None
        return round_obj

    round_obj = FederationRound(round_number=round_number, status="in_progress", started_at=datetime.utcnow())
    db.add(round_obj)
    db.flush()
    return round_obj


def _collect_submissions(db: Session, version: int) -> list[ClientSubmission]:
    return (
        db.query(ClientSubmission)
        .join(Client, Client.id == ClientSubmission.client_id)
        .filter(
            ClientSubmission.version == version,
            ClientSubmission.used_in_aggregation.is_(False),
            Client.is_active.is_(True),
        )
        .all()
    )


def maybe_run_aggregation(db: Session, current_version: int) -> bool:
    if not _aggregation_condition_met(db, current_version):
        return False
    result = run_aggregation(db)
    return result is not None


def run_aggregation(db: Session) -> int | None:
    current_model = _get_current_model(db)
    current_version = current_model.version if current_model else 0
    submissions = _collect_submissions(db, version=current_version)
    if not submissions:
        return None

    next_version = current_version + 1
    round_obj = _get_or_create_round(db, round_number=next_version)

    by_client = {}
    for sub in submissions:
        by_client.setdefault(sub.client_id, []).append(sub)

    per_client_aggregates: list[RoundClientAggregate] = []
    for client_id, client_submissions in by_client.items():
        aggregate_payload = {
            field: average_vectors([getattr(sub, field) for sub in client_submissions])
            for field in WEIGHT_FIELDS
        }
        aggregate = RoundClientAggregate(
            round_id=round_obj.id,
            client_id=client_id,
            version=next_version,
            **aggregate_payload,
        )
        per_client_aggregates.append(aggregate)
        db.add(aggregate)

    global_payload = {
        field: average_vectors([getattr(aggregate, field) for aggregate in per_client_aggregates])
        for field in WEIGHT_FIELDS
    }

    if current_model:
        current_model.is_current = False

    new_global_model = GlobalModelVersion(
        round_id=round_obj.id,
        version=next_version,
        is_current=True,
        **global_payload,
    )
    db.add(new_global_model)

    for submission in submissions:
        submission.used_in_aggregation = True
        submission.round_id = round_obj.id

    round_obj.status = "completed"
    round_obj.finished_at = datetime.utcnow()

    db.commit()
    return next_version

