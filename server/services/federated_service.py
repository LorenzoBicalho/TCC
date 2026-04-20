from datetime import datetime

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from db.config import settings
from db.schemas import Client, ClientSubmission, FederationRound, GlobalModelVersion, RoundClientAggregate
from db.validators import SubmitWeightsRequest, WeightPayload
from utils import average_vectors

WEIGHT_FIELDS = (
    "c",
    "p",
    "s",
    "q",
    "accuracy",
    "mean_percentage_error",
    "cluster_aggressive",
    "cluster_normal",
    "cluster_calm",
)


def model_to_weight_payload(model: GlobalModelVersion | RoundClientAggregate) -> WeightPayload:
    data = {field: getattr(model, field) for field in WEIGHT_FIELDS}
    return WeightPayload.model_validate(data)


def get_current_global_model(db: Session) -> GlobalModelVersion | None:
    return db.scalar(select(GlobalModelVersion).where(GlobalModelVersion.is_current.is_(True)))


def count_submissions_for_client_version(db: Session, client_id, version: int) -> int:
    q = select(func.count(ClientSubmission.id)).where(
        and_(ClientSubmission.client_id == client_id, ClientSubmission.version == version)
    )
    return int(db.scalar(q) or 0)


def submission_counts_by_client_for_version(db: Session, version: int) -> list[tuple]:
    return (
        db.query(ClientSubmission.client_id, func.count(ClientSubmission.id))
        .join(Client, Client.id == ClientSubmission.client_id)
        .filter(ClientSubmission.version == version, Client.is_active.is_(True))
        .group_by(ClientSubmission.client_id)
        .all()
    )


def count_active_clients(db: Session) -> int:
    return int(db.scalar(select(func.count(Client.id)).where(Client.is_active.is_(True))) or 0)


def aggregation_condition_met_for_version(db: Session, current_version: int) -> bool:
    active_clients_count = count_active_clients(db)
    if active_clients_count == 0:
        return False

    per_client_count = submission_counts_by_client_for_version(db, current_version)
    if not per_client_count:
        return False

    submitted_clients = len(per_client_count)
    if submitted_clients == active_clients_count:
        return True

    counts = sorted((int(c) for _, c in per_client_count), reverse=True)
    lead = counts[0] - counts[-1]
    ratio = submitted_clients / active_clients_count
    return lead >= settings.min_submission_lead and ratio >= settings.min_clients_ratio_for_aggregation


def get_or_create_round(db: Session, round_number: int) -> FederationRound:
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


def collect_unused_submissions(db: Session, version: int) -> list[ClientSubmission]:
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


def insert_client_submission(db: Session, client: Client, payload: SubmitWeightsRequest) -> None:
    # print(f"payload: {payload}")
    w = payload.weights
    submission = ClientSubmission(
        client_id=client.id,
        round_id=None,
        version=payload.version,
        c=w.c,
        p=w.p,
        s=w.s,
        q=w.q,
        cluster_aggressive=w.cluster_aggressive,
        cluster_normal=w.cluster_normal,
        cluster_calm=w.cluster_calm,
    )
    db.add(submission)
    db.commit()


def run_aggregation(db: Session) -> int | None:
    current_model = get_current_global_model(db)
    current_version = current_model.version if current_model else 0
    submissions = collect_unused_submissions(db, current_version)
    if not submissions:
        return None

    next_version = current_version + 1
    round_obj = get_or_create_round(db, next_version)

    by_client: dict = {}
    for sub in submissions:
        by_client.setdefault(sub.client_id, []).append(sub)

    per_client_aggregates: list[RoundClientAggregate] = []
    for client_id, client_subs in by_client.items():
        aggregate_payload = {
            field: average_vectors([getattr(s, field) for s in client_subs]) for field in WEIGHT_FIELDS
        }
        agg = RoundClientAggregate(
            round_id=round_obj.id,
            client_id=client_id,
            version=next_version,
            **aggregate_payload,
        )
        per_client_aggregates.append(agg)
        db.add(agg)

    global_payload = {
        field: average_vectors([getattr(a, field) for a in per_client_aggregates]) for field in WEIGHT_FIELDS
    }

    if current_model:
        current_model.is_current = False

    new_global = GlobalModelVersion(
        round_id=round_obj.id,
        version=next_version,
        is_current=True,
        **global_payload,
    )
    db.add(new_global)

    for sub in submissions:
        sub.used_in_aggregation = True
        sub.round_id = round_obj.id

    round_obj.status = "completed"
    round_obj.finished_at = datetime.utcnow()

    db.commit()
    return next_version


def try_run_aggregation_if_ready(db: Session, current_version: int) -> bool:
    if not aggregation_condition_met_for_version(db, current_version):
        return False
    run_aggregation(db)
    return True
