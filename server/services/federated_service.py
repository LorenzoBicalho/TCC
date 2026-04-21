from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.config import settings
from db.schemas import (
    Client,
    ClientSubmission,
    FederationRound,
    GlobalModelVersion,
)
from db.validators import SubmitWeightsRequest, WeightPayload
from utils import average_vectors

import client_service

WEIGHT_FIELDS = (
    "c",
    "p",
    "s",
    "q",
    "cluster_aggressive",
    "cluster_normal",
    "cluster_calm",
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def model_to_weight_payload(model: GlobalModelVersion) -> WeightPayload:
    data = {field: getattr(model, field) for field in WEIGHT_FIELDS}
    return WeightPayload.model_validate(data)


def get_current_global_model(db: Session) -> GlobalModelVersion | None:
    return db.scalar(
        select(GlobalModelVersion).where(GlobalModelVersion.is_current.is_(True))
    )

def get_or_create_round(db: Session, round_number: int) -> FederationRound:
    round_obj = db.scalar(
        select(FederationRound).where(FederationRound.round_number == round_number)
    )
    if round_obj:
        if round_obj.status != "in_progress":
            round_obj.status = "in_progress"
            round_obj.started_at = round_obj.started_at or datetime.now(timezone.utc)
            round_obj.finished_at = None
        return round_obj

    round_obj = FederationRound(
        round_number=round_number,
        status="in_progress",
        started_at=datetime.now(timezone.utc),
    )
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


def check_aggregation_condition(db: Session, current_version: int) -> bool:
    active_clients_count = client_service.count_active_clients(db)
    if active_clients_count == 0:
        return False

    per_client_count = client_service.submission_counts_by_client_for_version(
        db, current_version
    )
    if not per_client_count:
        return False

    submitted_clients = len(per_client_count)
    if submitted_clients == active_clients_count:
        return True

    counts = sorted((int(c) for _, c in per_client_count), reverse=True)
    lead = counts[0] - counts[-1]
    ratio = submitted_clients / active_clients_count
    return (
        lead >= settings.min_submission_lead
        and ratio >= settings.min_clients_ratio_for_aggregation
    )


# ── Aggregation strategies ─────────────────────────────────────────────────────

def fed_avg(by_client: dict[str, list[ClientSubmission]]) -> dict:
    """
    Standard Federated Averaging: each client is first averaged internally
    (in case it submitted multiple batches), then all clients are averaged equally.
    """
    client_averages = [
        {field: average_vectors([getattr(s, field) for s in subs]) for field in WEIGHT_FIELDS}
        for subs in by_client.values()
    ]
    return {
        field: average_vectors([ca[field] for ca in client_averages])
        for field in WEIGHT_FIELDS
    }


def fed_avg_weighted(by_client: dict[str, list[ClientSubmission]]) -> dict:
    """
    Weighted Federated Averaging: clients with more samples contribute more
    to the global model.
    """
    client_averages = {
        client_id: {
            field: average_vectors([getattr(s, field) for s in subs])
            for field in WEIGHT_FIELDS
        }
        for client_id, subs in by_client.items()
    }
    total_samples = sum(
        sum(s.num_samples or 1 for s in subs) for subs in by_client.values()
    )

    aggregated = {field: None for field in WEIGHT_FIELDS}
    for client_id, subs in by_client.items():
        client_samples = sum(s.num_samples or 1 for s in subs)
        w = client_samples / total_samples
        avg = client_averages[client_id]
        for field in WEIGHT_FIELDS:
            vec = avg[field]
            if aggregated[field] is None:
                aggregated[field] = (
                    [v * w for v in vec]
                    if isinstance(vec[0], float)
                    else [[v * w for v in row] for row in vec]
                )
            else:
                prev = aggregated[field]
                aggregated[field] = (
                    [p + v * w for p, v in zip(prev, vec)]
                    if isinstance(vec[0], float)
                    else [[p + v * w for p, v in zip(pr, vr)] for pr, vr in zip(prev, vec)]
                )
    return aggregated


AGGREGATION_STRATEGIES = {
    "FedAvg": fed_avg,
    "FedAvgWeighted": fed_avg_weighted,
    # plug in FedProx, SCAFFOLD, FedAdam, FedDyn here when ready
}


# ── Main aggregation runner ────────────────────────────────────────────────────

def run_aggregation(db: Session, current_model: GlobalModelVersion | None) -> int | None:
    current_version = current_model.version if current_model else 0
    submissions = collect_unused_submissions(db, current_version)
    if not submissions:
        return None

    next_version = current_version + 1
    round_obj = get_or_create_round(db, next_version)

    # Group submissions by client — a client may have sent multiple batches
    # since the last FL round, so each strategy handles per-client averaging internally
    by_client: dict = {}
    for sub in submissions:
        by_client.setdefault(sub.client_id, []).append(sub)

    # Run configured aggregation strategy
    strategy_name = getattr(settings, "aggregation_type", "FedAvg")
    strategy_fn = AGGREGATION_STRATEGIES.get(strategy_name)
    if strategy_fn is None:
        raise ValueError(f"Unknown aggregation strategy: '{strategy_name}'")

    global_payload = strategy_fn(by_client)

    # Retire current model and persist new global version
    if current_model:
        current_model.is_current = False

    new_global = GlobalModelVersion(
        round_id=round_obj.id,
        version=next_version,
        is_current=True,
        **global_payload,
    )
    db.add(new_global)

    # Mark all used submissions
    for sub in submissions:
        sub.used_in_aggregation = True
        sub.round_id = round_obj.id

    round_obj.status = "completed"
    round_obj.finished_at = datetime.now(timezone.utc)

    db.commit()
    return next_version