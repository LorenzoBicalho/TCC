from sqlalchemy.orm import Session

from db.config import settings
from db.validators import (
    LatestModelResponse,
    SubmitWeightsRequest,
    SubmitWeightsResponse,
)
from services import client_service
from services import federated_service


def get_latest_global_model(
    db: Session, device_identifier: str, client_version: int
) -> LatestModelResponse:
    client = client_service.get_active_by_device_identifier(db, device_identifier)
    if not client:
        raise ValueError("Client is not registered or is inactive.")

    current = federated_service.get_current_global_model(db)
    if not current:
        return LatestModelResponse(has_update=False, current_version=0, model=None)

    if client_version < current.version:
        return LatestModelResponse(
            has_update=True,
            current_version=current.version,
            model=federated_service.model_to_weight_payload(current),
        )

    return LatestModelResponse(
        has_update=False,
        current_version=current.version,
        model=None,
    )


def submit_client_weights(db: Session, payload: SubmitWeightsRequest) -> SubmitWeightsResponse:
    client = client_service.get_active_by_device_identifier(db, payload.device_identifier)
    if not client:
        raise ValueError("Client is not registered or is inactive.")

    current = federated_service.get_current_global_model(db)
    current_version = current.version if current else 0

    if payload.version != current_version:
        return SubmitWeightsResponse(
            status="outdated",
            detail="Submission discarded because client version is outdated.",
            current_version=current_version,
            latest_model=federated_service.model_to_weight_payload(current) if current else None,
        )

    # n_existing = client_service.count_submissions_for_client_version(
    #     db, client.id, payload.version
    # )
    # if n_existing >= settings.max_submissions_per_client_per_version:
    #     return SubmitWeightsResponse(
    #         status="ignored",
    #         detail="Submission ignored because this client reached the submission limit for this version.",
    #         current_version=current_version,
    #         latest_model=None,
    #     )

    client_service.insert_client_submission(db, client, payload)

    return SubmitWeightsResponse(
        status="success",
        detail="Submission stored successfully.",
        current_version=current_version,
        latest_model=None,
    )

def run_aggregation(db: Session, bypass: bool = False) -> int | None:
    """Run global aggregation. When bypass is False, gated by check_aggregation_condition."""
    current_model = federated_service.get_current_global_model(db)
    current_version = current_model.version if current_model else 0

    if not bypass:
        aggregation_available = federated_service.check_aggregation_condition(db, current_version)
        if not aggregation_available:
            return None

    return federated_service.run_aggregation(db, current_model)
