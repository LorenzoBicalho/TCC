from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from controllers.federated import (
    get_latest_global_model,
    run_aggregation,
    submit_client_weights,
)
from controllers.client import register_client
from controllers.telemetry import submit_client_telemetry


from db.validators import (
    AggregateResponse,
    ClientRegisterRequest,
    ClientResponse,
    LatestModelRequest,
    LatestModelResponse,
    SubmitWeightsRequest,
    SubmitWeightsResponse,
    TelemetryRequest
)
from db.session import SessionLocal, ensure_schema, get_db

import logging
import traceback

logger = logging.getLogger(__name__)

app = FastAPI(title="Federated Learning Server")
ensure_schema()
DbDependency = Annotated[Session, Depends(get_db)]


def run_aggregation_background() -> None:
    """Use a fresh DB session; the request-scoped session is closed before tasks run."""
    db = SessionLocal()
    try:
        run_aggregation(db, bypass=False)
    except Exception:
        logger.exception("Background aggregation failed")
    finally:
        db.close()


@app.post("/clients", response_model=ClientResponse)
def register_client_endpoint(payload: ClientRegisterRequest, db: DbDependency):
    return register_client(db, payload.device_identifier, payload.description)


@app.post("/model/latest", response_model=LatestModelResponse)
def latest_model_endpoint(payload: LatestModelRequest, db: DbDependency):
    try:
        return get_latest_global_model(db, payload.device_identifier, payload.client_version)
    except ValueError as exc:
        logger.error("403 error detail: %s", traceback.format_exc())
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/weights", response_model=SubmitWeightsResponse)
def submit_weights_endpoint(payload: SubmitWeightsRequest, db: DbDependency, background_tasks: BackgroundTasks):
    try:
        # background_tasks.add_task(run_aggregation_background)
        return submit_client_weights(db, payload)

    except ValueError as exc:
        logger.error("403 error detail: %s", traceback.format_exc())
        raise HTTPException(status_code=403, detail=str(exc)) from exc

@app.post("/federated/aggregate", response_model=AggregateResponse)
def aggregate_endpoint(db: DbDependency):
    # Operators may aggregate pending submissions without waiting for client-ratio gates.
    new_version = run_aggregation(db, bypass=True)
    if new_version is None:
        return AggregateResponse(
            status="skipped",
            detail="No unused submissions for the current global model version.",
            new_version=None,
        )
    return AggregateResponse(status="success", detail="Aggregation completed.", new_version=new_version)

@app.post("/telemetry")
def submit_telemetry_endpoint(payload: TelemetryRequest, db: DbDependency):
    try:
        return submit_client_telemetry(db, payload)
    except ValueError as exc:
        logger.error("403 error detail: %s", traceback.format_exc())
        raise HTTPException(status_code=403, detail=str(exc)) from exc