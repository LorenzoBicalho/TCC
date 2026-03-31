from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from db.models import Base
from db.schemas import (
    AggregateResponse,
    ClientRegisterRequest,
    ClientResponse,
    LatestModelRequest,
    LatestModelResponse,
    SubmitWeightsRequest,
    SubmitWeightsResponse,
)
from db.session import engine, get_db
from services.federated_service import (
    get_latest_model_for_client,
    register_client,
    run_aggregation,
    submit_client_weights,
)

app = FastAPI(title="Federated Learning Server")
Base.metadata.create_all(bind=engine)
DbDependency = Annotated[Session, Depends(get_db)]


@app.post("/clients", response_model=ClientResponse)
def register_client_endpoint(payload: ClientRegisterRequest, db: DbDependency):
    return register_client(db, payload.device_identifier, payload.description)


@app.post("/model/latest", response_model=LatestModelResponse)
def latest_model_endpoint(payload: LatestModelRequest, db: DbDependency):
    try:
        return get_latest_model_for_client(db, payload.device_identifier, payload.client_version)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/weights", response_model=SubmitWeightsResponse)
def submit_weights_endpoint(payload: SubmitWeightsRequest, db: DbDependency):
    try:
        return submit_client_weights(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/federated/aggregate", response_model=AggregateResponse)
def aggregate_endpoint(db: DbDependency):
    new_version = run_aggregation(db)
    if new_version is None:
        return AggregateResponse(status="skipped", detail="No eligible submissions to aggregate.", new_version=None)
    return AggregateResponse(status="success", detail="Aggregation completed.", new_version=new_version)