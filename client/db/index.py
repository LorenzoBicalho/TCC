# A posição desse arquivo e o nome dele faz sentido?

from typing import Annotated
from sqlalchemy.orm import Session
from db.schemas import (
    AggregateResponse,
    ClientRegisterRequest,
    ClientResponse,
    LatestModelRequest,
    LatestModelResponse,
    SubmitWeightsRequest,
    SubmitWeightsResponse,
)
from db.session import ensure_schema, get_db
db: Session
ensure_schema()

def insert_client_submission(payload: SubmitWeightsRequest) -> None:
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

def insert_data(payload):
    pass