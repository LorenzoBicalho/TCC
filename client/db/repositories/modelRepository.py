from sqlalchemy.orm import Session

from db.schemas import GlobalModel
from db.session import SessionLocal, ensure_schema
from db.validators import WeightPayload
from utils import utils

ensure_schema()

def insert_global_model(payload: WeightPayload, version: int) -> None:
    with SessionLocal() as db:
        submission = GlobalModel(
            version=version,
            c=utils.get_field(payload, "c"),
            p=utils.get_field(payload, "p"),
            s=utils.get_field(payload, "s"),
            q=utils.get_field(payload, "q"),
            cluster_aggressive=utils.get_field(payload, "cluster_aggressive"),
            cluster_normal=utils.get_field(payload, "cluster_normal"),
            cluster_calm=utils.get_field(payload, "cluster_calm"),
        )
        db.add(submission)
        db.commit()

# def insert_local_model(payload: WeightPayload, version: int) -> None:
#     with SessionLocal() as db:
#         submission = LocalModel(
#             version=version,
#             c=payload.c,
#             p=payload.p,
#             s=payload.s,
#             q=payload.q,
#             cluster_aggressive=payload.cluster_aggressive,
#             cluster_normal=payload.cluster_normal,
#             cluster_calm=payload.cluster_calm,
#         )
#         db.add(submission)
#         db.commit()

def get_global_model():
    with SessionLocal() as db:
        return (
            db.query(GlobalModel)
            .order_by(GlobalModel.version.desc())
            .first()
        )

def flatten_2d(matrix):
    if matrix is None:
        return None
    return [item for row in matrix for item in row]

def get_global_params(model):
    if model is None:
        return None

    return {
        "c": flatten_2d(utils.get_field(model, "c")),
        "p": flatten_2d(utils.get_field(model, "p")),
        "s": flatten_2d(utils.get_field(model, "s")),
        "q": utils.get_field(model, "q"),
        "cluster_aggressive": utils.get_field(model, "cluster_aggressive"),
        "cluster_normal": utils.get_field(model, "cluster_normal"),
        "cluster_calm": utils.get_field(model, "cluster_calm"),
    }

def delete_all_models() -> None:
    with SessionLocal() as db:
        db.query(GlobalModel).delete()
        db.commit()