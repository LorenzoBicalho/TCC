from sqlalchemy.orm import Session

from db.schemas import GlobalModel
from db.session import SessionLocal, ensure_schema
from db.validators import WeightPayload

ensure_schema()

def insert_global_model(payload: WeightPayload, version: int) -> None:
    with SessionLocal() as db:
        submission = GlobalModel(
            version=version,
            c=payload.get('c'),
            p=payload.get('p'),
            s=payload.get('s'),
            q=payload.get('q'),
            cluster_aggressive=payload.get('cluster_aggressive'),
            cluster_normal=payload.get('cluster_normal'),
            cluster_calm=payload.get('cluster_calm'),
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
        "c": flatten_2d(model.get('c')),  # [25]
        "p": flatten_2d(model.get('p')),  # [25]
        "s": flatten_2d(model.get('s')),  # [25]
        "q": list(model.get('q')),   # [5]
        "cluster_aggressive": list(model.get('cluster_aggressive')),  # [5]
        "cluster_normal": list(model.get('cluster_normal')),  # [5]
        "cluster_calm": list(model.get('cluster_calm')),  # [5]
    }

def delete_all_models() -> None:
    with SessionLocal() as db:
        db.query(GlobalModel).delete()
        db.commit()