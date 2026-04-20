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

def get_global_params():
    model = get_global_model()
    if model is None:
        return None
    return {
        "c": model.c,
        "p": model.p,
        "s": model.s,
        "q": model.q,
        "cluster_aggressive": model.cluster_aggressive,
        "cluster_normal": model.cluster_normal,
        "cluster_calm": model.cluster_calm,
    }

def delete_all_models() -> None:
    with SessionLocal() as db:
        db.query(GlobalModel).delete()
        db.commit()