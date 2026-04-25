from db.schemas import Features
from db.session import SessionLocal, ensure_schema
from db.validators import FeaturePayload
import uuid


ensure_schema()

def insert_data(payload: FeaturePayload, session_id: uuid) -> None:
    with SessionLocal() as db:
        record = Features(
            speed=payload.speed,
            acc_long=payload.acc_long,
            acc_lat=payload.acc_lat,
            engine_speed=payload.engine_speed,
            throttle_position=payload.throttle_position,
            session_id=session_id
        )
        db.add(record)
        db.commit()

def get_data():
    with SessionLocal() as db:
        return (
            db.query(Features)
            .order_by(Features.id.desc())
            .all()
        )

def get_data_count():

    with SessionLocal() as db:

        return (
            db.query(Features)
            .count()
        )

def delete_all_data() -> None:
    with SessionLocal() as db:
        db.query(Features).delete()
        db.commit()