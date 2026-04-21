from datetime import datetime, timezone

from sqlalchemy.orm import Session

from db.schemas import Client, Telemetry
from db.validators import TelemetryRequest


def insert_telemetry(db: Session, client: Client, payload: TelemetryRequest) -> int:

    now = datetime.now(timezone.utc)

    rows = [
        Telemetry(
            local_id=          row.local_id,
            client_id=         client.id,
            session_id=        payload.session_id,
            created_at=        row.created_at,
            submitted_at=      now,             
            speed=             row.speed,
            acc_long=          row.acc_long,
            acc_lat=           row.acc_lat,
            engine_speed=      row.engine_speed,
            throttle_position= row.throttle_position,
            version=           payload.version,
            classification=    row.classification,
        )
        for row in payload.telemetry
    ]

    db.bulk_save_objects(rows)
    db.commit()
    return len(rows)