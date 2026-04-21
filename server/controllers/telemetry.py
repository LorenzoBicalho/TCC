from services import telemetry_service, client_service
from db.validators import TelemetryRow
from db.schemas import Client, Telemetry
from db.validators import TelemetryRequest
from datetime import datetime, timezone
from sqlalchemy.orm import Session


def submit_client_telemetry(db: Session, payload: TelemetryRequest):
    client = client_service.get_active_by_device_identifier(db, payload.device_identifier)
    if not client:
        raise ValueError("Client is not registered or is inactive.")

    inserted = telemetry_service.insert_telemetry(db, client, payload)
    return {"inserted": inserted}