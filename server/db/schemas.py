import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.types import Float
from sqlalchemy import Index

class Base(DeclarativeBase):
    pass
class WeightsMixin:
    c = Column(ARRAY(Float, dimensions=2), nullable=False)     # [5][5]
    p = Column(ARRAY(Float, dimensions=2), nullable=False)     # [5][5]
    s = Column(ARRAY(Float, dimensions=2), nullable=False)     # [5][5]
    q = Column(ARRAY(Float), nullable=False)                   # [5]
    accuracy = Column(Float, nullable=True)
    mean_percentage_error = Column(Float, nullable=True)
    cluster_aggressive = Column(ARRAY(Float), nullable=False)  # [5]
    cluster_normal     = Column(ARRAY(Float), nullable=False)  # [5]
    cluster_calm       = Column(ARRAY(Float), nullable=False)  # [5]

# Tables
class Client(Base):
    __tablename__ = "clients"

    id                = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_identifier = Column(String(255), nullable=False, unique=True)  # e.g. MAC address or serial
    description       = Column(Text, nullable=True)
    registered_at     = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    is_active         = Column(Boolean, nullable=False, default=True)

    submissions = relationship("ClientSubmission", back_populates="client")
    telemetry   = relationship("Telemetry", back_populates="client")


    def __repr__(self):
        return f"<Client id={self.id} device={self.device_identifier}>"

class ClientSubmission(WeightsMixin, Base):
    __tablename__ = "client_submissions"

    id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    round_id  = Column(UUID(as_uuid=True), ForeignKey("federation_rounds.id"), nullable=True)
    version   = Column(Integer, nullable=False)  # model version the client was running
    num_samples = Column(Integer, nullable=True)

    submitted_at        = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    used_in_aggregation = Column(Boolean, nullable=False, default=False)

    client = relationship("Client", back_populates="submissions")
    round  = relationship("FederationRound", back_populates="submissions")

    def __repr__(self):
        return f"<ClientSubmission id={self.id} client={self.client_id} version={self.version}>"

class FederationRound(Base):
    __tablename__ = "federation_rounds"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_number = Column(Integer, nullable=False, unique=True)
    aggregation_type = Column(String(50), nullable=True, default="Avg")  # avg | other types of federatedd
    status       = Column(String(50), nullable=True, default="pending")  # pending | in_progress | completed
    started_at   = Column(DateTime(timezone=True), nullable=True)
    finished_at  = Column(DateTime(timezone=True), nullable=True)

    submissions  = relationship("ClientSubmission", back_populates="round")
    global_model = relationship("GlobalModelVersion", back_populates="round", uselist=False)

    def __repr__(self):
        return f"<FederationRound round={self.round_number} status={self.status}>"

class GlobalModelVersion(WeightsMixin, Base):
    __tablename__ = "global_model_versions"

    id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_id = Column(UUID(as_uuid=True), ForeignKey("federation_rounds.id"), nullable=False, unique=True)
    version  = Column(Integer, nullable=False, unique=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    is_current = Column(Boolean, nullable=False, default=False)  # only one row True at a time

    round = relationship("FederationRound", back_populates="global_model")

    def __repr__(self):
        return f"<GlobalModelVersion version={self.version} is_current={self.is_current}>"

class Telemetry(Base):

    __tablename__ = "telemetry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    local_id = Column(Integer)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    submitted_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    speed = Column(Float, nullable=True)
    acc_long = Column(Float, nullable=True)
    acc_lat = Column(Float, nullable=True)
    engine_speed = Column(Float, nullable=True)
    throttle_position = Column(Float, nullable=True)

    version        = Column(Integer, nullable=False)
    classification = Column(Integer, nullable=False)

    client = relationship("Client", back_populates="telemetry")

    def __repr__(self):

        return f"<Telemetry id={self.local_id} from client {self.client_id}>"

Index('ix_telemetry_client_time', Telemetry.client_id, Telemetry.created_at.desc())
Index('ix_telemetry_session', Telemetry.session_id)