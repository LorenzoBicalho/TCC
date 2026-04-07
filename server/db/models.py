import uuid
from datetime import datetime

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


class Base(DeclarativeBase):
    pass
class WeightsMixin:
    c = Column(ARRAY(Float, dimensions=2), nullable=False)     # [5][5]
    p = Column(ARRAY(Float, dimensions=2), nullable=False)     # [5][5]
    s = Column(ARRAY(Float, dimensions=2), nullable=False)     # [5][5]
    q = Column(ARRAY(Float), nullable=False)                   # [5]
    cluster_aggressive = Column(ARRAY(Float), nullable=False)  # [5]
    cluster_normal     = Column(ARRAY(Float), nullable=False)  # [5]
    cluster_calm       = Column(ARRAY(Float), nullable=False)  # [5]

# Tables
class Client(Base):
    __tablename__ = "clients"

    id                = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_identifier = Column(String(255), nullable=False, unique=True)  # e.g. MAC address or serial
    description       = Column(Text, nullable=True)
    registered_at     = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active         = Column(Boolean, nullable=False, default=True)

    submissions = relationship("ClientSubmission", back_populates="client")
    aggregates  = relationship("RoundClientAggregate", back_populates="client")

    def __repr__(self):
        return f"<Client id={self.id} device={self.device_identifier}>"


class FederationRound(Base):
    __tablename__ = "federation_rounds"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_number = Column(Integer, nullable=False, unique=True)
    status       = Column(String(50), nullable=False, default="pending")  # pending | in_progress | completed
    started_at   = Column(DateTime, nullable=True)
    finished_at  = Column(DateTime, nullable=True)

    submissions  = relationship("ClientSubmission", back_populates="round")
    aggregates   = relationship("RoundClientAggregate", back_populates="round")
    global_model = relationship("GlobalModelVersion", back_populates="round", uselist=False)

    def __repr__(self):
        return f"<FederationRound round={self.round_number} status={self.status}>"


class ClientSubmission(WeightsMixin, Base):
    __tablename__ = "client_submissions"

    id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    round_id  = Column(UUID(as_uuid=True), ForeignKey("federation_rounds.id"), nullable=True)
    version   = Column(Integer, nullable=False)  # model version the client was running

    submitted_at        = Column(DateTime, nullable=False, default=datetime.utcnow)
    used_in_aggregation = Column(Boolean, nullable=False, default=False)

    client = relationship("Client", back_populates="submissions")
    round  = relationship("FederationRound", back_populates="submissions")

    def __repr__(self):
        return f"<ClientSubmission id={self.id} client={self.client_id} version={self.version}>"


class RoundClientAggregate(WeightsMixin, Base):
    __tablename__ = "round_client_aggregates"

    id        = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_id  = Column(UUID(as_uuid=True), ForeignKey("federation_rounds.id"), nullable=False)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False)
    version   = Column(Integer, nullable=False)  # version produced after this aggregation

    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    round  = relationship("FederationRound", back_populates="aggregates")
    client = relationship("Client", back_populates="aggregates")

    def __repr__(self):
        return f"<RoundClientAggregate round={self.round_id} client={self.client_id}>"


class GlobalModelVersion(WeightsMixin, Base):
    __tablename__ = "global_model_versions"

    id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    round_id = Column(UUID(as_uuid=True), ForeignKey("federation_rounds.id"), nullable=False, unique=True)
    version  = Column(Integer, nullable=False, unique=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_current = Column(Boolean, nullable=False, default=False)  # only one row True at a time

    round = relationship("FederationRound", back_populates="global_model")

    def __repr__(self):
        return f"<GlobalModelVersion version={self.version} is_current={self.is_current}>"