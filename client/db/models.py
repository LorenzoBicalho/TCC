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
    Float
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

class GlobalModel(WeightsMixin, Base):
    __tablename__ = "global_model"

    id       = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version  = Column(Integer, nullable=False, unique=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<GlobalModelVersion version={self.version} is_current={self.is_current}>"

class OBD2Data(Base):
    __tablename__ = "obd2_data"

    id                = Column(Integer, autoincremet=True, primary_key=True) # Check if incremet is right 
    version           = Column(Integer, nullable=False, unique=True)
    created_at        = Column(DateTime, nullable=False, default=datetime.utcnow)
    speed             = Column(Float, nullable=True)
    acc_long          = Column(Float, nullable=True)
    acc_lat           = Column(Float, nullable=True)
    engine_speed      = Column(Float, nullable=True)
    throttle_position = Column(Float, nullable=True)

    def __repr__(self):
        return f"<GlobalModelVersion version={self.version} is_current={self.is_current}>"