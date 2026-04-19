import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    Null,
    String,
    Float,
    null,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass

class WeightsMixin:

    c = Column(JSON, nullable=False)  # [[...], [...]]
    p = Column(JSON, nullable=False)
    s = Column(JSON, nullable=False)

    q = Column(JSON, nullable=False)  # [...]

    cluster_aggressive = Column(JSON, nullable=False)
    cluster_normal = Column(JSON, nullable=False)
    cluster_calm = Column(JSON, nullable=False)

class GlobalModel(WeightsMixin, Base):

    __tablename__ = "global_model_table"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    version = Column(
        Integer,
        nullable=False,
        unique=True,
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    def __repr__(self):
        return f"<GlobalModel version={self.version}>"

# class LocalModel(WeightsMixin, Base):

#     __tablename__ = "local_model_table"

#     id = Column(
#         String,
#         primary_key=True,
#         default=lambda: str(uuid.uuid4()),
#     )

#     version = Column(
#         Integer,
#         nullable=False,
#         unique=True,
#     )

#     created_at = Column(
#         DateTime,
#         nullable=False,
#         default=datetime.utcnow,
#     )

#     submitted_at = Column(
#         DateTime,
#         nullable=True,
#         default=null,
#     )

#     def __repr__(self):

#         return f"<LocalModel version={self.version}>"

class Features(Base):

    __tablename__ = "features_table"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    speed = Column(Float, nullable=True)
    acc_long = Column(Float, nullable=True)
    acc_lat = Column(Float, nullable=True)
    engine_speed = Column(Float, nullable=True)
    throttle_position = Column(Float, nullable=True)

    def __repr__(self):

        return f"<OBD2Data id={self.id}>"