import argparse

from sqlalchemy import select

from db.models import Base, FederationRound, GlobalModelVersion
from db.session import SessionLocal, engine


def _constant_vector(size: int, value: float) -> list[float]:
    return [float(value) for _ in range(size)]


def seed_initial_global_model(force: bool = False) -> None:
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        has_rows = db.scalar(select(GlobalModelVersion.id).limit(1)) is not None
        if has_rows and not force:
            print("Initial data already exists. Use --force to recreate.")
            return

        if force:
            db.query(GlobalModelVersion).delete()
            db.query(FederationRound).filter(FederationRound.round_number == 0).delete()
            db.commit()

        bootstrap_round = FederationRound(round_number=0, status="completed")
        db.add(bootstrap_round)
        db.flush()

        model = GlobalModelVersion(
            round_id=bootstrap_round.id,
            version=0,
            is_current=True,
            c=_constant_vector(10, 0.5),
            p=_constant_vector(50, 0.0),
            s=_constant_vector(50, 1.0),
            q=_constant_vector(50, 0.0),
            cluster_aggressive=_constant_vector(15, 0.0),
            cluster_normal=_constant_vector(15, 0.0),
            cluster_calm=_constant_vector(15, 0.0),
        )
        db.add(model)
        db.commit()
        print("Initial global model seeded successfully (version=0).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap federated database state.")
    parser.add_argument("--force", action="store_true", help="Recreate initial global model row.")
    args = parser.parse_args()
    seed_initial_global_model(force=args.force)