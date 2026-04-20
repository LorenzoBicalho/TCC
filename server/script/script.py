import argparse
import os
import sys
from typing import Optional, Dict, List

import numpy as np
from sqlalchemy import select

# Allow running this file directly from `server/script`.
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from db.schemas import FederationRound, GlobalModelVersion
from db.session import SessionLocal, ensure_schema

num_rules = int(os.getenv("NUM_RULES"))
num_features = int(os.getenv("NUM_FEATURES"))

CENTROID_DEFINITIONS = {
    "aggressive": np.array([4.0, 1.3, 500.0, 12.0, 0.1], dtype=np.float64),
    "normal": np.array([6.0, 1.4, 1000.0, 13.5, 0.3], dtype=np.float64),
    "calm": np.array([14.0, 3.2, 2000.0, 25.0, 0.6], dtype=np.float64),
}


def _load_from_npz(
    npz_path: str,
    num_features: int,
    num_rules: int
) -> Optional[Dict[str, np.ndarray]]:

    try:
        with np.load(npz_path) as data:

            c = data.get("c")
            s = data.get("s")
            p = data.get("p")
            q = data.get("q")

            if any(v is None for v in (c, s, p, q)):
                return None

            c = np.asarray(c)
            s = np.asarray(s)
            p = np.asarray(p)
            q = np.asarray(q)

            if (
                c.size != num_features * num_rules
                or s.size != num_features * num_rules
                or p.size != num_features * num_rules
                or q.size != num_rules
            ):
                return None

            return {
                "c": c.reshape(num_features, num_rules),
                "s": s.reshape(num_features, num_rules),
                "p": p.reshape(num_features, num_rules),
                "q": q.reshape(num_rules),
            }

    except Exception as e:
        print(f"[WARN] Failed to load NPZ file: {npz_path}")
        print(f"[WARN] Exception: {e}")
        return None


def load_initial_params(
    num_features: int,
    num_rules: int,
    prefer_paths: Optional[List[str]] = None
) -> Optional[Dict[str, np.ndarray]]:

    paths: List[str] = []

    if prefer_paths:
        paths.extend(prefer_paths)

    # default path
    default_path = os.path.join(SERVER_DIR, "script", "final_params_central.npz")

    if not os.path.exists(default_path):
        print(f"[WARN] Could not find path: {default_path}. Falling back to random initialization.")
        return None

    loaded = _load_from_npz(
        default_path,
        num_features,
        num_rules
    )

    if loaded is None:
        print("[WARN] Could not load parameters from file. Falling back to random initialization.")
        return None

    print(f"[INFO] Initial parameters loaded from file: {default_path}")
    return loaded


def initialize_global_model(init_mode: str = "load"):

    params = None

    if init_mode == "load":

        params = load_initial_params(
            num_features,
            num_rules
        )

    if params is None:

        print("[INFO] Using RANDOM initialization for model parameters.")

        params = {
            "c": np.random.rand(num_features, num_rules),
            "s": 0.2 + 0.3 * np.random.rand(num_features, num_rules),
            "p": np.random.randn(num_features, num_rules) * 0.1,
            "q": np.random.randn(num_rules) * 0.1,
        }

    model = {
        **params,
        "centroids": {
            name: vector.copy() for name, vector in CENTROID_DEFINITIONS.items()
        },
    }

    return model


def _constant_vector(size: int, value: float) -> list[float]:
    return [float(value) for _ in range(size)]


def seed_initial_global_model(
    force: bool = False,
    init_mode: str = "load"
) -> None:

    ensure_schema()

    with SessionLocal() as db:

        has_rows = (
            db.scalar(
                select(GlobalModelVersion.id).limit(1)
            )
            is not None
        )

        if has_rows and not force:

            print(
                "Initial data already exists. Use --force to recreate."
            )
            return

        if force:

            db.query(GlobalModelVersion).delete()

            db.query(FederationRound).filter(
                FederationRound.round_number == 0
            ).delete()

            db.commit()

        bootstrap_round = FederationRound(
            round_number=0,
            status="completed"
        )

        db.add(bootstrap_round)
        db.flush()

        model_params = initialize_global_model(
            init_mode=init_mode
        )

        model = GlobalModelVersion(
            round_id=bootstrap_round.id,
            version=1,
            is_current=True,

            c=model_params["c"].tolist(),
            p=model_params["p"].tolist(),
            s=model_params["s"].tolist(),
            q=model_params["q"].tolist(),

            cluster_aggressive=np.asarray(model_params["centroids"]["aggressive"]).flatten().tolist(),
            cluster_normal=np.asarray(model_params["centroids"]["normal"]).flatten().tolist(),
            cluster_calm=np.asarray(model_params["centroids"]["calm"]).flatten().tolist(),
        )

        db.add(model)
        db.commit()

        print(
            "Initial global model seeded successfully (version=1)."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Bootstrap federated database state."
    )

    parser.add_argument(
        "--init",
        choices=["random", "load"],
        default="load",
        help="How to initialize global model parameters",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate initial global model row.",
    )

    args = parser.parse_args()

    seed_initial_global_model(
        force=args.force,
        init_mode=args.init
    )