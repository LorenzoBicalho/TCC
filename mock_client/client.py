import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# --- Normalization utilities (inlined) ---
FEATURES = [
    "speed",
    "acc_norm",
    "engine_speed",
    "throttle_position",
    "delta_acc_lat",
]
MIN_VALUES = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
MAX_VALUES = np.array([120.0, 5.0, 10000.0, 100.0, 3.0], dtype=np.float32)
SCALE_VALUES = 1.0 / (MAX_VALUES - MIN_VALUES)

NUM_FEATURES = len(FEATURES)
NUM_RULES = 5
Q_LEN = NUM_RULES  # must match server q length (NUM_RULES)
CENTROID_LEN = NUM_FEATURES  # each cluster_* on server (LENGTH_CENTROIDS / NUM_FEATURES)

CLIENT_LETTERS = tuple(chr(ord("A") + i) for i in range(10))  # A..J


def normalize_np(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.clip((X - MIN_VALUES) * SCALE_VALUES, 0.0, 1.0)


def normalize_df(df: pd.DataFrame) -> np.ndarray:
    return normalize_np(df[FEATURES].values)


def calys(x, num_rules, params):
    c, s, p, q = params["c"], params["s"], params["p"], params["q"]
    rule_outputs = q + np.dot(x, p)
    diff = x[:, None] - c
    exponent = -0.5 * (diff**2) / (s**2)
    rule_weights = np.exp(exponent).prod(axis=0)
    numerator = np.sum(rule_weights * rule_outputs)
    denominator = np.sum(rule_weights)
    return numerator / (denominator + 1e-8)


def evaluate_models(
    global_params,
    local_params,
    X_test,
    y_test,
    client_X_train,
    client_y_train,
    client_X_test,
    client_y_test,
    num_rules,
    round_num,
    strategy_name="FedAvg",
):
    y_pred_global = np.array([calys(X_test[i], num_rules, global_params) for i in range(len(X_test))])
    y_pred_global_rounded = np.clip(np.round(y_pred_global), 1, 3)

    global_metrics = {
        "global_test_accuracy": accuracy_score(y_test, y_pred_global_rounded),
        "global_test_mse": mean_squared_error(y_test, y_pred_global),
        "global_test_f1_score": f1_score(y_test, y_pred_global_rounded, average="weighted", zero_division=0),
        "global_test_precision": precision_score(y_test, y_pred_global_rounded, average="weighted", zero_division=0),
        "global_test_recall": recall_score(y_test, y_pred_global_rounded, average="weighted", zero_division=0),
    }

    y_pred_local_train = np.array(
        [calys(client_X_train[i], num_rules, local_params) for i in range(len(client_X_train))]
    )
    y_pred_local_rounded_train = np.clip(np.round(y_pred_local_train), 1, 3)
    y_pred_local_test = np.array(
        [calys(client_X_test[i], num_rules, local_params) for i in range(len(client_X_test))]
    )
    y_pred_local_rounded_test = np.clip(np.round(y_pred_local_test), 1, 3)

    local_metrics = {
        "local_train_accuracy": accuracy_score(client_y_train, y_pred_local_rounded_train),
        "local_train_mse": mean_squared_error(client_y_train, y_pred_local_train),
        "local_train_f1_score": f1_score(client_y_train, y_pred_local_rounded_train, average="weighted", zero_division=0),
        "local_train_precision": precision_score(client_y_train, y_pred_local_rounded_train, average="weighted", zero_division=0),
        "local_train_recall": recall_score(client_y_train, y_pred_local_rounded_train, average="weighted", zero_division=0),
        "local_test_accuracy": accuracy_score(client_y_test, y_pred_local_rounded_test),
        "local_test_mse": mean_squared_error(client_y_test, y_pred_local_test),
        "local_test_f1_score": f1_score(client_y_test, y_pred_local_rounded_test, average="weighted", zero_division=0),
        "local_test_precision": precision_score(client_y_test, y_pred_local_rounded_test, average="weighted", zero_division=0),
        "local_test_recall": recall_score(client_y_test, y_pred_local_rounded_test, average="weighted", zero_division=0),
    }

    report_global = {"round": round_num, **global_metrics}
    report_local = {"round": round_num, **local_metrics}

    print(
        f"\n--- AVALIAÇÃO - {strategy_name.upper()} - RODADA {round_num} ---"
        f"Global recebido (Validação) -> "
        f"Acc: {report_global['global_test_accuracy']:.4f}, "
        f"MSE: {report_global['global_test_mse']:.4f}, "
        f"F1: {report_global['global_test_f1_score']:.4f}"
        f"Local treinado (Treino) -> "
        f"Acc: {report_local['local_train_accuracy']:.4f}, "
        f"MSE: {report_local['local_train_mse']:.4f}, "
        f"F1: {report_local['local_train_f1_score']:.4f}"
        f"Local treinado (Validação) -> "
        f"Acc: {report_local['local_test_accuracy']:.4f}, "
        f"MSE: {report_local['local_test_mse']:.4f}, "
        f"F1: {report_local['local_test_f1_score']:.4f}"
    )

    return report_global, report_local


def evaluate_local_only(local_params, client_X_train, client_y_train, client_X_test, client_y_test, num_rules, round_num, strategy_name="FedAvg"):
    y_pred_local_train = np.array(
        [calys(client_X_train[i], num_rules, local_params) for i in range(len(client_X_train))]
    )
    y_pred_local_rounded_train = np.clip(np.round(y_pred_local_train), 1, 3)
    y_pred_local_test = np.array(
        [calys(client_X_test[i], num_rules, local_params) for i in range(len(client_X_test))]
    )
    y_pred_local_rounded_test = np.clip(np.round(y_pred_local_test), 1, 3)
    local_metrics = {
        "local_train_accuracy": accuracy_score(client_y_train, y_pred_local_rounded_train),
        "local_train_mse": mean_squared_error(client_y_train, y_pred_local_train),
        "local_train_f1_score": f1_score(client_y_train, y_pred_local_rounded_train, average="weighted", zero_division=0),
        "local_train_precision": precision_score(client_y_train, y_pred_local_rounded_train, average="weighted", zero_division=0),
        "local_train_recall": recall_score(client_y_train, y_pred_local_rounded_train, average="weighted", zero_division=0),
        "local_test_accuracy": accuracy_score(client_y_test, y_pred_local_rounded_test),
        "local_test_mse": mean_squared_error(client_y_test, y_pred_local_test),
        "local_test_f1_score": f1_score(client_y_test, y_pred_local_rounded_test, average="weighted", zero_division=0),
        "local_test_precision": precision_score(client_y_test, y_pred_local_rounded_test, average="weighted", zero_division=0),
        "local_test_recall": recall_score(client_y_test, y_pred_local_rounded_test, average="weighted", zero_division=0),
    }
    report_local = {"round": round_num, **local_metrics}

    print(
        f"\n--- AVALIAÇÃO - {strategy_name.upper()} - RODADA {round_num} ---"
        f"Local treinado (Treino) -> "
        f"Acc: {report_local['local_train_accuracy']:.4f}, "
        f"MSE: {report_local['local_train_mse']:.4f}, "
        f"F1: {report_local['local_train_f1_score']:.4f}"
        f"Local treinado (Validação) -> "
        f"Acc: {report_local['local_test_accuracy']:.4f}, "
        f"MSE: {report_local['local_test_mse']:.4f}, "
        f"F1: {report_local['local_test_f1_score']:.4f}"
    )

    return report_local


def calys_deriv(x, num_rules, params):
    c, s, p, q = params["c"], params["s"], params["p"], params["q"]
    rule_outputs = q + np.dot(x, p)
    diff = x[:, None] - c
    exponent = -0.5 * (diff**2) / (s**2)
    rule_weights = np.exp(exponent).prod(axis=0)
    numerator = np.sum(rule_weights * rule_outputs)
    denominator = np.sum(rule_weights)
    output = numerator / (denominator + 1e-8)
    return output, rule_weights, rule_outputs, denominator


def treinamento(X_train, y_train, params, num_rules, mu, alpha, epochs, shuffle: bool = False, grad_clip: float | None = None):
    num_features = X_train.shape[1]
    c, s, p, q = params["c"].copy(), params["s"].copy(), params["p"].copy(), params["q"].copy()
    c_g, s_g, p_g, q_g = params["c"], params["s"], params["p"], params["q"]

    for _ in range(epochs):
        if shuffle:
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
        else:
            indices = range(len(X_train))
        for k in indices:
            x, target = X_train[k], y_train[k]
            ys, w, y, b = calys_deriv(x, num_rules, {"c": c, "s": s, "p": p, "q": q})
            error = ys - target

            dys_dw = (y - ys) / (b + 1e-8)
            dys_dy = w / (b + 1e-8)

            diff = x[:, None] - c
            dw_dc = w * diff / (s**2)
            dw_ds = w * (diff**2) / (s**3)

            prox_term_c = mu * (c - c_g)
            prox_term_s = mu * (s - s_g)
            prox_term_p = mu * (p - p_g)
            prox_term_q = mu * (q - q_g)

            grad_c = error * dw_dc * dys_dw[None, :] + prox_term_c
            grad_s = error * dw_ds * dys_dw[None, :] + prox_term_s
            grad_p = error * (x[:, None]) * dys_dy[None, :] + prox_term_p
            grad_q = error * dys_dy + prox_term_q

            if grad_clip is not None and grad_clip > 0:
                g = float(grad_clip)
                grad_c = np.clip(grad_c, -g, g)
                grad_s = np.clip(grad_s, -g, g)
                grad_p = np.clip(grad_p, -g, g)
                grad_q = np.clip(grad_q, -g, g)

            c -= alpha * grad_c
            s -= alpha * grad_s
            p -= alpha * grad_p
            q -= alpha * grad_q

            c = np.clip(c, 0.0, 1.0)
            s = np.clip(s, 1e-3, 10.0)
            p = np.clip(p, -5.0, 5.0)
            q = np.clip(q, -5.0, 5.0)

    return {"c": c, "s": s, "p": p, "q": q}


def _pad_truncate(arr: np.ndarray, length: int) -> list[float]:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    out = np.zeros(length, dtype=np.float64)
    n = min(length, flat.size)
    out[:n] = flat[:n]
    return out.tolist()


def _nested_matrix_for_payload(mat: np.ndarray) -> list[list[float]]:
    """c/p/s as NUM_FEATURES×NUM_RULES nested lists (matches server WeightPayload)."""
    flat = np.asarray(mat, dtype=np.float64).reshape(-1)
    cells = NUM_FEATURES * NUM_RULES
    out = np.zeros(cells, dtype=np.float64)
    n = min(cells, flat.size)
    out[:n] = flat[:n]
    return out.reshape(NUM_FEATURES, NUM_RULES).tolist()


def weight_payload_from_arrays(local_params: dict, centroids: np.ndarray | None) -> dict:
    """Build JSON body fragment matching server WeightPayload (sizes aligned with server/.env)."""
    if centroids is None or centroids.shape != (3, NUM_FEATURES):
        centroids = np.tile(np.linspace(0.2, 0.8, NUM_FEATURES), (3, 1)).astype(np.float64)

    return {
        "c": _nested_matrix_for_payload(local_params["c"]),
        "p": _nested_matrix_for_payload(local_params["p"]),
        "s": _nested_matrix_for_payload(local_params["s"]),
        "q": _pad_truncate(local_params["q"], Q_LEN),
        "cluster_aggressive": _pad_truncate(centroids[0], CENTROID_LEN),
        "cluster_normal": _pad_truncate(centroids[1], CENTROID_LEN),
        "cluster_calm": _pad_truncate(centroids[2], CENTROID_LEN),
    }


def global_params_from_weight_payload(model: dict) -> tuple[dict, np.ndarray]:
    """Server lists -> ANFIS matrices + (3, num_features) centroids."""
    c = np.asarray(model["c"], dtype=np.float64).reshape(NUM_FEATURES, NUM_RULES)
    s = np.asarray(model["s"], dtype=np.float64).reshape(NUM_FEATURES, NUM_RULES)
    p = np.asarray(model["p"], dtype=np.float64).reshape(NUM_FEATURES, NUM_RULES)
    q = np.asarray(model["q"], dtype=np.float64).reshape(NUM_RULES)
    cents = np.stack(
        [
            np.asarray(model["cluster_aggressive"], dtype=np.float64).reshape(CENTROID_LEN)[:NUM_FEATURES],
            np.asarray(model["cluster_normal"], dtype=np.float64).reshape(CENTROID_LEN)[:NUM_FEATURES],
            np.asarray(model["cluster_calm"], dtype=np.float64).reshape(CENTROID_LEN)[:NUM_FEATURES],
        ]
    )
    return {"c": c, "s": s, "p": p, "q": q}, cents


def default_global_params(rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """When server has no weights to send (e.g. first pull at version 0), match seed-style init."""
    c = rng.uniform(0.1, 0.9, size=(NUM_FEATURES, NUM_RULES)).astype(np.float64)
    s = rng.uniform(0.5, 2.0, size=(NUM_FEATURES, NUM_RULES)).astype(np.float64)
    p = rng.uniform(-0.5, 0.5, size=(NUM_FEATURES, NUM_RULES)).astype(np.float64)
    q = rng.uniform(-1.0, 1.0, size=(NUM_RULES,)).astype(np.float64)
    cents = np.tile(np.linspace(0.2, 0.8, NUM_FEATURES), (3, 1)).astype(np.float64)
    return {"c": c, "s": s, "p": p, "q": q}, cents


def sample_client_rows(df_all: pd.DataFrame, client_letter: str, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    pool = df_all[df_all["client"] == client_letter]
    if pool.empty:
        raise ValueError(f"No rows for client '{client_letter}' in CSV (column 'client').")
    if len(pool) >= n_samples:
        return pool.sample(n=n_samples, random_state=int(rng.integers(0, 2**31 - 1)))
    print(f"[Aviso] Cliente {client_letter} tem apenas {len(pool)} amostras; usando todas.")
    return pool


def device_identifier(client_letter: str) -> str:
    return f"mock_client_{client_letter}"


def print_http_response(r: requests.Response) -> None:
    """Print status and JSON (or raw body) from the server."""
    print(f"HTTP {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except (ValueError, requests.exceptions.JSONDecodeError):
        print(r.text if r.text else "(empty body)")


def api_register(
    session: requests.Session,
    base_url: str,
    letter: str,
    description: str | None = None,
) -> requests.Response:
    url = base_url.rstrip("/") + "/clients"
    return session.post(
        url,
        json={
            "device_identifier": device_identifier(letter),
            "description": description if description is not None else f"Mock driving client {letter}",
        },
        timeout=60,
    )


def api_latest(session: requests.Session, base_url: str, letter: str, client_version: int) -> dict:
    url = base_url.rstrip("/") + "/model/latest"
    r = session.post(
        url,
        json={"device_identifier": device_identifier(letter), "client_version": client_version},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def api_submit_weights(
    session: requests.Session, base_url: str, letter: str, version: int, weights: dict
) -> requests.Response:
    url = base_url.rstrip("/") + "/weights"
    return session.post(
        url,
        json={"device_identifier": device_identifier(letter), "version": version, "weights": weights},
        timeout=120,
    )


def zero_weight_payload() -> dict:
    """Valid dummy weights for POST /weights (same lengths as server WeightPayload)."""
    z = np.zeros((NUM_FEATURES, NUM_RULES), dtype=np.float64)
    qz = np.zeros(NUM_RULES, dtype=np.float64)
    return weight_payload_from_arrays({"c": z, "s": z, "p": z, "q": qz}, centroids=None)


def main(
    client_letter: str,
    base_url: str,
    storegbl: bool,
    alpha: float = 0.01,
    epochs: int = 2,
    shuffle: bool = False,
    grad_clip: float | None = None,
    mu: float = 0.0,
    n_train_samples: int = 1600,
    seed: int | None = None,
    rounds: int = 1,
):
    orig_alpha = alpha
    alpha = float(max(1e-8, abs(alpha)))
    if orig_alpha != alpha:
        print(f"[Aviso] Alpha inválido ({orig_alpha}); usando |alpha|={alpha}")

    rng = np.random.default_rng(seed)
    print(
        f"Cliente {client_letter} (device={device_identifier(client_letter)}) "
        f"-> {base_url} | {n_train_samples} amostras aleatórias para treino | "
        f"alpha={alpha}, epochs={epochs}, mu={mu}, rounds={rounds}"
    )

    root = Path(__file__).resolve().parent
    csv_path = root / "all_data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    data = pd.read_csv(csv_path)
    client_data = sample_client_rows(data, client_letter, n_train_samples, rng)

    features = FEATURES
    global_X = normalize_df(data[features])
    global_y = data["cluster_id"].values

    client_X_full = normalize_df(client_data[features])

    session = requests.Session()
    reg = api_register(session, base_url, client_letter)
    print_http_response(reg)
    reg.raise_for_status()

    client_version = 0
    history_global: list[dict] = []
    history_local: list[dict] = []

    for round_num in range(1, rounds + 1):
        latest = api_latest(session, base_url, client_letter, client_version)
        current_version = int(latest["current_version"])
        model_json = latest.get("model")

        if model_json is not None:
            global_params, cents = global_params_from_weight_payload(model_json)
        else:
            global_params, cents = default_global_params(rng)
            print(
                f"[Aviso] Servidor não enviou pesos (ex.: mesma versão). "
                f"Usando init local para treinar; current_version={current_version}."
            )

        global_params = {**global_params, "centroids": cents}

        client_y_train = np.empty(len(client_X_full), dtype=int)
        for i in range(len(client_X_full)):
            x = client_X_full[i]
            dists = np.linalg.norm(x - cents, axis=1)
            client_y_train[i] = int(np.argmin(dists)) + 1

        _, client_X_test, _, client_y_test = train_test_split(
            client_X_full, client_y_train, test_size=0.3, stratify=client_y_train, random_state=seed
        )

        local_params = treinamento(
            client_X_full,
            client_y_train,
            global_params,
            NUM_RULES,
            mu,
            alpha,
            epochs,
            shuffle=shuffle,
            grad_clip=grad_clip,
        )

        if storegbl:
            report_global, report_local = evaluate_models(
                global_params,
                local_params,
                global_X,
                global_y,
                client_X_full,
                client_y_train,
                client_X_test,
                client_y_test,
                NUM_RULES,
                round_num,
            )
            history_global.append(report_global)
            history_local.append(report_local)
        else:
            report_local = evaluate_local_only(
                local_params,
                client_X_full,
                client_y_train,
                client_X_test,
                client_y_test,
                NUM_RULES,
                round_num,
            )
            history_local.append(report_local)

        # print(f"local_params: {local_params}")
        weights_body = weight_payload_from_arrays(local_params, global_params.get("centroids"))
        # print(f"weights_body: {weights_body}")
        submit_resp = api_submit_weights(session, base_url, client_letter, client_version, weights_body)
        print_http_response(submit_resp)
        try:
            submit = submit_resp.json()
        except ValueError:
            submit_resp.raise_for_status()
            break

        status = submit.get("status")
        if status == "outdated":
            client_version = int(submit.get("current_version", current_version))
            lm = submit.get("latest_model")
            if lm:
                gp, ce = global_params_from_weight_payload(lm)
                global_params = {**gp, "centroids": ce}
            print(f"[Servidor] Pesos rejeitados (outdated). Nova versão alvo: {client_version}. Rode outra rodada.")
            continue
        if status == "ignored":
            print(f"[Servidor] Submissão ignorada: {submit.get('detail', '')}")
            break

        client_version = int(submit.get("current_version", current_version))
        print(
            f"Cliente {client_letter}: submissão OK (aggregation_triggered={submit.get('aggregation_triggered')}). "
            f"Próxima versão local: {client_version}"
        )

    base_out = os.path.join(root, "resultados", "FedAvg", "anfis")
    os.makedirs(base_out, exist_ok=True)
    local_csv = os.path.join(base_out, f"client_{client_letter}_history.csv")
    if history_local:
        pd.DataFrame(history_local).to_csv(local_csv, index=False)
    if storegbl and history_global:
        global_csv = os.path.join(base_out, "global_history.csv")
        pd.DataFrame(history_global).to_csv(global_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock client HTTP para o servidor FastAPI (server/main.py).")
    parser.add_argument(
        "--request",
        required=True,
        choices=("clients", "weights"),
        help="Endpoint: POST /clients ou POST /weights (ver server/main.py).",
    )
    parser.add_argument(
        "--client",
        required=True,
        choices=CLIENT_LETTERS,
        help="Letra do cliente no CSV (coluna 'client'): A até J; vira device_identifier mock_client_<letra>.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="URL base do FastAPI (sem barra final).",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Apenas com --request clients: descrição opcional no cadastro.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0,
        help="Apenas com --request weights (sem --train): versão do modelo enviada ao servidor.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Apenas com --request weights: treina no CSV e envia pesos reais (fluxo completo).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1600,
        help="Com --request weights --train: amostras aleatórias desse cliente.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Semente para reprodutibilidade (treino).")
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Com --request weights --train: rodadas (pull + treino + push).",
    )
    parser.add_argument(
        "--storegbl",
        action="store_true",
        help="Com --train: avaliar também modelo global no CSV inteiro.",
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--mu", type=float, default=0.0, help="FedProx mu (servidor HTTP não envia mu).")
    args = parser.parse_args()

    session = requests.Session()

    if args.request == "clients":
        r = api_register(session, args.base_url, args.client, description=args.description)
        print_http_response(r)
        if r.status_code >= 400:
            sys.exit(1)
        sys.exit(0)

    # --request weights
    if args.train:
        main(
            client_letter=args.client,
            base_url=args.base_url,
            storegbl=args.storegbl,
            alpha=args.alpha,
            epochs=args.epochs,
            shuffle=args.shuffle,
            grad_clip=args.grad_clip,
            mu=args.mu,
            n_train_samples=args.samples,
            seed=args.seed,
            rounds=args.rounds,
        )
        sys.exit(0)

    r = api_submit_weights(
        session,
        args.base_url,
        args.client,
        args.version,
        zero_weight_payload(),
    )
    print_http_response(r)
    if r.status_code >= 400:
        sys.exit(1)
