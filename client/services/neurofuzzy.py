def classificate(data):
    # chama calys
    # set buzzer pinout

def calys(x, params):
    c, s, p, q = params["c"], params["s"], params["p"], params["q"]
    rule_outputs = q + np.dot(x, p)
    diff = x[:, None] - c
    exponent = -0.5 * (diff**2) / (s**2)
    rule_weights = np.exp(exponent).prod(axis=0)
    numerator = np.sum(rule_weights * rule_outputs)
    denominator = np.sum(rule_weights)
    output = numerator / (denominator + 1e-8)
    return output, rule_weights, rule_outputs, denominator

def normalize_np(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.clip((X - MIN_VALUES) * SCALE_VALUES, 0.0, 1.0)


def normalize_df(df: pd.DataFrame) -> np.ndarray:
    return normalize_np(df[FEATURES].values)

def train_model(X_train, y_train, params, num_rules, mu, alpha, epochs, shuffle: bool = False, grad_clip: float | None = None):
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
            ys, w, y, b = calys(x, num_rules, {"c": c, "s": s, "p": p, "q": q})
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
