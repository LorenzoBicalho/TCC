import numpy as np

from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from db.repositories import featuresRepository, modelRepository
from utils.utils import EPSILON, get_field, dict_to_feature_vector
import config


def normalize_matrix(X):
    if isinstance(X, dict):
        X = dict_to_feature_vector(X)

    if not isinstance(X, np.ndarray):
        raise TypeError(
            f"Expected X to be a numpy.ndarray or dict, got {type(X).__name__}."
        )

    try:
        X = X.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric values convertible to float.") from exc

    expected_features = len(config.FEATURE_ORDER)
    if X.ndim == 1:
        if X.shape[0] != expected_features:
            raise ValueError(
                f"Expected X to have {expected_features} features, got {X.shape[0]}."
            )
    elif X.ndim == 2:
        if X.shape[1] != expected_features:
            raise ValueError(
                "Expected X to have shape (n_samples, "
                f"{expected_features}), got {X.shape}."
            )
    else:
        raise ValueError(
            f"Expected X to be 1D or 2D, got {X.ndim}D input."
        )

    print(f'X : {X}')
    denom = (config.MAX_VALUES - config.MIN_VALUES)
    if denom.shape[0] != expected_features:
        raise ValueError(
            "Feature configuration mismatch: MIN_VALUES/MAX_VALUES length must "
            f"match FEATURE_ORDER length ({expected_features})."
        )
    denom[denom == 0] = 1.0

    X_norm = (X - config.MIN_VALUES) / denom
    X_norm = np.clip(X_norm, 0.0, 1.0)

    return X_norm

def calys(x, params):
    """
    Forward pass of the Sugeno neuro-fuzzy model.
    """
    c = np.array(get_field(params, "c"), dtype=float)
    p = np.array(get_field(params, "p"), dtype=float)
    s = np.array(get_field(params, "s"), dtype=float)
    q = np.array(get_field(params, "q"), dtype=float)

    rule_outputs = q + np.dot(x, p)

    diff = x[:, None] - c

    exponent = -0.5 * (diff ** 2) / (s ** 2)

    rule_weights = np.exp(exponent).prod(axis=0)

    numerator = np.sum(rule_weights * rule_outputs)

    denominator = np.sum(rule_weights)

    output = numerator / (denominator + EPSILON)

    return output, rule_weights, rule_outputs, denominator


def map_clusters_to_classes(y_class, cluster_idx):
    """
    Map each cluster to its dominant class label.
    """

    y_class = np.array(y_class)
    cluster_idx = np.array(cluster_idx)

    unique_clusters = np.unique(cluster_idx)

    cluster_to_class = {}
    class_counts_per_cluster = {}

    for cluster in unique_clusters:
        mask = cluster_idx == cluster
        classes_in_cluster = y_class[mask]

        class_counts = Counter(classes_in_cluster)

        class_counts_per_cluster[cluster] = class_counts

    for cluster in unique_clusters:
        class_counts = class_counts_per_cluster[cluster]

        if class_counts:
            most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
            cluster_to_class[cluster] = most_common_class
        else:
            cluster_to_class[cluster] = -1

    return cluster_to_class


def get_training_data():
    """
    Prepare dataset for training and validation.
    """
    data = featuresRepository.get_data()

    features = config.FEATURE_ORDER

    X = data[features].values.astype(float)

    num_clusters = config.NUM_CLUSTERS

    normalized_inputs = normalize_matrix(X)

    num_samples = len(X)

    y_class = np.zeros(num_samples)

    current_model = modelRepository.get_global_model()

    for i in range(num_samples):
        y_raw, *_ = calys(normalized_inputs[i], current_model)

        y_rounded = np.round(y_raw)

        y_class[i] = max(1, min(3, y_rounded))

    scaler_z = StandardScaler()
    X_norm = scaler_z.fit_transform(X)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    cluster_idx = kmeans.fit_predict(X_norm)
    cluster_to_class = map_clusters_to_classes(y_class, cluster_idx)

    outputs = np.vectorize(cluster_to_class.get)(cluster_idx)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

    train_idx, val_idx = next(sss.split(normalized_inputs, outputs))

    X_train = normalized_inputs[train_idx]
    X_val = normalized_inputs[val_idx]

    Y_train = outputs[train_idx]
    Y_val = outputs[val_idx]

    return X_train, X_val, Y_train, Y_val, current_model['version']


def evaluate_model(params, X_val, Y_val):
    """
    Evaluate trained model performance.
    """

    y_pred = np.zeros(len(X_val))

    for i in range(len(X_val)):
        y_pred[i], *_ = calys(X_val[i], params)

    y_pred_rounded = np.clip(np.round(y_pred), 1, 3)

    accuracy = np.mean(y_pred_rounded == Y_val) * 100

    error_percent = np.mean(np.abs((Y_val - y_pred) / (Y_val + EPSILON))) * 100

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Mean Percentage Error: {error_percent:.2f}%")

    metrics = {
        "accuracy": accuracy,
        "mean_percentage_error": error_percent,
    }

    return metrics


def train_model(alpha=0.001, max_epochs=10):
    """
    Train the neuro-fuzzy model using stochastic gradient descent.
    """
    num_rules = config.NUM_RULES

    X_train, X_val, Y_train, Y_val, version = get_training_data()

    num_samples, num_features = X_train.shape

    xmin = np.zeros(num_features)
    xmax = np.ones(num_features)

    c = np.random.uniform(xmin[:, None], xmax[:, None], size=(num_features, num_rules))
    s = np.random.rand(num_features, num_rules)
    p = np.random.randn(num_features, num_rules) * 0.1
    q = np.random.randn(num_rules) * 0.1

    params = {"c": c, "s": s, "p": p, "q": q}

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")

        total_error = 0

        indices = np.random.permutation(num_samples)

        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]

        for k in range(num_samples):
            x = X_train_shuffled[k]
            target = Y_train_shuffled[k]

            ys, w, y, b = calys(x, params)

            error = ys - target

            total_error += error ** 2

            dys_dw = (y - ys) / (b + EPSILON)
            dys_dy = w / (b + EPSILON)

            diff = x[:, None] - c

            dw_dc = w * diff / (s ** 2)
            dw_ds = w * (diff ** 2) / (s ** 3)
            dy_dp = x[:, None]

            c -= alpha * error * dw_dc * dys_dw[None, :]
            s -= alpha * error * dw_ds * dys_dw[None, :]
            p -= alpha * error * dy_dp * dys_dy[None, :]
            q -= alpha * error * dys_dy

        mse = total_error / num_samples

        print(f"Epoch {epoch + 1}, MSE: {mse:.6f}")

    trained_params = {"c": c, "s": s, "p": p, "q": q}

    metrics = evaluate_model(trained_params, X_val, Y_val)

    return trained_params, metrics, num_samples, version
