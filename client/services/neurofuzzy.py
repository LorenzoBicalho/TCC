import numpy as np

from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from db.repositories import featuresRepository, modelRepository
from utils.utils import EPSILON, get_field
import config

def calys(x, params):
    """
    Forward pass of the Sugeno neuro-fuzzy model.
    """

    c = np.array(get_field(params, "c"), dtype=float)
    p = np.array(get_field(params, "p"), dtype=float)
    s = np.array(get_field(params, "s"), dtype=float)
    q = np.array(get_field(params, "q"), dtype=float)
    print(c)
    print(p)
    print(s)
    print(q)

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

    features = [
        "speed",
        "acc_long",
        "acc_lat",
        "engine_speed",
        "throttle_position",
    ]

    X = data[features].values

    num_clusters = config.NUM_CLUSTERS

    scaler_minmax = MinMaxScaler()
    normalized_inputs = scaler_minmax.fit_transform(X)

    num_samples = len(X)

    y_class = np.zeros(num_samples)

    current_model = modelRepository.get_global_model()

    for i in range(num_samples):
        y_raw, *_ = calys(normalized_inputs[i], current_model)

        y_rounded = np.round(y_raw)

        # Ensure class is within the valid range {1, 2, 3}
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

    xmin = np.min(X_train, axis=0)
    xmax = np.max(X_train, axis=0)

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
