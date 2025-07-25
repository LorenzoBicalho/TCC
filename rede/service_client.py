# src/fuzzy_clustering_pipeline.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

def map_clusters_to_classes(y_class, cluster_idx):
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

def calys(x, num_inputs, num_rules, centers, sigmas, weights, biases):
    # Calcula os y_j = q_j + sum_i p(i,j)*x(i) para todos j de uma vez
    rule_outputs = biases + np.dot(x, weights)  # shape (num_rules,)

    # Calcula os exponentes para todos i,j
    diff = x[:, None] - centers  # shape (num_inputs, num_rules)
    exponent = -0.5 * (diff ** 2) / (sigmas ** 2)  # shape (num_inputs, num_rules)

    # Calcula os pesos w_j = prod_i exp(...)
    rule_weights = np.exp(exponent).prod(axis=0)  # shape (num_rules,)

    numerator = np.sum(rule_weights * rule_outputs)
    denominator = np.sum(rule_weights)

    output = numerator / (denominator + 1e-8)
    return output, rule_weights, rule_outputs, denominator

def treinamento(X_train, y_train, num_rules, alpha=0.01, max_epochs=30):
    num_samples, num_features = X_train.shape
    
    # Inicialização dos parâmetros
    xmin = np.min(X_train, axis=0)
    xmax = np.max(X_train, axis=0)
    c = np.random.uniform(xmin[:, None], xmax[:, None], size=(num_features, num_rules))
    s = np.random.rand(num_features, num_rules)
    p = np.random.randn(num_features, num_rules) * 0.1
    q = np.random.randn(num_rules) * 0.1
    
    for epoch in range(max_epochs):
        total_error = 0
        # Embaralhar os dados (opcional, para estocasticidade)
        indices = np.random.permutation(num_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        for k in range(num_samples):
            x = X_train_shuffled[k]
            target = y_train_shuffled[k]
            
            # Forward pass usando calys otimizado
            ys, w, y, b = calys(x, num_features, num_rules, c, s, p, q)
            error = ys - target
            total_error += error ** 2
            
            # Derivadas parciais
            dys_dw = (y - ys) / (b + 1e-8)  # shape (num_rules,)
            dys_dy = w / (b + 1e-8)  # shape (num_rules,)
            
            # Diferenças e derivadas vetoriais
            diff = x[:, None] - c  # shape (num_features, num_rules)
            dw_dc = w * diff / (s ** 2)  # shape (num_features, num_rules)
            dw_ds = w * (diff ** 2) / (s ** 3)  # shape (num_features, num_rules)
            dy_dp = x[:, None]  # shape (num_features, num_rules)
            
            # Atualização dos parâmetros
            c -= alpha * error * dw_dc * dys_dw[None, :]  # shape (num_features, num_rules)
            s -= alpha * error * dw_ds * dys_dw[None, :]  # shape (num_features, num_rules)
            p -= alpha * error * dy_dp * dys_dy[None, :]  # shape (num_features, num_rules)
            q -= alpha * error * dys_dy  # shape (num_rules,)
        
        # Monitoramento do erro médio por época
        mse = total_error / num_samples
        print(f"Epoch {epoch + 1}/{max_epochs}, MSE: {mse:.6f}")
    
    return c, s, p, q, total_error / num_samples

def main():
    num_rules = 10
    num_clusters = 3  # Número de clusters do KMeans

    # Leitura dos dados e seleção de features
    data = pd.read_csv('drivers/clustered_data.csv')
    features = ['fuel_consumption', 'speed', 'acc_norm', 'throttle_position', 'engine_speed', 'deflection']
    X = data[features].values

    # Leitura dos parâmetros para o modelo fuzzy (calys)
    c = pd.read_csv('parametros_globais/c.csv', header=None).values
    s = pd.read_csv('parametros_globais/s.csv', header=None).values
    p = pd.read_csv('parametros_globais/p.csv', header=None).values
    q = pd.read_csv('parametros_globais/q.csv', header=None).values.flatten()

    print("Parâmetros lidos:")
    print(f"c: {c.shape}, s: {s.shape}, p: {p.shape}, q: {q.shape}")

    # Normalização para o modelo fuzzy (MinMax para [0, 1])
    scaler_minmax = MinMaxScaler()
    normalized_inputs = scaler_minmax.fit_transform(X)

    # Cálculo da classe com base na saída do sistema fuzzy
    num_samples = len(X)
    y_class = np.zeros(num_samples)
    for i in range(num_samples):
        y_raw, *_ = calys(normalized_inputs[i], X.shape[1], num_rules, c, s, p, q)
        y_rounded = np.round(y_raw)
        y_class[i] = max(1, min(3, y_rounded))  # Garante que y_class ∈ {1, 2, 3}

    # Nova normalização (StandardScaler) para aplicar o KMeans com melhor desempenho
    scaler_z = StandardScaler()
    X_norm = scaler_z.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    cluster_idx = kmeans.fit_predict(X_norm)

    # Mapeamento de clusters para classes (com base em y_class dominante)
    cluster_to_class = map_clusters_to_classes(y_class, cluster_idx)
    outputs = np.vectorize(cluster_to_class.get)(cluster_idx)

    # Divisão estratificada em treino/validação usando os outputs do cluster
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(sss.split(normalized_inputs, outputs))

    X_train, X_val = normalized_inputs[train_idx], normalized_inputs[val_idx]
    y_train, y_val = outputs[train_idx], outputs[val_idx]

    c, s, p, q, _ = treinamento(X_train, y_train, num_rules)

    pd.DataFrame(c).to_csv('parametros_locais/c.csv', index=False, header=None)
    pd.DataFrame(s).to_csv('parametros_locais/s.csv', index=False, header=None)
    pd.DataFrame(p).to_csv('parametros_locais/p.csv', index=False, header=None)
    pd.DataFrame(q).to_csv('parametros_locais/q.csv', index=False, header=None)

    y_pred = np.zeros(len(X_val))
    for i in range(len(X_val)):
        y_pred[i], *_ = calys(X_val[i], X_val.shape[1], num_rules, c, s, p, q)

    # Avaliação
    y_pred_rounded = np.clip(np.round(y_pred), 1, 3)  # Garante que está no intervalo esperado
    accuracy = np.mean(y_pred_rounded == y_val) * 100

    epsilon = 1e-8  # Evita divisão por zero
    error_percent = np.mean(np.abs((y_val - y_pred) / (y_val + epsilon))) * 100

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Mean Percentage Error: {error_percent:.2f}%")

if __name__ == "__main__":
    main()
