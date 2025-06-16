% kmeans_analysis.m
% Análise de clusters KMeans com dados de direção

close all;
clear;
clc;

%% 1. Carregar os dados
T = readtable('labeled_data2.csv');

%% 2. Remover linhas com velocidade igual a zero
linhas_antes = height(T);
T = T(T.speed > 0, :);
linhas_depois = height(T);
fprintf('Removidas %d linhas com speed = 0\n', linhas_antes - linhas_depois);


%% 3. Selecionar variáveis numéricas para clustering
X = [T.fuel_consumption, T.speed, T.acc_norm,...
     T.throttle_position, T.engine_speed, T.deflection];

% Verificar valores ausentes
if any(any(ismissing(X)))
    warning('Existem valores ausentes nas variáveis numéricas. Serão removidos.');
    validRows = all(~ismissing(X), 2);
    T = T(validRows, :);
    X = X(validRows, :);
end

%% 4. Remover outliers (usando Z-score com threshold 3)
% X_z = normalize(X);
% z_thresh = 3;
% outlier_mask = any(abs(X_z) > z_thresh, 2);
% if any(outlier_mask)
%     fprintf('Removendo %d outliers com Z-score > %.1f...\n', sum(outlier_mask), z_thresh);
% end
% T = T(~outlier_mask, :);
% X = X(~outlier_mask, :);
% X_z = X_z(~outlier_mask, :);

%% 5. Normalizar novamente os dados (Z-score)
X_norm = normalize(X); % já estava normalizado, mas mantemos a consistência

k = 3;

%% 6. Executar KMeans
[cluster_idx, centroids] = kmeans(X_norm, k, 'Replicates', 10);

%% 7. Calcular distância ao centróide
dist_to_centroid = vecnorm(X_norm - centroids(cluster_idx, :), 2, 2);

%% 8. Adicionar resultados à tabela
T.cluster_id = cluster_idx;
T.cluster_score = dist_to_centroid;

%% 9. Análise cruzada
det = string(T.deterministic_class);
fuz = string(T.fuzzy_class);
jrk = string(T.jerk_class);
grp = string(T.group_match_id);

for i = 1:k
    fprintf('\n====================\n');
    fprintf('Cluster %d\n', i);
    fprintf('Total de pontos: %d\n', sum(cluster_idx == i));

    % Subconjuntos
    det_i = det(cluster_idx == i);
    fuz_i = fuz(cluster_idx == i);
    jrk_i = jrk(cluster_idx == i);
    grp_i = grp(cluster_idx == i);

    % Tabelas de frequência
    fprintf('\nDeterministic:\n');
    disp(array2table(tabulate(det_i), 'VariableNames', {'Classe', 'Contagem', 'Percentual'}));

    fprintf('Fuzzy:\n');
    disp(array2table(tabulate(fuz_i), 'VariableNames', {'Classe', 'Contagem', 'Percentual'}));

    fprintf('Jerk:\n');
    disp(array2table(tabulate(jrk_i), 'VariableNames', {'Classe', 'Contagem', 'Percentual'}));

    fprintf('Grupo (Match):\n');
    disp(array2table(tabulate(string(grp_i)), 'VariableNames', {'Grupo', 'Contagem', 'Percentual'}));
end

%% 10. Salvar nova tabela com cluster e score
writetable(T, 'clustered_data2.csv');
