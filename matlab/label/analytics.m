close all;
clear;
clc;

% === Carregar dados ===
T = readtable('clustered_data2.csv');

% === Remover linhas com velocidade 0 ===
linhas_antes = height(T);
T = T(T.speed > 0, :);
linhas_depois = height(T);
fprintf('Removidas %d linhas com speed = 0\n', linhas_antes - linhas_depois);

% === Variáveis de interesse ===
k = numel(unique(T.cluster_id));  % identificar K dinamicamente
cluster_idx = T.cluster_id;
det = string(T.deterministic_class);
fuz = string(T.fuzzy_class);
jrk = string(T.jerk_class);
grp = string(T.group_match_id);
n = height(T);

% Contagem de ocorrências por grupo e cálculo de médias
unique_grps = unique(grp);

% Inicializar tabela de resumo
summary_table = table();

for i = 1:length(unique_grps)
    g = unique_grps(i);
    idx = grp == g;

    % Contagem e percentual
    count = sum(idx);
    percent = 100 * count / height(T);

    % Médias das variáveis
    mean_vals = mean([T.fuel_consumption(idx), T.speed(idx), T.acc_norm(idx), ...
                      T.throttle_position(idx), T.engine_speed(idx), T.deflection(idx)]);

    % Adicionar à tabela de resumo
    summary_table = [summary_table; table(g, count, percent, mean_vals(1), mean_vals(2), ...
        mean_vals(3), mean_vals(4), mean_vals(5), mean_vals(6), ...
        'VariableNames', {'Grupo', 'Contagem', 'Percentual', ...
        'Media_fuel', 'Media_speed', 'Media_acc', ...
        'Media_throttle', 'Media_engine', 'Media_deflection'})];
end

% Exibir tabela final
disp('Resumo por grupo com médias das variáveis:');
disp(summary_table);

% === Análise de Coincidência (special treatment for det) ===

% Special comparison for det (Calmo matches Normal)
compare_det = @(x) (det == x) | (det == "Normal" & x == "Calmo");

% All three classifiers agree (with det's special case)
all_equal = compare_det(fuz) & compare_det(jrk) & (fuz == jrk);
count_all_equal = sum(all_equal);

% Two classifiers agree (with det's special case)
two_agree = ~all_equal & ...
           (compare_det(fuz) | ...         % det matches fuz
            compare_det(jrk) | ...         % det matches jrk
            (fuz == jrk));                 % fuz matches jrk (exact)
count_two_agree = sum(two_agree);

count_none_equal = n - count_all_equal - count_two_agree;

fprintf('\nCoincidência entre classificadores (det: Calmo≈Normal):\n');
fprintf('Total de observações         : %d\n', n);
fprintf('Coincidência total (3/3)     : %d (%.2f%%)\n', count_all_equal, 100*count_all_equal/n);
fprintf('Coincidência parcial (2/3)   : %d (%.2f%%)\n', count_two_agree, 100*count_two_agree/n);
fprintf('Nenhuma coincidência (0/3)   : %d (%.2f%%)\n', count_none_equal, 100*count_none_equal/n);

% === Função para análise por classe ===
classes_alvo = ["Agressivo", "Normal", "Calmo"];
for i = 1:length(classes_alvo)
    cls = classes_alvo(i);
    if cls == "Calmo"
        det_match = (det == "Normal");
    else
        det_match = (det == cls);
    end
    fuz_match = (fuz == cls);
    jrk_match = (jrk == cls);

    match_count = double(det_match) + double(fuz_match) + double(jrk_match);

    count_all = sum(match_count == 3);
    count_two = sum(match_count == 2);
    count_one = sum(match_count == 1);
    count_zero = sum(match_count == 0);

    fprintf('\nAnálise de Classificação "%s":\n', cls);
    fprintf('Todos classificaram "%s"  : %d (%.2f%%)\n', cls, count_all, 100*count_all/n);
    fprintf('Dois classificaram "%s"   : %d (%.2f%%)\n', cls, count_two, 100*count_two/n);
    fprintf('Um classificou "%s"       : %d (%.2f%%)\n', cls, count_one, 100*count_one/n);
    fprintf('Nenhum classificou "%s"   : %d (%.2f%%)\n', cls, count_zero, 100*count_zero/n);
end

% === Análise por Cluster ===
for i = 1:k
    fprintf('\n====================\n');
    fprintf('Cluster %d\n', i);
    fprintf('Total de pontos: %d\n', sum(cluster_idx == i));

    % Subconjuntos
    idx = (cluster_idx == i);
    det_i = det(idx);
    fuz_i = fuz(idx);
    jrk_i = jrk(idx);
    grp_i = grp(idx);

    % Tabelas de frequência
    % fprintf('\nDeterministic:\n');
    % disp(array2table(tabulate(det_i), 'VariableNames', {'Classe', 'Contagem', 'Percentual'}));
    % 
    % fprintf('Fuzzy:\n');
    % disp(array2table(tabulate(fuz_i), 'VariableNames', {'Classe', 'Contagem', 'Percentual'}));
    % 
    % fprintf('Jerk:\n');
    % disp(array2table(tabulate(jrk_i), 'VariableNames', {'Classe', 'Contagem', 'Percentual'}));

    fprintf('Grupo (Match):\n');
    disp(array2table(tabulate(grp_i), 'VariableNames', {'Grupo', 'Contagem', 'Percentual'}));

    % Variáveis para cálculo de média
    mean_vals = mean([T.fuel_consumption(idx), T.speed(idx), T.acc_norm(idx), ...
                      T.throttle_position(idx), T.engine_speed(idx), T.deflection(idx)]);
    var_names = {'Fuel_Consumption', 'Speed', 'Acc_Norm', ...
                 'Throttle_Position', 'Engine_Speed', 'Deflection'};
    T_mean = array2table(mean_vals, 'VariableNames', var_names);
    disp('Médias das variáveis do cluster:');
    disp(T_mean);
end
