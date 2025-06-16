close all;
clear all;
clc;

% planilha = 'raw_db/KIA A.csv';
planilha = 'raw_db/Driving Data(A-J).csv';

% Ler os dados do arquivo CSV
data = readtable(planilha);

time_index = 52;
time_series = table2array(data(:, time_index));

% Criar a coluna 'run'
run = zeros(size(time_series));
current_run = 1;
run(1) = current_run;

for i = 2:length(time_series)
    if time_series(i) < time_series(i-1)
        current_run = current_run + 1;
    end
    run(i) = current_run;
end

data.run = run;
data.id = (1:height(data))';

% Posições a excluir
% Adiciona +1 a todas as posições porque a nova coluna 'run' foi inserida ao final
colsToRemove = [6, 8, 9, 10, 11, 18, 22, 23, 25, 26, 27, 28, 29, 30, ...
                33, 41, 42, 43, 46, 47, 54];

processed_db = data;
processed_db(:, colsToRemove) = [];

% Definição dos índices das colunas (baseado na ordem do Excel/A-Z)
fuel_consumption_index = 1;
throttle_position_index = 3;  % C
engine_speed_index = 8;       % H
vehicle_speed_index = 26;     % Z
acc_longitudinal_index = 27;  % AA
acc_lateral_index = 29;       % AC
steering_angle_index = 31;    % AE
time_index = 32;              % AF
run_index = 34;               % AH
id_index = 35;                % AI

data = processed_db;
unique_run = unique(data{:, run_index});
output_table = [];

% Fuzzy system
fuzzy_system = readfis('fuzzy_logic.fis');

% Parâmetros do escore determinístico
w_throttle = 0.33;
w_rpm = 0.33;
w_deflection = 0.33;

threshold_throttle = 38;
threshold_rpm = 3300;
threshold_deflection = 355;

max_throttle = 100;
max_rpm = 6000;
max_deflection = 500;

% Jerk normalization
J_norm = 0.2732;

for i = 1:length(unique_run)
    run_val = unique_run(i);
    subset = data(data{:, run_index} == run_val, :);

    % Coleta de colunas
    id_values = subset{:, id_index};
    run_values = subset{:, run_index};
    time_values = subset{:, time_index};
    fuel_consumption_values = subset{:, fuel_consumption_index};
    speed_values = subset{:, vehicle_speed_index};
    acc_long_values = subset{:, acc_longitudinal_index};
    acc_lat_values = subset{:, acc_lateral_index};
    throttle_position_values = subset{:, throttle_position_index};
    engine_speed_values = subset{:, engine_speed_index};
    steering_angle_values = subset{:, steering_angle_index};

    % === Feature 1: acc_norm ===
    acc_norm = calcular_acc_norm(acc_long_values, acc_lat_values);

    % === Feature 2: deflection ===
    deflection = [0; abs(diff(steering_angle_values))];

    % === Classificação Fuzzy ===
    fuzzy_input = [acc_norm, speed_values];
    fuzzy_score = evalfis(fuzzy_system, fuzzy_input);
    fuzzy_class = strings(length(fuzzy_score), 1);
    fuzzy_class(fuzzy_score < 0.33) = "Calmo";
    fuzzy_class(fuzzy_score >= 0.33 & fuzzy_score < 0.66) = "Normal";
    fuzzy_class(fuzzy_score >= 0.66) = "Agressivo";

    % === Classificação Determinística ===
    throttle_norm = max(0, min((throttle_position_values - threshold_throttle) / (max_throttle - threshold_throttle), 1));
    rpm_norm = max(0, min((engine_speed_values - threshold_rpm) / (max_rpm - threshold_rpm), 1));
    deflection_norm = max(0, min((deflection - threshold_deflection) / (max_deflection - threshold_deflection), 1));

    deterministic_score = w_throttle * throttle_norm + w_rpm * rpm_norm + w_deflection * deflection_norm;
    deterministic_class = strings(length(deterministic_score), 1);
    deterministic_class(deterministic_score == 0) = "Normal";
    deterministic_class(deterministic_score > 0) = "Agressivo";

    % === Classificação Jerk ===
    speed_mps = speed_values / 3.6;
    acc_temp = [0; abs(diff(speed_mps))];
    jerk = [0; abs(diff(acc_temp))];

    gamma = zeros(length(jerk), 1);
    for j = 10:length(jerk)
        window = jerk(j-9:j);
        gamma(j) = std(window) / J_norm;
    end

    jerk_score = gamma;
    jerk_class = strings(length(jerk_score), 1);
    jerk_class(jerk_score < 0.5) = "Calmo";
    jerk_class(jerk_score >= 0.5 & jerk_score < 1) = "Normal";
    jerk_class(jerk_score >= 1) = "Agressivo";

    % === Criar tabela por iteração ===
    T = table(id_values, run_values, time_values, fuel_consumption_values, ...
        speed_values, acc_long_values, acc_lat_values, acc_norm, ...
        throttle_position_values, engine_speed_values, steering_angle_values, deflection, ...
        deterministic_score, fuzzy_score, jerk_score, ...
        deterministic_class, fuzzy_class, jerk_class, ...
        'VariableNames', {'id', 'run', 'time', 'fuel_consumption', ...
        'speed', 'acc_long', 'acc_lat', 'acc_norm', ...
        'throttle_position', 'engine_speed', 'steering_angle', 'deflection', ...
        'deterministic_score', 'fuzzy_score', 'jerk_score', ...
        'deterministic_class', 'fuzzy_class', 'jerk_class'});

    % === Agrupamento Triplo ===
    det = string(T.deterministic_class);
    fuz = string(T.fuzzy_class);
    jrk = string(T.jerk_class);

    n = height(T);
    group_ids = zeros(n, 1);
    match_strength = zeros(n, 1);

    for k = 1:n
        d = det(k); f = fuz(k); j = jrk(k);

        matches = [
            d == "Normal",    f == "Calmo",     j == "Calmo";
            d == "Normal",    f == "Normal",    j == "Normal";
            d == "Agressivo", f == "Agressivo", j == "Agressivo"
        ];

        for g = 1:3
            match_count = sum(matches(g, :));
            if match_count == 3
                match_strength(k) = 3;
                group_ids(k) = g;
                break;
            elseif match_count == 2
                match_strength(k) = 2;
                group_ids(k) = g;
            end
        end
    end

    % Adiciona ao final da tabela
    T.group_match_strength = match_strength;
    T.group_match_id = group_ids;

    % Concatena no final
    output_table = [output_table; T];
end

writetable(output_table, 'labeled_data2.csv');

% Função para calcular aceleração combinada
function acc_norm = calcular_acc_norm(acc_long, acc_lat)
    acc_norm = sqrt(acc_long.^2 + acc_lat.^2);
end


