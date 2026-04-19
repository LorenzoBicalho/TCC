close all
clear
clc

%% Parâmetros do modelo
num_rules = 10;           % Número de regras fuzzy
max_epochs = 100;        % Número máximo de épocas
alpha = 0.01;             % Taxa de aprendizado

%% Carregar dados
planilha = '../Label/clustered_data.csv';
data = readtable(planilha);

% Separar entradas (X) e saídas (y)
inputs = [data.fuel_consumption, data.speed, data.acc_norm,...
          data.throttle_position, data.engine_speed, data.deflection];
outputs = data.cluster_id;

% Normalizar entradas (Min-Max)
min_inputs = min(inputs);
max_inputs = max(inputs);
normalized_inputs = (inputs - min_inputs) ./ (max_inputs - min_inputs);

%% Dividir dados de forma estratificada (considerando desequilíbrio entre classes)
cv = cvpartition(outputs, 'HoldOut', 0.3); % 70% treino, 30% validação, estratificado
trainIdx = training(cv);
valIdx = test(cv);

X_train = normalized_inputs(trainIdx, :);
y_train = outputs(trainIdx);

X_val = normalized_inputs(valIdx, :);
y_val = outputs(valIdx);

[num_samples, num_features] = size(X_train);

%% Inicialização dos parâmetros fuzzy
xmin = min(X_train);
xmax = max(X_train);

c = zeros(num_features, num_rules);
s = zeros(num_features, num_rules);
p = zeros(num_features, num_rules);
q = zeros(1, num_rules);

melhores_c = c;
melhores_s = s;
melhores_p = p;
melhores_q = q;

for j = 1:num_rules
    for i = 1:num_features
        c(i, j) = xmin(i) + (xmax(i) - xmin(i)) * rand;
        s(i, j) = rand;
        p(i, j) = rand;
    end
    q(j) = rand;
end

%% Treinamento
melhor_mse = inf; % Inicializa com infinito
mse_epoch = zeros(1, max_epochs);
for epoch = 1:max_epochs
    total_error = 0;
    for k = 1:num_samples
        x = X_train(k, :);
        target = y_train(k);
        
        [ys, w, y, b] = calys(x, num_features, num_rules, c, s, p, q);
        error = ys - target;
        total_error = total_error + error^2;
        
        for j = 1:num_rules
            dys_dw = (y(j) - ys) / (b + eps);
            dys_dy = w(j) / (b + eps);
            
            for i = 1:num_features
                dw_dc = w(j) * (x(i) - c(i, j)) / (s(i, j)^2);
                dw_ds = w(j) * (x(i) - c(i, j))^2 / (s(i, j)^3);
                dy_dp = x(i);

                c(i, j) = c(i, j) - alpha * error * dys_dw * dw_dc;
                s(i, j) = s(i, j) - alpha * error * dys_dw * dw_ds;
                p(i, j) = p(i, j) - alpha * error * dys_dy * dy_dp;
            end
            q(j) = q(j) - alpha * error * dys_dy;
        end
    end
    mse_epoch(epoch) = total_error / num_samples;
end

%% Avaliação no conjunto de validação
num_val = size(X_val, 1);
y_pred = zeros(num_val, 1);

for k = 1:num_val
    x = X_val(k, :);
    [y_pred(k), ~, ~, ~] = calys(x, num_features, num_rules, c, s, p, q);
end

% Classe prevista (arredondamento)
y_pred_class = round(y_pred);
y_pred_class = min(y_pred_class, 3); % Limita valor máximo a 3
y_pred_class = max(y_pred_class, 1); % Limita valor mínimo a 1

% Acurácia
accuracy = sum(y_pred_class == y_val) / length(y_val) * 100;

% Erro percentual médio
error_percent = mean(abs((y_val - y_pred) ./ (y_val + eps))) * 100;

% Matriz de Confusão - Clusters
figure;
confusionchart(y_val, y_pred_class);
title('Matriz de Confusão - Clusters');

figure;
plot(1:max_epochs, mse_epoch, 'b-', 'LineWidth', 1.5);
xlabel('Épocas');
ylabel('Erro Quadrático Médio (MSE)');
title('Convergência do Treinamento');
grid on;

%% Resultado
fprintf('Acurácia no conjunto de validação: %.2f%%\n', accuracy);
fprintf('Erro Percentual Médio: %.2f%%\n', error_percent);

%% Função auxiliar
function [ys, w, y, b] = calys(x, n, m, c, s, p, q)
    a = 0; b = 0;
    y = zeros(1, m);
    w = zeros(1, m);
    
    for j = 1:m
        y(j) = q(j);
        w(j) = 1;
        for i = 1:n
            y(j) = y(j) + p(i, j) * x(i);
            w(j) = w(j) * exp(-0.5 * ((x(i) - c(i, j))^2 / (s(i, j)^2)));
        end
        a = a + w(j) * y(j);
        b = b + w(j);
    end
    ys = a / (b + eps);
end

% Salva todas as variáveis no formato binário do MATLAB
save('parametros_treino.mat', 'melhores_c', 'melhores_s', 'melhores_p', 'melhores_q');

writematrix(melhores_c, 'melhores_c.csv');
writematrix(melhores_s, 'melhores_s.csv');
writematrix(melhores_p, 'melhores_p.csv');
writematrix(melhores_q, 'melhores_q.csv');  % mesmo que seja vetor

fileID = fopen('parametros.txt','w');

fprintf(fileID, '--- c ---\n');
fprintf(fileID, [repmat('%.6f\t', 1, size(melhores_c, 1)) '\n'], melhores_c');

fprintf(fileID, '\n--- s ---\n');
fprintf(fileID, [repmat('%.6f\t', 1, size(melhores_s, 1)) '\n'], melhores_s');

fprintf(fileID, '\n--- p ---\n');
fprintf(fileID, [repmat('%.6f\t', 1, size(melhores_p, 1)) '\n'], melhores_p');

fprintf(fileID, '\n--- q ---\n');
fprintf(fileID, '%.6f\t', melhores_q);
fprintf(fileID, '\n');

fclose(fileID);

