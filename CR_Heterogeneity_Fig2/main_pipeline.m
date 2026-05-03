%%

% load Exp0

trainS = load(fullfile('data','train.mat'), 'Ht', 'gainHt');
validS = load(fullfile('data','valid.mat'), 'Ht', 'gainHt');
testS  = load(fullfile('data','test.mat'),  'Ht', 'gainHt');

trainData = trainS.Ht;
validData = validS.Ht;
testData  = testS.Ht;


%%

Xtr = reshape(trainData, size(trainData,1), []);
Xva = reshape(validData, size(validData,1), []);
Xte = reshape(testData, size(testData,1), []);

C = Xtr' * Xtr;          % [1024, 1024]
[V, D] = eig(C);
[lambda, idx] = sort(diag(D), 'descend');
V = V(:, idx);

%%

% test

rng(0);
idx = randperm(size(Xtr,1), min(2000, size(Xtr,1)));
Xsub = Xtr(idx, :);

check = verify_nmse_energy_vs_direct(Xsub, V, [1/32, 1/16, 1/8]);

%%

% main data

cList = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4];

resulttr = estimate_nmse_from_basis_energy_light(Xtr, V, cList, 1e-5, true);
resultva = estimate_nmse_from_basis_energy_light(Xva, V, cList, 1e-5, true);
resultte = estimate_nmse_from_basis_energy_light(Xte, V, cList, 1e-5, true);

nmseListTr = resulttr.nmseMean;
nmseListVa = resultva.nmseMean;
nmseListTe = resultte.nmseMean;

%%

% main result per scenario

scenarioNames = {'CDL-B', 'CDL-E', 'C2:Indoor5G', 'C2:Outdoor2G'};
cList = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4];

outtr = estimate_nmse_by_scenario_quarters(Xtr, V, cList, scenarioNames, 1e-5, true);
outva = estimate_nmse_by_scenario_quarters(Xva, V, cList, scenarioNames, 1e-5, true);
outte = estimate_nmse_by_scenario_quarters(Xte, V, cList, scenarioNames, 1e-5, true);

%%

plot_nmse_by_scenario_db(outtr, 'TitleStr', 'Train NMSE vs c');
plot_nmse_by_scenario_db(outva, 'TitleStr', 'Valid NMSE vs c');
plot_nmse_by_scenario_db(outte, 'TitleStr', 'Test NMSE vs c');

%%

plot_nmse_by_scenario_tvt_db(outtr, outva, outte, ...
    'IncludeGlobal', true, ...
    'XMode', 'c', ...
    'FigureTitle', 'Scenario-wise NMSE-vs-c [dB]');

%%

summary_te = compare_loglinear_models_quarters(outte, 'Verbose', true);
summary_va = compare_loglinear_models_quarters(outva, 'Verbose', true);
summary_tr = compare_loglinear_models_quarters(outtr, 'Verbose', true);

