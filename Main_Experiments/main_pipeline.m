%%
clear; clc; close all;

%% ============================================================
% Stage 0. Global setup
%% ============================================================

cRef = 1/16;
cList = [1/128, 1/96, 1/64, 1/48, 1/32, 1/24, 1/16, 1/12, 1/8, 1/6, 1/4];

resourceLimitedFeedbackRegime = true;

if resourceLimitedFeedbackRegime
cRef = cRef/4;
cList = cList/4;
end

lambda = 1e-2;
obsSize = 100;

K_va = 40;
K_te = 4;
seed_va = 0;
seed_te = 0;

SNRdB = 10;
SNR = 10^(SNRdB/10);

% fast sum-rate evaluation options
maxDropsEval = 10;

doValidationPlots = false;
doConsistencyCheck = true;
doBruteForceOracle = true;


KList_fig4 = [2 4 6 8 10 12 16];
seed_fig4 = 0;

alpha_fig4 = 1.0;
maxDropsEval_fig4 = 5;
doBruteForceOracle_fig4 = false;
KmaxBruteForce_fig4 = 6;

% figure generation / saving switches
writeFig3 = false;
writeFig4 = false;
writeFig5 = true;

saveFig3Data = false;
saveFig4Data = false;
saveFig5Data = true;

saveFigPdf = false;
saveFigPng = false;
saveFigMatlabFig = false;

useTimestampForSave = true;
runTimestamp = datestr(now, 'yyyymmdd_HHMMSS');


%% ============================================================
% Stage 1. Load train data and build global basis V
%% ============================================================

if (~exist('V', 'var'))
    trainS = load(fullfile('data','train.mat'), 'Ht', 'gainHt');
    trainData = trainS.Ht; clear trainS;

    Xtr = reshape(trainData, size(trainData,1), []);
    C = Xtr' * Xtr;
    [V, D] = eig(C);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);

    clear trainData Xtr C D idx;
end

%% ============================================================
% Stage 2. Load validation data and train descriptor -> summary
%% ============================================================

if (~exist('validData', 'var'))
    validS = load(fullfile('data','valid.mat'), 'Ht', 'gainHt');
    validData = validS.Ht; clear validS;
end

out_va = build_user_descriptor_curve_dataset( ...
    validData, V, cRef, cList, K_va, seed_va, obsSize);

Phi_va = out_va.Phi;
Yab_va = out_va.fitParamLogLinear;
Ycurve_va = out_va.nmseCurveDb; %#ok<NASGU>

[model, ~] = fit_ridge_phi_to_yab(Phi_va, Yab_va, lambda);

%% ============================================================
% Stage 3. Validation diagnostics (optional)
%% ============================================================

if doValidationPlots
    Yhat_va = predict_ridge_phi_to_yab(model, Phi_va);

    metrics = evaluate_yab_prediction(Yab_va, Yhat_va);
    disp(metrics)

    metricsOrd = evaluate_order_preservation(Yab_va, Yhat_va, cList);
    disp(metricsOrd)

    Ycurvehat_va = predict_curve_from_ab(Yhat_va, cList);
    metricsCurve = evaluate_curve_prediction(out_va.nmseCurveDb, Ycurvehat_va, cList);
    disp(metricsCurve)

    plot_user_cr_distortion_curves(out_va)
    plot_yab_scatter(Yab_va)
    plot_yab_scatter(Yab_va, ...
        'ScenarioId', [out_va.user.scenarioId]', ...
        'Yhat', Yhat_va, ...
        'TitleStr', 'True vs predicted [a,b]')
    plot_yab_true_vs_pred(Yab_va, Yhat_va)
end

%% ============================================================
% Stage 4. Load test data and build test user dataset
%% ============================================================

if (~exist('testS', 'var'))
    testS = load(fullfile('data','test.mat'), 'Ht', 'gainHt', 'Horg', 'gainHorg');
    testData = testS.Ht;
end

Nsub_full = size(testS.Horg, 4);
subIdxEval = unique(round(linspace(1, Nsub_full, 32)));

out_te = build_user_descriptor_curve_dataset( ...
    testData, V, cRef, cList, K_te, seed_te, obsSize);

Phi_te = out_te.Phi;
Ycurve_te = out_te.nmseCurveDb;
Yhat_te = predict_ridge_phi_to_yab(model, Phi_te);

DhatDb_te  = predict_curve_from_ab(Yhat_te, cList);
DhatLin_te = 10.^(DhatDb_te / 10);
DtrueLin_te = 10.^(Ycurve_te / 10);

Btot = K_te * cRef;

%% ============================================================
% Stage 5. Solve policies in NMSE/surrogate domain
%% ============================================================

% uniform baseline under the same total budget
cPerUserBudget = Btot / K_te;
feasibleUniformIdx = find(cList <= cPerUserBudget + 1e-12);
if isempty(feasibleUniformIdx)
    error('No feasible uniform operating point exists for the given Btot/K.');
end

uniformIdx = feasibleUniformIdx(end);
policy_uniform.idx = uniformIdx * ones(K_te,1);
policy_uniform.cAlloc = cList(policy_uniform.idx).';
policy_uniform.usedBudget = sum(policy_uniform.cAlloc);

% summary-driven exact policy
policy_summary_exact = exact_min_sum_nmse_policy(DhatLin_te, cList, Btot);

% curve-oracle exact policy
policy_oracle_curve = exact_min_sum_nmse_policy(DtrueLin_te, cList, Btot);

% greedy heuristic policy
policy_heuristic = greedy_min_sum_nmse_policy_alljump(DhatLin_te, cList, Btot);

policyCell = {
    policy_uniform, ...
    policy_summary_exact, ...
    policy_oracle_curve, ...
    policy_heuristic
    };

policyName = {
    'uniform', ...
    'summary_exact', ...
    'oracle_curve', ...
    'heuristic'
    };

%% ============================================================
% Stage 6. Consistency check: summary exact vs heuristic
%% ============================================================

if doConsistencyCheck
    pol_exact = policy_summary_exact;
    pol_greedy = policy_heuristic;

    rowIdx = (1:K_te).';

    pred_exact = sum(DhatLin_te(sub2ind(size(DhatLin_te), rowIdx, pol_exact.idx(:))));
    pred_greedy = sum(DhatLin_te(sub2ind(size(DhatLin_te), rowIdx, pol_greedy.idx(:))));

    true_exact = sum(DtrueLin_te(sub2ind(size(DtrueLin_te), rowIdx, pol_exact.idx(:))));
    true_greedy = sum(DtrueLin_te(sub2ind(size(DtrueLin_te), rowIdx, pol_greedy.idx(:))));

    fprintf('\n=== Consistency check: summary exact vs heuristic ===\n');
    fprintf('Predicted sum NMSE (exact)   = %.10f\n', pred_exact);
    fprintf('Predicted sum NMSE (greedy)  = %.10f\n', pred_greedy);
    fprintf('True sum NMSE (exact alloc)  = %.10f\n', true_exact);
    fprintf('True sum NMSE (greedy alloc) = %.10f\n', true_greedy);
    fprintf('Used budget (exact)          = %.10f\n', pol_exact.usedBudget);
    fprintf('Used budget (greedy)         = %.10f\n', pol_greedy.usedBudget);
    fprintf('Allocation mismatch count    = %d / %d\n', ...
        sum(pol_exact.idx ~= pol_greedy.idx), K_te);

    if pred_exact <= pred_greedy + 1e-10
        fprintf('PASS: exact solver is no worse than greedy on surrogate objective.\n');
    else
        fprintf('FAIL: greedy beat exact on surrogate objective. Check solver.\n');
    end
end

%% ============================================================
% Stage 7. Compare policies in NMSE domain
%% ============================================================

fprintf('\n============================================================\n');
fprintf('Policy allocations\n');
fprintf('============================================================\n');

numPolicy = numel(policyCell);

for p = 1:numPolicy
    fprintf('%s:\n', policyName{p});
    disp(policyCell{p}.cAlloc')
end

trueNmse = zeros(1, numPolicy);
predNmse = zeros(1, numPolicy);

rowIdx = (1:K_te).';

for p = 1:numPolicy
    idxp = policyCell{p}.idx(:);
    trueNmse(p) = sum(DtrueLin_te(sub2ind(size(DtrueLin_te), rowIdx, idxp)));
    predNmse(p) = sum(DhatLin_te(sub2ind(size(DhatLin_te), rowIdx, idxp)));
end

Tnmse = table(policyName(:), trueNmse(:), predNmse(:), ...
    'VariableNames', {'policy','trueSumNmse','predSumNmse'});
disp(Tnmse)

iUniform = find(strcmp(policyName, 'uniform'));
iSummary = find(strcmp(policyName, 'summary_exact'));
iCurve   = find(strcmp(policyName, 'oracle_curve'));
iHeur    = find(strcmp(policyName, 'heuristic'));

oracleRecoveryNmse_summary = ...
    (trueNmse(iUniform) - trueNmse(iSummary)) / ...
    (trueNmse(iUniform) - trueNmse(iCurve) + eps);

oracleRecoveryNmse_heur = ...
    (trueNmse(iUniform) - trueNmse(iHeur)) / ...
    (trueNmse(iUniform) - trueNmse(iCurve) + eps);

fprintf('NMSE curve-oracle recovery (summary exact) = %.2f %%\n', ...
    100 * oracleRecoveryNmse_summary);
fprintf('NMSE curve-oracle recovery (heuristic)     = %.2f %%\n', ...
    100 * oracleRecoveryNmse_heur);

%% ============================================================
% Stage 8. Precompute sum-rate bank
%% ============================================================

fprintf('\nPrecomputing Hhat bank for sum-rate evaluation...\n');

[Htrue_bank, Hhat_bank, numDrops] = build_sumrate_bank( ...
    testS, out_te, V, cList, ...
    'MaxDrops', maxDropsEval, ...
    'Verbose', true);

%% ============================================================
% Stage 9. Sum-rate evaluation for current four policies
%% ============================================================

sumRateMat = zeros(numDrops, numPolicy);
userRateCell = cell(numPolicy, 1);

fprintf('\nEvaluating sum rate for current policies...\n');

for p = 1:numPolicy
    out_sr = eval_policy_sumrate_from_bank( ...
        policyCell{p}, Htrue_bank, Hhat_bank, SNR, subIdxEval);

    sumRateMat(:,p) = out_sr.sumRateVec;
    userRateCell{p} = out_sr.userRateMat;
end

meanSumRate = mean(sumRateMat, 1);
stdSumRate  = std(sumRateMat, 0, 1);

Tsr = table(policyName(:), meanSumRate(:), stdSumRate(:), ...
    'VariableNames', {'policy','meanSumRate','stdSumRate'});
disp(Tsr)

curveOracleRecoverySr_summary = ...
    (meanSumRate(iSummary) - meanSumRate(iUniform)) / ...
    (meanSumRate(iCurve)   - meanSumRate(iUniform) + eps);

curveOracleRecoverySr_heur = ...
    (meanSumRate(iHeur) - meanSumRate(iUniform)) / ...
    (meanSumRate(iCurve) - meanSumRate(iUniform) + eps);

fprintf('Sum-rate curve-oracle recovery (summary exact) = %.2f %%\n', ...
    100 * curveOracleRecoverySr_summary);
fprintf('Sum-rate curve-oracle recovery (heuristic)     = %.2f %%\n', ...
    100 * curveOracleRecoverySr_heur);

%% ============================================================
% Stage 10. True brute-force sum-rate oracle
%% ============================================================

if doBruteForceOracle
    oracle_bruteforce = brute_force_sumrate_oracle( ...
        Htrue_bank, Hhat_bank, cList, Btot, SNR, ...
        'SubIdx', subIdxEval, ...
        'Verbose', true);

    policy_oracle_bruteforce.idx = oracle_bruteforce.idx;
    policy_oracle_bruteforce.cAlloc = oracle_bruteforce.cAlloc;
    policy_oracle_bruteforce.usedBudget = oracle_bruteforce.usedBudget;

    policyCell = {
        policy_uniform, ...
        policy_summary_exact, ...
        policy_oracle_curve, ...
        policy_heuristic, ...
        policy_oracle_bruteforce
        };

    policyName = {
        'uniform', ...
        'summary_exact', ...
        'oracle_curve', ...
        'heuristic', ...
        'oracle_bruteforce'
        };

    numPolicy = numel(policyCell);

    fprintf('\n============================================================\n');
    fprintf('Policy allocations including brute-force oracle\n');
    fprintf('============================================================\n');

    for p = 1:numPolicy
        fprintf('%s:\n', policyName{p});
        disp(policyCell{p}.cAlloc')
    end

    % Re-evaluate NMSE table with brute-force oracle included
    trueNmse = zeros(1, numPolicy);
    predNmse = zeros(1, numPolicy);

    for p = 1:numPolicy
        idxp = policyCell{p}.idx(:);
        trueNmse(p) = sum(DtrueLin_te(sub2ind(size(DtrueLin_te), rowIdx, idxp)));
        predNmse(p) = sum(DhatLin_te(sub2ind(size(DhatLin_te), rowIdx, idxp)));
    end

    Tnmse = table(policyName(:), trueNmse(:), predNmse(:), ...
        'VariableNames', {'policy','trueSumNmse','predSumNmse'});
    disp(Tnmse)

    % Re-evaluate sum-rate table with brute-force oracle included
    sumRateMat = zeros(numDrops, numPolicy);
    userRateCell = cell(numPolicy, 1);

    fprintf('\nEvaluating sum rate including brute-force oracle...\n');

    for p = 1:numPolicy
        out_sr = eval_policy_sumrate_from_bank( ...
            policyCell{p}, Htrue_bank, Hhat_bank, SNR, subIdxEval);

        sumRateMat(:,p) = out_sr.sumRateVec;
        userRateCell{p} = out_sr.userRateMat;
    end

    meanSumRate = mean(sumRateMat, 1);
    stdSumRate  = std(sumRateMat, 0, 1);

    Tsr = table(policyName(:), meanSumRate(:), stdSumRate(:), ...
        'VariableNames', {'policy','meanSumRate','stdSumRate'});
    disp(Tsr)

    iUniform = find(strcmp(policyName, 'uniform'));
    iSummary = find(strcmp(policyName, 'summary_exact'));
    iCurve   = find(strcmp(policyName, 'oracle_curve'));
    iHeur    = find(strcmp(policyName, 'heuristic'));
    iBF      = find(strcmp(policyName, 'oracle_bruteforce'));

    bfRecoverySr_summary = ...
        (meanSumRate(iSummary) - meanSumRate(iUniform)) / ...
        (meanSumRate(iBF)      - meanSumRate(iUniform) + eps);

    bfRecoverySr_curve = ...
        (meanSumRate(iCurve) - meanSumRate(iUniform)) / ...
        (meanSumRate(iBF)    - meanSumRate(iUniform) + eps);

    bfRecoverySr_heur = ...
        (meanSumRate(iHeur) - meanSumRate(iUniform)) / ...
        (meanSumRate(iBF)   - meanSumRate(iUniform) + eps);

    fprintf('Sum-rate brute-force-oracle recovery (summary exact) = %.2f %%\n', ...
        100 * bfRecoverySr_summary);
    fprintf('Sum-rate brute-force-oracle recovery (curve oracle)  = %.2f %%\n', ...
        100 * bfRecoverySr_curve);
    fprintf('Sum-rate brute-force-oracle recovery (heuristic)     = %.2f %%\n', ...
        100 * bfRecoverySr_heur);
end

%% ============================================================
% Stage 11. Final summary table
%% ============================================================

summaryTable = table( ...
    policyName(:), ...
    trueNmse(:), ...
    predNmse(:), ...
    meanSumRate(:), ...
    stdSumRate(:), ...
    'VariableNames', ...
    {'policy','trueSumNmse','predSumNmse','meanSumRate','stdSumRate'});

disp(summaryTable)





%% ============================================================
% Fig. 3. Budget sweep, plotting, and saving
%% ============================================================

if writeFig3

alphaList_fig3 = 0.50:0.10:1.50;
numBudget_fig3 = numel(alphaList_fig3);

policyNamesBase_fig3 = {'uniform','summary_exact','oracle_curve','heuristic'};
if doBruteForceOracle
    policyNames_fig3 = [policyNamesBase_fig3, {'oracle_bruteforce'}];
else
    policyNames_fig3 = policyNamesBase_fig3;
end
numPolicy_fig3 = numel(policyNames_fig3);

budgetList_fig3 = nan(numBudget_fig3,1);
meanSumRateSweep_fig3 = nan(numBudget_fig3, numPolicy_fig3);
stdSumRateSweep_fig3  = nan(numBudget_fig3, numPolicy_fig3);
trueNmseSweep_fig3    = nan(numBudget_fig3, numPolicy_fig3);
predNmseSweep_fig3    = nan(numBudget_fig3, numPolicy_fig3);

curveRecovery_summary_fig3 = nan(numBudget_fig3,1);
curveRecovery_heur_fig3    = nan(numBudget_fig3,1);

bfRecovery_summary_fig3 = nan(numBudget_fig3,1);
bfRecovery_curve_fig3   = nan(numBudget_fig3,1);
bfRecovery_heur_fig3    = nan(numBudget_fig3,1);

policyAllocSweep_fig3 = cell(numBudget_fig3, numPolicy_fig3);
rowIdx_fig3 = (1:K_te).';

fprintf('\n============================================================\n');
fprintf('Fig. 3 budget sweep starts\n');
fprintf('============================================================\n');

for ib = 1:numBudget_fig3
    alpha_fig3 = alphaList_fig3(ib);
    Btot_fig3 = alpha_fig3 * K_te * cRef;
    budgetList_fig3(ib) = Btot_fig3;

    fprintf('\n[Fig.3] %2d / %2d : alpha = %.2f, Btot = %.6f\n', ...
        ib, numBudget_fig3, alpha_fig3, Btot_fig3);

    cPerUserBudget_fig3 = Btot_fig3 / K_te;
    feasibleUniformIdx_fig3 = find(cList <= cPerUserBudget_fig3 + 1e-12);

    if isempty(feasibleUniformIdx_fig3)
        warning('No feasible uniform point at alpha = %.2f. Skipping.', alpha_fig3);
        continue;
    end

    uniformIdx_fig3 = feasibleUniformIdx_fig3(end);
    policy_uniform_fig3.idx = uniformIdx_fig3 * ones(K_te,1);
    policy_uniform_fig3.cAlloc = cList(policy_uniform_fig3.idx).';
    policy_uniform_fig3.usedBudget = sum(policy_uniform_fig3.cAlloc);

    try
        policy_summary_exact_fig3 = exact_min_sum_nmse_policy(DhatLin_te, cList, Btot_fig3);
        policy_oracle_curve_fig3  = exact_min_sum_nmse_policy(DtrueLin_te, cList, Btot_fig3);
        policy_heuristic_fig3     = greedy_min_sum_nmse_policy_alljump(DhatLin_te, cList, Btot_fig3);
    catch ME
        warning('Policy solve failed at alpha = %.2f: %s', alpha_fig3, ME.message);
        continue;
    end

    policyCell_fig3 = {
        policy_uniform_fig3, ...
        policy_summary_exact_fig3, ...
        policy_oracle_curve_fig3, ...
        policy_heuristic_fig3
        };

    if doBruteForceOracle
        try
            oracle_bruteforce_fig3 = brute_force_sumrate_oracle( ...
                Htrue_bank, Hhat_bank, cList, Btot_fig3, SNR, ...
                'SubIdx', subIdxEval, ...
                'Verbose', false);

            policy_oracle_bruteforce_fig3.idx = oracle_bruteforce_fig3.idx;
            policy_oracle_bruteforce_fig3.cAlloc = oracle_bruteforce_fig3.cAlloc;
            policy_oracle_bruteforce_fig3.usedBudget = oracle_bruteforce_fig3.usedBudget;

            policyCell_fig3{end+1} = policy_oracle_bruteforce_fig3;
        catch ME
            warning('Brute-force oracle failed at alpha = %.2f: %s', alpha_fig3, ME.message);

            policyCell_fig3{numPolicy_fig3} = struct( ...
                'idx', nan(K_te,1), ...
                'cAlloc', nan(K_te,1), ...
                'usedBudget', nan);
        end
    end

    for p = 1:numel(policyCell_fig3)
        idxp_fig3 = policyCell_fig3{p}.idx(:);

        if any(isnan(idxp_fig3))
            continue;
        end

        trueNmseSweep_fig3(ib,p) = sum(DtrueLin_te(sub2ind(size(DtrueLin_te), rowIdx_fig3, idxp_fig3)));
        predNmseSweep_fig3(ib,p) = sum(DhatLin_te(sub2ind(size(DhatLin_te), rowIdx_fig3, idxp_fig3)));
        policyAllocSweep_fig3{ib,p} = policyCell_fig3{p}.cAlloc;
    end

    for p = 1:numel(policyCell_fig3)
        idxp_fig3 = policyCell_fig3{p}.idx(:);

        if any(isnan(idxp_fig3))
            continue;
        end

        out_sr_fig3 = eval_policy_sumrate_from_bank( ...
            policyCell_fig3{p}, Htrue_bank, Hhat_bank, SNR, subIdxEval);

        meanSumRateSweep_fig3(ib,p) = out_sr_fig3.meanSumRate;
        stdSumRateSweep_fig3(ib,p)  = out_sr_fig3.stdSumRate;
    end

    iUniform_fig3 = find(strcmp(policyNames_fig3, 'uniform'));
    iSummary_fig3 = find(strcmp(policyNames_fig3, 'summary_exact'));
    iCurve_fig3   = find(strcmp(policyNames_fig3, 'oracle_curve'));
    iHeur_fig3    = find(strcmp(policyNames_fig3, 'heuristic'));

    curveRecovery_summary_fig3(ib) = ...
        (meanSumRateSweep_fig3(ib,iSummary_fig3) - meanSumRateSweep_fig3(ib,iUniform_fig3)) / ...
        (meanSumRateSweep_fig3(ib,iCurve_fig3)   - meanSumRateSweep_fig3(ib,iUniform_fig3) + eps);

    curveRecovery_heur_fig3(ib) = ...
        (meanSumRateSweep_fig3(ib,iHeur_fig3) - meanSumRateSweep_fig3(ib,iUniform_fig3)) / ...
        (meanSumRateSweep_fig3(ib,iCurve_fig3) - meanSumRateSweep_fig3(ib,iUniform_fig3) + eps);

    if doBruteForceOracle
        iBF_fig3 = find(strcmp(policyNames_fig3, 'oracle_bruteforce'));

        if ~isnan(meanSumRateSweep_fig3(ib,iBF_fig3))
            bfRecovery_summary_fig3(ib) = ...
                (meanSumRateSweep_fig3(ib,iSummary_fig3) - meanSumRateSweep_fig3(ib,iUniform_fig3)) / ...
                (meanSumRateSweep_fig3(ib,iBF_fig3)      - meanSumRateSweep_fig3(ib,iUniform_fig3) + eps);

            bfRecovery_curve_fig3(ib) = ...
                (meanSumRateSweep_fig3(ib,iCurve_fig3) - meanSumRateSweep_fig3(ib,iUniform_fig3)) / ...
                (meanSumRateSweep_fig3(ib,iBF_fig3)    - meanSumRateSweep_fig3(ib,iUniform_fig3) + eps);

            bfRecovery_heur_fig3(ib) = ...
                (meanSumRateSweep_fig3(ib,iHeur_fig3) - meanSumRateSweep_fig3(ib,iUniform_fig3)) / ...
                (meanSumRateSweep_fig3(ib,iBF_fig3)   - meanSumRateSweep_fig3(ib,iUniform_fig3) + eps);
        end
    end
end

fig3 = struct();
fig3.alphaList = alphaList_fig3(:);
fig3.budgetList = budgetList_fig3;
fig3.policyNames = policyNames_fig3;
fig3.meanSumRateSweep = meanSumRateSweep_fig3;
fig3.stdSumRateSweep = stdSumRateSweep_fig3;
fig3.trueNmseSweep = trueNmseSweep_fig3;
fig3.predNmseSweep = predNmseSweep_fig3;
fig3.curveRecovery_summary = curveRecovery_summary_fig3;
fig3.curveRecovery_heur = curveRecovery_heur_fig3;
fig3.bfRecovery_summary = bfRecovery_summary_fig3;
fig3.bfRecovery_curve = bfRecovery_curve_fig3;
fig3.bfRecovery_heur = bfRecovery_heur_fig3;
fig3.policyAllocSweep = policyAllocSweep_fig3;

disp('=== Fig. 3 mean sum-rate sweep ===');
disp(array2table(fig3.meanSumRateSweep, ...
    'VariableNames', matlab.lang.makeValidName(fig3.policyNames)))

if doBruteForceOracle
    disp('=== Fig. 3 brute-force-oracle recovery ===');
    disp(table(fig3.alphaList, fig3.bfRecovery_summary, fig3.bfRecovery_curve, fig3.bfRecovery_heur, ...
        'VariableNames', {'alpha','summary_exact','curve_oracle','heuristic'}))
else
    disp('=== Fig. 3 curve-oracle recovery ===');
    disp(table(fig3.alphaList, fig3.curveRecovery_summary, fig3.curveRecovery_heur, ...
        'VariableNames', {'alpha','summary_exact','heuristic'}))
end

fig3a = figure;
hold on; grid on; box on;

for p = 1:numPolicy_fig3
    plot(fig3.budgetList, fig3.meanSumRateSweep(:,p), '-o', 'LineWidth', 1.5, ...
        'DisplayName', fig3.policyNames{p});
end

xlabel('Total feedback budget');
ylabel('Mean MU-MIMO sum rate');
title('Fig. 3(a): Sum-rate performance versus total feedback budget');
legend('Location','best');

fig3b = figure;
hold on; grid on; box on;

if doBruteForceOracle
    plot(fig3.budgetList, 100*fig3.bfRecovery_summary, '-o', 'LineWidth', 1.5, ...
        'DisplayName', 'Summary exact');
    plot(fig3.budgetList, 100*fig3.bfRecovery_curve, '-s', 'LineWidth', 1.5, ...
        'DisplayName', 'Curve oracle');
    plot(fig3.budgetList, 100*fig3.bfRecovery_heur, '-^', 'LineWidth', 1.5, ...
        'DisplayName', 'Heuristic');
    ylabel('Recovery relative to brute-force oracle (%)');
else
    plot(fig3.budgetList, 100*fig3.curveRecovery_summary, '-o', 'LineWidth', 1.5, ...
        'DisplayName', 'Summary exact');
    plot(fig3.budgetList, 100*fig3.curveRecovery_heur, '-^', 'LineWidth', 1.5, ...
        'DisplayName', 'Heuristic');
    ylabel('Recovery relative to curve oracle (%)');
end

xlabel('Total feedback budget');
title('Fig. 3(b): Recovery versus total feedback budget');
legend('Location','best');

figOutDir = fullfile('Figures', 'fig3');
if ~exist(figOutDir, 'dir')
    mkdir(figOutDir);
end

if useTimestampForSave
    suffix = ['_' runTimestamp];
else
    suffix = '';
end

base3a = ['fig3a_sumrate_vs_budget' suffix];
base3b = ['fig3b_recovery_vs_budget' suffix];

if saveFigMatlabFig
    saveas(fig3a, fullfile(figOutDir, [base3a '.fig']));
    saveas(fig3b, fullfile(figOutDir, [base3b '.fig']));
end

if saveFigPng
    saveas(fig3a, fullfile(figOutDir, [base3a '.png']));
    saveas(fig3b, fullfile(figOutDir, [base3b '.png']));
end

if saveFigPdf
    exportgraphics(fig3a, fullfile(figOutDir, [base3a '.pdf']), ...
        'ContentType', 'vector');
    exportgraphics(fig3b, fullfile(figOutDir, [base3b '.pdf']), ...
        'ContentType', 'vector');
end

if saveFig3Data
    dataOutDir = fullfile('results', 'fig3');
    if ~exist(dataOutDir, 'dir')
        mkdir(dataOutDir);
    end

    fig3save = struct();
    fig3save.alphaList = alphaList_fig3(:);
    fig3save.budgetList = budgetList_fig3;
    fig3save.policyNames = policyNames_fig3;

    fig3save.meanSumRateSweep = meanSumRateSweep_fig3;
    fig3save.stdSumRateSweep  = stdSumRateSweep_fig3;
    fig3save.trueNmseSweep    = trueNmseSweep_fig3;
    fig3save.predNmseSweep    = predNmseSweep_fig3;

    fig3save.curveRecovery_summary = curveRecovery_summary_fig3;
    fig3save.curveRecovery_heur    = curveRecovery_heur_fig3;

    fig3save.bfRecovery_summary = bfRecovery_summary_fig3;
    fig3save.bfRecovery_curve   = bfRecovery_curve_fig3;
    fig3save.bfRecovery_heur    = bfRecovery_heur_fig3;

    fig3save.policyAllocSweep = policyAllocSweep_fig3;

    fig3save.meta = struct();
    fig3save.meta.timestamp = runTimestamp;
    fig3save.meta.cRef = cRef;
    fig3save.meta.cList = cList;
    fig3save.meta.K_te = K_te;
    fig3save.meta.SNRdB = SNRdB;
    fig3save.meta.SNR = SNR;
    fig3save.meta.maxDropsEval = maxDropsEval;
    fig3save.meta.subIdxEval = subIdxEval;
    fig3save.meta.doBruteForceOracle = doBruteForceOracle;

    dataFile = fullfile(dataOutDir, ['fig3_data_compact' suffix '.mat']);
    save(dataFile, 'fig3save');

    fprintf('Compact Fig. 3 data saved: %s\n', dataFile);
end

else
    fprintf('\n[Fig.3] Skipped because writeFig3 = false.\n');
end


%% ============================================================
% Fig. 4: Complexity scaling sweep, plotting, and saving
%% ============================================================

if writeFig4

Nsub_full = size(testS.Horg, 4);
subIdxEval_fig4 = unique(round(linspace(1, Nsub_full, 16)));

numK_fig4 = numel(KList_fig4);

time_summary_exact_fig4 = nan(numK_fig4,1);
time_heuristic_fig4     = nan(numK_fig4,1);
time_bruteforce_fig4    = nan(numK_fig4,1);

numFeasible_bruteforce_fig4 = nan(numK_fig4,1);

predNmse_summary_exact_fig4 = nan(numK_fig4,1);
predNmse_heuristic_fig4     = nan(numK_fig4,1);

fprintf('\n============================================================\n');
fprintf('Fig. 4 complexity sweep starts\n');
fprintf('============================================================\n');

for ik = 1:numK_fig4
    K_fig4 = KList_fig4(ik);
    Btot_fig4 = alpha_fig4 * K_fig4 * cRef;

    fprintf('\n[Fig.4] %2d / %2d : K = %d, Btot = %.6f\n', ...
        ik, numK_fig4, K_fig4, Btot_fig4);

    out_te_fig4 = build_user_descriptor_curve_dataset( ...
        testData, V, cRef, cList, K_fig4, seed_fig4, obsSize);

    Phi_te_fig4 = out_te_fig4.Phi;
    Ycurve_te_fig4 = out_te_fig4.nmseCurveDb;
    Yhat_te_fig4 = predict_ridge_phi_to_yab(model, Phi_te_fig4);

    DhatDb_te_fig4  = predict_curve_from_ab(Yhat_te_fig4, cList);
    DhatLin_te_fig4 = 10.^(DhatDb_te_fig4 / 10);

    tExact = tic;
    policy_summary_exact_fig4 = exact_min_sum_nmse_policy( ...
        DhatLin_te_fig4, cList, Btot_fig4);
    time_summary_exact_fig4(ik) = toc(tExact);

    tHeur = tic;
    policy_heuristic_fig4 = greedy_min_sum_nmse_policy_alljump( ...
        DhatLin_te_fig4, cList, Btot_fig4);
    time_heuristic_fig4(ik) = toc(tHeur);

    rowIdx_fig4 = (1:K_fig4).';

    predNmse_summary_exact_fig4(ik) = sum(DhatLin_te_fig4( ...
        sub2ind(size(DhatLin_te_fig4), rowIdx_fig4, policy_summary_exact_fig4.idx(:))));

    predNmse_heuristic_fig4(ik) = sum(DhatLin_te_fig4( ...
        sub2ind(size(DhatLin_te_fig4), rowIdx_fig4, policy_heuristic_fig4.idx(:))));

    if doBruteForceOracle_fig4 && (K_fig4 <= KmaxBruteForce_fig4)
        fprintf('  Building bank for brute-force oracle...\n');

        [Htrue_bank_fig4, Hhat_bank_fig4, ~] = build_sumrate_bank( ...
            testS, out_te_fig4, V, cList, ...
            'MaxDrops', maxDropsEval_fig4, ...
            'Verbose', false);

        tBF = tic;
        oracle_bruteforce_fig4 = brute_force_sumrate_oracle( ...
            Htrue_bank_fig4, Hhat_bank_fig4, cList, Btot_fig4, SNR, ...
            'SubIdx', subIdxEval_fig4, ...
            'Verbose', false, ...
            'MaxRuntimeSec', 300, ...
            'MinBudgetFraction', 0.5);
        time_bruteforce_fig4(ik) = toc(tBF);

        numFeasible_bruteforce_fig4(ik) = oracle_bruteforce_fig4.numFeasible;
    end

    fprintf('  runtime summary_exact = %.6f s\n', time_summary_exact_fig4(ik));
    fprintf('  runtime heuristic     = %.6f s\n', time_heuristic_fig4(ik));

    if ~isnan(time_bruteforce_fig4(ik))
        fprintf('  runtime brute_force   = %.6f s\n', time_bruteforce_fig4(ik));
        fprintf('  feasible allocations  = %d\n', numFeasible_bruteforce_fig4(ik));
    end
end

fig4 = struct();
fig4.KList = KList_fig4(:);
fig4.BtotList = alpha_fig4 * KList_fig4(:) * cRef;
fig4.time_summary_exact = time_summary_exact_fig4;
fig4.time_heuristic = time_heuristic_fig4;
fig4.time_bruteforce = time_bruteforce_fig4;
fig4.numFeasible_bruteforce = numFeasible_bruteforce_fig4;
fig4.predNmse_summary_exact = predNmse_summary_exact_fig4;
fig4.predNmse_heuristic = predNmse_heuristic_fig4;

Tfig4 = table( ...
    fig4.KList, ...
    fig4.BtotList, ...
    fig4.time_summary_exact, ...
    fig4.time_heuristic, ...
    fig4.time_bruteforce, ...
    fig4.numFeasible_bruteforce, ...
    'VariableNames', ...
    {'K','Btot','time_summary_exact','time_heuristic','time_bruteforce','numFeasible_bruteforce'});

disp(Tfig4)

fig4a = figure;
hold on; grid on; box on;

plot(fig4.KList, fig4.time_summary_exact, '-o', 'LineWidth', 1.5, ...
    'DisplayName', 'Summary exact');
plot(fig4.KList, fig4.time_heuristic, '-^', 'LineWidth', 1.5, ...
    'DisplayName', 'Greedy heuristic');

validBF = ~isnan(fig4.time_bruteforce);

if any(validBF)
    plot(fig4.KList(validBF), fig4.time_bruteforce(validBF), '-s', 'LineWidth', 1.5, ...
        'DisplayName', 'Brute-force oracle');
end

set(gca, 'YScale', 'log');
xlabel('Number of users $K$', 'Interpreter', 'latex');
ylabel('Runtime (s)');
title('Fig. 4(a): Complexity scaling versus number of users');
legend('Location', 'best');

fig4b = figure;
hold on; grid on; box on;

if any(validBF)
    plot(fig4.KList(validBF), fig4.numFeasible_bruteforce(validBF), '-s', ...
        'LineWidth', 1.5, 'DisplayName', 'Brute-force oracle');
else
    text(0.5, 0.5, 'Brute-force oracle disabled', ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'center');
end

set(gca, 'YScale', 'log');
xlabel('Number of users $K$', 'Interpreter', 'latex');
ylabel('Number of feasible allocations');
title('Fig. 4(b): Feasible search-space growth');
legend('Location', 'best');

figOutDir = fullfile('Figures', 'fig4');
if ~exist(figOutDir, 'dir')
    mkdir(figOutDir);
end

if useTimestampForSave
    suffix = ['_' runTimestamp];
else
    suffix = '';
end

base4a = ['fig4a_runtime_vs_K' suffix];
base4b = ['fig4b_searchspace_vs_K' suffix];

if saveFigMatlabFig
    saveas(fig4a, fullfile(figOutDir, [base4a '.fig']));
    saveas(fig4b, fullfile(figOutDir, [base4b '.fig']));
end

if saveFigPng
    saveas(fig4a, fullfile(figOutDir, [base4a '.png']));
    saveas(fig4b, fullfile(figOutDir, [base4b '.png']));
end

if saveFigPdf
    exportgraphics(fig4a, fullfile(figOutDir, [base4a '.pdf']), ...
        'ContentType', 'vector');
    exportgraphics(fig4b, fullfile(figOutDir, [base4b '.pdf']), ...
        'ContentType', 'vector');
end

if saveFig4Data
    dataOutDir = fullfile('results', 'fig4');
    if ~exist(dataOutDir, 'dir')
        mkdir(dataOutDir);
    end

    fig4save = struct();
    fig4save.KList = fig4.KList;
    fig4save.BtotList = fig4.BtotList;
    fig4save.time_summary_exact = fig4.time_summary_exact;
    fig4save.time_heuristic = fig4.time_heuristic;
    fig4save.time_bruteforce = fig4.time_bruteforce;
    fig4save.numFeasible_bruteforce = fig4.numFeasible_bruteforce;
    fig4save.predNmse_summary_exact = fig4.predNmse_summary_exact;
    fig4save.predNmse_heuristic = fig4.predNmse_heuristic;

    fig4save.meta = struct();
    fig4save.meta.timestamp = runTimestamp;
    fig4save.meta.cRef = cRef;
    fig4save.meta.cList = cList;
    fig4save.meta.SNRdB = SNRdB;
    fig4save.meta.SNR = SNR;
    fig4save.meta.alpha_fig4 = alpha_fig4;
    fig4save.meta.seed_fig4 = seed_fig4;
    fig4save.meta.maxDropsEval_fig4 = maxDropsEval_fig4;
    fig4save.meta.subIdxEval_fig4 = subIdxEval_fig4;
    fig4save.meta.doBruteForceOracle_fig4 = doBruteForceOracle_fig4;
    fig4save.meta.KmaxBruteForce_fig4 = KmaxBruteForce_fig4;

    dataFile = fullfile(dataOutDir, ['fig4_data_compact' suffix '.mat']);
    save(dataFile, 'fig4save');

    fprintf('Compact Fig. 4 data saved: %s\n', dataFile);
end

else
    fprintf('\n[Fig.4] Skipped because writeFig4 = false.\n');
end



%% ============================================================
% Fig. 5: K-sweep performance scaling
%% ============================================================



if writeFig5

KList_fig5 = [2 4 6 8 10 12 16];
alpha_fig5 = 1.0;
seed_fig5 = 0;

maxDropsEval_fig5 = 5;
NsubEval_fig5 = 16;

doBruteForceOracle_fig5 = true;
KmaxBruteForce_fig5 = 6;

Nsub_full = size(testS.Horg, 4);
subIdxEval_fig5 = unique(round(linspace(1, Nsub_full, NsubEval_fig5)));

policyNamesBase_fig5 = {'uniform','summary_exact','oracle_curve','heuristic'};
if doBruteForceOracle_fig5
    policyNames_fig5 = [policyNamesBase_fig5, {'oracle_bruteforce'}];
else
    policyNames_fig5 = policyNamesBase_fig5;
end

numK_fig5 = numel(KList_fig5);
numPolicy_fig5 = numel(policyNames_fig5);

meanSumRate_fig5 = nan(numK_fig5, numPolicy_fig5);
stdSumRate_fig5  = nan(numK_fig5, numPolicy_fig5);

trueNmse_fig5 = nan(numK_fig5, numPolicy_fig5);
predNmse_fig5 = nan(numK_fig5, numPolicy_fig5);

gainOverUniform_fig5 = nan(numK_fig5, numPolicy_fig5);
bfRecovery_fig5 = nan(numK_fig5, numPolicy_fig5);
curveRecovery_fig5 = nan(numK_fig5, numPolicy_fig5);

policyAlloc_fig5 = cell(numK_fig5, numPolicy_fig5);

fprintf('\n============================================================\n');
fprintf('Fig. 5 K-sweep performance scaling starts\n');
fprintf('============================================================\n');

for ik = 1:numK_fig5

    K_fig5 = KList_fig5(ik);
    Btot_fig5 = alpha_fig5 * K_fig5 * cRef;

    fprintf('\n[Fig.5] %2d / %2d : K = %d, alpha = %.2f, Btot = %.6f\n', ...
        ik, numK_fig5, K_fig5, alpha_fig5, Btot_fig5);

    %% Build K-user test dataset
    out_te_fig5 = build_user_descriptor_curve_dataset( ...
        testData, V, cRef, cList, K_fig5, seed_fig5, obsSize);

    Phi_te_fig5 = out_te_fig5.Phi;
    Ycurve_te_fig5 = out_te_fig5.nmseCurveDb;

    Yhat_te_fig5 = predict_ridge_phi_to_yab(model, Phi_te_fig5);

    DhatDb_te_fig5  = predict_curve_from_ab(Yhat_te_fig5, cList);
    DhatLin_te_fig5 = 10.^(DhatDb_te_fig5 / 10);
    DtrueLin_te_fig5 = 10.^(Ycurve_te_fig5 / 10);

    rowIdx_fig5 = (1:K_fig5).';

    %% Uniform policy
    cPerUserBudget_fig5 = Btot_fig5 / K_fig5;
    feasibleUniformIdx_fig5 = find(cList <= cPerUserBudget_fig5 + 1e-12);

    if isempty(feasibleUniformIdx_fig5)
        warning('[Fig.5] No feasible uniform point for K = %d. Skipping.', K_fig5);
        continue;
    end

    uniformIdx_fig5 = feasibleUniformIdx_fig5(end);
    policy_uniform_fig5.idx = uniformIdx_fig5 * ones(K_fig5,1);
    policy_uniform_fig5.cAlloc = cList(policy_uniform_fig5.idx).';
    policy_uniform_fig5.usedBudget = sum(policy_uniform_fig5.cAlloc);

    %% Surrogate-domain policies
    try
        policy_summary_exact_fig5 = exact_min_sum_nmse_policy( ...
            DhatLin_te_fig5, cList, Btot_fig5);

        policy_oracle_curve_fig5 = exact_min_sum_nmse_policy( ...
            DtrueLin_te_fig5, cList, Btot_fig5);

        policy_heuristic_fig5 = greedy_min_sum_nmse_policy_alljump( ...
            DhatLin_te_fig5, cList, Btot_fig5);
    catch ME
        warning('[Fig.5] Policy solve failed for K = %d: %s', K_fig5, ME.message);
        continue;
    end

    policyCell_fig5 = {
        policy_uniform_fig5, ...
        policy_summary_exact_fig5, ...
        policy_oracle_curve_fig5, ...
        policy_heuristic_fig5
        };

    %% Build sum-rate bank
    fprintf('[Fig.5] Building sum-rate bank for K = %d...\n', K_fig5);

    [Htrue_bank_fig5, Hhat_bank_fig5, numDrops_fig5] = build_sumrate_bank( ...
        testS, out_te_fig5, V, cList, ...
        'MaxDrops', maxDropsEval_fig5, ...
        'Verbose', false);

    %% Optional brute-force oracle for small K
    if doBruteForceOracle_fig5 && (K_fig5 <= KmaxBruteForce_fig5)

        fprintf('[Fig.5] Running brute-force oracle for K = %d...\n', K_fig5);

        try
            oracle_bruteforce_fig5 = brute_force_sumrate_oracle( ...
                Htrue_bank_fig5, Hhat_bank_fig5, cList, Btot_fig5, SNR, ...
                'SubIdx', subIdxEval_fig5, ...
                'Verbose', false, ...
                'MaxRuntimeSec', 300, ...
                'MinBudgetFraction', 0.5);

            policy_oracle_bruteforce_fig5.idx = oracle_bruteforce_fig5.idx;
            policy_oracle_bruteforce_fig5.cAlloc = oracle_bruteforce_fig5.cAlloc;
            policy_oracle_bruteforce_fig5.usedBudget = oracle_bruteforce_fig5.usedBudget;

            policyCell_fig5{end+1} = policy_oracle_bruteforce_fig5;

        catch ME
            warning('[Fig.5] Brute-force oracle failed for K = %d: %s', ...
                K_fig5, ME.message);

            policyCell_fig5{end+1} = struct( ...
                'idx', nan(K_fig5,1), ...
                'cAlloc', nan(K_fig5,1), ...
                'usedBudget', nan);
        end

    elseif doBruteForceOracle_fig5
        policyCell_fig5{end+1} = struct( ...
            'idx', nan(K_fig5,1), ...
            'cAlloc', nan(K_fig5,1), ...
            'usedBudget', nan);
    end

    %% Evaluate NMSE and sum rate
    for p = 1:numel(policyCell_fig5)

        idxp_fig5 = policyCell_fig5{p}.idx(:);

        if any(isnan(idxp_fig5))
            continue;
        end

        trueNmse_fig5(ik,p) = sum(DtrueLin_te_fig5( ...
            sub2ind(size(DtrueLin_te_fig5), rowIdx_fig5, idxp_fig5)));

        predNmse_fig5(ik,p) = sum(DhatLin_te_fig5( ...
            sub2ind(size(DhatLin_te_fig5), rowIdx_fig5, idxp_fig5)));

        policyAlloc_fig5{ik,p} = policyCell_fig5{p}.cAlloc;

        out_sr_fig5 = eval_policy_sumrate_from_bank( ...
            policyCell_fig5{p}, Htrue_bank_fig5, Hhat_bank_fig5, SNR, subIdxEval_fig5);

        meanSumRate_fig5(ik,p) = out_sr_fig5.meanSumRate;
        stdSumRate_fig5(ik,p)  = out_sr_fig5.stdSumRate;
    end

    %% Gain / recovery metrics
    iUniform_fig5 = find(strcmp(policyNames_fig5, 'uniform'));
    iSummary_fig5 = find(strcmp(policyNames_fig5, 'summary_exact'));
    iCurve_fig5   = find(strcmp(policyNames_fig5, 'oracle_curve'));
    iHeur_fig5    = find(strcmp(policyNames_fig5, 'heuristic'));

    for p = 1:numPolicy_fig5
        gainOverUniform_fig5(ik,p) = ...
            (meanSumRate_fig5(ik,p) - meanSumRate_fig5(ik,iUniform_fig5)) / ...
            (meanSumRate_fig5(ik,iUniform_fig5) + eps);
    end

    for p = 1:numPolicy_fig5
        curveRecovery_fig5(ik,p) = ...
            (meanSumRate_fig5(ik,p) - meanSumRate_fig5(ik,iUniform_fig5)) / ...
            (meanSumRate_fig5(ik,iCurve_fig5) - meanSumRate_fig5(ik,iUniform_fig5) + eps);
    end

    if doBruteForceOracle_fig5
        iBF_fig5 = find(strcmp(policyNames_fig5, 'oracle_bruteforce'));

        if ~isnan(meanSumRate_fig5(ik,iBF_fig5))
            for p = 1:numPolicy_fig5
                bfRecovery_fig5(ik,p) = ...
                    (meanSumRate_fig5(ik,p) - meanSumRate_fig5(ik,iUniform_fig5)) / ...
                    (meanSumRate_fig5(ik,iBF_fig5) - meanSumRate_fig5(ik,iUniform_fig5) + eps);
            end
        end
    end

    fprintf('[Fig.5] K = %d done. Uniform = %.6g, Summary = %.6g, Heuristic = %.6g, Curve oracle = %.6g\n', ...
        K_fig5, ...
        meanSumRate_fig5(ik,iUniform_fig5), ...
        meanSumRate_fig5(ik,iSummary_fig5), ...
        meanSumRate_fig5(ik,iHeur_fig5), ...
        meanSumRate_fig5(ik,iCurve_fig5));
end

%% Package results
fig5 = struct();
fig5.KList = KList_fig5(:);
fig5.alpha = alpha_fig5;
fig5.BtotList = alpha_fig5 * KList_fig5(:) * cRef;
fig5.policyNames = policyNames_fig5;

fig5.meanSumRate = meanSumRate_fig5;
fig5.stdSumRate = stdSumRate_fig5;

fig5.trueNmse = trueNmse_fig5;
fig5.predNmse = predNmse_fig5;

fig5.gainOverUniform = gainOverUniform_fig5;
fig5.curveRecovery = curveRecovery_fig5;
fig5.bfRecovery = bfRecovery_fig5;

fig5.policyAlloc = policyAlloc_fig5;

disp('=== Fig. 5 mean sum-rate versus K ===');
disp(array2table(fig5.meanSumRate, ...
    'VariableNames', matlab.lang.makeValidName(fig5.policyNames), ...
    'RowNames', compose('K%d', fig5.KList)))

disp('=== Fig. 5 gain over uniform (%) ===');
disp(array2table(100*fig5.gainOverUniform, ...
    'VariableNames', matlab.lang.makeValidName(fig5.policyNames), ...
    'RowNames', compose('K%d', fig5.KList)))

%% Plot 1: mean sum-rate versus K
fig5a = figure;
hold on; grid on; box on;

for p = 1:numPolicy_fig5
    plot(fig5.KList, fig5.meanSumRate(:,p), '-o', ...
        'LineWidth', 1.5, ...
        'DisplayName', fig5.policyNames{p});
end

xlabel('Number of users K');
ylabel('Mean MU-MIMO sum rate');
title('K-sweep sum-rate performance');
legend('Location','best');

%% Plot 2: gain over uniform versus K
fig5b = figure;
hold on; grid on; box on;

for p = 1:numPolicy_fig5
    if strcmp(fig5.policyNames{p}, 'uniform')
        continue;
    end

    plot(fig5.KList, 100*fig5.gainOverUniform(:,p), '-o', ...
        'LineWidth', 1.5, ...
        'DisplayName', fig5.policyNames{p});
end

xlabel('Number of users K');
ylabel('Gain over uniform allocation (%)');
title('K-sweep gain over uniform allocation');
legend('Location','best');

%% Plot 3: recovery versus K
fig5c = figure;
hold on; grid on; box on;

iSummary_fig5 = find(strcmp(policyNames_fig5, 'summary_exact'));
iHeur_fig5    = find(strcmp(policyNames_fig5, 'heuristic'));
iCurve_fig5   = find(strcmp(policyNames_fig5, 'oracle_curve'));

plot(fig5.KList, 100*fig5.curveRecovery(:,iSummary_fig5), '-o', ...
    'LineWidth', 1.5, 'DisplayName', 'Summary exact');
plot(fig5.KList, 100*fig5.curveRecovery(:,iHeur_fig5), '-^', ...
    'LineWidth', 1.5, 'DisplayName', 'Greedy heuristic');
plot(fig5.KList, 100*fig5.curveRecovery(:,iCurve_fig5), '-s', ...
    'LineWidth', 1.5, 'DisplayName', 'Curve oracle');

xlabel('Number of users K');
ylabel('Recovery relative to curve oracle (%)');
title('K-sweep recovery relative to curve oracle');
legend('Location','best');

%% Save figures and data
figOutDir = fullfile('Figures', 'fig5');
if ~exist(figOutDir, 'dir')
    mkdir(figOutDir);
end

if useTimestampForSave
    suffix = ['_' runTimestamp];
else
    suffix = '';
end

base5a = ['fig5a_sumrate_vs_K' suffix];
base5b = ['fig5b_gain_vs_K' suffix];
base5c = ['fig5c_recovery_vs_K' suffix];

if saveFigMatlabFig
    saveas(fig5a, fullfile(figOutDir, [base5a '.fig']));
    saveas(fig5b, fullfile(figOutDir, [base5b '.fig']));
    saveas(fig5c, fullfile(figOutDir, [base5c '.fig']));
end

if saveFigPng
    saveas(fig5a, fullfile(figOutDir, [base5a '.png']));
    saveas(fig5b, fullfile(figOutDir, [base5b '.png']));
    saveas(fig5c, fullfile(figOutDir, [base5c '.png']));
end

if saveFigPdf
    exportgraphics(fig5a, fullfile(figOutDir, [base5a '.pdf']), ...
        'ContentType', 'vector');
    exportgraphics(fig5b, fullfile(figOutDir, [base5b '.pdf']), ...
        'ContentType', 'vector');
    exportgraphics(fig5c, fullfile(figOutDir, [base5c '.pdf']), ...
        'ContentType', 'vector');
end

if saveFig5Data
    dataOutDir = fullfile('results', 'fig5');
    if ~exist(dataOutDir, 'dir')
        mkdir(dataOutDir);
    end

    fig5save = fig5;
    fig5save.meta = struct();
    fig5save.meta.timestamp = runTimestamp;
    fig5save.meta.cRef = cRef;
    fig5save.meta.cList = cList;
    fig5save.meta.SNRdB = SNRdB;
    fig5save.meta.SNR = SNR;
    fig5save.meta.alpha_fig5 = alpha_fig5;
    fig5save.meta.seed_fig5 = seed_fig5;
    fig5save.meta.maxDropsEval_fig5 = maxDropsEval_fig5;
    fig5save.meta.NsubEval_fig5 = NsubEval_fig5;
    fig5save.meta.subIdxEval_fig5 = subIdxEval_fig5;
    fig5save.meta.doBruteForceOracle_fig5 = doBruteForceOracle_fig5;
    fig5save.meta.KmaxBruteForce_fig5 = KmaxBruteForce_fig5;

    dataFile = fullfile(dataOutDir, ['fig5_data_compact' suffix '.mat']);
    save(dataFile, 'fig5save');

    fprintf('Compact Fig. 5 data saved: %s\n', dataFile);
end

else
    fprintf('\n[Fig.5] Skipped because writeFig5 = false.\n');
end

