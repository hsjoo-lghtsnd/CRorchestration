K_te = 16;
seed_te = 0;

out_te = build_user_descriptor_curve_dataset( ...
    testData, V, cRef, cList, K_te, seed_te, obsSize);

Phi_te = out_te.Phi;
Ycurve_te = out_te.nmseCurveDb;
Yhat_te = predict_ridge_phi_to_yab(model, Phi_te);

DhatDb_te  = predict_curve_from_ab(Yhat_te, cList);
DhatLin_te = 10.^(DhatDb_te / 10);
DtrueLin_te = 10.^(Ycurve_te / 10);

Btot = K_te * cRef;

pol_exact = exact_min_sum_nmse_policy(DhatLin_te, cList, Btot);
pol_greedy = greedy_min_sum_nmse_policy_alljump(DhatLin_te, cList, Btot);

rowIdx = (1:K_te).';

pred_exact = sum(DhatLin_te(sub2ind(size(DhatLin_te), rowIdx, pol_exact.idx(:))));
pred_greedy = sum(DhatLin_te(sub2ind(size(DhatLin_te), rowIdx, pol_greedy.idx(:))));

true_exact = sum(DtrueLin_te(sub2ind(size(DtrueLin_te), rowIdx, pol_exact.idx(:))));
true_greedy = sum(DtrueLin_te(sub2ind(size(DtrueLin_te), rowIdx, pol_greedy.idx(:))));

fprintf('K = %d\n', K_te);
fprintf('Allocation mismatch count = %d / %d\n', sum(pol_exact.idx ~= pol_greedy.idx), K_te);
fprintf('Predicted NMSE exact      = %.10f\n', pred_exact);
fprintf('Predicted NMSE greedy     = %.10f\n', pred_greedy);
fprintf('True NMSE exact alloc     = %.10f\n', true_exact);
fprintf('True NMSE greedy alloc    = %.10f\n', true_greedy);


alphaList = 0.6:0.1:1.4;

for ia = 1:numel(alphaList)
    Btot = alphaList(ia) * K_te * cRef;

    pol_exact = exact_min_sum_nmse_policy(DhatLin_te, cList, Btot);
    pol_greedy = greedy_min_sum_nmse_policy_alljump(DhatLin_te, cList, Btot);

    mismatch = sum(pol_exact.idx ~= pol_greedy.idx);

    fprintf('alpha=%.2f, mismatch=%d\n', alphaList(ia), mismatch);
end

alpha = 0.60;
Btot = alpha * K_te * cRef;

pol_exact = exact_min_sum_nmse_policy(DhatLin_te, cList, Btot);
pol_greedy = greedy_min_sum_nmse_policy_alljump(DhatLin_te, cList, Btot);

rowIdx = (1:K_te).';

pred_exact = sum(DhatLin_te(sub2ind(size(DhatLin_te), rowIdx, pol_exact.idx(:))));
pred_greedy = sum(DhatLin_te(sub2ind(size(DhatLin_te), rowIdx, pol_greedy.idx(:))));

true_exact = sum(DtrueLin_te(sub2ind(size(DtrueLin_te), rowIdx, pol_exact.idx(:))));
true_greedy = sum(DtrueLin_te(sub2ind(size(DtrueLin_te), rowIdx, pol_greedy.idx(:))));

fprintf('alpha=%.2f\n', alpha);
fprintf('mismatch = %d\n', sum(pol_exact.idx ~= pol_greedy.idx));
fprintf('pred exact  = %.10f\n', pred_exact);
fprintf('pred greedy = %.10f\n', pred_greedy);
fprintf('true exact  = %.10f\n', true_exact);
fprintf('true greedy = %.10f\n', true_greedy);

Tdiff = table((1:K_te)', pol_exact.cAlloc, pol_greedy.cAlloc, ...
    'VariableNames', {'user','c_exact','c_greedy'});
disp(Tdiff(Tdiff.c_exact ~= Tdiff.c_greedy, :))





%% ============================================================
% Check exact vs greedy at K = 16, alpha = 0.60
% Requires in workspace:
%   testS, testData, V, model, cRef, cList, obsSize, SNR
%   build_user_descriptor_curve_dataset
%   exact_min_sum_nmse_policy
%   greedy_min_sum_nmse_policy_alljump
%   build_sumrate_bank
%   eval_policy_sumrate_from_bank
%% ============================================================

K_chk = 16;
seed_chk = 0;
alpha_chk = 0.60;
Btot_chk = alpha_chk * K_chk * cRef;

maxDropsEval_chk = 10;
Nsub_full_chk = size(testS.Horg, 4);
subIdxEval_chk = unique(round(linspace(1, Nsub_full_chk, 32)));

%% 1) rebuild test-user dataset for K = 16
out_te_chk = build_user_descriptor_curve_dataset( ...
    testData, V, cRef, cList, K_chk, seed_chk, obsSize);

Phi_te_chk = out_te_chk.Phi;
Ycurve_te_chk = out_te_chk.nmseCurveDb;
Yhat_te_chk = predict_ridge_phi_to_yab(model, Phi_te_chk);

DhatDb_te_chk  = predict_curve_from_ab(Yhat_te_chk, cList);
DhatLin_te_chk = 10.^(DhatDb_te_chk / 10);
DtrueLin_te_chk = 10.^(Ycurve_te_chk / 10);

%% 2) solve exact and greedy policies
pol_exact = exact_min_sum_nmse_policy(DhatLin_te_chk, cList, Btot_chk);
pol_greedy = greedy_min_sum_nmse_policy_alljump(DhatLin_te_chk, cList, Btot_chk);

%% 3) NMSE-side comparison
rowIdx_chk = (1:K_chk).';

pred_exact = sum(DhatLin_te_chk(sub2ind(size(DhatLin_te_chk), rowIdx_chk, pol_exact.idx(:))));
pred_greedy = sum(DhatLin_te_chk(sub2ind(size(DhatLin_te_chk), rowIdx_chk, pol_greedy.idx(:))));

true_exact = sum(DtrueLin_te_chk(sub2ind(size(DtrueLin_te_chk), rowIdx_chk, pol_exact.idx(:))));
true_greedy = sum(DtrueLin_te_chk(sub2ind(size(DtrueLin_te_chk), rowIdx_chk, pol_greedy.idx(:))));

fprintf('\n=== alpha = %.2f, K = %d : exact vs greedy ===\n', alpha_chk, K_chk);
fprintf('Predicted sum NMSE (exact)   = %.10f\n', pred_exact);
fprintf('Predicted sum NMSE (greedy)  = %.10f\n', pred_greedy);
fprintf('True sum NMSE (exact alloc)  = %.10f\n', true_exact);
fprintf('True sum NMSE (greedy alloc) = %.10f\n', true_greedy);
fprintf('Allocation mismatch count    = %d / %d\n', ...
    sum(pol_exact.idx ~= pol_greedy.idx), K_chk);

Tdiff = table((1:K_chk)', pol_exact.cAlloc, pol_greedy.cAlloc, ...
    'VariableNames', {'user','c_exact','c_greedy'});
disp(Tdiff(Tdiff.c_exact ~= Tdiff.c_greedy, :))

%% 4) rebuild sum-rate bank for K = 16
[Htrue_bank_chk, Hhat_bank_chk, numDrops_chk] = build_sumrate_bank( ...
    testS, out_te_chk, V, cList, ...
    'MaxDrops', maxDropsEval_chk, ...
    'Verbose', true);

fprintf('numDrops used = %d\n', numDrops_chk);

%% 5) sum-rate comparison using the rebuilt bank
out_sr_exact = eval_policy_sumrate_from_bank( ...
    pol_exact, Htrue_bank_chk, Hhat_bank_chk, SNR, subIdxEval_chk);

out_sr_greedy = eval_policy_sumrate_from_bank( ...
    pol_greedy, Htrue_bank_chk, Hhat_bank_chk, SNR, subIdxEval_chk);

fprintf('Mean sum rate (exact)        = %.10f\n', out_sr_exact.meanSumRate);
fprintf('Mean sum rate (greedy)       = %.10f\n', out_sr_greedy.meanSumRate);
fprintf('Std sum rate  (exact)        = %.10f\n', out_sr_exact.stdSumRate);
fprintf('Std sum rate  (greedy)       = %.10f\n', out_sr_greedy.stdSumRate);
fprintf('Absolute sum-rate gap        = %.10e\n', ...
    out_sr_exact.meanSumRate - out_sr_greedy.meanSumRate);

srGapVec = out_sr_exact.sumRateVec - out_sr_greedy.sumRateVec;
fprintf('Mean per-drop gap            = %.10e\n', mean(srGapVec));
fprintf('Max  per-drop gap            = %.10e\n', max(abs(srGapVec)));






%% ============================================================
% Sweep check: exact vs greedy at K = 16 over multiple budgets
%
% Requires in workspace:
%   testS, testData, V, model, cRef, cList, obsSize, SNR
%   build_user_descriptor_curve_dataset
%   exact_min_sum_nmse_policy
%   greedy_min_sum_nmse_policy_alljump
%   build_sumrate_bank
%   eval_policy_sumrate_from_bank
%% ============================================================

K_chk = 16;
seed_chk = 0;

alphaList_chk = 0.50:0.10:0.80;   % tighten or widen as needed
maxDropsEval_chk = 10;

Nsub_full_chk = size(testS.Horg, 4);
subIdxEval_chk = unique(round(linspace(1, Nsub_full_chk, 32)));

%% 1) Rebuild test-user dataset for K = 16
out_te_chk = build_user_descriptor_curve_dataset( ...
    testData, V, cRef, cList, K_chk, seed_chk, obsSize);

Phi_te_chk = out_te_chk.Phi;
Ycurve_te_chk = out_te_chk.nmseCurveDb;
Yhat_te_chk = predict_ridge_phi_to_yab(model, Phi_te_chk);

DhatDb_te_chk  = predict_curve_from_ab(Yhat_te_chk, cList);
DhatLin_te_chk = 10.^(DhatDb_te_chk / 10);
DtrueLin_te_chk = 10.^(Ycurve_te_chk / 10);

%% 2) Rebuild sum-rate bank for K = 16
[Htrue_bank_chk, Hhat_bank_chk, numDrops_chk] = build_sumrate_bank( ...
    testS, out_te_chk, V, cList, ...
    'MaxDrops', maxDropsEval_chk, ...
    'Verbose', true);

fprintf('numDrops used = %d\n', numDrops_chk);

%% 3) Sweep containers
numAlpha_chk = numel(alphaList_chk);

mismatchCount_chk = nan(numAlpha_chk,1);

predExact_chk = nan(numAlpha_chk,1);
predGreedy_chk = nan(numAlpha_chk,1);
predGap_chk = nan(numAlpha_chk,1);

trueExact_chk = nan(numAlpha_chk,1);
trueGreedy_chk = nan(numAlpha_chk,1);
trueGap_chk = nan(numAlpha_chk,1);

srExact_chk = nan(numAlpha_chk,1);
srGreedy_chk = nan(numAlpha_chk,1);
srGap_chk = nan(numAlpha_chk,1);

stdExact_chk = nan(numAlpha_chk,1);
stdGreedy_chk = nan(numAlpha_chk,1);

allocExact_chk = cell(numAlpha_chk,1);
allocGreedy_chk = cell(numAlpha_chk,1);

rowIdx_chk = (1:K_chk).';

%% 4) Main sweep
fprintf('\n============================================================\n');
fprintf('Sweep check: exact vs greedy for K = %d\n', K_chk);
fprintf('============================================================\n');

for ia = 1:numAlpha_chk
    alpha_chk = alphaList_chk(ia);
    Btot_chk = alpha_chk * K_chk * cRef;

    fprintf('\nalpha = %.2f, Btot = %.6f\n', alpha_chk, Btot_chk);

    % Solve policies
    pol_exact = exact_min_sum_nmse_policy(DhatLin_te_chk, cList, Btot_chk);
    pol_greedy = greedy_min_sum_nmse_policy_alljump(DhatLin_te_chk, cList, Btot_chk);

    allocExact_chk{ia} = pol_exact.cAlloc;
    allocGreedy_chk{ia} = pol_greedy.cAlloc;

    % Allocation mismatch
    mismatchCount_chk(ia) = sum(pol_exact.idx ~= pol_greedy.idx);

    % Predicted NMSE
    predExact_chk(ia) = sum(DhatLin_te_chk(sub2ind(size(DhatLin_te_chk), rowIdx_chk, pol_exact.idx(:))));
    predGreedy_chk(ia) = sum(DhatLin_te_chk(sub2ind(size(DhatLin_te_chk), rowIdx_chk, pol_greedy.idx(:))));
    predGap_chk(ia) = predGreedy_chk(ia) - predExact_chk(ia);   % positive means exact better on surrogate

    % True NMSE
    trueExact_chk(ia) = sum(DtrueLin_te_chk(sub2ind(size(DtrueLin_te_chk), rowIdx_chk, pol_exact.idx(:))));
    trueGreedy_chk(ia) = sum(DtrueLin_te_chk(sub2ind(size(DtrueLin_te_chk), rowIdx_chk, pol_greedy.idx(:))));
    trueGap_chk(ia) = trueGreedy_chk(ia) - trueExact_chk(ia);   % positive means exact better on true NMSE

    % Sum-rate
    out_sr_exact = eval_policy_sumrate_from_bank( ...
        pol_exact, Htrue_bank_chk, Hhat_bank_chk, SNR, subIdxEval_chk);

    out_sr_greedy = eval_policy_sumrate_from_bank( ...
        pol_greedy, Htrue_bank_chk, Hhat_bank_chk, SNR, subIdxEval_chk);

    srExact_chk(ia) = out_sr_exact.meanSumRate;
    srGreedy_chk(ia) = out_sr_greedy.meanSumRate;
    srGap_chk(ia) = srGreedy_chk(ia) - srExact_chk(ia);         % positive means greedy better on sum rate

    stdExact_chk(ia) = out_sr_exact.stdSumRate;
    stdGreedy_chk(ia) = out_sr_greedy.stdSumRate;

    fprintf('  mismatch count      = %d / %d\n', mismatchCount_chk(ia), K_chk);
    fprintf('  pred gap (g-e)      = %+0.10e\n', predGap_chk(ia));
    fprintf('  true gap (g-e)      = %+0.10e\n', trueGap_chk(ia));
    fprintf('  sum-rate gap (g-e)  = %+0.10e\n', srGap_chk(ia));
end

%% 5) Summary table
Tsweep_chk = table( ...
    alphaList_chk(:), ...
    mismatchCount_chk, ...
    predExact_chk, predGreedy_chk, predGap_chk, ...
    trueExact_chk, trueGreedy_chk, trueGap_chk, ...
    srExact_chk, srGreedy_chk, srGap_chk, ...
    stdExact_chk, stdGreedy_chk, ...
    'VariableNames', ...
    {'alpha','mismatchCount', ...
     'predExact','predGreedy','predGap_greedy_minus_exact', ...
     'trueExact','trueGreedy','trueGap_greedy_minus_exact', ...
     'srExact','srGreedy','srGap_greedy_minus_exact', ...
     'stdExact','stdGreedy'});

disp(Tsweep_chk)

%% 6) Optional: show mismatched users for the budgets that differ
for ia = 1:numAlpha_chk
    if mismatchCount_chk(ia) > 0
        alpha_chk = alphaList_chk(ia);
        Btot_chk = alpha_chk * K_chk * cRef;

        pol_exact = exact_min_sum_nmse_policy(DhatLin_te_chk, cList, Btot_chk);
        pol_greedy = greedy_min_sum_nmse_policy_alljump(DhatLin_te_chk, cList, Btot_chk);

        fprintf('\n--- Detailed mismatch at alpha = %.2f ---\n', alpha_chk);
        Tdiff_chk = table((1:K_chk)', pol_exact.cAlloc, pol_greedy.cAlloc, ...
            'VariableNames', {'user','c_exact','c_greedy'});
        disp(Tdiff_chk(Tdiff_chk.c_exact ~= Tdiff_chk.c_greedy, :))
    end
end