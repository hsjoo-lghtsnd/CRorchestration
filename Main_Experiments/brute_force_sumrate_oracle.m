function out = brute_force_sumrate_oracle(Htrue_bank, Hhat_bank, cList, Btot, SNR, varargin)
%BRUTE_FORCE_SUMRATE_ORACLE True brute-force oracle for sum-rate maximization
%
% Solves approximately:
%   max_{c_k in cList, sum_k c_k <= Btot} mean_drop U(c)
%
% with:
%   - exhaustive feasible search with budget pruning
%   - early stop if runtime exceeds a given limit
%   - initial incumbent from uniform allocation
%   - optional lower bound on tested total budget
%
% Input:
%   Htrue_bank : cell(K, numDrops), each entry [1, Nt, Nr, Nsub]
%   Hhat_bank  : cell(K, numDrops, M), each entry [1, Nt, Nr, Nsub]
%   cList      : [1, M] admissible operating points, ascending
%   Btot       : scalar total budget
%   SNR        : linear SNR
%
% Name-value options:
%   'SubIdx'           : subcarrier indices for fast evaluation
%   'Verbose'          : true/false, default = true
%   'MaxRuntimeSec'    : stop search after this many seconds, default = 300
%   'MinBudgetFraction': only evaluate allocations with usedBudget >=
%                        MinBudgetFraction * Btot, default = 0.5
%
% Output:
%   out.idx             : [K,1] best operating-point indices found
%   out.cAlloc          : [K,1] best operating-point values
%   out.usedBudget      : scalar
%   out.meanSumRate     : scalar best mean sum rate found
%   out.stdSumRate      : scalar std of sum rate across drops
%   out.sumRateVec      : [numDrops,1] sum rate per drop for best allocation
%   out.userRateMat     : [numDrops,K] user rates for best allocation
%   out.numFeasible     : number of feasible allocations actually evaluated
%   out.numVisited      : number of leaf allocations reached
%   out.didTimeout      : true if search stopped by time limit
%   out.elapsedSec      : elapsed wall-clock time
%   out.initializedFromUniform : true if initial incumbent was uniform
%
% Notes:
%   - If timeout happens, returns the best incumbent found so far.
%   - Search is still exact only if didTimeout == false and
%     MinBudgetFraction does not exclude any truly optimal point.

    p = inputParser;
    addParameter(p, 'SubIdx', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'Verbose', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'MaxRuntimeSec', 300, @(x) isscalar(x) && x > 0);
    addParameter(p, 'MinBudgetFraction', 0.5, @(x) isscalar(x) && x >= 0 && x <= 1);
    parse(p, varargin{:});

    subIdx = p.Results.SubIdx;
    verbose = logical(p.Results.Verbose);
    maxRuntimeSec = p.Results.MaxRuntimeSec;
    minBudgetFraction = p.Results.MinBudgetFraction;

    cList = cList(:).';
    M = numel(cList);

    K = size(Htrue_bank, 1);
    numDrops = size(Htrue_bank, 2);

    if size(Hhat_bank,1) ~= K || size(Hhat_bank,2) ~= numDrops || size(Hhat_bank,3) ~= M
        error('Hhat_bank size must be [K, numDrops, M] consistent with Htrue_bank and cList.');
    end

    if any(diff(cList) <= 0)
        error('cList must be strictly increasing.');
    end

    minBudget = K * cList(1);
    if Btot + 1e-12 < minBudget
        error('Btot is smaller than minimum feasible budget K*min(cList).');
    end

    minTestBudget = max(minBudget, minBudgetFraction * Btot);

    % Best-so-far record
    bestMeanSumRate = -inf;
    bestStdSumRate = NaN;
    bestIdx = [];
    bestSumRateVec = [];
    bestUserRateMat = [];
    initializedFromUniform = false;

    numFeasible = 0;   % number of allocations actually evaluated
    numVisited = 0;    % number of feasible leaves reached (before budget-floor filtering)
    idxCur = ones(K,1);
    didTimeout = false;
    tStart = tic;

    if verbose
        fprintf('\n=== brute_force_sumrate_oracle ===\n');
        fprintf('K = %d, M = %d, numDrops = %d\n', K, M, numDrops);
        fprintf('Btot = %.10f\n', Btot);
        fprintf('MaxRuntimeSec = %.2f\n', maxRuntimeSec);
        fprintf('Min tested budget = max(K*min(c), %.2f*Btot) = %.10f\n', ...
            minBudgetFraction, minTestBudget);
    end

    % ------------------------------------------------------------
    % Initial incumbent: uniform baseline (largest feasible uniform c <= Btot/K)
    % ------------------------------------------------------------
    cPerUserBudget = Btot / K;
    feasibleUniformIdx = find(cList <= cPerUserBudget + 1e-12, 1, 'last');

    if ~isempty(feasibleUniformIdx)
        policy_uniform.idx = feasibleUniformIdx * ones(K,1);
        policy_uniform.cAlloc = cList(policy_uniform.idx).';
        policy_uniform.usedBudget = sum(policy_uniform.cAlloc);

        if policy_uniform.usedBudget >= minTestBudget - 1e-12
            out_sr0 = eval_policy_sumrate_from_bank( ...
                policy_uniform, Htrue_bank, Hhat_bank, SNR, subIdx);

            bestMeanSumRate = out_sr0.meanSumRate;
            bestStdSumRate = out_sr0.stdSumRate;
            bestIdx = policy_uniform.idx;
            bestSumRateVec = out_sr0.sumRateVec;
            bestUserRateMat = out_sr0.userRateMat;
            initializedFromUniform = true;

            if verbose
                fprintf('Initialized incumbent from uniform: meanSR = %.10f, budget = %.10f\n', ...
                    bestMeanSumRate, policy_uniform.usedBudget);
            end
        end
    end

    recurse_user(1, 0);

    elapsedSec = toc(tStart);

    if isempty(bestIdx)
        error('No feasible allocation was evaluated before termination.');
    end

    out.idx = bestIdx;
    out.cAlloc = cList(bestIdx).';
    out.usedBudget = sum(out.cAlloc);
    out.meanSumRate = bestMeanSumRate;
    out.stdSumRate = bestStdSumRate;
    out.sumRateVec = bestSumRateVec;
    out.userRateMat = bestUserRateMat;
    out.numFeasible = numFeasible;
    out.numVisited = numVisited;
    out.didTimeout = didTimeout;
    out.elapsedSec = elapsedSec;
    out.initializedFromUniform = initializedFromUniform;

    % ============================================================
    % Nested recursive enumeration with budget pruning
    % ============================================================
    function recurse_user(kUser, budgetSoFar)
        % timeout guard
        if toc(tStart) > maxRuntimeSec
            didTimeout = true;
            return;
        end

        if kUser > K
            numVisited = numVisited + 1;

            usedBudgetCur = sum(cList(idxCur));

            % skip very small-total allocations
            if usedBudgetCur < minTestBudget - 1e-12
                return;
            end

            % Evaluate this feasible allocation
            numFeasible = numFeasible + 1;

            policy_tmp.idx = idxCur;
            policy_tmp.cAlloc = cList(idxCur).';
            policy_tmp.usedBudget = usedBudgetCur;

            out_sr = eval_policy_sumrate_from_bank( ...
                policy_tmp, Htrue_bank, Hhat_bank, SNR, subIdx);

            meanSR = out_sr.meanSumRate;

            if meanSR > bestMeanSumRate
                bestMeanSumRate = meanSR;
                bestStdSumRate = out_sr.stdSumRate;
                bestIdx = idxCur;
                bestSumRateVec = out_sr.sumRateVec;
                bestUserRateMat = out_sr.userRateMat;

                if verbose
                    fprintf('New best: meanSR = %.10f, usedBudget = %.10f, idx = %s, elapsed = %.2fs\n', ...
                        bestMeanSumRate, usedBudgetCur, mat2str(bestIdx.'), toc(tStart));
                end
            end
            return;
        end

        usersLeftAfter = K - kUser;

        % iterate finer-to-coarser so good incumbents are found earlier
        for m = M:-1:1
            cNow = cList(m);
            newBudget = budgetSoFar + cNow;

            % feasibility lower bound for remaining users
            minRemaining = usersLeftAfter * cList(1);
            if newBudget + minRemaining > Btot + 1e-12
                continue;
            end

            % optional upper optimism test for budget floor:
            % even if all remaining users take max c, can we still reach minTestBudget?
            maxRemaining = usersLeftAfter * cList(end);
            if newBudget + maxRemaining < minTestBudget - 1e-12
                continue;
            end

            idxCur(kUser) = m;
            recurse_user(kUser + 1, newBudget);

            if didTimeout
                return;
            end
        end
    end
end