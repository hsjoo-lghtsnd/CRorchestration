function out = exact_min_sum_nmse_policy(Dlin, cList, Btot)
%EXACT_MIN_SUM_NMSE_POLICY
% Exact global search for
%   min sum_k D_k(c_k)
% s.t.
%   sum_k c_k <= Btot,  c_k in cList
%
% Input:
%   Dlin  : [K, M] linear-scale NMSE (predicted or true)
%   cList : [1, M] admissible CR values, ascending
%   Btot  : scalar total budget
%
% Output:
%   out.idx         : [K,1] selected operating-point indices
%   out.cAlloc      : [K,1] selected CR values
%   out.usedBudget  : scalar used budget
%   out.sumNmse     : scalar exact minimum objective value

    [K, M] = size(Dlin);
    cList = cList(:).';

    if numel(cList) ~= M
        error('cList length must match size(Dlin,2).');
    end
    if any(diff(cList) <= 0)
        error('cList must be strictly increasing.');
    end

    minBudget = K * cList(1);
    if Btot + 1e-12 < minBudget
        error('Btot is smaller than minimum feasible budget K*min(cList).');
    end

    tol = 1e-12;

    % ------------------------------------------------------------
    % Reachable budget states by stage
    % states{k+1} = reachable total budgets using first k users
    % ------------------------------------------------------------
    states = cell(K+1, 1);
    states{1} = 0;   % 0 users -> budget 0

    for k = 1:K
        prevStates = states{k};
        newStates = [];

        for b = prevStates
            newStates = [newStates, b + cList]; %#ok<AGROW>
        end

        newStates = unique(round(newStates, 12));
        newStates = newStates(newStates <= Btot + tol);
        states{k+1} = sort(newStates);
    end

    if isempty(states{K+1})
        error('No feasible terminal budget states found.');
    end

    % ------------------------------------------------------------
    % DP tables per stage
    % dp{k+1}(j) = min cost using first k users ending at states{k+1}(j)
    % ------------------------------------------------------------
    dp = cell(K+1, 1);
    parentBudget = cell(K+1, 1);
    parentChoice = cell(K+1, 1);

    dp{1} = 0;                  % only one state: budget 0
    parentBudget{1} = 0;
    parentChoice{1} = 0;

    for k = 1:K
        prevStates = states{k};
        currStates = states{k+1};

        dp{k+1} = inf(1, numel(currStates));
        parentBudget{k+1} = zeros(1, numel(currStates));
        parentChoice{k+1} = zeros(1, numel(currStates));

        for jPrev = 1:numel(prevStates)
            prevCost = dp{k}(jPrev);
            if ~isfinite(prevCost)
                continue;
            end

            bPrev = prevStates(jPrev);

            for m = 1:M
                bNow = round(bPrev + cList(m), 12);
                if bNow > Btot + tol
                    continue;
                end

                jNow = find(abs(currStates - bNow) <= tol, 1);
                if isempty(jNow)
                    continue;
                end

                cand = prevCost + Dlin(k, m);
                if cand < dp{k+1}(jNow)
                    dp{k+1}(jNow) = cand;
                    parentBudget{k+1}(jNow) = jPrev;
                    parentChoice{k+1}(jNow) = m;
                end
            end
        end
    end

    % ------------------------------------------------------------
    % Best terminal state
    % ------------------------------------------------------------
    [bestVal, jBest] = min(dp{K+1});
    if ~isfinite(bestVal)
        error('No feasible exact solution found.');
    end

    % ------------------------------------------------------------
    % Backtracking
    % ------------------------------------------------------------
    idx = zeros(K,1);
    jCur = jBest;

    for k = K:-1:1
        idx(k) = parentChoice{k+1}(jCur);
        jCur = parentBudget{k+1}(jCur);
    end

    cAlloc = cList(idx).';
    usedBudget = sum(cAlloc);

    out.idx = idx;
    out.cAlloc = cAlloc;
    out.usedBudget = usedBudget;
    out.sumNmse = bestVal;
    out.terminalBudgetStates = states{K+1};
end