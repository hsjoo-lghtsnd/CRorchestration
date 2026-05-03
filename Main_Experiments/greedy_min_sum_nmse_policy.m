function out = greedy_min_sum_nmse_policy(Dlin, cList, Btot)
%GREEDY_MIN_SUM_NMSE_POLICY
% Minimize sum_k D_k(c_k) under sum_k c_k <= Btot, with B(c)=c.
%
% Dlin  : [K, M] predicted or true linear NMSE
% cList : [1, M] admissible CR values, ascending
% Btot  : scalar total CR budget

    [K, M] = size(Dlin);
    cList = cList(:).';

    if numel(cList) ~= M
        error('cList length must match size(Dlin,2).');
    end

    % Start from coarsest CR
    idx = ones(K,1);
    usedBudget = K * cList(1);

    if usedBudget > Btot + 1e-12
        error('Btot is smaller than minimum feasible budget K*min(cList).');
    end

    while true
        bestScore = -inf;
        bestUser = NaN;

        for k = 1:K
            m = idx(k);

            if m >= M
                continue;
            end

            oldC = cList(m);
            newC = cList(m+1);
            addCost = newC - oldC;

            if usedBudget + addCost > Btot + 1e-12
                continue;
            end

            % Since objective is minimize NMSE, gain is reduction in NMSE
            gain = Dlin(k,m) - Dlin(k,m+1);
            score = gain / addCost;

            if score > bestScore
                bestScore = score;
                bestUser = k;
            end
        end

        if isnan(bestUser) || bestScore <= 0
            break;
        end

        oldIdx = idx(bestUser);
        idx(bestUser) = oldIdx + 1;
        usedBudget = usedBudget + cList(oldIdx+1) - cList(oldIdx);
    end

    cAlloc = cList(idx).';

    out.idx = idx;
    out.cAlloc = cAlloc;
    out.usedBudget = usedBudget;
    out.sumPredNmse = sum(Dlin(sub2ind(size(Dlin), (1:K)', idx)));
end