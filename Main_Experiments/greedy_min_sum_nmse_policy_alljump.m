function out = greedy_min_sum_nmse_policy_alljump(Dlin, cList, Btot)
%GREEDY_MIN_SUM_NMSE_POLICY_ALLJUMP
% Minimize sum_k D_k(c_k) under sum_k c_k <= Btot, with B(c)=c.
%
% At each iteration, considers all feasible jumps c_m -> c_l, l > m,
% and selects the jump with the largest NMSE reduction per added CR cost.
%
% Dlin  : [K, M] linear-scale NMSE curve
% cList : [1, M] admissible CR values, ascending
% Btot  : total feedback budget

    [K, M] = size(Dlin);
    cList = cList(:).';

    if numel(cList) ~= M
        error('cList length must match size(Dlin,2).');
    end

    if any(diff(cList) <= 0)
        error('cList must be strictly increasing.');
    end

    idx = ones(K,1);                 % start from coarsest CR
    usedBudget = K * cList(1);

    if usedBudget > Btot + 1e-12
        error('Btot is smaller than K*min(cList).');
    end

    while true
        bestScore = -inf;
        bestGain = -inf;
        bestUser = NaN;
        bestNextIdx = NaN;

        for k = 1:K
            m0 = idx(k);

            for m1 = (m0+1):M
                addCost = cList(m1) - cList(m0);

                if usedBudget + addCost > Btot + 1e-12
                    continue;
                end

                gain = Dlin(k,m0) - Dlin(k,m1);  % NMSE reduction

                if gain <= 0
                    continue;
                end

                score = gain / addCost;          % reduction per CR cost

                if score > bestScore || ...
                   (abs(score - bestScore) < 1e-15 && gain > bestGain)
                    bestScore = score;
                    bestGain = gain;
                    bestUser = k;
                    bestNextIdx = m1;
                end
            end
        end

        if isnan(bestUser)
            break;
        end

        oldIdx = idx(bestUser);
        idx(bestUser) = bestNextIdx;
        usedBudget = usedBudget + cList(bestNextIdx) - cList(oldIdx);
    end

    rowIdx = (1:K)';
    linIdx = sub2ind(size(Dlin), rowIdx, idx);

    out.idx = idx;
    out.cAlloc = cList(idx).';
    out.usedBudget = usedBudget;
    out.sumNmse = sum(Dlin(linIdx));
    out.meanNmse = mean(Dlin(linIdx));
    out.selectedNmse = Dlin(linIdx);
end