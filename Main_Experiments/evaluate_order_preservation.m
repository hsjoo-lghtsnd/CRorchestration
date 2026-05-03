function metrics = evaluate_order_preservation(Yab, Yhat, cList)
%EVALUATE_ORDER_PRESERVATION
% Check whether predicted [a,b] preserves user ordering.
%
% Input:
%   Yab   : [N,2] true [a,b]
%   Yhat  : [N,2] predicted [a,b]
%   cList : [1,M] CR list
%
% Output:
%   metrics: struct with rank-correlation and top-k overlap results

    arguments
        Yab double
        Yhat double
        cList double
    end

    if size(Yab,2) ~= 2 || size(Yhat,2) ~= 2
        error('Yab and Yhat must be [N,2].');
    end

    N = size(Yab,1);

    aTrue = Yab(:,1); bTrue = Yab(:,2);
    aHat  = Yhat(:,1); bHat  = Yhat(:,2);

    metrics = struct();

    % 1) a and b ordering
    metrics.spearman_a = corr(aTrue, aHat, 'Type', 'Spearman');
    metrics.spearman_b = corr(bTrue, bHat, 'Type', 'Spearman');

    metrics.kendall_a = corr(aTrue, aHat, 'Type', 'Kendall');
    metrics.kendall_b = corr(bTrue, bHat, 'Type', 'Kendall');

    % 2) marginal gain ordering at each CR step
    gainTrue = compute_marginal_gain_from_ab(Yab, cList);
    gainHat  = compute_marginal_gain_from_ab(Yhat, cList);
    % size = [N, M-1]

    M1 = size(gainTrue, 2);
    metrics.spearman_gain = zeros(1, M1);
    metrics.kendall_gain = zeros(1, M1);

    topFracList = [0.1, 0.2, 0.25];
    metrics.topOverlap_gain = zeros(M1, numel(topFracList));

    for m = 1:M1
        gt = gainTrue(:,m);
        gh = gainHat(:,m);

        metrics.spearman_gain(m) = corr(gt, gh, 'Type', 'Spearman');
        metrics.kendall_gain(m) = corr(gt, gh, 'Type', 'Kendall');

        for t = 1:numel(topFracList)
            q = max(1, round(N * topFracList(t)));
            idxTrue = topk_indices(gt, q);
            idxHat  = topk_indices(gh, q);

            metrics.topOverlap_gain(m,t) = numel(intersect(idxTrue, idxHat)) / q;
        end
    end

    metrics.mean_spearman_gain = mean(metrics.spearman_gain);
    metrics.mean_kendall_gain = mean(metrics.kendall_gain);
end

function gain = compute_marginal_gain_from_ab(Yab, cList)
%COMPUTE_MARGINAL_GAIN_FROM_AB
% gain(:,m) = predicted distortion reduction from cList(m) to cList(m+1)
%
% Uses linear NMSE:
% D(c) = 10^((a - b log2(c))/10)

    a = Yab(:,1);
    b = Yab(:,2);

    x = log2(cList(:))';                 % [1,M]
    Ddb = a - b .* x;                    % [N,M]
    Dlin = 10.^(Ddb / 10);

    gain = Dlin(:,1:end-1) - Dlin(:,2:end);
end

function idx = topk_indices(x, k)
%TOPK_INDICES Return indices of top-k largest entries.

    [~, order] = sort(x, 'descend');
    idx = order(1:k);
end
