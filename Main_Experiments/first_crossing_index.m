function idx = first_crossing_index(cumFrac, threshold)
%FIRST_CROSSING_INDEX First index where cumulative fraction exceeds threshold.

    [B, D] = size(cumFrac);
    idx = zeros(B, 1);

    mask = cumFrac >= threshold;

    for i = 1:B
        j = find(mask(i,:), 1, 'first');
        if isempty(j)
            j = D;
        end
        idx(i) = j;
    end
end