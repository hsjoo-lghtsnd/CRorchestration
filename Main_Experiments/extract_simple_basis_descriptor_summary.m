function desc = extract_simple_basis_descriptor_summary(XhatObs, V)
%EXTRACT_SIMPLE_BASIS_DESCRIPTOR_SUMMARY
% Simple observable descriptor summary from reconstructed CSI samples.
%
% XhatObs : [Nobs, D]

    coeff = XhatObs * V;
    e = abs(coeff).^2;

    totalE = sum(abs(XhatObs).^2, 2);
    totalE = max(totalE, eps);

    cumFrac = cumsum(e, 2) ./ totalE;

    L90 = first_crossing_index(cumFrac, 0.90);
    L95 = first_crossing_index(cumFrac, 0.95);
    L99 = first_crossing_index(cumFrac, 0.99);

    D = size(V, 2);

    desc = struct();

    desc.L90Mean = mean(L90);
    desc.L95Mean = mean(L95);
    desc.L99Mean = mean(L99);

    desc.L90Median = median(L90);
    desc.L95Median = median(L95);
    desc.L99Median = median(L99);

    desc.c90Mean = mean(L90 / D);
    desc.c95Mean = mean(L95 / D);
    desc.c99Mean = mean(L99 / D);

    desc.c90Median = median(L90 / D);
    desc.c95Median = median(L95 / D);
    desc.c99Median = median(L99 / D);

    desc.energyMean = mean(totalE);
    desc.energyStd = std(totalE);

    % Compact feature vector
    desc.vector = [
        desc.c90Mean, ...
        desc.c95Mean, ...
        desc.c99Mean, ...
        desc.c90Median, ...
        desc.c95Median, ...
        desc.c99Median, ...
        desc.energyMean, ...
        desc.energyStd
    ];
end