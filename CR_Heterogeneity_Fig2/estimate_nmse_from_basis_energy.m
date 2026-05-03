function result = estimate_nmse_from_basis_energy(X, V, cList, tol, verbose)
%ESTIMATE_NMSE_FROM_BASIS_ENERGY
% Estimate average NMSE for multiple compression ratios using basis energy.
%
% Inputs:
%   X     : [Nsample, Ndim], each row should be l2-normalized
%   V     : [Ndim, Ndim], full unitary/orthonormal basis
%   cList : list of compression ratios, e.g. [1/32, 1/16, 1/8, 1/4]
%   tol   : tolerance for row norm check
%
% Output:
%   result.cList
%   result.LList
%   result.capturedEnergyMean
%   result.nmseMean
%   result.rowNormStats
%   result.validity

    if nargin < 4 || isempty(tol)
        tol = 1e-8;
    end
    if nargin < 5
        verbose = true;
    end

    [Nsample, Ndim] = size(X);

    assert(ismatrix(X), 'X must be 2-D');
    assert(ismatrix(V), 'V must be 2-D');
    assert(size(V,1) == Ndim && size(V,2) == Ndim, ...
        'V must be of size [Ndim, Ndim]');
    assert(isvector(cList) && ~isempty(cList), ...
        'cList must be a nonempty vector');

    cList = cList(:).';
    assert(all(cList > 0 & cList <= 1), ...
        'All compression ratios must satisfy 0 < c <= 1');

    % 1) row norm check
    rowNormStats = check_row_unit_norm(X, tol, false);

    % 2) basis coefficients
    Z = X * V;    % [Nsample, Ndim]

    % 3) energy per coefficient and cumulative captured energy
    E = abs(Z).^2;                 % [Nsample, Ndim]
    Ecum = cumsum(E, 2);           % cumulative energy up to column k

    % 4) convert c to retained rank L
    LList = max(1, round(cList * Ndim));
    LList = min(LList, Ndim);

    capturedEnergyMean = zeros(size(LList));
    nmseMean = zeros(size(LList));

    for i = 1:numel(LList)
        L = LList(i);
        captured = Ecum(:, L);              % [Nsample, 1]
        capturedEnergyMean(i) = mean(captured);
        nmseMean(i) = 1 - capturedEnergyMean(i);
    end

    % 5) optional consistency check
    totalEnergyMean = mean(Ecum(:, end));

    validity = struct();
    validity.row_norm_ok = rowNormStats.is_ok;
    validity.mean_total_coeff_energy = totalEnergyMean;
    validity.unitarity_consistency_error = abs(totalEnergyMean - 1);

    result = struct();
    result.cList = cList;
    result.LList = LList;
    result.capturedEnergyMean = capturedEnergyMean;
    result.nmseMean = nmseMean;
    result.rowNormStats = rowNormStats;
    result.validity = validity;

    if verbose
        fprintf('[estimate_nmse_from_basis_energy]\n');
        fprintf('  Nsample = %d\n', Nsample);
        fprintf('  Ndim    = %d\n', Ndim);
        fprintf('  row_norm_ok = %d\n', validity.row_norm_ok);
        fprintf('  mean total coeff energy = %.12f\n', validity.mean_total_coeff_energy);
        fprintf('  consistency error       = %.3e\n', validity.unitarity_consistency_error);
        fprintf('\n');
        fprintf('   c           L        captured_energy_mean      nmse_mean\n');
        fprintf('-------------------------------------------------------------\n');
        for i = 1:numel(LList)
            fprintf('%10.6f   %4d      %.12f      %.12f\n', ...
                cList(i), LList(i), capturedEnergyMean(i), nmseMean(i));
        end
    end
end