function result = estimate_nmse_from_basis_energy_light(X, V, cList, tol, verbose)
% X     : [Nsample, Ndim], row-normalized
% V     : [Ndim, Ndim] full orthonormal/unitary basis
% cList : compression ratios, e.g. [1/32 1/16 1/8]
%
% result fields:
%   .cList
%   .LList
%   .capturedEnergyMean
%   .nmseMean
%   .rowNormStats

    if nargin < 4 || isempty(tol)
        tol = 1e-5;
    end
    if nargin < 5
        verbose = true;
    end

    [Nsample, Ndim] = size(X);
    cList = cList(:).';

    % row norm check
    rowEnergy = sum(abs(X).^2, 2);
    rowNorms = sqrt(rowEnergy);
    absErr = abs(rowNorms - 1);

    rowNormStats = struct();
    rowNormStats.is_ok = all(absErr <= tol);
    rowNormStats.max_abs_err = max(absErr);
    rowNormStats.mean_abs_err = mean(absErr);
    rowNormStats.min_norm = min(rowNorms);
    rowNormStats.max_norm = max(rowNorms);
    rowNormStats.mean_norm = mean(rowNorms);

    % c -> retained rank
    LList = max(1, round(cList * Ndim));
    LList = min(LList, Ndim);
    Lmax = max(LList);

    % only compute up to max needed rank
    Vuse = V(:, 1:Lmax);
    Z = X * Vuse;                     % [Nsample, Lmax]
    Ecum = cumsum(abs(Z).^2, 2);      % cumulative captured energy

    capturedEnergyMean = zeros(size(LList), 'like', rowEnergy);
    nmseMean = zeros(size(LList), 'like', rowEnergy);

    for i = 1:numel(LList)
        L = LList(i);
        capturedMean = mean(Ecum(:, L));
        capturedEnergyMean(i) = capturedMean;
        nmseMean(i) = mean(rowEnergy - Ecum(:, L));
    end

    result = struct();
    result.cList = cList;
    result.LList = LList;
    result.capturedEnergyMean = capturedEnergyMean;
    result.nmseMean = nmseMean;
    result.rowNormStats = rowNormStats;

    if verbose
        fprintf('[estimate_nmse_from_basis_energy_light]\n');
        fprintf('  Nsample = %d\n', Nsample);
        fprintf('  Ndim    = %d\n', Ndim);
        fprintf('  row_norm_ok = %d\n', rowNormStats.is_ok);
        fprintf('  max row-norm abs err = %.3e\n', rowNormStats.max_abs_err);
        fprintf('\n');
        fprintf('    c         L      captured_energy      nmse\n');
        fprintf('---------------------------------------------------\n');
        for i = 1:numel(LList)
            fprintf('%8.5f   %4d   %.12e   %.12e\n', ...
                cList(i), LList(i), capturedEnergyMean(i), nmseMean(i));
        end
    end
end