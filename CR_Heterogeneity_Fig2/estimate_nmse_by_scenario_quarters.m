function out = estimate_nmse_by_scenario_quarters(X, V, cList, scenarioNames, tol, verbose)
%ESTIMATE_NMSE_BY_SCENARIO_QUARTERS
% Split X into 4 equal contiguous scenario blocks and estimate NMSE per scenario.
%
% Inputs:
%   X             : [Nsample, Ndim]
%   V             : [Ndim, Ndim]
%   cList         : compression ratio list
%   scenarioNames : 1x4 cell, e.g. {'S1','S2','S3','S4'}
%   tol           : row norm tolerance
%   verbose       : true/false
%
% Output:
%   out.global
%   out.scenarios(i)
%       .name
%       .index_range
%       .result   % result from estimate_nmse_from_basis_energy_light

    if nargin < 4 || isempty(scenarioNames)
        scenarioNames = {'scenario1','scenario2','scenario3','scenario4'};
    end
    if nargin < 5 || isempty(tol)
        tol = 1e-5;
    end
    if nargin < 6
        verbose = true;
    end

    Nsample = size(X, 1);
    assert(mod(Nsample, 4) == 0, 'Nsample must be divisible by 4.');
    assert(numel(scenarioNames) == 4, 'scenarioNames must have length 4.');

    blockSize = Nsample / 4;

    out = struct();
    out.global = estimate_nmse_from_basis_energy_light(X, V, cList, tol, false);
    out.scenarios = repmat(struct(), 1, 4);

    for s = 1:4
        idxStart = (s-1)*blockSize + 1;
        idxEnd   = s*blockSize;
        idx = idxStart:idxEnd;

        Xs = X(idx, :);
        res = estimate_nmse_from_basis_energy_light(Xs, V, cList, tol, false);

        out.scenarios(s).name = scenarioNames{s};
        out.scenarios(s).index_range = [idxStart, idxEnd];
        out.scenarios(s).result = res;
    end

    nmseMatrix = zeros(4, numel(cList));
    for s = 1:4
        nmseMatrix(s, :) = out.scenarios(s).result.nmseMean;
    end

    out.nmseMatrix = nmseMatrix;
    out.cList = out.global.cList;
    out.LList = out.global.LList;
    out.scenarioNames = scenarioNames;

    if verbose
        fprintf('[estimate_nmse_by_scenario_quarters]\n');
        fprintf('  Nsample = %d\n', Nsample);
        fprintf('  blockSize = %d\n\n', blockSize);

        fprintf('  Global:\n');
        for i = 1:numel(out.global.cList)
            fprintf('    c = %.5f, L = %d, NMSE = %.12e\n', ...
                out.global.cList(i), out.global.LList(i), out.global.nmseMean(i));
        end

        fprintf('\n  Per scenario:\n');
        for s = 1:4
            fprintf('  [%d] %s (rows %d:%d)\n', ...
                s, out.scenarios(s).name, ...
                out.scenarios(s).index_range(1), out.scenarios(s).index_range(2));

            r = out.scenarios(s).result;
            for i = 1:numel(r.cList)
                fprintf('      c = %.5f, L = %d, NMSE = %.12e\n', ...
                    r.cList(i), r.LList(i), r.nmseMean(i));
            end
        end
    end
end