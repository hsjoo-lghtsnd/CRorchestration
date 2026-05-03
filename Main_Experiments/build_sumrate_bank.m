function [Htrue_bank, Hhat_bank, numDrops] = build_sumrate_bank(testS, out_te, V, cList, varargin)
%BUILD_SUMRATE_BANK Precompute true/reconstructed CSI banks for sum-rate eval
%
% Input:
%   testS.Ht    : [Nsample, Nt, Nr, Ntap]
%   testS.Horg  : [Nsample, Nt, Nr, Nsub]
%   out_te      : structure with out_te.splitIdx{k}.curveGlobalIndices
%   V           : basis used by reconstruct_from_basis
%   cList       : admissible CR values
%
% Name-value options:
%   'MaxDrops'  : max number of upcoming samples per user to use
%                 default = min available across users
%   'Verbose'   : true/false, default = true
%
% Output:
%   Htrue_bank  : cell(K, numDrops), each [1, Nt, Nr, Nsub]
%   Hhat_bank   : cell(K, numDrops, M), each [1, Nt, Nr, Nsub]
%   numDrops    : scalar number of drops actually used

    p = inputParser;
    addParameter(p, 'MaxDrops', [], @(x) isempty(x) || (isscalar(x) && x >= 1));
    addParameter(p, 'Verbose', true, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});

    maxDropsOpt = p.Results.MaxDrops;
    verbose = logical(p.Results.Verbose);

    K = numel(out_te.splitIdx);
    M = numel(cList);
    Nsub = size(testS.Horg, 4);

    % Determine common number of evaluation drops
    numDrops = inf;
    for k = 1:K
        numDrops = min(numDrops, numel(out_te.splitIdx{k}.curveGlobalIndices));
    end
    if ~isempty(maxDropsOpt)
        numDrops = min(numDrops, maxDropsOpt);
    end

    if verbose
        fprintf('Building sum-rate bank: K=%d, M=%d, numDrops=%d\n', K, M, numDrops);
    end

    Htrue_bank = cell(K, numDrops);
    Hhat_bank  = cell(K, numDrops, M);

    for k = 1:K
        if verbose
            fprintf('  user %d / %d\n', k, K);
        end

        for d = 1:numDrops
            sampleIdx = out_te.splitIdx{k}.curveGlobalIndices(d);

            % True frequency-domain CSI
            Htrue_bank{k,d} = testS.Horg(sampleIdx,:,:,:);   % [1, Nt, Nr, Nsub]

            % Delay-domain CSI sample
            Ht_sample = testS.Ht(sampleIdx,:,:,:);           % [1, Nt, Nr, Ntap]
            Xsample = reshape(Ht_sample, 1, []);

            for m = 1:M
                % Reconstruct from basis at cList(m)
                Xhat = reconstruct_from_basis(Xsample, V, cList(m));
                Ht_hat = reshape(Xhat, size(Ht_sample));     % [1, Nt, Nr, Ntap]

                % Convert to frequency domain
                Hhat_bank{k,d,m} = delay_to_freq_csi(Ht_hat, Nsub);  % [1, Nt, Nr, Nsub]
            end
        end
    end
end