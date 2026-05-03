function check_dataset_overlap(trainFile, validFile, testFile, opts)
%CHECK_DATASET_OVERLAP
% Check exact duplicates / exact overlaps / near overlaps for saved dataset splits.
%
% Inputs:
%   trainFile, validFile, testFile : .mat file paths
%   opts fields (all optional):
%       .roundDigits      = 10        % rounding digits for exact-match fingerprint
%       .numProbe         = 500       % # random probes for near-overlap
%       .cosThresh        = 0.9999    % near-overlap cosine threshold
%       .useGainInExact   = true      % include gainHt in exact comparison
%       .checkHorg        = false     % also run near-overlap on Horg for test split
%       .rngSeed          = 1         % for reproducible probe sampling
%       .verbose          = true
%
% Example:
%   opts = struct('roundDigits',10,'numProbe',1000,'cosThresh',0.9999,...
%                 'useGainInExact',true,'checkHorg',false);
%   check_dataset_overlap( ...
%       'dataset_E2E4_mix/combined/train_E2E4_mix.mat', ...
%       'dataset_E2E4_mix/combined/valid_E2E4_mix.mat', ...
%       'dataset_E2E4_mix/combined/test_E2E4_mix.mat', opts);

    if nargin < 4 || isempty(opts)
        opts = struct();
    end

    if ~isfield(opts, 'roundDigits') || isempty(opts.roundDigits)
        opts.roundDigits = 10;
    end
    if ~isfield(opts, 'numProbe') || isempty(opts.numProbe)
        opts.numProbe = 500;
    end
    if ~isfield(opts, 'cosThresh') || isempty(opts.cosThresh)
        opts.cosThresh = 0.9999;
    end
    if ~isfield(opts, 'useGainInExact') || isempty(opts.useGainInExact)
        opts.useGainInExact = true;
    end
    if ~isfield(opts, 'checkHorg') || isempty(opts.checkHorg)
        opts.checkHorg = false;
    end
    if ~isfield(opts, 'rngSeed') || isempty(opts.rngSeed)
        opts.rngSeed = 1;
    end
    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = true;
    end

    rng(opts.rngSeed, 'twister');

    fprintf('=== Loading files ===\n');
    train = load(trainFile, 'Ht', 'gainHt');
    valid = load(validFile, 'Ht', 'gainHt');
    test  = load(testFile,  'Ht', 'gainHt');
    if opts.checkHorg
        tmp = load(testFile, 'Horg', 'gainHorg');
        if isfield(tmp, 'Horg')
            test.Horg = tmp.Horg;
            test.gainHorg = tmp.gainHorg;
        else
            warning('checkHorg=true, but Horg not found in test file.');
            opts.checkHorg = false;
        end
    end

    % -----------------------------
    % 1) Internal exact duplicates
    % -----------------------------
    fprintf('\n=== Internal exact duplicate check ===\n');
    report_exact_duplicates(train.Ht, train.gainHt, 'train', opts.roundDigits, opts.useGainInExact);
    report_exact_duplicates(valid.Ht, valid.gainHt, 'valid', opts.roundDigits, opts.useGainInExact);
    report_exact_duplicates(test.Ht,  test.gainHt,  'test',  opts.roundDigits, opts.useGainInExact);

    % -----------------------------
    % 2) Exact overlap across splits
    % -----------------------------
    fprintf('\n=== Exact overlap across splits ===\n');
    report_exact_overlap(train.Ht, train.gainHt, valid.Ht, valid.gainHt, ...
        'train', 'valid', opts.roundDigits, opts.useGainInExact);
    report_exact_overlap(train.Ht, train.gainHt, test.Ht, test.gainHt, ...
        'train', 'test', opts.roundDigits, opts.useGainInExact);
    report_exact_overlap(valid.Ht, valid.gainHt, test.Ht, test.gainHt, ...
        'valid', 'test', opts.roundDigits, opts.useGainInExact);

    % -----------------------------
    % 3) Near overlap across splits (Ht)
    % -----------------------------
    fprintf('\n=== Near-overlap check on normalized Ht ===\n');
    report_near_overlap(train.Ht, valid.Ht, 'train', 'valid', opts.numProbe, opts.cosThresh);
    report_near_overlap(train.Ht, test.Ht,  'train', 'test',  opts.numProbe, opts.cosThresh);
    report_near_overlap(valid.Ht, test.Ht,  'valid', 'test',  opts.numProbe, opts.cosThresh);

    % -----------------------------
    % 4) Optional near overlap on Horg inside test only
    % -----------------------------
    if opts.checkHorg
        fprintf('\n=== Optional internal near-overlap check on test Horg ===\n');
        report_internal_near_duplicates(test.Horg, 'test.Horg', opts.numProbe, opts.cosThresh);
    end

    fprintf('\nDone.\n');
end


% ============================================================
function report_exact_duplicates(H, gain, nameStr, roundDigits, useGain)
    K = build_exact_keys(H, gain, roundDigits, useGain);

    [~, ia, ic] = unique(K, 'rows', 'stable');
    nTotal = size(K, 1);
    nUnique = numel(ia);
    nDup = nTotal - nUnique;

    fprintf('[%s] total=%d, unique=%d, duplicates=%d\n', ...
        nameStr, nTotal, nUnique, nDup);

    if nDup > 0
        counts = accumarray(ic, 1);
        dupGroups = find(counts > 1);
        fprintf('  duplicate groups = %d\n', numel(dupGroups));
        fprintf('  max multiplicity = %d\n', max(counts));
    end
end


% ============================================================
function report_exact_overlap(H1, gain1, H2, gain2, name1, name2, roundDigits, useGain)
    K1 = build_exact_keys(H1, gain1, roundDigits, useGain);
    K2 = build_exact_keys(H2, gain2, roundDigits, useGain);

    [tf, loc] = ismember(K1, K2, 'rows');
    nOverlap = sum(tf);

    fprintf('[%s vs %s] exact overlap = %d / %d\n', ...
        name1, name2, nOverlap, size(K1,1));

    if nOverlap > 0
        fprintf('  first few overlaps:\n');
        idx1 = find(tf);
        nShow = min(5, numel(idx1));
        for i = 1:nShow
            fprintf('    %s(%d) == %s(%d)\n', ...
                name1, idx1(i), name2, loc(idx1(i)));
        end
    end
end


% ============================================================
function K = build_exact_keys(H, gain, roundDigits, useGain)
% Build row-wise keys for exact-ish comparison after rounding.
% H : [N, ...] complex
% gain : [N,1] or [N]
%
% Output:
%   K : [N, D] real matrix for unique/ismember rows

    N = size(H,1);
    X = reshape(H, N, []);
    X = [real(X), imag(X)];

    scale = 10^roundDigits;
    X = round(X * scale) / scale;

    if useGain
        g = reshape(gain, N, 1);
        g = round(g * scale) / scale;
        K = [X, g];
    else
        K = X;
    end
end


% ============================================================
function report_near_overlap(H1, H2, name1, name2, numProbe, cosThresh)
% H1, H2 are assumed already normalized per instance (Fro norm = 1)
% We probe random rows from H1 and find max cosine similarity in H2.

    N1 = size(H1,1);
    N2 = size(H2,1);

    nProbe = min(numProbe, N1);
    idx = randperm(N1, nProbe);

    X1 = reshape(H1(idx,:,:,:), nProbe, []);
    X2 = reshape(H2, N2, []);

    % Convert to double for stable GEMM
    X1 = double(X1);
    X2 = double(X2);

    % Normalize again just in case
    X1 = row_normalize(X1);
    X2 = row_normalize(X2);

    % complex cosine similarity magnitude
    G = abs(X1 * X2.');
    maxSim = max(G, [], 2);

    fprintf('[%s vs %s] probes=%d\n', name1, name2, nProbe);
    fprintf('  max cosine similarity: mean=%.6f, median=%.6f, max=%.6f\n', ...
        mean(maxSim), median(maxSim), max(maxSim));
    fprintf('  # above %.6f = %d\n', cosThresh, sum(maxSim >= cosThresh));
end


% ============================================================
function report_internal_near_duplicates(H, nameStr, numProbe, cosThresh)
% For internal check: probe rows in H against the full H, excluding self-match.

    N = size(H,1);
    nProbe = min(numProbe, N);
    idx = randperm(N, nProbe);

    Xall = reshape(H, N, []);
    Xprobe = Xall(idx, :);

    Xall = double(Xall);
    Xprobe = double(Xprobe);

    Xall = row_normalize(Xall);
    Xprobe = row_normalize(Xprobe);

    G = abs(Xprobe * Xall.');

    % remove self-match
    for i = 1:nProbe
        G(i, idx(i)) = -inf;
    end

    maxSim = max(G, [], 2);

    fprintf('[%s internal] probes=%d\n', nameStr, nProbe);
    fprintf('  nearest-neighbor cosine similarity: mean=%.6f, median=%.6f, max=%.6f\n', ...
        mean(maxSim), median(maxSim), max(maxSim));
    fprintf('  # above %.6f = %d\n', cosThresh, sum(maxSim >= cosThresh));
end


% ============================================================
function X = row_normalize(X)
    nrm = sqrt(sum(abs(X).^2, 2));
    nrm(nrm == 0) = 1;
    X = X ./ nrm;
end