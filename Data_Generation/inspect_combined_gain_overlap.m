function report = inspect_combined_gain_overlap(combinedRoot)
%INSPECT_COMBINED_GAIN_OVERLAP
% Check self-overlap and cross-overlap among combined train/valid/test files
% using gainHt sequence only.
%
% Assumptions:
%   combinedRoot/
%       train_E2E4_mix.mat
%       valid_E2E4_mix.mat
%       test_E2E4_mix.mat
%
% Main outputs:
%   - self-overlap stats within each split
%   - cross-overlap stats across split pairs
%   - repeated / overlapping index runs to see whether duplication occurs
%     as contiguous blocks
%
% Notes:
%   - This is a suspicious-pattern detector based on gainHt only.
%   - If two different CSI samples happen to have the same gain, this code
%     may mark them as overlapping candidates.
%   - Exact-equality is used by default because duplicated samples caused by
%     seed / append issues are likely to preserve identical single-precision
%     gain values.

    if nargin < 1 || isempty(combinedRoot)
        combinedRoot = fullfile('dataset_E2E4_mix', 'combined');
    end

    files.train = fullfile(combinedRoot, 'train_E2E4_mix.mat');
    files.valid = fullfile(combinedRoot, 'valid_E2E4_mix.mat');
    files.test  = fullfile(combinedRoot, 'test_E2E4_mix.mat');

    splitNames = {'train','valid','test'};

    % ------------------------------------------------------------
    % 1) Load gains
    % ------------------------------------------------------------
    gains = struct();
    Ns = struct();

    for i = 1:numel(splitNames)
        sname = splitNames{i};
        assert(isfile(files.(sname)), 'Missing file: %s', files.(sname));

        mf = matfile(files.(sname));
        g = mf.gainHt(:,1);
        g = g(:);

        gains.(sname) = g;
        Ns.(sname) = numel(g);

        fprintf('[%s] loaded %d gain values from %s\n', sname, Ns.(sname), files.(sname));
    end

    report = struct();
    report.self = struct();
    report.cross = struct();

    fprintf('\n==================================================\n');
    fprintf('SELF-OVERLAP CHECK\n');
    fprintf('==================================================\n');

    % ------------------------------------------------------------
    % 2) Self-overlap check within each split
    % ------------------------------------------------------------
    for i = 1:numel(splitNames)
        sname = splitNames{i};
        g = gains.(sname);

        selfRep = analyze_self_overlap(g, sname);
        report.self.(sname) = selfRep;
    end

    fprintf('\n==================================================\n');
    fprintf('CROSS-OVERLAP CHECK\n');
    fprintf('==================================================\n');

    % ------------------------------------------------------------
    % 3) Cross-overlap across splits
    % ------------------------------------------------------------
    pairList = {
        'train', 'valid';
        'train', 'test';
        'valid', 'test';
        };

    for k = 1:size(pairList,1)
        a = pairList{k,1};
        b = pairList{k,2};

        crossRep = analyze_cross_overlap(gains.(a), gains.(b), a, b);
        key = sprintf('%s_vs_%s', a, b);
        report.cross.(key) = crossRep;
    end

    fprintf('\n==================================================\n');
    fprintf('SUMMARY\n');
    fprintf('==================================================\n');

    for i = 1:numel(splitNames)
        sname = splitNames{i};
        r = report.self.(sname);

        fprintf('[SELF | %s] N=%d, unique=%d, duplicate_entries=%d, duplicate_ratio=%.4f%%\n', ...
            sname, r.N, r.nUnique, r.nDuplicateEntries, 100*r.duplicateEntryRatio);
    end

    for k = 1:size(pairList,1)
        a = pairList{k,1};
        b = pairList{k,2};
        key = sprintf('%s_vs_%s', a, b);
        r = report.cross.(key);

        fprintf('[CROSS | %s-%s] overlap_unique=%d, overlap_ratio_vs_%s=%.4f%%, overlap_ratio_vs_%s=%.4f%%\n', ...
            a, b, r.nOverlapUnique, a, 100*r.overlapRatioA, b, 100*r.overlapRatioB);
    end
end


% ============================================================
function rep = analyze_self_overlap(g, splitName)
% Analyze repeated gain values within one split.

    N = numel(g);
    [u, ~, ic] = unique(g, 'stable');
    counts = accumarray(ic, 1);

    dupMaskUnique = counts > 1;
    dupValues = u(dupMaskUnique);
    nUnique = numel(u);
    nDupUnique = sum(dupMaskUnique);

    if isempty(dupValues)
        repeatedPos = [];
    else
        repeatedPos = find(ismember(g, dupValues));
    end

    nDuplicateEntries = numel(repeatedPos);
    duplicateEntryRatio = nDuplicateEntries / N;

    runs = find_contiguous_runs(repeatedPos);

    rep = struct();
    rep.N = N;
    rep.nUnique = nUnique;
    rep.nDuplicateUnique = nDupUnique;
    rep.nDuplicateEntries = nDuplicateEntries;
    rep.duplicateEntryRatio = duplicateEntryRatio;
    rep.dupValues = dupValues;
    rep.repeatedPositions = repeatedPos;
    rep.runs = runs;

    fprintf('\n[SELF | %s]\n', splitName);
    fprintf('  total N                  : %d\n', N);
    fprintf('  unique(gain)             : %d\n', nUnique);
    fprintf('  repeated unique gains    : %d\n', nDupUnique);
    fprintf('  repeated entry count     : %d\n', nDuplicateEntries);
    fprintf('  repeated entry ratio     : %.4f%%\n', 100*duplicateEntryRatio);

    if ~isempty(runs)
        fprintf('  contiguous repeated-position runs (first up to 10):\n');
        print_runs(runs, 10);
    else
        fprintf('  no contiguous repeated-position runs found.\n');
    end
end


% ============================================================
function rep = analyze_cross_overlap(gA, gB, nameA, nameB)
% Analyze overlap of gain values across two splits.

    [uA, ~, icA] = unique(gA, 'stable');
    [uB, ~, icB] = unique(gB, 'stable');

    overlapVals = intersect(uA, uB, 'stable');

    posA = find(ismember(gA, overlapVals));
    posB = find(ismember(gB, overlapVals));

    runsA = find_contiguous_runs(posA);
    runsB = find_contiguous_runs(posB);

    rep = struct();
    rep.NA = numel(gA);
    rep.NB = numel(gB);
    rep.nUniqueA = numel(uA);
    rep.nUniqueB = numel(uB);
    rep.nOverlapUnique = numel(overlapVals);
    rep.overlapRatioA = numel(overlapVals) / numel(uA);
    rep.overlapRatioB = numel(overlapVals) / numel(uB);
    rep.overlapValues = overlapVals;
    rep.positionsA = posA;
    rep.positionsB = posB;
    rep.runsA = runsA;
    rep.runsB = runsB;

    % Additional "entry-wise" overlap count by repeated membership
    rep.nEntryOverlapA = numel(posA);
    rep.nEntryOverlapB = numel(posB);
    rep.entryOverlapRatioA = numel(posA) / numel(gA);
    rep.entryOverlapRatioB = numel(posB) / numel(gB);

    % Count frequency of each overlap value in each split
    cntA = accumarray(icA, 1);
    cntB = accumarray(icB, 1);

    [tfA, locA] = ismember(overlapVals, uA);
    [tfB, locB] = ismember(overlapVals, uB);
    assert(all(tfA) && all(tfB), 'Internal overlap mapping error.');

    rep.countsOverlapA = cntA(locA);
    rep.countsOverlapB = cntB(locB);

    fprintf('\n[CROSS | %s vs %s]\n', nameA, nameB);
    fprintf('  unique(%s)              : %d\n', nameA, numel(uA));
    fprintf('  unique(%s)              : %d\n', nameB, numel(uB));
    fprintf('  overlap unique gains    : %d\n', rep.nOverlapUnique);
    fprintf('  overlap ratio vs %s     : %.4f%%\n', nameA, 100*rep.overlapRatioA);
    fprintf('  overlap ratio vs %s     : %.4f%%\n', nameB, 100*rep.overlapRatioB);
    fprintf('  entry overlap ratio %s  : %.4f%%\n', nameA, 100*rep.entryOverlapRatioA);
    fprintf('  entry overlap ratio %s  : %.4f%%\n', nameB, 100*rep.entryOverlapRatioB);

    if ~isempty(runsA)
        fprintf('  %s overlap-position runs (first up to 10):\n', nameA);
        print_runs(runsA, 10);
    else
        fprintf('  no overlap-position runs in %s.\n', nameA);
    end

    if ~isempty(runsB)
        fprintf('  %s overlap-position runs (first up to 10):\n', nameB);
        print_runs(runsB, 10);
    else
        fprintf('  no overlap-position runs in %s.\n', nameB);
    end
end


% ============================================================
function runs = find_contiguous_runs(idx)
% Return contiguous runs [start, end, length] for sorted positions.

    idx = idx(:);
    if isempty(idx)
        runs = zeros(0,3);
        return;
    end

    idx = sort(idx);
    d = diff(idx);

    runStartPos = [1; find(d > 1) + 1];
    runEndPos   = [find(d > 1); numel(idx)];

    runStarts = idx(runStartPos);
    runEnds   = idx(runEndPos);
    runLens   = runEnds - runStarts + 1;

    runs = [runStarts, runEnds, runLens];
end


% ============================================================
function print_runs(runs, maxShow)
    nShow = min(size(runs,1), maxShow);
    for i = 1:nShow
        fprintf('    run %2d: [%d, %d] (len=%d)\n', ...
            i, runs(i,1), runs(i,2), runs(i,3));
    end
    if size(runs,1) > maxShow
        fprintf('    ... (%d more runs omitted)\n', size(runs,1) - maxShow);
    end
end