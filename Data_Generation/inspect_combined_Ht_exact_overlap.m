function report = inspect_combined_Ht_exact_overlap(combinedRoot)
%INSPECT_COMBINED_HT_EXACT_OVERLAP
% 2-stage overlap check on combined files:
%   1) use gainHt exact equality to generate candidate pairs
%   2) confirm with exact Ht equality
%
% Output:
%   report.self.train / valid / test
%   report.cross.train_vs_valid / train_vs_test / valid_vs_test

    if nargin < 1 || isempty(combinedRoot)
        combinedRoot = fullfile('dataset_E2E4_mix', 'combined');
    end

    files.train = fullfile(combinedRoot, 'train_E2E4_mix.mat');
    files.valid = fullfile(combinedRoot, 'valid_E2E4_mix.mat');
    files.test  = fullfile(combinedRoot, 'test_E2E4_mix.mat');

    splitNames = {'train','valid','test'};
    mfs = struct();
    gains = struct();

    for i = 1:numel(splitNames)
        sname = splitNames{i};
        assert(isfile(files.(sname)), 'Missing file: %s', files.(sname));
        mfs.(sname) = matfile(files.(sname));
        g = mfs.(sname).gainHt(:,1);
        gains.(sname) = g(:);
        fprintf('[%s] loaded gainHt (%d samples)\n', sname, numel(gains.(sname)));
    end

    report = struct();
    report.self = struct();
    report.cross = struct();

    fprintf('\n==================================================\n');
    fprintf('SELF exact-Ht check\n');
    fprintf('==================================================\n');

    for i = 1:numel(splitNames)
        sname = splitNames{i};
        report.self.(sname) = self_exact_check(mfs.(sname), gains.(sname), sname);
    end

    fprintf('\n==================================================\n');
    fprintf('CROSS exact-Ht check\n');
    fprintf('==================================================\n');

    pairList = {
        'train', 'valid';
        'train', 'test';
        'valid', 'test';
        };

    for k = 1:size(pairList,1)
        a = pairList{k,1};
        b = pairList{k,2};
        key = sprintf('%s_vs_%s', a, b);
        report.cross.(key) = cross_exact_check(mfs.(a), gains.(a), a, ...
                                               mfs.(b), gains.(b), b);
    end
end


% ============================================================
function rep = self_exact_check(mf, g, splitName)
% Within one split:
%   - find duplicated gain values
%   - among those candidates, test exact Ht equality

    [u, ~, ic] = unique(g, 'stable');
    counts = accumarray(ic, 1);
    dupValMask = counts > 1;
    dupVals = u(dupValMask);

    exactPairs = zeros(0,2,'uint32');
    candidateGroups = 0;
    candidatePairs = 0;

    for iv = 1:numel(dupVals)
        idx = find(g == dupVals(iv));
        if numel(idx) < 2
            continue;
        end

        candidateGroups = candidateGroups + 1;

        % Compare all pairs within the same gain group
        for p = 1:numel(idx)-1
            Hi = mf.Ht(idx(p),:,:,:);
            for q = p+1:numel(idx)
                candidatePairs = candidatePairs + 1;
                Hj = mf.Ht(idx(q),:,:,:);

                if isequaln(Hi, Hj)
                    exactPairs(end+1,:) = uint32([idx(p), idx(q)]); %#ok<AGROW>
                end
            end
        end
    end

    rep = struct();
    rep.N = numel(g);
    rep.nDupGainValues = numel(dupVals);
    rep.nCandidateGroups = candidateGroups;
    rep.nCandidatePairs = candidatePairs;
    rep.exactPairs = exactPairs;
    rep.nExactPairs = size(exactPairs,1);

    fprintf('\n[SELF | %s]\n', splitName);
    fprintf('  duplicated gain values     : %d\n', rep.nDupGainValues);
    fprintf('  candidate groups           : %d\n', rep.nCandidateGroups);
    fprintf('  candidate pairs checked    : %d\n', rep.nCandidatePairs);
    fprintf('  exact Ht-equal pairs       : %d\n', rep.nExactPairs);

    if rep.nExactPairs > 0
        disp('  first exact pairs (up to 10):');
        disp(rep.exactPairs(1:min(10,end),:));
        runs = find_pair_runs_self(rep.exactPairs);
        rep.runs = runs;
        if ~isempty(runs)
            fprintf('  contiguous-pair runs (first up to 10):\n');
            print_pair_runs(runs, 10);
        end
    else
        rep.runs = zeros(0,5);
    end
end


% ============================================================
function rep = cross_exact_check(mfA, gA, nameA, mfB, gB, nameB)
% Across two splits:
%   - intersect duplicated/overlapping gain values
%   - test exact Ht equality for candidate cross-pairs

    overlapVals = intersect(unique(gA,'stable'), unique(gB,'stable'), 'stable');

    exactPairs = zeros(0,2,'uint32');
    candidateGroups = 0;
    candidatePairs = 0;

    for iv = 1:numel(overlapVals)
        idxA = find(gA == overlapVals(iv));
        idxB = find(gB == overlapVals(iv));

        if isempty(idxA) || isempty(idxB)
            continue;
        end

        candidateGroups = candidateGroups + 1;

        for p = 1:numel(idxA)
            Ha = mfA.Ht(idxA(p),:,:,:);
            for q = 1:numel(idxB)
                candidatePairs = candidatePairs + 1;
                Hb = mfB.Ht(idxB(q),:,:,:);

                if isequaln(Ha, Hb)
                    exactPairs(end+1,:) = uint32([idxA(p), idxB(q)]); %#ok<AGROW>
                end
            end
        end
    end

    rep = struct();
    rep.NA = numel(gA);
    rep.NB = numel(gB);
    rep.nOverlapGainValues = numel(overlapVals);
    rep.nCandidateGroups = candidateGroups;
    rep.nCandidatePairs = candidatePairs;
    rep.exactPairs = exactPairs;
    rep.nExactPairs = size(exactPairs,1);

    fprintf('\n[CROSS | %s vs %s]\n', nameA, nameB);
    fprintf('  overlapping gain values    : %d\n', rep.nOverlapGainValues);
    fprintf('  candidate groups           : %d\n', rep.nCandidateGroups);
    fprintf('  candidate pairs checked    : %d\n', rep.nCandidatePairs);
    fprintf('  exact Ht-equal pairs       : %d\n', rep.nExactPairs);

    if rep.nExactPairs > 0
        disp('  first exact pairs (up to 10):');
        disp(rep.exactPairs(1:min(10,end),:));
        runs = find_pair_runs_cross(rep.exactPairs);
        rep.runs = runs;
        if ~isempty(runs)
            fprintf('  contiguous-pair runs (first up to 10):\n');
            print_pair_runs(runs, 10);
        end
    else
        rep.runs = zeros(0,5);
    end
end


% ============================================================
function runs = find_pair_runs_self(pairs)
% Detect run-like structure in self pairs [i,j]
% Returns rows: [start_i, end_i, start_j, end_j, len]

    if isempty(pairs)
        runs = zeros(0,5);
        return;
    end

    pairs = sortrows(double(pairs), [1 2]);
    keep = true(size(pairs,1),1);

    runStart = 1;
    out = [];

    for k = 2:size(pairs,1)
        di = pairs(k,1) - pairs(k-1,1);
        dj = pairs(k,2) - pairs(k-1,2);

        if ~(di == 1 && dj == 1)
            runEnd = k-1;
            out = [out; summarize_pair_run(pairs(runStart:runEnd,:))]; %#ok<AGROW>
            runStart = k;
        end
    end
    out = [out; summarize_pair_run(pairs(runStart:end,:))];

    runs = out;
end


% ============================================================
function runs = find_pair_runs_cross(pairs)
% Same format as self, but for cross pairs [iA, iB]

    if isempty(pairs)
        runs = zeros(0,5);
        return;
    end

    pairs = sortrows(double(pairs), [1 2]);

    runStart = 1;
    out = [];

    for k = 2:size(pairs,1)
        d1 = pairs(k,1) - pairs(k-1,1);
        d2 = pairs(k,2) - pairs(k-1,2);

        if ~(d1 == 1 && d2 == 1)
            runEnd = k-1;
            out = [out; summarize_pair_run(pairs(runStart:runEnd,:))]; %#ok<AGROW>
            runStart = k;
        end
    end
    out = [out; summarize_pair_run(pairs(runStart:end,:))];

    runs = out;
end


% ============================================================
function row = summarize_pair_run(pairBlock)
    row = [pairBlock(1,1), pairBlock(end,1), ...
           pairBlock(1,2), pairBlock(end,2), ...
           size(pairBlock,1)];
end


% ============================================================
function print_pair_runs(runs, maxShow)
    nShow = min(size(runs,1), maxShow);
    for i = 1:nShow
        fprintf('    run %2d: A[%d:%d], B[%d:%d], len=%d\n', ...
            i, runs(i,1), runs(i,2), runs(i,3), runs(i,4), runs(i,5));
    end
    if size(runs,1) > maxShow
        fprintf('    ... (%d more runs omitted)\n', size(runs,1) - maxShow);
    end
end