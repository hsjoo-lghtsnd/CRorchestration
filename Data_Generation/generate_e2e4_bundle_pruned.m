function generate_e2e4_bundle_pruned(saveRoot)
%GENERATE_E2E4_BUNDLE_PRUNED
% Build dataset using 4 scenarios:
%   E2 scenario 2 = CDL-B
%   E2 scenario 5 = CDL-E
%   E4 scenario 2 = IndoorHall_5GHz
%   E4 scenario 3 = SemiUrban_CloselySpacedUser_2_6GHz
%
% New behavior:
% 1) Generate per-scenario train/valid/test files
% 2) For each scenario, inspect exact Ht overlap across train/valid/test
% 3) Drop-and-refill overlapped samples until clean (or max rounds)
% 4) Rebuild combined files
% 5) Run final combined overlap check
%
% Notes:
% - "Drop" is implemented as overwrite of selected rows with freshly
%   generated samples.
% - Refill is done row-by-row for robustness.
% - Priority rule in duplicate clusters:
%       keep train > valid > test
%       and keep the smallest index within the same split
%
% Dependencies:
% - load_environment(...)
% - inspect_combined_Ht_exact_overlap(...) is NOT required here
%
% Output folders:
%   saveRoot/per_scenario
%   saveRoot/combined

    if nargin < 1 || isempty(saveRoot)
        saveRoot = 'dataset_E2E4_mix';
    end

    % ------------------------------------------------------------
    % Scenario specification
    % ------------------------------------------------------------
    scenarioSpec = { ...
        struct('E', 2, 'scenario_choice', 2, 'tag', 'E2_CDL-B'), ...
        struct('E', 2, 'scenario_choice', 5, 'tag', 'E2_CDL-E'), ...
        struct('E', 4, 'scenario_choice', 2, 'tag', 'E4_IndoorHall_5GHz'), ...
        struct('E', 4, 'scenario_choice', 3, 'tag', 'E4_SemiUrbanCSU_2p6') ...
    };

    nScenario = numel(scenarioSpec);

    % per-scenario counts
    Nper.train = 30000;
    Nper.valid = 10000;
    Nper.test  = 5000;

    % total counts
    Ntot.train = nScenario * Nper.train;
    Ntot.valid = nScenario * Nper.valid;
    Ntot.test  = nScenario * Nper.test;

    % fixed split seeds
    splitSeed.train = 12345;
    splitSeed.valid = 78901;
    splitSeed.test  = 219876;

    % chunk sizes for simulation
    chunkSize.train = 5000;
    chunkSize.valid = 5000;
    chunkSize.test  = 1000;

    % chunk sizes for combining
    combineChunk.train = 5000;
    combineChunk.valid = 5000;
    combineChunk.test  = 1000;

    useSingle = true;

    % pruning / refill controls
    pruneOpts.maxRoundsPerScenario = 8;
    pruneOpts.maxRetryPerSample = 200;
    pruneOpts.refillSeedBase = 900000000;

    if ~exist(saveRoot, 'dir')
        mkdir(saveRoot);
    end
    perScenarioRoot = fullfile(saveRoot, 'per_scenario');
    combinedRoot = fullfile(saveRoot, 'combined');
    if ~exist(perScenarioRoot, 'dir'); mkdir(perScenarioRoot); end
    if ~exist(combinedRoot, 'dir'); mkdir(combinedRoot); end

    % Base dimension assumption for saved target
    commonOpts = struct();
    commonOpts.Nt = 32;
    commonOpts.Nr = 1;
    commonOpts.Ntap = 32;
    commonOpts.Nsub = 624;

    % ------------------------------------------------------------
    % 1) Ensure per-scenario files exist
    % ------------------------------------------------------------
    splitNames = {'train','valid','test'};

    for iSc = 1:nScenario
        spec = scenarioSpec{iSc};
        E = spec.E;
        scenario_choice = spec.scenario_choice;
        scenario_tag = spec.tag;

        fprintf('\n=====================================\n');
        fprintf('Scenario %d/%d: %s\n', iSc, nScenario, scenario_tag);
        fprintf('=====================================\n');

        E_opts_base = build_env_opts(E, scenario_choice);

        for iSplit = 1:numel(splitNames)
            splitName = splitNames{iSplit};
            Nthis = Nper.(splitName);
            chunk = chunkSize.(splitName);

            if strcmp(splitName, 'test')
                returnHorg = true;
                splitBaseSeed = splitSeed.test;
            elseif strcmp(splitName, 'valid')
                returnHorg = false;
                splitBaseSeed = splitSeed.valid;
            else
                returnHorg = false;
                splitBaseSeed = splitSeed.train;
            end

            scenarioFile = fullfile(perScenarioRoot, sprintf('%s_%s.mat', splitName, scenario_tag));

            if is_complete_split_file(scenarioFile, splitName, Nthis)
                fprintf('[%s | %s] Existing complete file found. Skip simulation.\n', splitName, scenario_tag);
                continue;
            end

            fprintf('[%s | %s] Generating missing/incomplete file: %s\n', splitName, scenario_tag, scenarioFile);

            if strcmp(splitName, 'test')
                initialize_test_file(scenarioFile, Nthis, E_opts_base, useSingle, true);
            else
                initialize_trainvalid_file(scenarioFile, Nthis, E_opts_base, useSingle, true);
            end
            write_split_meta(scenarioFile, spec, splitBaseSeed, Nthis, splitName, E_opts_base);

            localOffset = 0;
            nChunk = ceil(Nthis / chunk);

            for ic = 1:nChunk
                nNow = min(chunk, Nthis - localOffset);
                if nNow <= 0
                    break;
                end

                chunkSeed = splitBaseSeed + 100000 * iSc + ic;

                fprintf('[%s | %s] chunk %d/%d, n=%d, seed=%d\n', ...
                    splitName, scenario_tag, ic, nChunk, nNow, chunkSeed);

                E_opts_now = E_opts_base;
                E_opts_now.seed = chunkSeed;
                E_opts_now.returnHorg = returnHorg;

                if returnHorg
                    [Ht_raw, Horg_raw] = load_environment(E, nNow, scenario_choice, E_opts_now);
                    [Ht_norm, gainHt] = normalize_instances_fro(Ht_raw, useSingle);
                    [Horg_norm, gainHorg] = normalize_instances_fro(Horg_raw, useSingle);

                    write_test_chunk(scenarioFile, localOffset + 1, Ht_norm, gainHt, Horg_norm, gainHorg);
                    clear Ht_raw Horg_raw Ht_norm Horg_norm gainHt gainHorg;
                else
                    [Ht_raw, ~] = load_environment(E, nNow, scenario_choice, E_opts_now);
                    [Ht_norm, gainHt] = normalize_instances_fro(Ht_raw, useSingle);

                    write_trainvalid_chunk(scenarioFile, localOffset + 1, Ht_norm, gainHt);
                    clear Ht_raw Ht_norm gainHt;
                end

                localOffset = localOffset + nNow;
            end

            assert(localOffset == Nthis, 'Local split size mismatch: %s %s', splitName, scenario_tag);
        end
    end

    % ------------------------------------------------------------
    % 2) Per-scenario prune + refill
    % ------------------------------------------------------------
    for iSc = 1:nScenario
        spec = scenarioSpec{iSc};
        fprintf('\n=====================================\n');
        fprintf('Prune/refill scenario %d/%d: %s\n', iSc, nScenario, spec.tag);
        fprintf('=====================================\n');

        prune_per_scenario_overlap(saveRoot, spec, Nper, splitSeed, pruneOpts, useSingle);
    end

    % ------------------------------------------------------------
    % 3) Rebuild combined files
    % ------------------------------------------------------------
    combinedTrainFile = fullfile(combinedRoot, 'train_E2E4_mix.mat');
    combinedValidFile = fullfile(combinedRoot, 'valid_E2E4_mix.mat');
    combinedTestFile  = fullfile(combinedRoot, 'test_E2E4_mix.mat');

    fprintf('\nRebuilding combined files from per-scenario files...\n');

    initialize_trainvalid_file(combinedTrainFile, Ntot.train, commonOpts, useSingle, true);
    initialize_trainvalid_file(combinedValidFile, Ntot.valid, commonOpts, useSingle, true);
    initialize_test_file(combinedTestFile, Ntot.test, commonOpts, useSingle, true);

    write_combined_meta(combinedTrainFile, scenarioSpec, splitSeed.train, Ntot.train, 'train', commonOpts);
    write_combined_meta(combinedValidFile, scenarioSpec, splitSeed.valid, Ntot.valid, 'valid', commonOpts);
    write_combined_meta(combinedTestFile,  scenarioSpec, splitSeed.test,  Ntot.test,  'test',  commonOpts);

    combinedOffset.train = 0;
    combinedOffset.valid = 0;
    combinedOffset.test  = 0;

    for iSc = 1:nScenario
        spec = scenarioSpec{iSc};
        scenario_tag = spec.tag;

        srcTrain = fullfile(perScenarioRoot, sprintf('train_%s.mat', scenario_tag));
        assert(is_complete_split_file(srcTrain, 'train', Nper.train), 'Missing/incomplete source: %s', srcTrain);
        combinedOffset.train = append_split_into_combined( ...
            srcTrain, combinedTrainFile, 'train', combinedOffset.train, combineChunk.train);

        srcValid = fullfile(perScenarioRoot, sprintf('valid_%s.mat', scenario_tag));
        assert(is_complete_split_file(srcValid, 'valid', Nper.valid), 'Missing/incomplete source: %s', srcValid);
        combinedOffset.valid = append_split_into_combined( ...
            srcValid, combinedValidFile, 'valid', combinedOffset.valid, combineChunk.valid);

        srcTest = fullfile(perScenarioRoot, sprintf('test_%s.mat', scenario_tag));
        assert(is_complete_split_file(srcTest, 'test', Nper.test), 'Missing/incomplete source: %s', srcTest);
        combinedOffset.test = append_split_into_combined( ...
            srcTest, combinedTestFile, 'test', combinedOffset.test, combineChunk.test);
    end

    assert(combinedOffset.train == Ntot.train, 'Combined train size mismatch.');
    assert(combinedOffset.valid == Ntot.valid, 'Combined valid size mismatch.');
    assert(combinedOffset.test  == Ntot.test,  'Combined test size mismatch.');

    % ------------------------------------------------------------
    % 4) Final combined exact-overlap check
    % ------------------------------------------------------------
    fprintf('\n=====================================\n');
    fprintf('Final combined overlap check\n');
    fprintf('=====================================\n');

    combinedReport = inspect_combined_Ht_exact_overlap_local(combinedRoot);
    totalExact = combinedReport.totalExactPairs;

    if totalExact == 0
        fprintf('Final combined check passed: no exact duplicates detected.\n');
    else
        warning('Final combined check still found %d exact pairs.', totalExact);
    end

    fprintf('\nAll done.\n');
    fprintf('Combined outputs:\n');
    fprintf('  %s\n', combinedTrainFile);
    fprintf('  %s\n', combinedValidFile);
    fprintf('  %s\n', combinedTestFile);
end


% ============================================================
function prune_per_scenario_overlap(saveRoot, spec, Nper, splitSeed, pruneOpts, useSingle)

    splitNames = {'train','valid','test'};
    perScenarioRoot = fullfile(saveRoot, 'per_scenario');

    for roundIdx = 1:pruneOpts.maxRoundsPerScenario
        fprintf('\n[prune] scenario=%s, round=%d/%d\n', ...
            spec.tag, roundIdx, pruneOpts.maxRoundsPerScenario);

        rep = inspect_per_scenario_Ht_exact_overlap_local(saveRoot, spec.tag);
        totalExact = rep.totalExactPairs;

        if totalExact == 0
            fprintf('[prune] no exact duplicates remain for %s\n', spec.tag);
            return;
        end

        nodes = collect_nodes_from_report(rep);
        edges = collect_edges_from_report(rep);
        clusters = build_duplicate_clusters(nodes, edges);

        replaceTargets = choose_replacement_targets(clusters);

        if isempty(replaceTargets)
            fprintf('[prune] no replace targets found. stop.\n');
            return;
        end

        fprintf('[prune] replace targets = %d rows\n', numel(replaceTargets));

        % group by split
        for iSplit = 1:numel(splitNames)
            splitName = splitNames{iSplit};
            idxThis = find(strcmp({replaceTargets.split}, splitName));
            if isempty(idxThis)
                continue;
            end

            localRows = [replaceTargets(idxThis).idx];
            localRows = unique(localRows(:).', 'stable');

            if strcmp(splitName, 'test')
                returnHorg = true;
                splitBaseSeed = splitSeed.test;
            elseif strcmp(splitName, 'valid')
                returnHorg = false;
                splitBaseSeed = splitSeed.valid;
            else
                returnHorg = false;
                splitBaseSeed = splitSeed.train;
            end

            fileThis = fullfile(perScenarioRoot, sprintf('%s_%s.mat', splitName, spec.tag));
            assert(is_complete_split_file(fileThis, splitName, Nper.(splitName)), ...
                'Incomplete file before refill: %s', fileThis);

            fprintf('[prune] split=%s, replacing %d rows\n', splitName, numel(localRows));

            for k = 1:numel(localRows)
                rowIdx = localRows(k);

                [Ht_new, gainHt_new, Horg_new, gainHorg_new] = ...
                    generate_unique_row_for_split(saveRoot, spec, splitName, rowIdx, ...
                    returnHorg, splitBaseSeed, roundIdx, k, pruneOpts, useSingle);

                mf = matfile(fileThis, 'Writable', true);
                mf.Ht(rowIdx,:,:,:) = Ht_new;
                mf.gainHt(rowIdx,:) = reshape(gainHt_new, 1, 1);

                if strcmp(splitName, 'test')
                    mf.Horg(rowIdx,:,:,:) = Horg_new;
                    mf.gainHorg(rowIdx,:) = reshape(gainHorg_new, 1, 1);
                end
            end
        end
    end

    warning('Scenario %s reached max prune rounds without full cleanup.', spec.tag);
end


% ============================================================
function [Ht_new, gainHt_new, Horg_new, gainHorg_new] = ...
    generate_unique_row_for_split(saveRoot, spec, splitName, rowIdx, ...
    returnHorg, splitBaseSeed, roundIdx, localCounter, pruneOpts, useSingle)

    Horg_new = [];
    gainHorg_new = [];

    perScenarioRoot = fullfile(saveRoot, 'per_scenario');
    splitNames = {'train','valid','test'};

    E_opts_base = build_env_opts(spec.E, spec.scenario_choice);

    for retry = 1:pruneOpts.maxRetryPerSample
        seedNow = pruneOpts.refillSeedBase + ...
                  10000000 * roundIdx + ...
                  100000 * split_priority(splitName) + ...
                  1000 * localCounter + retry + splitBaseSeed;

        E_opts_now = E_opts_base;
        E_opts_now.seed = seedNow;
        E_opts_now.returnHorg = returnHorg;

        if returnHorg
            [Ht_raw, Horg_raw] = load_environment(spec.E, 1, spec.scenario_choice, E_opts_now);
            [Ht_new, gainHt_new] = normalize_instances_fro(Ht_raw, useSingle);
            [Horg_new, gainHorg_new] = normalize_instances_fro(Horg_raw, useSingle);
        else
            [Ht_raw, ~] = load_environment(spec.E, 1, spec.scenario_choice, E_opts_now);
            [Ht_new, gainHt_new] = normalize_instances_fro(Ht_raw, useSingle);
        end

        % check uniqueness against all train/valid/test rows of this scenario
        isDup = false;
        for iSplit = 1:numel(splitNames)
            sname = splitNames{iSplit};
            fileCheck = fullfile(perScenarioRoot, sprintf('%s_%s.mat', sname, spec.tag));
            mf = matfile(fileCheck);

            N = size(mf, 'Ht', 1);
            for ii = 1:N
                if strcmp(sname, splitName) && ii == rowIdx
                    continue;
                end
                if isequaln(mf.Ht(ii,:,:,:), Ht_new)
                    isDup = true;
                    break;
                end
            end
            if isDup
                break;
            end
        end

        if ~isDup
            return;
        end
    end

    error('Failed to generate a unique replacement for %s | %s | row=%d', ...
        spec.tag, splitName, rowIdx);
end


% ============================================================
function p = split_priority(splitName)
    switch splitName
        case 'train'
            p = 1;
        case 'valid'
            p = 2;
        case 'test'
            p = 3;
        otherwise
            error('Unknown split: %s', splitName);
    end
end


% ============================================================
function nodes = collect_nodes_from_report(rep)
    keys = {'train','valid','test'};
    nodeMap = containers.Map();
    nodes = struct('split', {}, 'idx', {}, 'key', {});

    % self
    for i = 1:numel(keys)
        s = keys{i};
        P = rep.self.(s).exactPairs;
        for r = 1:size(P,1)
            k1 = sprintf('%s:%d', s, P(r,1));
            k2 = sprintf('%s:%d', s, P(r,2));
            if ~isKey(nodeMap, k1)
                nodeMap(k1) = 1;
                nodes(end+1) = struct('split', s, 'idx', double(P(r,1)), 'key', k1); %#ok<AGROW>
            end
            if ~isKey(nodeMap, k2)
                nodeMap(k2) = 1;
                nodes(end+1) = struct('split', s, 'idx', double(P(r,2)), 'key', k2); %#ok<AGROW>
            end
        end
    end

    % cross
    crossKeys = fieldnames(rep.cross);
    for i = 1:numel(crossKeys)
        name = crossKeys{i};
        parts = strsplit(name, '_vs_');
        sA = parts{1};
        sB = parts{2};
        P = rep.cross.(name).exactPairs;
        for r = 1:size(P,1)
            k1 = sprintf('%s:%d', sA, P(r,1));
            k2 = sprintf('%s:%d', sB, P(r,2));
            if ~isKey(nodeMap, k1)
                nodeMap(k1) = 1;
                nodes(end+1) = struct('split', sA, 'idx', double(P(r,1)), 'key', k1); %#ok<AGROW>
            end
            if ~isKey(nodeMap, k2)
                nodeMap(k2) = 1;
                nodes(end+1) = struct('split', sB, 'idx', double(P(r,2)), 'key', k2); %#ok<AGROW>
            end
        end
    end
end


% ============================================================
function edges = collect_edges_from_report(rep)
    nodes = collect_nodes_from_report(rep);
    key2id = containers.Map();
    for i = 1:numel(nodes)
        key2id(nodes(i).key) = i;
    end

    edges = zeros(0,2);

    keys = {'train','valid','test'};
    for i = 1:numel(keys)
        s = keys{i};
        P = rep.self.(s).exactPairs;
        for r = 1:size(P,1)
            k1 = sprintf('%s:%d', s, P(r,1));
            k2 = sprintf('%s:%d', s, P(r,2));
            edges(end+1,:) = [key2id(k1), key2id(k2)]; %#ok<AGROW>
        end
    end

    crossKeys = fieldnames(rep.cross);
    for i = 1:numel(crossKeys)
        name = crossKeys{i};
        parts = strsplit(name, '_vs_');
        sA = parts{1};
        sB = parts{2};
        P = rep.cross.(name).exactPairs;
        for r = 1:size(P,1)
            k1 = sprintf('%s:%d', sA, P(r,1));
            k2 = sprintf('%s:%d', sB, P(r,2));
            edges(end+1,:) = [key2id(k1), key2id(k2)]; %#ok<AGROW>
        end
    end
end


% ============================================================
function clusters = build_duplicate_clusters(nodes, edges)
    if isempty(nodes)
        clusters = {};
        return;
    end

    if isempty(edges)
        clusters = cell(numel(nodes),1);
        for i = 1:numel(nodes)
            clusters{i} = nodes(i);
        end
        return;
    end

    G = graph(edges(:,1), edges(:,2));
    compId = conncomp(G);

    clusters = cell(max(compId),1);
    for c = 1:max(compId)
        idx = find(compId == c);
        clusters{c} = nodes(idx);
    end
end


% ============================================================
function replaceTargets = choose_replacement_targets(clusters)
    replaceTargets = struct('split', {}, 'idx', {}, 'key', {});

    for c = 1:numel(clusters)
        members = clusters{c};
        if numel(members) <= 1
            continue;
        end

        score = zeros(numel(members),1);
        for k = 1:numel(members)
            score(k) = 1000000 * split_priority(members(k).split) + members(k).idx;
        end

        [~, keepPos] = min(score);

        for k = 1:numel(members)
            if k == keepPos
                continue;
            end
            replaceTargets(end+1) = members(k); %#ok<AGROW>
        end
    end
end


% ============================================================
function rep = inspect_per_scenario_Ht_exact_overlap_local(saveRoot, scenarioTag)

    perScenarioRoot = fullfile(saveRoot, 'per_scenario');

    files.train = fullfile(perScenarioRoot, sprintf('train_%s.mat', scenarioTag));
    files.valid = fullfile(perScenarioRoot, sprintf('valid_%s.mat', scenarioTag));
    files.test  = fullfile(perScenarioRoot, sprintf('test_%s.mat',  scenarioTag));

    splitNames = {'train','valid','test'};
    mfs = struct();
    gains = struct();

    for i = 1:numel(splitNames)
        sname = splitNames{i};
        mfs.(sname) = matfile(files.(sname));
        gains.(sname) = mfs.(sname).gainHt(:,1);
        gains.(sname) = gains.(sname)(:);
    end

    rep = struct();
    rep.self = struct();
    rep.cross = struct();

    for i = 1:numel(splitNames)
        sname = splitNames{i};
        rep.self.(sname) = self_exact_check_local(mfs.(sname), gains.(sname));
    end

    pairList = {
        'train', 'valid';
        'train', 'test';
        'valid', 'test';
        };

    totalExact = 0;
    for k = 1:size(pairList,1)
        a = pairList{k,1};
        b = pairList{k,2};
        key = sprintf('%s_vs_%s', a, b);
        rep.cross.(key) = cross_exact_check_local(mfs.(a), gains.(a), mfs.(b), gains.(b));
        totalExact = totalExact + rep.cross.(key).nExactPairs;
    end

    totalExact = totalExact + rep.self.train.nExactPairs + rep.self.valid.nExactPairs + rep.self.test.nExactPairs;
    rep.totalExactPairs = totalExact;
end


% ============================================================
function rep = inspect_combined_Ht_exact_overlap_local(combinedRoot)

    files.train = fullfile(combinedRoot, 'train_E2E4_mix.mat');
    files.valid = fullfile(combinedRoot, 'valid_E2E4_mix.mat');
    files.test  = fullfile(combinedRoot, 'test_E2E4_mix.mat');

    splitNames = {'train','valid','test'};
    mfs = struct();
    gains = struct();

    for i = 1:numel(splitNames)
        sname = splitNames{i};
        mfs.(sname) = matfile(files.(sname));
        gains.(sname) = mfs.(sname).gainHt(:,1);
        gains.(sname) = gains.(sname)(:);
    end

    rep = struct();
    rep.self = struct();
    rep.cross = struct();

    for i = 1:numel(splitNames)
        sname = splitNames{i};
        rep.self.(sname) = self_exact_check_local(mfs.(sname), gains.(sname));
    end

    pairList = {
        'train', 'valid';
        'train', 'test';
        'valid', 'test';
        };

    totalExact = 0;
    for k = 1:size(pairList,1)
        a = pairList{k,1};
        b = pairList{k,2};
        key = sprintf('%s_vs_%s', a, b);
        rep.cross.(key) = cross_exact_check_local(mfs.(a), gains.(a), mfs.(b), gains.(b));
        totalExact = totalExact + rep.cross.(key).nExactPairs;
    end

    totalExact = totalExact + rep.self.train.nExactPairs + rep.self.valid.nExactPairs + rep.self.test.nExactPairs;
    rep.totalExactPairs = totalExact;
end


% ============================================================
function rep = self_exact_check_local(mf, g)

    [u, ~, ic] = unique(g, 'stable');
    counts = accumarray(ic, 1);
    dupVals = u(counts > 1);

    exactPairs = zeros(0,2,'uint32');
    candidatePairs = 0;

    for iv = 1:numel(dupVals)
        idx = find(g == dupVals(iv));
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

    rep.nCandidatePairs = candidatePairs;
    rep.exactPairs = exactPairs;
    rep.nExactPairs = size(exactPairs,1);
end


% ============================================================
function rep = cross_exact_check_local(mfA, gA, mfB, gB)

    overlapVals = intersect(unique(gA,'stable'), unique(gB,'stable'), 'stable');

    exactPairs = zeros(0,2,'uint32');
    candidatePairs = 0;

    for iv = 1:numel(overlapVals)
        idxA = find(gA == overlapVals(iv));
        idxB = find(gB == overlapVals(iv));

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

    rep.nCandidatePairs = candidatePairs;
    rep.exactPairs = exactPairs;
    rep.nExactPairs = size(exactPairs,1);
end


% ============================================================
function E_opts = build_env_opts(E, scenario_choice)
    E_opts = struct();
    E_opts.Nsub = 624;
    E_opts.Ntap = 32;
    E_opts.Nt = 32;
    E_opts.Nr = 1;
    E_opts.maxRetry = 200;
    E_opts.Ntap_gen = 250;

    if E == 2
        if scenario_choice == 2
            E_opts.delaySpread = 300e-9;
        elseif scenario_choice == 5
            E_opts.delaySpread = 30e-9;
        end
    end
end


% ============================================================
function tf = is_complete_split_file(filename, splitName, expectedN)

    tf = false;
    if ~isfile(filename)
        return;
    end

    try
        info = whos('-file', filename);
        names = {info.name};

        if strcmp(splitName, 'test')
            required = {'Ht','gainHt','Horg','gainHorg'};
        else
            required = {'Ht','gainHt'};
        end

        for i = 1:numel(required)
            if ~ismember(required{i}, names)
                return;
            end
        end

        mf = matfile(filename);

        if size(mf, 'Ht', 1) ~= expectedN
            return;
        end

        gainHt = mf.gainHt(:, :);
        gainHt = gainHt(:);

        if numel(gainHt) ~= expectedN
            return;
        end

        if nnz(gainHt ~= 0) ~= expectedN
            return;
        end

        if strcmp(splitName, 'test')
            if size(mf, 'Horg', 1) ~= expectedN
                return;
            end

            gainHorg = mf.gainHorg(:, :);
            gainHorg = gainHorg(:);

            if numel(gainHorg) ~= expectedN
                return;
            end

            if nnz(gainHorg ~= 0) ~= expectedN
                return;
            end
        end

        tf = true;
    catch
        tf = false;
    end
end


% ============================================================
function newOffset = append_split_into_combined(srcFile, dstFile, splitName, currentOffset, chunkSize)

    src = matfile(srcFile);
    dst = matfile(dstFile, 'Writable', true);

    N = size(src, 'Ht', 1);
    nChunk = ceil(N / chunkSize);

    for ic = 1:nChunk
        startIdx = (ic-1)*chunkSize + 1;
        endIdx = min(ic*chunkSize, N);
        idx = startIdx:endIdx;
        nNow = numel(idx);

        Ht = src.Ht(idx,:,:,:);
        gainHt = src.gainHt(idx,:);

        dstIdx = currentOffset + idx;

        dst.Ht(dstIdx,:,:,:) = Ht;
        dst.gainHt(dstIdx,:) = gainHt;

        if strcmp(splitName, 'test')
            Horg = src.Horg(idx,:,:,:);
            gainHorg = src.gainHorg(idx,:);
            dst.Horg(dstIdx,:,:,:) = Horg;
            dst.gainHorg(dstIdx,:) = gainHorg;
        end
    end

    newOffset = currentOffset + N;
end


% ============================================================
function [Hnorm, gain] = normalize_instances_fro(H, useSingle)
    N = size(H, 1);
    H2 = reshape(H, N, []);
    gain = sqrt(sum(abs(H2).^2, 2));
    gain(gain == 0) = 1;

    Hnorm2 = H2 ./ gain;
    Hnorm = reshape(Hnorm2, size(H));

    if useSingle
        Hnorm = single(Hnorm);
        gain = single(gain);
    end
end


% ============================================================
function initialize_trainvalid_file(filename, N, E_opts, useSingle, overwrite)
    if overwrite && isfile(filename)
        delete(filename);
    end

    Nt = E_opts.Nt;
    Nr = E_opts.Nr;
    Ntap = E_opts.Ntap;

    if useSingle
        zc = complex(single(0), single(0));
        zg = single(0);
    else
        zc = complex(0, 0);
        zg = 0;
    end

    mf = matfile(filename, 'Writable', true);
    mf.Ht(1:N, 1:Nt, 1:Nr, 1:Ntap) = zc;
    mf.gainHt(1:N, 1:1) = zg;
end


function initialize_test_file(filename, N, E_opts, useSingle, overwrite)
    if overwrite && isfile(filename)
        delete(filename);
    end

    Nt = E_opts.Nt;
    Nr = E_opts.Nr;
    Ntap = E_opts.Ntap;
    Nsub = E_opts.Nsub;

    if useSingle
        zc = complex(single(0), single(0));
        zg = single(0);
    else
        zc = complex(0, 0);
        zg = 0;
    end

    mf = matfile(filename, 'Writable', true);
    mf.Ht(1:N, 1:Nt, 1:Nr, 1:Ntap) = zc;
    mf.gainHt(1:N, 1:1) = zg;
    mf.Horg(1:N, 1:Nt, 1:Nr, 1:Nsub) = zc;
    mf.gainHorg(1:N, 1:1) = zg;
end


% ============================================================
function write_trainvalid_chunk(filename, startIdx, Ht, gainHt)
    mf = matfile(filename, 'Writable', true);
    n = size(Ht, 1);
    idx = startIdx:(startIdx + n - 1);
    mf.Ht(idx, :, :, :) = Ht;
    mf.gainHt(idx, :) = reshape(gainHt, [], 1);
end


function write_test_chunk(filename, startIdx, Ht, gainHt, Horg, gainHorg)
    mf = matfile(filename, 'Writable', true);
    n = size(Ht, 1);
    idx = startIdx:(startIdx + n - 1);
    mf.Ht(idx, :, :, :) = Ht;
    mf.gainHt(idx, :) = reshape(gainHt, [], 1);
    mf.Horg(idx, :, :, :) = Horg;
    mf.gainHorg(idx, :) = reshape(gainHorg, [], 1);
end


% ============================================================
function write_split_meta(filename, spec, baseSeed, N, splitName, E_opts)
    mf = matfile(filename, 'Writable', true);

    meta = struct();
    meta.split = splitName;
    meta.E = spec.E;
    meta.scenario_choice = spec.scenario_choice;
    meta.scenario_tag = spec.tag;
    meta.count = N;
    meta.base_seed = baseSeed;
    meta.E_opts = E_opts;
    meta.description = 'Per-instance Fro-normalized dataset. gainHt/gainHorg store original Fro norms.';

    mf.meta = meta;
end


function write_combined_meta(filename, scenarioSpec, baseSeed, N, splitName, E_opts)
    mf = matfile(filename, 'Writable', true);

    meta = struct();
    meta.split = splitName;
    meta.count = N;
    meta.base_seed = baseSeed;
    meta.E_opts = E_opts;
    meta.scenarioSpec = scenarioSpec;
    meta.description = 'Combined dataset across E2 scenario 2/5 and E4 scenario 2/3. Per-instance Fro-normalized dataset. gainHt/gainHorg store original Fro norms.';

    mf.meta = meta;
end