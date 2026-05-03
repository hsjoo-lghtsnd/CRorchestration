function [userIdx, meta] = make_scenario_balanced_user_indices(Nsample, K, seed)
%MAKE_SCENARIO_BALANCED_USER_INDICES
% Partition scenario-ordered samples into K synthetic users by index only.
%
% Assumption:
%   Samples are ordered as four equal scenario blocks:
%   scenario 1: 1 : N/4
%   scenario 2: N/4+1 : N/2
%   scenario 3: N/2+1 : 3N/4
%   scenario 4: 3N/4+1 : N
%
% Output:
%   userIdx{k}.globalIndices : sample indices assigned to user k
%   userIdx{k}.scenarioId    : scenario index of user k

    arguments
        Nsample (1,1) {mustBeInteger, mustBePositive}
        K (1,1) {mustBeInteger, mustBePositive}
        seed (1,1) {mustBeInteger, mustBeNonnegative} = 0
    end

    rng(seed);

    nScenario = 4;

    if mod(Nsample, nScenario) ~= 0
        error('Nsample must be divisible by 4.');
    end

    nPerScenario = Nsample / nScenario;

    % cyclic assignment: K=8 -> 1 2 3 4 1 2 3 4
    userScenario = mod((1:K) - 1, nScenario) + 1;
    usersPerScenario = accumarray(userScenario(:), 1, [nScenario, 1]);

    userIdx = cell(1, K);

    for s = 1:nScenario
        usersInScenario = find(userScenario == s);
        nUserS = numel(usersInScenario);

        scenarioStart = (s - 1) * nPerScenario + 1;
        scenarioEnd   = s * nPerScenario;
        scenarioRows  = scenarioStart:scenarioEnd;

        permRows = scenarioRows(randperm(nPerScenario));

        if nUserS == 0
            continue;
        end

        splitEdges = round(linspace(0, nPerScenario, nUserS + 1));

        for u = 1:nUserS
            k = usersInScenario(u);

            st = splitEdges(u) + 1;
            ed = splitEdges(u + 1);

            idx = permRows(st:ed);

            userIdx{k} = struct();
            userIdx{k}.userId = k;
            userIdx{k}.scenarioId = s;
            userIdx{k}.scenarioUserId = u;
            userIdx{k}.numUsersInScenario = nUserS;
            userIdx{k}.globalIndices = idx(:);
            userIdx{k}.localIndices = idx(:) - (s - 1) * nPerScenario;
            userIdx{k}.numSamples = numel(idx);
        end
    end

    meta = struct();
    meta.Nsample = Nsample;
    meta.K = K;
    meta.seed = seed;
    meta.nScenario = nScenario;
    meta.nPerScenario = nPerScenario;
    meta.userScenario = userScenario;
    meta.usersPerScenario = usersPerScenario;

    fprintf('\n[make_scenario_balanced_user_indices]\n');
    fprintf('  Nsample = %d, K = %d, seed = %d\n', Nsample, K, seed);
    fprintf('  usersPerScenario = ');
    fprintf('%d ', usersPerScenario);
    fprintf('\n');

    for k = 1:K
        fprintf('  user %2d | scenario %d | samples %d\n', ...
            userIdx{k}.userId, userIdx{k}.scenarioId, userIdx{k}.numSamples);
    end
end