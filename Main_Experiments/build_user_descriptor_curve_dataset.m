function out = build_user_descriptor_curve_dataset(data, V, cRef, cList, K, seed, obsSize)
%BUILD_USER_DESCRIPTOR_CURVE_DATASET
% Build per-user observable descriptor summaries and hidden CR-distortion labels.
%
% data : [Nsample, Nt, Nr, Ntap]
% V    : [D, D] basis learned from training data
%
% Output:
%   out.user(k).descriptorSummary
%   out.user(k).nmseCurve
%   out.user(k).nmseCurveDb
%   out.user(k).fitParamLogLinear = [a, b]
%
% Model:
%   D_dB(c) ~= a - b log2(c)

    arguments
        data {mustBeNumeric}
        V {mustBeNumeric}
        cRef (1,1) double {mustBePositive}
        cList (1,:) double {mustBePositive}
        K (1,1) {mustBeInteger, mustBePositive}
        seed (1,1) {mustBeInteger, mustBeNonnegative}
        obsSize (1,1) {mustBeInteger, mustBePositive}
    end

    dataSize = size(data);
    Nsample = dataSize(1);
    D = size(V, 1);

    X = reshape(data, Nsample, []);

    if size(X,2) ~= D
        error('Flattened data dimension does not match size(V,1).');
    end

    [userIdx, meta] = make_scenario_balanced_user_indices(Nsample, K, seed);

    splitIdx = cell(1, K);
    for k = 1:K
        splitIdx{k} = split_user_indices_for_observation_and_curve(userIdx{k}, obsSize);
    end

    % Precompute fixed-cRef reconstruction for observable descriptor extraction
    XhatRef = reconstruct_from_basis(X, V, cRef);

    out = struct();
    out.cRef = cRef;
    out.cList = cList;
    out.K = K;
    out.seed = seed;
    out.obsSize = obsSize;
    out.meta = meta;
    out.userIdx = userIdx;
    out.splitIdx = splitIdx;
    out.user = struct([]);

    for k = 1:K
        idxObs = splitIdx{k}.obsGlobalIndices;
        idxCurve = splitIdx{k}.curveGlobalIndices;

        % Observable part: xApp only sees reconstructed CSI at cRef
        XhatObs = XhatRef(idxObs, :);

        descriptorSummary = extract_simple_basis_descriptor_summary( ...
            XhatObs, V);

        % Hidden label part: offline CR-distortion curve
        nmseCurve = zeros(1, numel(cList));

        Xcurve = X(idxCurve, :);

        for m = 1:numel(cList)
            c = cList(m);
            XhatCurve = reconstruct_from_basis(Xcurve, V, c);

            nmseCurve(m) = sum(abs(Xcurve(:) - XhatCurve(:)).^2) / ...
                           max(sum(abs(Xcurve(:)).^2), eps);
        end

        nmseCurveDb = 10 * log10(max(nmseCurve, eps));

        fitParam = fit_loglinear_cr_nmse(cList, nmseCurveDb);

        out.user(k).userId = k;
        out.user(k).scenarioId = userIdx{k}.scenarioId;
        out.user(k).obsIndices = idxObs;
        out.user(k).curveIndices = idxCurve;

        out.user(k).descriptorSummary = descriptorSummary;
        out.user(k).nmseCurve = nmseCurve;
        out.user(k).nmseCurveDb = nmseCurveDb;
        out.user(k).fitParamLogLinear = fitParam; % [a, b]
    end

    % Matrix form for regression
    out.Phi = vertcat_descriptor(out.user);
    out.nmseCurve = vertcat(out.user.nmseCurve);
    out.nmseCurveDb = vertcat(out.user.nmseCurveDb);
    out.fitParamLogLinear = vertcat(out.user.fitParamLogLinear);
end

function Phi = vertcat_descriptor(userStruct)
%VERTCAT_DESCRIPTOR Stack descriptor vectors from out.user.

    K = numel(userStruct);
    nDesc = numel(userStruct(1).descriptorSummary.vector);

    Phi = zeros(K, nDesc);

    for k = 1:K
        Phi(k,:) = userStruct(k).descriptorSummary.vector;
    end
end