function [Hset, meta, cfg] = generate_cost2100_dataset(network, scenario, Nt, Nr, Ntap, Ntrain, opts)
%GENERATE_COST2100_DATASET
% Generate COST2100 wideband MIMO impulse-response samples in the form
%   Hset : [Ntrain, Nt, Nr, Ntap]  (complex)
%
% Modified version:
% - Adds duplicate rejection on raw hFixed samples
% - Adds lightweight signature screening before exact H comparison
% - Adds optional cap on accepted samples per outer run
%
% Required inputs:
%   network   : one of
%       'IndoorHall_5GHz'
%       'SemiUrban_300MHz'
%       'Indoor_CloselySpacedUser_2_6GHz'
%       'SemiUrban_CloselySpacedUser_2_6GHz'
%       'SemiUrban_VLA_2_6GHz'
%   scenario  : 'LOS' or 'NLOS'  (subject to network limitations)
%   Nt        : # Tx antennas
%   Nr        : # Rx antennas
%   Ntap      : fixed tap length in output
%   Ntrain    : # samples to collect
%
% Optional opts struct fields:
%   opts.seed                  : RNG seed (default: [])
%   opts.linkMode              : 'auto', 'Single', or 'Multiple' (default: 'auto')
%   opts.maxOuterRuns          : max # repeated COST2100 runs if one run is not enough (default: 200)
%   opts.txRot                 : [azi ele] in rad (default: [0 0])
%   opts.rxRot                 : [azi ele] in rad (default: [0 0])
%   opts.verbose               : true/false (default: true)
%   opts.maxRetry              : retry count for "No active cluster" (default: 20)
%
% Added options:
%   opts.rejectDuplicates      : true/false, reject exact duplicate hFixed (default: true)
%   opts.signatureTol          : tolerance used in signature rounding (default: 1e-12)
%   opts.maxSamplesPerOuterRun : max accepted samples from one outer run (default: inf)
%   opts.storeSignatures       : true/false, store accepted signatures in cfg (default: false)
%
% Outputs:
%   Hset : [Ntrain, Nt, Nr, Ntap]
%   meta : struct array with per-sample bookkeeping
%   cfg  : struct containing the COST2100 setup used and debug counters
%
% Notes:
% - Duplicate rejection is done on raw hFixed before any downstream normalization.
% - Screening strategy:
%     1) build a cheap numeric signature
%     2) compare only against prior samples with the same signature
%     3) confirm with isequaln(...)
%
% Example:
%   opts.seed = 7;
%   opts.rejectDuplicates = true;
%   opts.maxSamplesPerOuterRun = 10;
%   [Hset, meta, cfg] = generate_cost2100_dataset( ...
%       'Indoor_CloselySpacedUser_2_6GHz', 'LOS', 32, 1, 32, 500, opts);

    if nargin < 7 || isempty(opts)
        opts = struct();
    end

    % -------------------------
    % Validate scalar inputs
    % -------------------------
    validateattributes(network, {'char','string'}, {'nonempty'}, mfilename, 'network', 1);
    validateattributes(scenario, {'char','string'}, {'nonempty'}, mfilename, 'scenario', 2);
    validateattributes(Nt, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'Nt', 3);
    validateattributes(Nr, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'Nr', 4);
    validateattributes(Ntap, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'Ntap', 5);
    validateattributes(Ntrain, {'numeric'}, {'scalar','integer','positive'}, mfilename, 'Ntrain', 6);

    network  = char(string(network));
    scenario = upper(char(string(scenario)));

    % -------------------------
    % Parse options
    % -------------------------
    if ~isfield(opts, 'seed');                  opts.seed = [];               end
    if ~isfield(opts, 'linkMode');              opts.linkMode = 'auto';       end
    if ~isfield(opts, 'maxOuterRuns');          opts.maxOuterRuns = 2000;     end
    if ~isfield(opts, 'txRot');                 opts.txRot = [0 0];           end
    if ~isfield(opts, 'rxRot');                 opts.rxRot = [0 0];           end
    if ~isfield(opts, 'verbose');               opts.verbose = true;          end
    if ~isfield(opts, 'maxRetry');              opts.maxRetry = 20;           end

    if ~isfield(opts, 'rejectDuplicates');      opts.rejectDuplicates = true; end
    if ~isfield(opts, 'signatureTol');          opts.signatureTol = 1e-12;    end
    if ~isfield(opts, 'maxSamplesPerOuterRun'); opts.maxSamplesPerOuterRun = inf; end
    if ~isfield(opts, 'storeSignatures');       opts.storeSignatures = false; end

    opts.linkMode = char(string(opts.linkMode));

    if ~isempty(opts.seed)
        rng('default');
        rng(opts.seed, 'twister');
    end

    % -------------------------
    % Build default COST2100 setup
    % -------------------------
    cfg = local_default_setup(network, scenario, opts.linkMode);

    % attach debug / run options
    cfg.rejectDuplicates = opts.rejectDuplicates;
    cfg.signatureTol = opts.signatureTol;
    cfg.maxSamplesPerOuterRun = opts.maxSamplesPerOuterRun;

    % -------------------------
    % Preallocate outputs
    % -------------------------
    Hset = complex(zeros(Ntrain, Nt, Nr, Ntap));

    metaTemplate = struct( ...
        'sample_index', [], ...
        'outer_run', [], ...
        'link_index', [], ...
        'snapshot_index', [], ...
        'raw_num_taps', [], ...
        'network', network, ...
        'scenario', cfg.scenario, ...
        'linkMode', cfg.linkMode, ...
        'accepted', true, ...
        'rejection_reason', '', ...
        'signature', []);
    meta = repmat(metaTemplate, Ntrain, 1);

    % bookkeeping for duplicate rejection
    acceptedCount = 0;
    duplicateRejectCount = 0;
    perOuterAccepted = 0;

    % signature table aligned with accepted samples 1:acceptedCount
    acceptedSignature = zeros(Ntrain, 4);  % [nnzAbs, fro2, realSum, imagSum]

    % -------------------------
    % Build dipole array responses
    % -------------------------
    [Gtx, Grx] = local_build_dipole_arrays(Nt, Nr);

    % -------------------------
    % Collect samples
    % -------------------------
    outerRun = 0;

    while acceptedCount < Ntrain
        outerRun = outerRun + 1;
        if outerRun > opts.maxOuterRuns
            error('Stopped after %d COST2100 runs. Collected only %d / %d accepted samples.', ...
                opts.maxOuterRuns, acceptedCount, Ntrain);
        end

        perOuterAccepted = 0;

        if opts.verbose
            fprintf('[generate_cost2100_dataset] COST2100 run %d...\n', outerRun);
        end

        if ~isempty(opts.maxRetry)
            maxRetry = opts.maxRetry;
        else
            maxRetry = 20;
        end
        retry = 0;

        while true
            try
                if ~isempty(opts.seed)
                    rng('default');
                    rng(opts.seed + 1000*outerRun + retry, 'twister');
                end

                [paraEx, ~, link, ~] = cost2100( ...
                    cfg.network, ...
                    cfg.scenario, ...
                    cfg.freq, ...
                    cfg.snapRate, ...
                    cfg.snapNum, ...
                    cfg.BSPosCenter, ...
                    cfg.BSPosSpacing, ...
                    cfg.BSPosNum, ...
                    cfg.MSPos, ...
                    cfg.MSVelo);

                break;

            catch ME
                if contains(ME.message, 'No active cluster', 'IgnoreCase', true)
                    retry = retry + 1;
                    if opts.verbose
                        fprintf('[generate_cost2100_dataset] No active cluster -> retry %d/%d\n', retry, maxRetry);
                    end
                    if retry >= maxRetry
                        fprintf('[generate_cost2100_dataset] No active cluster. Try another seed or increase maxRetry.\n');
                        rethrow(ME);
                    end
                else
                    rethrow(ME);
                end
            end
        end

        % Flatten link array
        linkList = link(:);

        stopOuter = false;

        for iLink = 1:numel(linkList)
            if ~isfield(linkList(iLink), 'channel') || isempty(linkList(iLink).channel)
                continue;
            end

            nSnap = numel(linkList(iLink).channel);
            for iSnap = 1:nSnap
                if acceptedCount >= Ntrain
                    break;
                end

                if perOuterAccepted >= opts.maxSamplesPerOuterRun
                    stopOuter = true;
                    break;
                end

                channel = linkList(iLink).channel{iSnap};
                hraw = get_H(channel, Gtx, Grx, opts.txRot, opts.rxRot, paraEx);

                % Expect wideband 3-D output
                if ndims(hraw) ~= 3
                    error('Expected wideband get_H output to be 3-D, got ndims=%d.', ndims(hraw));
                end

                % Robustly interpret hraw as [tau, Nt, Nr]
                if size(hraw,2) == Nt && size(hraw,3) == Nr
                    % do nothing
                elseif size(hraw,2) == Nr && size(hraw,3) == Nt
                    hraw = permute(hraw, [1 3 2]);
                else
                    error(['Unexpected get_H size: [%d %d %d]. ', ...
                           'Could not match requested Nt=%d, Nr=%d.'], ...
                          size(hraw,1), size(hraw,2), size(hraw,3), Nt, Nr);
                end

                rawNumTaps = size(hraw, 1);

                % Convert [tau, Nt, Nr] -> [Nt, Nr, tau]
                h3 = permute(hraw, [2 3 1]);

                % Force tap axis to Ntap
                hFixed = complex(zeros(Nt, Nr, Ntap));
                L = min(rawNumTaps, Ntap);
                hFixed(:,:,1:L) = h3(:,:,1:L);

                % -------------------------
                % Duplicate rejection
                % -------------------------
                isDup = false;
                sig = local_signature(hFixed, opts.signatureTol);

                if opts.rejectDuplicates && acceptedCount > 0
                    cand = find( ...
                        acceptedSignature(1:acceptedCount,1) == sig(1) & ...
                        acceptedSignature(1:acceptedCount,2) == sig(2) & ...
                        acceptedSignature(1:acceptedCount,3) == sig(3) & ...
                        acceptedSignature(1:acceptedCount,4) == sig(4));

                    if ~isempty(cand)
                        for ic = 1:numel(cand)
                            if isequaln(Hset(cand(ic),:,:,:), hFixed)
                                isDup = true;
                                break;
                            end
                        end
                    end
                end

                if isDup
                    duplicateRejectCount = duplicateRejectCount + 1;
                    if opts.verbose
                        fprintf('[generate_cost2100_dataset] Rejected duplicate at outer=%d, link=%d, snap=%d\n', ...
                            outerRun, iLink, iSnap);
                    end
                    continue;
                end

                % -------------------------
                % Accept sample
                % -------------------------
                acceptedCount = acceptedCount + 1;
                perOuterAccepted = perOuterAccepted + 1;

                Hset(acceptedCount,:,:,:) = hFixed;
                acceptedSignature(acceptedCount,:) = sig;

                meta(acceptedCount).sample_index   = acceptedCount;
                meta(acceptedCount).outer_run      = outerRun;
                meta(acceptedCount).link_index     = iLink;
                meta(acceptedCount).snapshot_index = iSnap;
                meta(acceptedCount).raw_num_taps   = rawNumTaps;
                meta(acceptedCount).accepted       = true;
                meta(acceptedCount).rejection_reason = '';
                meta(acceptedCount).signature      = sig;

                if opts.verbose
                    fprintf('[generate_cost2100_dataset] Accepted %d/%d | outer=%d link=%d snap=%d\n', ...
                        acceptedCount, Ntrain, outerRun, iLink, iSnap);
                end
            end

            if acceptedCount >= Ntrain || stopOuter
                break;
            end
        end
    end

    % trim in case of defensive over-allocation assumptions
    Hset = Hset(1:acceptedCount,:,:,:);
    meta = meta(1:acceptedCount);

    cfg.acceptedCount = acceptedCount;
    cfg.duplicateRejectCount = duplicateRejectCount;
    cfg.totalSeenCount = acceptedCount + duplicateRejectCount;
    cfg.acceptanceRate = acceptedCount / max(1, cfg.totalSeenCount);

    if opts.storeSignatures
        cfg.acceptedSignature = acceptedSignature(1:acceptedCount,:);
    end

    if opts.verbose
        fprintf('[generate_cost2100_dataset] Done. Accepted %d samples.\n', acceptedCount);
        fprintf('[generate_cost2100_dataset] Rejected duplicates: %d\n', duplicateRejectCount);
        fprintf('[generate_cost2100_dataset] Acceptance rate: %.4f\n', cfg.acceptanceRate);
        fprintf('Output size = [%d, %d, %d, %d]\n', size(Hset,1), size(Hset,2), size(Hset,3), size(Hset,4));
    end
end


% =========================================================================
function sig = local_signature(h, tol)
%LOCAL_SIGNATURE
% Cheap signature to shortlist duplicate candidates.
%
% Output:
%   sig = [nnzAbs, fro2, realSum, imagSum]
%
% Values are rounded to a tolerance-scaled grid to stabilize exact
% floating-point matching in the signature stage.

    v = h(:);

    nnzAbs = sum(abs(v) > 0);
    fro2 = sum(abs(v).^2);
    realSum = sum(real(v));
    imagSum = sum(imag(v));

    sig = [nnzAbs, fro2, realSum, imagSum];
    sig = round(sig ./ tol) .* tol;
end


% =========================================================================
function cfg = local_default_setup(network, scenario, linkMode)
% Replicates the tested defaults in demo_model.m, but exposed as a helper.
%
% linkMode is only used to choose a suitable default geometry.
% The cost2100() function itself does not take linkMode explicitly.

    network = char(string(network));
    scenario = upper(char(string(scenario)));
    linkMode = char(string(linkMode));

    validNetworks = { ...
        'IndoorHall_5GHz', ...
        'SemiUrban_300MHz', ...
        'Indoor_CloselySpacedUser_2_6GHz', ...
        'SemiUrban_CloselySpacedUser_2_6GHz', ...
        'SemiUrban_VLA_2_6GHz'};

    if ~ismember(network, validNetworks)
        error('Unsupported network: %s', network);
    end

    if ~ismember(scenario, {'LOS','NLOS'})
        error('scenario must be ''LOS'' or ''NLOS''.');
    end

    % Auto-select link mode
    if strcmpi(linkMode, 'auto')
        switch network
            case {'IndoorHall_5GHz', 'SemiUrban_300MHz'}
                linkMode = 'Single';
            case {'Indoor_CloselySpacedUser_2_6GHz', 'SemiUrban_CloselySpacedUser_2_6GHz', 'SemiUrban_VLA_2_6GHz'}
                linkMode = 'Multiple';
            otherwise
                linkMode = 'Single';
        end
    end

    % Default configuration
    cfg = struct();
    cfg.network = network;
    cfg.linkMode = linkMode;
    cfg.scenario = scenario;

    switch network
        case 'IndoorHall_5GHz'
            if ~strcmpi(linkMode, 'Single')
                error('IndoorHall_5GHz does not support multiple links in demo defaults.');
            end
            if ~strcmpi(scenario, 'LOS')
                error('IndoorHall_5GHz demo defaults support LOS only.');
            end

            cfg.freq = [-10e6 10e6] + 5.3e9;
            cfg.snapRate = 1;
            cfg.snapNum = 100;
            cfg.MSPos = [10 5 0];
            cfg.MSVelo = -[0 0.1 0];
            cfg.BSPosCenter = [10 10 0];
            cfg.BSPosSpacing = [0 0 0];
            cfg.BSPosNum = 1;

        case 'SemiUrban_300MHz'
            if strcmpi(linkMode, 'Multiple') && ~strcmpi(scenario, 'LOS')
                error('SemiUrban_300MHz demo defaults support multiple-link LOS only.');
            end

            cfg.freq = [2.75e8 2.95e8];
            cfg.snapRate = 1;
            cfg.snapNum = 100;
            cfg.BSPosCenter = [0 0 0];
            cfg.BSPosSpacing = [0 0 0];
            cfg.BSPosNum = 1;

            if strcmpi(linkMode, 'Single')
                cfg.MSPos = [100 -200 0];
                cfg.MSVelo = [-0.2 0.9 0];
            else
                cfg.MSPos = [100 -200 0;
                             120 -200 0];
                cfg.MSVelo = [-0.2 0.9 0;
                              -0.2 0.9 0];
            end

        case 'Indoor_CloselySpacedUser_2_6GHz'
            cfg.freq = [2.58e9 2.62e9];
            cfg.snapRate = 50;
            cfg.snapNum = 50;

            MSPos = [ ...
                -2.5600, 1.7300, 2.2300; ...
                -3.0800, 1.7300, 2.2300; ...
                -2.5600, 2.6200, 2.5800; ...
                -4.6400, 1.7300, 2.2300; ...
                -2.5600, 4.4000, 3.3000; ...
                -3.0800, 3.5100, 2.9400; ...
                -3.6000, 4.4000, 3.3000; ...
                -4.1200, 4.4000, 3.3000; ...
                -4.1200, 2.6200, 2.5800];

            BSPosCenter = [0.30 -4.37 3.20];
            center = mean(MSPos, 1);
            cfg.MSPos = MSPos - repmat(center, size(MSPos,1), 1);
            cfg.MSVelo = repmat([-0.25, 0, 0], size(MSPos,1), 1);
            cfg.BSPosCenter = BSPosCenter - center;
            cfg.BSPosSpacing = [0 0 0];
            cfg.BSPosNum = 1;

            if strcmpi(linkMode, 'Single')
                cfg.MSPos = cfg.MSPos(1,:);
                cfg.MSVelo = cfg.MSVelo(1,:);
            end

        case 'SemiUrban_CloselySpacedUser_2_6GHz'
            cfg.freq = [2.58e9 2.62e9];
            cfg.snapRate = 10;
            cfg.snapNum = 5;

            MSPos = [ ...
                -27,   -10,   0; ...
                -28.5,  -8.5, 0; ...
                -27,    -8.5, 0; ...
                -25.5,  -8.5, 0; ...
                -28.5, -10,   0; ...
                -25.5, -10,   0; ...
                -28.5, -11.5, 0; ...
                -27,   -11.5, 0; ...
                -25.5, -11.5, 0];

            MSVelo = [ ...
                 0.4,   0.3, 0; ...
                 0.3,  -0.4, 0; ...
                -0.5,   0.1, 0; ...
                -0.3,  -0.4, 0; ...
                -0.4,  -0.2, 0; ...
                 0.3,   0.4, 0; ...
                 0.3,  -0.3, 0; ...
                -0.45,  0.1, 0; ...
                 0.4,  -0.3, 0];

            cfg.BSPosCenter = [0 0 8];
            cfg.BSPosSpacing = [0 0 0];
            cfg.BSPosNum = 1;

            if strcmpi(linkMode, 'Single')
                cfg.MSPos = MSPos(1,:);
                cfg.MSVelo = MSVelo(1,:);
            else
                cfg.MSPos = MSPos;
                cfg.MSVelo = MSVelo;
            end

        case 'SemiUrban_VLA_2_6GHz'
            cfg.freq = [2.57e9 2.62e9];
            cfg.snapRate = 1;
            cfg.snapNum = 1;
            cfg.BSPosCenter = [0 0 0];
            cfg.BSPosSpacing = [0.0577 0 0];
            cfg.BSPosNum = 128;

            MSPos = [30 30 0; -10 -10 0];
            MSVelo = [0 0 0; 0 0 0];

            if strcmpi(linkMode, 'Single')
                cfg.MSPos = MSPos(1,:);
                cfg.MSVelo = MSVelo(1,:);
            else
                cfg.MSPos = MSPos;
                cfg.MSVelo = MSVelo;
            end
    end
end


% =========================================================================
function [Gtx, Grx] = local_build_dipole_arrays(Ntx, Nrx)
%LOCAL_BUILD_DIPOLE_ARRAYS
% Safe replacement for COST2100 get_dipole_G.m

    validateattributes(Ntx, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(Nrx, {'numeric'}, {'scalar','integer','positive'});

    PhiDeg   = 0:360;
    ThetaDeg = -90:90;

    Phi   = PhiDeg   * pi / 180;
    Theta = ThetaDeg * pi / 180;

    Np = numel(Phi);
    Ne = numel(Theta);

    G = zeros(Np, Ne);

    for ip = 1:Np
        for ie = 1:Ne
            theta = Theta(ie);
            if abs(theta - pi/2) < 1e-12 || abs(theta + pi/2) < 1e-12
                G(ip, ie) = 0;
            else
                G(ip, ie) = cos((pi/2) * sin(theta)) / cos(theta);
            end
        end
    end

    Anttx = complex(zeros(Ntx, Np, Ne));
    for n = 1:Ntx
        for ip = 1:Np
            phi = Phi(ip);
            Anttx(n, ip, :) = G(ip, :) * exp(-1j * pi * cos(phi) * (n - 1));
        end
    end

    Antrx = complex(zeros(Nrx, Np, Ne));
    for n = 1:Nrx
        for ip = 1:Np
            phi = Phi(ip);
            Antrx(n, ip, :) = G(ip, :) * exp(-1j * pi * cos(phi) * (n - 1));
        end
    end

    Gtx = struct();
    Gtx.antennaResponse = Anttx;
    Gtx.minResponse     = min(abs(Anttx(:)));
    Gtx.azimuthRange    = PhiDeg;
    Gtx.elevationRange  = ThetaDeg;
    Gtx.dangle          = 1;

    Grx = struct();
    Grx.antennaResponse = Antrx;
    Grx.minResponse     = min(abs(Antrx(:)));
    Grx.azimuthRange    = PhiDeg;
    Grx.elevationRange  = ThetaDeg;
    Grx.dangle          = 1;
end