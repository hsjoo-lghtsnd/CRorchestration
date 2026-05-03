function [sumRate, userRate] = evaluate_drop_zf_sumrate_fast(Htrue_drop, Hhat_drop, SNR, subIdx)
%EVALUATE_DROP_ZF_SUMRATE_FAST Fast MU-ZF sum-rate evaluation for one drop
%
% Input:
%   Htrue_drop : [K, Nt, Nr, Nsub]
%   Hhat_drop  : [K, Nt, Nr, Nsub]
%   SNR        : linear SNR
%   subIdx     : optional subcarrier indices to evaluate
%
% Output:
%   sumRate    : scalar
%   userRate   : [1, K]
%
% Notes:
%   - Uses only RX branch 1 (single_rx assumption)
%   - ZF precoder is built from Hhat, rate is evaluated on Htrue
%   - Faster than the original version by avoiding nested user-interference loops

    arguments
        Htrue_drop
        Hhat_drop
        SNR (1,1) double {mustBePositive}
        subIdx = []
    end

    if isempty(subIdx)
        subIdx = 1:size(Htrue_drop, 4);
    end

    [K, Nt, Nr, Nsub] = size(Htrue_drop); %#ok<ASGLU>
    assert(size(Hhat_drop,1) == K, 'Hhat_drop must have the same K as Htrue_drop.');
    assert(size(Hhat_drop,2) == Nt, 'Hhat_drop must have the same Nt as Htrue_drop.');
    assert(K <= Nt, 'ZF requires K <= Nt.');

    Nuse = numel(subIdx);
    userRate_sub = zeros(K, Nuse);
    Pk = SNR / K;

    for ii = 1:Nuse
        n = subIdx(ii);

        % Effective channels using only RX branch 1: [K, Nt]
        Htrue_eff = reshape(Htrue_drop(:,:,1,n), K, Nt);
        Hhat_eff  = reshape(Hhat_drop(:,:,1,n),  K, Nt);

        % ZF precoder from imperfect CSI
        % W = Hhat^H * inv(Hhat Hhat^H), with tiny diagonal loading
        Gram = Hhat_eff * Hhat_eff';
        W = Hhat_eff' / (Gram + 1e-12 * eye(K));

        % Normalize columns
        colNorm = vecnorm(W, 2, 1);
        colNorm = max(colNorm, 1e-12);
        W = W ./ colNorm;

        % Effective true channel after beamforming: [K, K]
        G = Htrue_eff * W;

        % Signal/interference decomposition
        signal = Pk * abs(diag(G)).^2;       % [K,1]
        totalPow = Pk * sum(abs(G).^2, 2);   % [K,1]
        interf = totalPow - signal;          % [K,1]

        sinr = signal ./ (1 + interf);
        userRate_sub(:, ii) = log2(1 + sinr);
    end

    userRate = mean(userRate_sub, 2).';   % [1, K]
    sumRate = sum(userRate);
end