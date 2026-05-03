function [sumRate, userRate] = evaluate_drop_zf_sumrate(Htrue_drop, Hhat_drop, SNR, streamMode)
%EVALUATE_DROP_ZF_SUMRATE Evaluate MU ZF sum rate for one drop
%
% Input:
%   Htrue_drop : [K, Nt, Nr, Nsub]
%   Hhat_drop  : [K, Nt, Nr, Nsub]
%   SNR        : linear SNR
%
% Output:
%   sumRate    : scalar
%   userRate   : [1, K]
%
% Notes:
%   - One stream per user
%   - Equal power across users
%   - Effective channel uses only RX branch 1 if Nr > 1
%   - ZF precoder is built from Hhat, rate evaluated on Htrue

    arguments
        Htrue_drop
        Hhat_drop
        SNR (1,1) double {mustBePositive}
        streamMode (1,1) string = "single_rx"
    end

    [K, Nt, Nr, Nsub] = size(Htrue_drop);
    assert(streamMode == "single_rx", 'Only single_rx mode is supported currently.');
    assert(K <= Nt, 'ZF requires K <= Nt.');

    userRate_sub = zeros(K, Nsub);

    for n = 1:Nsub
        % Effective user channel row vectors: [K, Nt]
        Htrue_eff = zeros(K, Nt);
        Hhat_eff  = zeros(K, Nt);

        for k = 1:K
            Htrue_k = squeeze(Htrue_drop(k,:,1,n)); % use first RX branch
            Hhat_k  = squeeze(Hhat_drop(k,:,1,n));
            Htrue_eff(k,:) = reshape(Htrue_k, 1, []);
            Hhat_eff(k,:)  = reshape(Hhat_k, 1, []);
        end

        % ZF precoder from imperfect CSI
        % W = Hhat^H * inv(Hhat Hhat^H)
        W = Hhat_eff' * pinv(Hhat_eff * Hhat_eff');

        % Normalize columns
        for k = 1:K
            wk = W(:,k);
            nk = norm(wk);
            if nk > 0
                W(:,k) = wk / nk;
            end
        end

        % Equal power allocation across K users
        Pk = SNR / K;

        for k = 1:K
            hk = Htrue_eff(k,:);    % [1 x Nt]
            wk = W(:,k);            % [Nt x 1]

            signal = Pk * abs(hk * wk)^2;

            interf = 0;
            for j = 1:K
                if j == k, continue; end
                wj = W(:,j);
                interf = interf + Pk * abs(hk * wj)^2;
            end

            sinr_k = signal / (1 + interf);
            userRate_sub(k,n) = log2(1 + sinr_k);
        end
    end

    userRate = mean(userRate_sub, 2).';   % [1, K]
    sumRate = sum(userRate);
end