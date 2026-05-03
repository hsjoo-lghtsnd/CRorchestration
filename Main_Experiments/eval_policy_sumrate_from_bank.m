function out = eval_policy_sumrate_from_bank(policy, Htrue_bank, Hhat_bank, SNR, subIdx)
%EVAL_POLICY_SUMRATE_FROM_BANK Evaluate one policy using precomputed banks
%
% Input:
%   policy.idx   : [K,1] selected operating-point indices
%   Htrue_bank   : cell(K, numDrops), each entry [1, Nt, Nr, Nsub]
%   Hhat_bank    : cell(K, numDrops, M), each entry [1, Nt, Nr, Nsub]
%   SNR          : linear SNR
%   subIdx       : optional subcarrier indices for fast evaluation
%
% Output:
%   out.sumRateVec  : [numDrops,1]
%   out.userRateMat : [numDrops,K]
%   out.meanSumRate : scalar
%   out.stdSumRate  : scalar

    arguments
        policy struct
        Htrue_bank
        Hhat_bank
        SNR (1,1) double {mustBePositive}
        subIdx = []
    end

    K = numel(policy.idx);
    numDrops = size(Htrue_bank, 2);

    % Infer dimensions from the first bank entry
    sampleH = Htrue_bank{1,1};
    [~, Nt, Nr, Nsub] = size(sampleH); %#ok<ASGLU>

    sumRateVec = zeros(numDrops, 1);
    userRateMat = zeros(numDrops, K);

    for d = 1:numDrops
        Htrue_drop = zeros(K, Nt, Nr, Nsub);
        Hhat_drop  = zeros(K, Nt, Nr, Nsub);

        for k = 1:K
            idxk = policy.idx(k);

            Htrue_drop(k,:,:,:) = Htrue_bank{k,d};
            Hhat_drop(k,:,:,:)  = Hhat_bank{k,d,idxk};
        end

        [sr, ur] = evaluate_drop_zf_sumrate_fast(Htrue_drop, Hhat_drop, SNR, subIdx);
        sumRateVec(d) = sr;
        userRateMat(d,:) = ur;
    end

    out.sumRateVec = sumRateVec;
    out.userRateMat = userRateMat;
    out.meanSumRate = mean(sumRateVec);
    out.stdSumRate = std(sumRateVec, 0, 1);
end