function [model, Yhat] = fit_ridge_phi_to_yab(Phi, Yab, lambda)
%FIT_RIDGE_PHI_TO_YAB Fit multi-output ridge regression from Phi to [a,b].
%
% Input:
%   Phi    : [N, P] descriptor matrix
%   Yab    : [N, 2] target matrix, columns are [a, b]
%   lambda : nonnegative scalar ridge coefficient
%
% Output:
%   model  : struct containing normalization stats and fitted weights
%   Yhat   : [N, 2] fitted prediction on the training data
%
% Model:
%   Y ~= PhiNorm * W + b
%
% Notes:
%   - Phi is standardized column-wise.
%   - Y is centered but not standardized.
%   - Bias term is handled separately and is not regularized.

    arguments
        Phi double
        Yab double
        lambda (1,1) double {mustBeNonnegative}
    end

    [N, P] = size(Phi);

    if size(Yab,1) ~= N
        error('Phi and Yab must have the same number of rows.');
    end

    if size(Yab,2) ~= 2
        error('Yab must be [N, 2], with columns [a, b].');
    end

    % Standardize Phi
    phiMean = mean(Phi, 1);
    phiStd = std(Phi, 0, 1);
    phiStd(phiStd < eps) = 1;

    PhiNorm = (Phi - phiMean) ./ phiStd;

    % Center Y
    yMean = mean(Yab, 1);
    Yc = Yab - yMean;

    % Ridge solution
    % W = argmin ||Yc - PhiNorm*W||_F^2 + lambda*||W||_F^2
    W = (PhiNorm' * PhiNorm + lambda * eye(P)) \ (PhiNorm' * Yc);

    % Predict on training set
    Yhat = PhiNorm * W + yMean;

    % Save model
    model = struct();
    model.lambda = lambda;
    model.phiMean = phiMean;
    model.phiStd = phiStd;
    model.yMean = yMean;
    model.W = W;
end