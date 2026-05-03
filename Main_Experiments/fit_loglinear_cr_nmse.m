function fitParam = fit_loglinear_cr_nmse(cList, nmseCurveDb)
%FIT_LOGLINEAR_CR_NMSE
% Fit D_dB(c) = a - b log2(c).

    x = log2(cList(:));
    y = nmseCurveDb(:);

    A = [ones(numel(x),1), -x];

    theta = A \ y;

    a = theta(1);
    b = theta(2);

    fitParam = [a, b];
end