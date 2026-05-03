function Ycurvehat = predict_curve_from_ab(Yab, cList)
%PREDICT_CURVE_FROM_AB Reconstruct CR-distortion curves from fitted [a,b].
%
% Input:
%   Yab   : [N, 2], columns = [a, b]
%   cList : [1, M] or [M, 1], compression ratios
%
% Output:
%   Ycurvehat : [N, M], predicted NMSE curve in dB
%
% Model:
%   D_dB(c) = a - b * log2(c)

    arguments
        Yab double
        cList double
    end

    if size(Yab,2) ~= 2
        error('Yab must be [N,2], columns = [a,b].');
    end

    a = Yab(:,1);              % [N,1]
    b = Yab(:,2);              % [N,1]
    x = log2(cList(:))';       % [1,M]

    Ycurvehat = a - b .* x;    % implicit expansion -> [N,M]
end