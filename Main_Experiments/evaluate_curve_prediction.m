function metrics = evaluate_curve_prediction(YcurveTrueDb, YcurveHatDb, cList)
%EVALUATE_CURVE_PREDICTION
% Compare predicted and true CR-distortion curves in dB domain.

    arguments
        YcurveTrueDb double
        YcurveHatDb double
        cList double
    end

    if ~isequal(size(YcurveTrueDb), size(YcurveHatDb))
        error('YcurveTrueDb and YcurveHatDb must have the same size.');
    end

    err = YcurveHatDb - YcurveTrueDb;

    metrics = struct();
    metrics.mae_curve_per_c = mean(abs(err), 1);
    metrics.rmse_curve_per_c = sqrt(mean(err.^2, 1));
    metrics.mae_curve_mean = mean(abs(err), 'all');
    metrics.rmse_curve_mean = sqrt(mean(err.^2, 'all'));

    % Rank correlation per CR point
    M = size(YcurveTrueDb, 2);
    metrics.spearman_curve = zeros(1, M);
    metrics.kendall_curve = zeros(1, M);

    for m = 1:M
        metrics.spearman_curve(m) = corr( ...
            YcurveTrueDb(:,m), YcurveHatDb(:,m), 'Type', 'Spearman');
        metrics.kendall_curve(m) = corr( ...
            YcurveTrueDb(:,m), YcurveHatDb(:,m), 'Type', 'Kendall');
    end

    metrics.mean_spearman_curve = mean(metrics.spearman_curve);
    metrics.mean_kendall_curve = mean(metrics.kendall_curve);

    % Marginal gain ordering
    gainTrue = compute_marginal_gain_from_curve_db(YcurveTrueDb);
    gainHat  = compute_marginal_gain_from_curve_db(YcurveHatDb);

    M1 = size(gainTrue, 2);
    metrics.spearman_gain = zeros(1, M1);
    metrics.kendall_gain = zeros(1, M1);

    for m = 1:M1
        metrics.spearman_gain(m) = corr(gainTrue(:,m), gainHat(:,m), ...
            'Type', 'Spearman');
        metrics.kendall_gain(m) = corr(gainTrue(:,m), gainHat(:,m), ...
            'Type', 'Kendall');
    end

    metrics.mean_spearman_gain = mean(metrics.spearman_gain);
    metrics.mean_kendall_gain = mean(metrics.kendall_gain);
end


function gain = compute_marginal_gain_from_curve_db(YcurveDb)
%COMPUTE_MARGINAL_GAIN_FROM_CURVE_DB
% gain(:,m) = distortion reduction from c_m to c_{m+1}
% YcurveDb: [N, M]

    Dlin = 10.^(YcurveDb / 10);
    gain = Dlin(:,1:end-1) - Dlin(:,2:end);
end
