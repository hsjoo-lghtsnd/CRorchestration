function metrics = evaluate_yab_prediction(Ytrue, Yhat)
%EVALUATE_YAB_PREDICTION Compute simple regression metrics for [a,b].

    arguments
        Ytrue double
        Yhat double
    end

    err = Yhat - Ytrue;

    metrics = struct();
    metrics.mae_ab = mean(abs(err), 1);
    metrics.rmse_ab = sqrt(mean(err.^2, 1));

    metrics.mae_mean = mean(abs(err), 'all');
    metrics.rmse_mean = sqrt(mean(err.^2, 'all'));
end