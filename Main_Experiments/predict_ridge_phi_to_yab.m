function Yhat = predict_ridge_phi_to_yab(model, Phi)
%PREDICT_RIDGE_PHI_TO_YAB Predict [a,b] from Phi using fitted ridge model.

    arguments
        model struct
        Phi double
    end

    PhiNorm = (Phi - model.phiMean) ./ model.phiStd;
    Yhat = PhiNorm * model.W + model.yMean;
end