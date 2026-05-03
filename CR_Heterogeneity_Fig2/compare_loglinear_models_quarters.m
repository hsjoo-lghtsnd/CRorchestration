function summary = compare_loglinear_models_quarters(out, varargin)
%COMPARE_LOGLINEAR_MODELS_QUARTERS
% Compare two models for quarter-wise NMSE curves in `out`:
%
% Model A:
%   log10(NMSE) ~ a * c + b
%
% Model B:
%   log10(NMSE) ~ a * log2(c) + b
%
% Input:
%   out : output struct from estimate_nmse_by_scenario_quarters
%
% Name-Value:
%   'Verbose' : true/false (default: true)
%
% Output:
%   summary.table
%   summary.global
%   summary.quarters

    p = inputParser;
    addParameter(p, 'Verbose', true, @(x) islogical(x) || isnumeric(x));
    parse(p, varargin{:});
    verbose = logical(p.Results.Verbose);

    c = out.cList(:);
    assert(all(c > 0), 'cList must be positive.');

    nScen = numel(out.scenarioNames);

    names = cell(nScen + 1, 1);
    r2_linearX = zeros(nScen + 1, 1);
    r2_logX = zeros(nScen + 1, 1);
    rmse_linearX = zeros(nScen + 1, 1);
    rmse_logX = zeros(nScen + 1, 1);
    slope_linearX = zeros(nScen + 1, 1);
    intercept_linearX = zeros(nScen + 1, 1);
    slope_logX = zeros(nScen + 1, 1);
    intercept_logX = zeros(nScen + 1, 1);
    betterModel = cell(nScen + 1, 1);

    % Global
    fitG = i_fit_one(c, out.global.nmseMean(:));
    names{1} = 'Global';
    r2_linearX(1) = fitG.r2_linearX;
    r2_logX(1) = fitG.r2_logX;
    rmse_linearX(1) = fitG.rmse_linearX;
    rmse_logX(1) = fitG.rmse_logX;
    slope_linearX(1) = fitG.p_linearX(1);
    intercept_linearX(1) = fitG.p_linearX(2);
    slope_logX(1) = fitG.p_logX(1);
    intercept_logX(1) = fitG.p_logX(2);
    betterModel{1} = fitG.betterModel;

    quarterFits = cell(1, nScen);

    % Quarters
    for s = 1:nScen
        fitS = i_fit_one(c, out.nmseMatrix(s, :).');
        quarterFits{s} = fitS;

        names{s+1} = out.scenarioNames{s};
        r2_linearX(s+1) = fitS.r2_linearX;
        r2_logX(s+1) = fitS.r2_logX;
        rmse_linearX(s+1) = fitS.rmse_linearX;
        rmse_logX(s+1) = fitS.rmse_logX;
        slope_linearX(s+1) = fitS.p_linearX(1);
        intercept_linearX(s+1) = fitS.p_linearX(2);
        slope_logX(s+1) = fitS.p_logX(1);
        intercept_logX(s+1) = fitS.p_logX(2);
        betterModel{s+1} = fitS.betterModel;
    end

    T = table(names, ...
        r2_linearX, r2_logX, ...
        rmse_linearX, rmse_logX, ...
        slope_linearX, intercept_linearX, ...
        slope_logX, intercept_logX, ...
        betterModel, ...
        'VariableNames', { ...
        'Name', ...
        'R2_linearX', 'R2_logX', ...
        'RMSE_linearX', 'RMSE_logX', ...
        'Slope_linearX', 'Intercept_linearX', ...
        'Slope_logX', 'Intercept_logX', ...
        'BetterModel'});

    summary = struct();
    summary.table = T;
    summary.global = fitG;
    summary.quarters = quarterFits;

    if verbose
        disp(T);
    end
end


function fit = i_fit_one(c, nmse)
    y = log10(nmse);

    % Model A: y ~ a*c + b
    x1 = c;
    p1 = polyfit(x1, y, 1);
    yhat1 = polyval(p1, x1);

    % Model B: y ~ a*log2(c) + b
    x2 = log2(c);
    p2 = polyfit(x2, y, 1);
    yhat2 = polyval(p2, x2);

    [r2_1, rmse_1] = i_metrics(y, yhat1);
    [r2_2, rmse_2] = i_metrics(y, yhat2);

    fit = struct();
    fit.p_linearX = p1;
    fit.yhat_linearX = yhat1;
    fit.r2_linearX = r2_1;
    fit.rmse_linearX = rmse_1;

    fit.p_logX = p2;
    fit.yhat_logX = yhat2;
    fit.r2_logX = r2_2;
    fit.rmse_logX = rmse_2;

    if r2_1 > r2_2
        fit.betterModel = 'linearX';
    elseif r2_1 < r2_2
        fit.betterModel = 'logX';
    else
        fit.betterModel = 'tie';
    end
end


function [r2, rmse] = i_metrics(y, yhat)
    resid = y - yhat;
    sse = sum(resid.^2);
    sst = sum((y - mean(y)).^2);
    if sst == 0
        r2 = 1;
    else
        r2 = 1 - sse / sst;
    end
    rmse = sqrt(mean(resid.^2));
end