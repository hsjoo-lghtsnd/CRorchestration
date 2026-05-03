function plot_yab_scatter(Yab, varargin)
%PLOT_YAB_SCATTER Scatter plot of fitted [a, b] parameters.
%
% Usage:
%   plot_yab_scatter(Yab)
%   plot_yab_scatter(Yab, 'ScenarioId', scenarioId)
%   plot_yab_scatter(Yab, 'Yhat', Yhat)
%
% Input:
%   Yab : [N, 2], columns = [a, b]
%
% Optional name-value:
%   'ScenarioId' : [N,1] scenario labels
%   'Yhat'       : [N,2] predicted [a,b]
%   'TitleStr'   : title string

    p = inputParser;
    addRequired(p, 'Yab', @isnumeric);
    addParameter(p, 'ScenarioId', [], @isnumeric);
    addParameter(p, 'Yhat', [], @isnumeric);
    addParameter(p, 'TitleStr', 'Scatter of fitted [a,b]', @(x) ischar(x) || isstring(x));
    parse(p, Yab, varargin{:});

    scenarioId = p.Results.ScenarioId;
    Yhat = p.Results.Yhat;
    TitleStr = char(p.Results.TitleStr);

    if size(Yab,2) ~= 2
        error('Yab must be [N,2].');
    end

    a = Yab(:,1);
    b = Yab(:,2);

    figure;
    hold on; grid on; box on;

    if isempty(scenarioId)
        scatter(a, b, 50, 'filled', 'DisplayName', 'True [a,b]');
    else
        uSc = unique(scenarioId(:))';
        for s = uSc
            idx = (scenarioId == s);
            scatter(a(idx), b(idx), 50, 'filled', ...
                'DisplayName', sprintf('Scenario %d', s));
        end
    end

    if ~isempty(Yhat)
        if ~isequal(size(Yhat), size(Yab))
            error('Yhat must have the same size as Yab.');
        end

        scatter(Yhat(:,1), Yhat(:,2), 60, 'x', 'LineWidth', 1.5, ...
            'DisplayName', 'Predicted [a,b]');

        % optional connection lines
        for i = 1:size(Yab,1)
            plot([Yab(i,1), Yhat(i,1)], [Yab(i,2), Yhat(i,2)], ...
                '-', 'LineWidth', 0.5, 'HandleVisibility', 'off');
        end
    end

    xlabel('a');
    ylabel('b');
    title(TitleStr);
    legend('Location', 'best');
end