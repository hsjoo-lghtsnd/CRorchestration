function h = plot_nmse_by_scenario_tvt_db(outtr, outva, outte, varargin)
%PLOT_NMSE_BY_SCENARIO_TVT_DB Plot train/valid/test scenario-wise NMSE in dB.
%
% Usage:
%   plot_nmse_by_scenario_tvt_db(outtr, outva, outte);
%
% Name-Value:
%   'IncludeGlobal' : true/false (default: true)
%   'XMode'         : 'c' or 'L' (default: 'c')
%   'FigureTitle'   : overall title (default: '')
%   'FloorDb'       : lower clipping floor in dB (default: -120)

    p = inputParser;
    addParameter(p, 'IncludeGlobal', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'XMode', 'c', @(x) any(strcmpi(x, {'c','L'})));
    addParameter(p, 'FigureTitle', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'FloorDb', -120, @(x) isnumeric(x) && isscalar(x));
    parse(p, varargin{:});

    includeGlobal = logical(p.Results.IncludeGlobal);
    xMode = lower(char(p.Results.XMode));
    figureTitle = char(p.Results.FigureTitle);
    floorDb = p.Results.FloorDb;

    h = figure;

    subplot(1,3,1);
    i_plot_single_axes_db(outtr, 'Train', includeGlobal, xMode, floorDb);

    subplot(1,3,2);
    i_plot_single_axes_db(outva, 'Valid', includeGlobal, xMode, floorDb);

    subplot(1,3,3);
    i_plot_single_axes_db(outte, 'Test', includeGlobal, xMode, floorDb);

    if ~isempty(figureTitle)
        sgtitle(figureTitle);
    end
end


function i_plot_single_axes_db(out, titleStr, includeGlobal, xMode, floorDb)

    cList = out.cList(:).';
    nmseMatrix = out.nmseMatrix;
    scenarioNames = out.scenarioNames;
    globalNmse = out.global.nmseMean(:).';

    if strcmp(xMode, 'c')
        x = cList;
        xLabelStr = 'c';
    else
        x = out.LList(:).';
        xLabelStr = 'L';
    end

    nmseDb = 10 * log10(max(nmseMatrix, 10^(floorDb/10)));
    globalDb = 10 * log10(max(globalNmse, 10^(floorDb/10)));

    hold on;
    grid on;
    box on;

    for s = 1:size(nmseMatrix, 1)
        plot(x, nmseDb(s, :), '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    end

    if includeGlobal
        plot(x, globalDb, '--s', 'LineWidth', 2.0, 'MarkerSize', 7);
        legendEntries = [scenarioNames(:); {'Global'}];
    else
        legendEntries = scenarioNames(:);
    end

    xlabel(xLabelStr);
    ylabel('NMSE [dB]');
    title(titleStr);
    legend(legendEntries, 'Location', 'best', 'Interpreter', 'none');
    hold off;
end