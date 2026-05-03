function h = plot_nmse_by_scenario_db(out, varargin)
%PLOT_NMSE_BY_SCENARIO_DB Plot NMSE-vs-c curves in dB scale.
%
% Usage:
%   plot_nmse_by_scenario_db(out);
%   plot_nmse_by_scenario_db(out, 'IncludeGlobal', true, 'TitleStr', 'Train');
%
% Required fields in out:
%   out.cList
%   out.nmseMatrix
%   out.scenarioNames
%   out.global.nmseMean
%
% Name-Value options:
%   'IncludeGlobal' : true/false (default: true)
%   'TitleStr'      : plot title (default: '')
%   'XMode'         : 'c' or 'L' (default: 'c')
%   'MarkerSize'    : marker size (default: 7)
%   'LineWidth'     : line width (default: 1.5)
%   'LegendLocation': legend location (default: 'best')
%   'FloorDb'       : lower clipping floor in dB (default: -120)

    p = inputParser;
    addParameter(p, 'IncludeGlobal', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'TitleStr', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'XMode', 'c', @(x) any(strcmpi(x, {'c','L'})));
    addParameter(p, 'MarkerSize', 7, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'LineWidth', 1.5, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'LegendLocation', 'best', @(x) ischar(x) || isstring(x));
    addParameter(p, 'FloorDb', -120, @(x) isnumeric(x) && isscalar(x));
    parse(p, varargin{:});

    includeGlobal = logical(p.Results.IncludeGlobal);
    titleStr = char(p.Results.TitleStr);
    xMode = lower(char(p.Results.XMode));
    markerSize = p.Results.MarkerSize;
    lineWidth = p.Results.LineWidth;
    legendLocation = char(p.Results.LegendLocation);
    floorDb = p.Results.FloorDb;

    assert(isfield(out, 'cList'), 'out.cList is missing.');
    assert(isfield(out, 'nmseMatrix'), 'out.nmseMatrix is missing.');
    assert(isfield(out, 'scenarioNames'), 'out.scenarioNames is missing.');
    assert(isfield(out, 'global') && isfield(out.global, 'nmseMean'), ...
        'out.global.nmseMean is missing.');

    cList = out.cList(:).';
    nmseMatrix = out.nmseMatrix;
    scenarioNames = out.scenarioNames;
    globalNmse = out.global.nmseMean(:).';

    Nscenario = size(nmseMatrix, 1);
    Nc = size(nmseMatrix, 2);

    assert(numel(cList) == Nc, 'Length of cList must match nmseMatrix columns.');
    assert(numel(scenarioNames) == Nscenario, ...
        'Number of scenarioNames must match nmseMatrix rows.');
    assert(numel(globalNmse) == Nc, ...
        'Length of global.nmseMean must match cList.');

    if strcmp(xMode, 'c')
        x = cList;
        xLabelStr = 'Compression ratio c';
    else
        assert(isfield(out, 'LList'), 'out.LList is required when XMode is ''L''.');
        x = out.LList(:).';
        xLabelStr = 'Retained rank L';
    end

    % dB conversion with floor
    nmseDb = 10 * log10(max(nmseMatrix, 10^(floorDb/10)));
    globalDb = 10 * log10(max(globalNmse, 10^(floorDb/10)));

    h = figure;
    hold on;
    grid on;
    box on;

    for s = 1:Nscenario
        plot(x, nmseDb(s, :), '-o', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
    end

    if includeGlobal
        plot(x, globalDb, '--s', 'LineWidth', lineWidth + 0.5, ...
            'MarkerSize', markerSize + 1);
        legendEntries = [scenarioNames(:); {'Global'}];
    else
        legendEntries = scenarioNames(:);
    end

    xlabel(xLabelStr);
    ylabel('NMSE [dB]');
    title(titleStr);
    legend(legendEntries, 'Location', legendLocation, 'Interpreter', 'none');

    hold off;
end