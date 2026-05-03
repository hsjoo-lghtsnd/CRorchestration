function h = plot_nmse_by_scenario(out, varargin)
%PLOT_NMSE_BY_SCENARIO Plot NMSE-vs-c curves from output struct.
%
% Usage:
%   plot_nmse_by_scenario(out);
%   plot_nmse_by_scenario(out, 'IncludeGlobal', true, 'TitleStr', 'Train');
%
% Required fields in out:
%   out.cList
%   out.nmseMatrix          % [Nscenario, Nc]
%   out.scenarioNames       % 1 x Nscenario cell
%   out.global.nmseMean     % [1, Nc] or [Nc, 1]
%
% Name-Value options:
%   'IncludeGlobal' : true/false (default: true)
%   'UseLogY'       : true/false (default: true)
%   'TitleStr'      : plot title (default: '')
%   'XMode'         : 'c' or 'L' (default: 'c')
%                     - 'c': x-axis is compression ratio c
%                     - 'L': x-axis is retained rank L
%   'MarkerSize'    : marker size (default: 7)
%   'LineWidth'     : line width (default: 1.5)
%   'LegendLocation': legend location (default: 'best')

    p = inputParser;
    addParameter(p, 'IncludeGlobal', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'UseLogY', true, @(x) islogical(x) || isnumeric(x));
    addParameter(p, 'TitleStr', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'XMode', 'c', @(x) any(strcmpi(x, {'c','L'})));
    addParameter(p, 'MarkerSize', 7, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'LineWidth', 1.5, @(x) isnumeric(x) && isscalar(x));
    addParameter(p, 'LegendLocation', 'best', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});

    includeGlobal = logical(p.Results.IncludeGlobal);
    useLogY = logical(p.Results.UseLogY);
    titleStr = char(p.Results.TitleStr);
    xMode = lower(char(p.Results.XMode));
    markerSize = p.Results.MarkerSize;
    lineWidth = p.Results.LineWidth;
    legendLocation = char(p.Results.LegendLocation);

    % Basic checks
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

    % X-axis data
    if strcmp(xMode, 'c')
        x = cList;
        xLabelStr = 'Compression ratio c';
    else
        assert(isfield(out, 'LList'), 'out.LList is required when XMode is ''L''.');
        x = out.LList(:).';
        xLabelStr = 'Retained rank L';
    end

    % Plot
    h = figure;
    hold on;
    grid on;
    box on;

    for s = 1:Nscenario
        y = nmseMatrix(s, :);
        if useLogY
            semilogy(x, y, '-o', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
        else
            plot(x, y, '-o', 'LineWidth', lineWidth, 'MarkerSize', markerSize);
        end
    end

    if includeGlobal
        if useLogY
            semilogy(x, globalNmse, '--s', 'LineWidth', lineWidth + 0.5, ...
                'MarkerSize', markerSize + 1);
        else
            plot(x, globalNmse, '--s', 'LineWidth', lineWidth + 0.5, ...
                'MarkerSize', markerSize + 1);
        end
        legendEntries = [scenarioNames(:); {'Global'}];
    else
        legendEntries = scenarioNames(:);
    end

    xlabel(xLabelStr);
    ylabel('NMSE');
    title(titleStr);
    legend(legendEntries, 'Location', legendLocation, 'Interpreter', 'none');

    hold off;
end