function plot_user_cr_distortion_curves(out, opts)
%PLOT_USER_CR_DISTORTION_CURVES Plot per-user CR-distortion curves.

    arguments
        out struct
        opts.YMode char
        opts.GroupByScenario logical
        opts.ShowMean logical
        opts.TitleStr char
    end

    if ~isfield(opts, 'YMode'), opts.YMode = 'db'; end
    if ~isfield(opts, 'GroupByScenario'), opts.GroupByScenario = true; end
    if ~isfield(opts, 'ShowMean'), opts.ShowMean = true; end
    if ~isfield(opts, 'TitleStr'), opts.TitleStr = 'Per-user CR-distortion curves'; end

    cList = out.cList;
    K = out.K;

    figure; hold on; grid on; box on;

    for k = 1:K
        if strcmpi(opts.YMode, 'db')
            y = out.user(k).nmseCurveDb;
            ylab = 'NMSE [dB]';
        else
            y = out.user(k).nmseCurve;
            ylab = 'NMSE';
        end

        if opts.GroupByScenario
            nameStr = sprintf('U%d/S%d', k, out.user(k).scenarioId);
        else
            nameStr = sprintf('User %d', k);
        end

        plot(cList, y, '-o', 'LineWidth', 1.0, ...
            'DisplayName', nameStr);
    end

    if opts.ShowMean
        if strcmpi(opts.YMode, 'db')
            yMean = mean(out.nmseCurveDb, 1);
        else
            yMean = mean(out.nmseCurve, 1);
        end

        plot(cList, yMean, 'k--s', 'LineWidth', 2.0, ...
            'DisplayName', 'Mean');
    end

    set(gca, 'XScale', 'log');
    xlabel('Compression ratio c');
    ylabel(ylab);
    title(opts.TitleStr);
    legend('Location', 'best');
end