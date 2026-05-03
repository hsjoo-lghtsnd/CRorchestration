function plot_yab_true_vs_pred(Ytrue, Yhat)
%PLOT_YAB_TRUE_VS_PRED True-vs-predicted scatter for a and b separately.

    if size(Ytrue,2) ~= 2 || size(Yhat,2) ~= 2
        error('Ytrue and Yhat must be [N,2].');
    end

    figure;
    hold on; grid on; box on;
    scatter(Ytrue(:,1), Yhat(:,1), 40, 'filled');
    xmin = min([Ytrue(:,1); Yhat(:,1)]);
    xmax = max([Ytrue(:,1); Yhat(:,1)]);
    plot([xmin xmax], [xmin xmax], 'k--', 'LineWidth', 1.2);
    xlabel('True a');
    ylabel('Predicted a');
    title('a: true vs predicted');

    figure;
    hold on; grid on; box on;
    scatter(Ytrue(:,2), Yhat(:,2), 40, 'filled');
    xmin = min([Ytrue(:,2); Yhat(:,2)]);
    xmax = max([Ytrue(:,2); Yhat(:,2)]);
    plot([xmin xmax], [xmin xmax], 'k--', 'LineWidth', 1.2);
    xlabel('True b');
    ylabel('Predicted b');
    title('b: true vs predicted');
end