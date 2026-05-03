function stats = check_row_unit_norm(X, tol, verbose)
%CHECK_ROW_UNIT_NORM Check whether each row of X has unit l2 norm.
%
% stats fields:
%   .is_ok
%   .max_abs_err
%   .mean_abs_err
%   .min_norm
%   .max_norm
%   .mean_norm
%   .row_norms

    if nargin < 2 || isempty(tol)
        tol = 1e-8;
    end
    if nargin < 3
        verbose = true;
    end

    rowNorms = sqrt(sum(abs(X).^2, 2));
    absErr = abs(rowNorms - 1);

    stats = struct();
    stats.is_ok = all(absErr <= tol);
    stats.max_abs_err = max(absErr);
    stats.mean_abs_err = mean(absErr);
    stats.min_norm = min(rowNorms);
    stats.max_norm = max(rowNorms);
    stats.mean_norm = mean(rowNorms);
    stats.row_norms = rowNorms;

    if verbose
        fprintf('[check_row_unit_norm]\n');
        fprintf('  tol          = %.3e\n', tol);
        fprintf('  is_ok        = %d\n', stats.is_ok);
        fprintf('  min_norm     = %.12f\n', stats.min_norm);
        fprintf('  max_norm     = %.12f\n', stats.max_norm);
        fprintf('  mean_norm    = %.12f\n', stats.mean_norm);
        fprintf('  max_abs_err  = %.3e\n', stats.max_abs_err);
        fprintf('  mean_abs_err = %.3e\n', stats.mean_abs_err);
    end
end