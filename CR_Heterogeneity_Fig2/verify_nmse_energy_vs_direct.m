function check = verify_nmse_energy_vs_direct(X, V, cList, absTol, relTol, verbose)

    if nargin < 4 || isempty(absTol)
        if isa(X, 'single') || isa(V, 'single')
            absTol = 1e-5;
        else
            absTol = 1e-8;
        end
    end
    if nargin < 5 || isempty(relTol)
        if isa(X, 'single') || isa(V, 'single')
            relTol = 1e-3;
        else
            relTol = 1e-6;
        end
    end
    if nargin < 6
        verbose = true;
    end

    [~, Ndim] = size(X);
    cList = cList(:).';
    LList = max(1, round(cList * Ndim));
    LList = min(LList, Ndim);

    rowEnergy = sum(abs(X).^2, 2);

    nmseEnergyMean = zeros(size(LList));
    nmseDirectMean = zeros(size(LList));
    absDiff = zeros(size(LList));
    relDiff = zeros(size(LList));
    passEach = false(size(LList));

    for i = 1:numel(LList)
        L = LList(i);
        VL = V(:,1:L);

        ZL = X * VL;
        capturedEnergy = sum(abs(ZL).^2, 2);
        nmseEnergy = rowEnergy - capturedEnergy;

        R = X - ZL * VL';
        nmseDirect = sum(abs(R).^2, 2);

        nmseEnergyMean(i) = mean(nmseEnergy);
        nmseDirectMean(i) = mean(nmseDirect);

        absDiff(i) = abs(nmseEnergyMean(i) - nmseDirectMean(i));
        relDiff(i) = absDiff(i) / max(abs(nmseDirectMean(i)), eps(class(nmseDirectMean(i))));
        passEach(i) = (absDiff(i) <= absTol) || (relDiff(i) <= relTol);
    end

    check = struct();
    check.cList = cList;
    check.LList = LList;
    check.nmseEnergyMean = nmseEnergyMean;
    check.nmseDirectMean = nmseDirectMean;
    check.absDiff = absDiff;
    check.relDiff = relDiff;
    check.absTol = absTol;
    check.relTol = relTol;
    check.passEach = passEach;
    check.isPass = all(passEach);

    if verbose
        fprintf('[verify_nmse_energy_vs_direct]\n');
        fprintf('  overall pass = %d\n', check.isPass);
        fprintf('  absTol = %.3e, relTol = %.3e\n\n', absTol, relTol);
        fprintf('    c         L      nmse_energy      nmse_direct      absDiff      relDiff\n');
        fprintf('-----------------------------------------------------------------------------\n');
        for i = 1:numel(LList)
            fprintf('%8.5f   %4d   %.12e   %.12e   %.3e   %.3e\n', ...
                cList(i), LList(i), nmseEnergyMean(i), nmseDirectMean(i), absDiff(i), relDiff(i));
        end
    end
end