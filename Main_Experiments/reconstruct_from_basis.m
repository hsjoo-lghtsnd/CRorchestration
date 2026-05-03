function Xhat = reconstruct_from_basis(X, V, c)
%RECONSTRUCT_FROM_BASIS Orthogonal projection reconstruction using first L columns.

    D = size(V, 2);
    L = max(1, floor(D * c));
    L = min(L, D);

    Vc = V(:, 1:L);

    % If V is orthonormal, Vc' is enough.
    % pinv(Vc) is safer if numerical orthogonality is uncertain.
    Xhat = X * Vc * pinv(Vc);
end