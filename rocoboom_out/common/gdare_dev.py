# Combined estimator and controller for multiplicative noise LQG

import autograd.numpy as np
import autograd.numpy.linalg as la
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from autograd import grad, jacobian

from utility.matrixmath import specrad


def vec(A):
    """Return the vectorized matrix A by stacking its columns."""
    return A.reshape(-1, order="F")


def mat(v, shape=None):
    """Return matricization i.e. the inverse operation of vec of vector v."""
    if shape is None:
        shape = (4, n, n)
    return np.reshape(v, shape, order='F')


def vmat(v, shape=None):
    """Return matricization i.e. the inverse operation of vec of vector v."""
    if shape is None:
        dim = int(np.sqrt(v.size))
        shape = dim, dim
    matrix = v.reshape(shape[1], shape[0]).T
    return matrix


def min_eig(X):
    return np.min(la.eig(X)[0])


def diff(X, Y):
    return la.norm(X-Y, ord=2, axis=(1, 2))


def rand_psd(d, s=None):
    # Generate a random d x d positive semidefinite matrix
    E = np.diag(npr.rand(d))
    U = npr.randn(d, d)
    V = la.qr(U)[0]
    P = V.dot(E.dot(V.T))
    if s is not None:
        P *= s/specrad(P)
    return P


def make_noise_data(Sigma, d1, d2):
    d12 = d1*d2
    a, V = la.eig(Sigma)
    Aa = np.zeros([d12, d1, d2])
    for i in range(d12):
        Aa[i] = V[:, i].reshape(d1, d2)
    return a, Aa


def make_system_data(n=4, m=3, p=2, r=0.99, s=1.0):
    # nominal system matrices
    A = npr.randn(n, n)
    A *= r/specrad(A)
    B = npr.randn(n, m)
    C = npr.randn(p, n)

    # Multiplicative noise variances
    SigmaA = rand_psd(n*n, s=0.001*s)  # State multiplicative noise covariance
    SigmaB = rand_psd(n*m, s=0.001*s)  # Input multiplicative noise covariance
    SigmaC = rand_psd(n*p, s=0.001*s)  # Output multiplicative noise covariance
    a, Aa = make_noise_data(SigmaA, n, n)
    b, Bb = make_noise_data(SigmaB, n, m)
    c, Cc = make_noise_data(SigmaC, p, n)

    # Penalty matrices
    Q = 1.0*np.eye(n)  # state cost
    R = 1.0*np.eye(m)  # action cost

    # Noise variances
    W = 0.1*np.eye(n)  # process noise
    V = 0.1*np.eye(p)  # sensor noise

    sysdata = dict(A=A, B=B, C=C, a=a, Aa=Aa, b=b, Bb=Bb, c=c, Cc=Cc, Q=Q, R=R, W=W, V=V, n=n, m=m, p=p)
    return sysdata


def example_sysdata(beta=0.1):
    # de Koning 1992 example
    # beta = 0.0 deterministic case, beta = 0.2 ms-compensatable, beta = 0.3 not ms-compensatable

    n = 2
    m = 1
    p = 1
    A = np.array([[0.7092, 0.3017],
                  [0.1814, 0.9525]])
    B = np.array([[0.7001],
                  [0.1593]])
    C = np.array([[0.3088, 0.5735]])

    a = np.copy(beta)[None]
    Aa = np.copy(A)[None]
    b = np.copy(beta)[None]
    Bb = np.copy(B)[None]
    c = np.copy(beta)[None]
    Cc = np.copy(C)[None]

    Q = np.diag([0.7350, 0.9820])
    R = 0.6644*np.eye(m)
    W = np.diag([0.5627, 0.7357])
    V = 0.2588*np.eye(p)

    sysdata = dict(A=A, B=B, C=C, a=a, Aa=Aa, b=b, Bb=Bb, c=c, Cc=Cc, Q=Q, R=R, W=W, V=V, n=n, m=m, p=p)
    return sysdata


def unpack(sysdata):
    return [sysdata[key] for key in ['A', 'B', 'C', 'a', 'Aa', 'b', 'Bb', 'c', 'Cc', 'Q', 'R', 'W', 'V', 'n', 'm', 'p']]


def qfun(X):
    A, B, C, a, Aa, b, Bb, c, Cc, Q, R, W, V, n, m, p = unpack(sysdata)
    X1, X2, X3, X4 = [X[i] for i in range(4)]

    # Control Q-function (G)
    # Penalty terms
    Gxx = np.copy(Q)
    Gxu = np.zeros([n, m])
    Gux = np.zeros([m, n])
    Guu = np.copy(R)

    # Mean terms
    Gxx += np.dot(A.T, np.dot(X1, A))
    Gxu += np.dot(A.T, np.dot(X1, B))
    Gux += np.dot(B.T, np.dot(X1, A))
    Guu += np.dot(B.T, np.dot(X1, B))

    # Noise terms
    for i in range(a.size):
        ai, Ai = a[i], Aa[i]
        Gxx += ai*np.dot(Ai.T, np.dot(X1, Ai))
        Gxx += ai*np.dot(Ai.T, np.dot(X2, Ai))
    for i in range(b.size):
        bi, Bi = b[i], Bb[i]
        Guu += bi*np.dot(Bi.T, np.dot(X1, Bi))
        Guu += bi*np.dot(Bi.T, np.dot(X2, Bi))

    # Estimator Q-function (H)
    # Penalty terms
    Hxx = np.copy(W)
    Hxy = np.zeros([n, p])
    Hyx = np.zeros([p, n])
    Hyy = np.copy(V)

    # Mean terms
    Hxx += np.dot(A, np.dot(X3, A.T))
    Hxy += np.dot(A, np.dot(X3, C.T))
    Hyx += np.dot(C, np.dot(X3, A.T))
    Hyy += np.dot(C, np.dot(X3, C.T))

    # Noise terms
    # TODO replace loops with einsum, similar to ricc()
    for i in range(a.size):
        ai, Ai = a[i], Aa[i]
        Hxx += ai*np.dot(Ai, np.dot(X3, Ai.T))
        Hxx += ai*np.dot(Ai, np.dot(X4, Ai.T))
    for i in range(c.size):
        ci, Ci = c[i], Cc[i]
        Hyy += ci*np.dot(Ci, np.dot(X3, Ci.T))
        Hyy += ci*np.dot(Ci, np.dot(X4, Ci.T))

    G = np.block([[Gxx, Gxu],
                  [Gux, Guu]])
    H = np.block([[Hxx, Hxy],
                  [Hyx, Hyy]])
    return G, H


def gain(X):
    n, m, p = [sysdata[key] for key in ['n', 'm', 'p']]
    G, H = qfun(X)
    Gxx = G[0:n, 0:n]
    Gxu = G[0:n, n:n+m]
    Gux = G[n:n+m, 0:n]
    Guu = G[n:n+m, n:n+m]

    Hxx = H[0:n, 0:n]
    Hxy = H[0:n, n:n+p]
    Hyx = H[n:n+p, 0:n]
    Hyy = H[n:n+p, n:n+p]

    # Compute gains
    K = -la.solve(Guu, Gux)  # Control gain  u = K*x
    L = la.solve(Hyy, Hyx).T  # Estimator gain  xhat = A*x + B*u + L*(y - C*xhat)
    return K, L


def ricc(X, sysdata):
    # Riccati operator for multiplicative noise LQG
    # See W.L. de Koning, TAC 1992  https://ieeexplore.ieee.org/document/135491
    A, B, C, a, Aa, b, Bb, c, Cc, Q, R, W, V, n, m, p = unpack(sysdata)

    # Control Q-function (G)
    # Mean terms
    Gxu = np.dot(A.T, np.dot(X[0], B))
    Gux = np.dot(B.T, np.dot(X[0], A))
    Guu = R + np.dot(B.T, np.dot(X[0], B)) \
          + np.einsum('x,xji,jk,xkl->il', b, Bb, X[0], Bb) \
          + np.einsum('x,xji,jk,xkl->il', b, Bb, X[1], Bb)

    # Estimator Q-function (H)
    # Mean terms
    Hxy = np.dot(A, np.dot(X[2], C.T))
    Hyx = np.dot(C, np.dot(X[2], A.T))
    Hyy = V + np.dot(C, np.dot(X[2], C.T)) \
          + np.einsum('x,xij,jk,xlk->il', c, Cc, X[2], Cc) \
          + np.einsum('x,xij,jk,xlk->il', c, Cc, X[3], Cc)

    # Compute gains
    K = -la.solve(Guu, Gux)  # Control gain  u = K*x
    L = la.solve(Hyy, Hyx).T  # Estimator gain  xhat = A*x + B*u + L*(y - C*xhat)

    # Closed-loop system matrices
    ABK = A + np.dot(B, K)
    ALC = A - np.dot(L, C)

    LX2L = np.dot(L.T, np.dot(X[1], L))
    KX4K = np.dot(K, np.dot(X[3], K.T))

    Gxx = Q + np.dot(A.T, np.dot(X[0], A)) \
          + np.einsum('x,xji,jk,xkl->il', a, Aa, X[0], Aa) \
          + np.einsum('x,xji,jk,xkl->il', a, Aa, X[1], Aa) \
          + np.einsum('x,xji,jk,xkl->il', c, Cc, LX2L, Cc)

    Hxx = W + np.dot(A, np.dot(X[2], A.T)) \
          + np.einsum('x,xij,jk,xlk->il', a, Aa, X[2], Aa) \
          + np.einsum('x,xij,jk,xlk->il', a, Aa, X[3], Aa) \
          + np.einsum('x,xij,jk,xlk->il', b, Bb, KX4K, Bb)

    # Form the RHS
    Z1 = np.dot(Gxu, la.solve(Guu, Gux))
    Z3 = np.dot(Hxy, la.solve(Hyy, Hyx))

    Y1 = Gxx - Z1
    Y2 = np.dot(ALC.T, np.dot(X[1], ALC)) + Z1
    Y3 = Hxx - Z3
    Y4 = np.dot(ABK, np.dot(X[3], ABK.T)) + Z3

    Y = np.stack([Y1, Y2, Y3, Y4])
    return Y


def gdlyap(sysdata, K, L, primal=True, solver='direct', check_stable=True, P0=None, max_iters=1000):
    """
    (G)eneralized (d)iscrete-time (Lyap)unov equation solver for input-state-output systems with multiplicative noise

    sysdata: dict, with all problem data for input-state-output systems with multiplicative noise
    K, L: matrices, control and estimator gains
    primal: bool, True for the 'prime' problem which yields the cost/value matrix,
                  False for the 'dual' problem which yields the steady-state second moment matrix
    solver: string, 'direct' is a one-shot solver that turns the matrix equation
                      into a vector system of equations and solves using generic linear system solver in numpy.linalg.
                    'smith' is a basic iterative solver, requires an initial guess and termination criterion,
                      most useful for large n, but convergence speed depends on the spectral radius.
                      Best results when warm-starting near the solution.
    check_stable: bool, True will check whether the closed-loop system is ms-stable
                    and return a matrix of np.inf if not ms-stable.
                    Only set to False if it is known beforehand that (K, L) is ms-stabilizing.
    P0: matrix, initial guess for Smith method
    max_iters: int, maximum number of iterations for Smith method
    """

    # Extract problem data
    A, B, C, a, Aa, b, Bb, c, Cc, Q, R, W, V, n, m, p = unpack(sysdata)

    # Intermediate quantities
    BK = np.dot(B, K)
    LC = np.dot(L, C)
    F = A + BK - LC

    if solver == 'direct':
        Phi = np.block([[A, BK],
                        [LC, F]])
        zA = np.zeros_like(A)
        zBK = np.zeros_like(BK)
        zLC = np.zeros_like(LC)
        zF = np.zeros_like(F)

        if primal:
            # Build the closed-loop quadratic cost transition operator
            PhiPhi = np.kron(Phi.T, Phi.T)
            for i in range(a.size):
                PhiAa = np.block([[Aa[i], zBK],
                                  [zLC, zF]])
                PhiPhi += a[i]*np.kron(PhiAa.T, PhiAa.T)
            for i in range(b.size):
                PhiBb = np.block([[zA, np.dot(Bb[i], K)],
                                  [zLC, zF]])
                PhiPhi += b[i]*np.kron(PhiBb.T, PhiBb.T)
            for i in range(c.size):
                PhiCc = np.block([[zA, zBK],
                                  [np.dot(L, Cc[i]), zF]])
                PhiPhi += c[i]*np.kron(PhiCc.T, PhiCc.T)
            # Build the penalty matrix
            Qprime = np.block([[Q, np.zeros([n, n])],
                               [np.zeros([n, n]), np.dot(K.T, np.dot(R, K))]])
            vQprime = vec(Qprime)
            # Solve
            if check_stable:
                r = specrad(PhiPhi)
                if r > 1:
                    return np.full((2*n, 2*n), np.inf)
            vP = la.solve(np.eye((2*n)*(2*n)) - PhiPhi, vQprime)
            P = vmat(vP)
            return P
        else:
            # Build the closed-loop second moment transition operator
            PhiPhi = np.kron(Phi, Phi)
            for i in range(a.size):
                PhiAa = np.block([[Aa[i], zBK],
                                  [zLC, zF]])
                PhiPhi += a[i]*np.kron(PhiAa, PhiAa)
            for i in range(b.size):
                PhiBb = np.block([[zA, np.dot(Bb[i], K)],
                                  [zLC, zF]])
                PhiPhi += b[i]*np.kron(PhiBb, PhiBb)
            for i in range(c.size):
                PhiCc = np.block([[zA, zBK],
                                  [np.dot(L, Cc[i]), zF]])
                PhiPhi += c[i]*np.kron(PhiCc, PhiCc)
            # Build the penalty matrix
            Wprime = np.block([[W, np.zeros([n, n])],
                               [np.zeros([n, n]), np.dot(L, np.dot(V, L.T))]])
            vWprime = vec(Wprime)
            # Solve
            if check_stable:
                r = specrad(PhiPhi)
                if r > 1:
                    return np.full((2*n, 2*n), np.inf)
            vS = la.solve(np.eye((2*n)*(2*n)) - PhiPhi, vWprime)
            S = vmat(vS)
            return S

    elif solver == 'smith':
        # Initialize
        if P0 is None:
            P = np.zeros([n + n, n + n])
        else:
            P = np.copy(P0)

        # Form the right-hand side to iterate as a fixed-point operator
        if primal:
            def rhs(P):
                # Extract blocks of P
                P11 = P[0:n, 0:n]
                P12 = P[0:n, n:n + n]
                P21 = P[n:n + n, 0:n]
                P22 = P[n:n + n, n:n + n]

                LP22L = np.dot(L.T, np.dot(P22, L))

                # Construct the rhs as block matrix X
                X11_1 = np.dot(A.T, np.dot(P11, A)) + np.einsum('x,xji,jk,xkl->il', a, Aa, P11, Aa)
                X11_2 = np.dot(A.T, np.dot(P12, LC))
                X11_3 = X11_2.T
                X11_4 = np.dot(C.T, np.dot(LP22L, C)) + np.einsum('x,xji,jk,xkl->il', c, Cc, LP22L, Cc)

                X12_1 = np.dot(A.T, np.dot(P11, BK))
                X12_2 = np.dot(A.T, np.dot(P12, F))
                X12_3 = np.dot(LC.T, np.dot(P21, BK))
                X12_4 = np.dot(LC.T, np.dot(P22, F))

                X22_1 = np.dot(BK.T, np.dot(P11, BK)) + np.dot(K.T,
                                                               np.dot(np.einsum('x,xji,jk,xkl->il', b, Bb, P11, Bb), K))
                X22_2 = np.dot(BK.T, np.dot(P12, F))
                X22_3 = X22_2.T
                X22_4 = np.dot(F.T, np.dot(P22, F))

                X11 = Q + X11_1 + X11_2 + X11_3 + X11_4
                X12 = X12_1 + X12_2 + X12_3 + X12_4
                X21 = X12.T
                X22 = np.dot(K.T, np.dot(R, K)) + X22_1 + X22_2 + X22_3 + X22_4

                X = np.block([[X11, X12],
                              [X21, X22]])
                return X
        else:
            def rhs(S):
                # Extract blocks of S
                S11 = S[0:n, 0:n]
                S12 = S[0:n, n:n + n]
                S21 = S[n:n + n, 0:n]
                S22 = S[n:n + n, n:n + n]

                # LP22L = np.dot(L.T, np.dot(S22, L))
                KS22K = np.dot(K, np.dot(S22, K.T))

                # Construct the rhs as block matrix X
                X11_1 = np.dot(A, np.dot(S11, A.T)) + np.einsum('x,xij,jk,xlk->il', a, Aa, S11, Aa)
                X11_2 = np.dot(A, np.dot(S12, BK.T))
                X11_3 = X11_2.T
                X11_4 = np.dot(B, np.dot(KS22K, B.T)) + np.einsum('x,xij,jk,xlk->il', b, Bb, KS22K, Bb)

                X12_1 = np.dot(A, np.dot(S11, LC.T))
                X12_2 = np.dot(A, np.dot(S12, F.T))
                X12_3 = np.dot(BK, np.dot(S21, LC.T))
                X12_4 = np.dot(BK, np.dot(S22, F.T))

                X22_1 = np.dot(LC, np.dot(S11, LC.T)) + np.dot(L, np.dot(np.einsum('x,xij,jk,xlk->il', c, Cc, S11, Cc),
                                                                         L.T))
                X22_2 = np.dot(LC, np.dot(S12, F.T))
                X22_3 = X22_2.T
                X22_4 = np.dot(F, np.dot(S22, F.T))

                X11 = W + X11_1 + X11_2 + X11_3 + X11_4
                X12 = X12_1 + X12_2 + X12_3 + X12_4
                X21 = X12.T
                X22 = np.dot(L, np.dot(V, L.T)) + X22_1 + X22_2 + X22_3 + X22_4

                X = np.block([[X11, X12],
                              [X21, X22]])
                return X

        # Iterate
        for i in range(max_iters):
            P = rhs(P)
        return P


def value(K, L, sysdata, *args, **kwargs):
    P = gdlyap(sysdata, K, L, primal=True, *args, **kwargs)
    S = gdlyap(sysdata, K, L, primal=False, *args, **kwargs)

    # Build value matrices for arbitrary compensator ( Z == X if everything works properly)
    if np.all(np.isfinite(P)) and np.all(np.isfinite(S)):
        X = np.zeros((4, n, n))
        X[0] = P[0:n, 0:n] - P[n:n + n, n:n + n]
        X[1] = P[n:n + n, n:n + n]
        X[2] = S[0:n, 0:n] - S[n:n + n, n:n + n]
        X[3] = S[n:n + n, n:n + n]
    else:
        X = np.full((4, n, n), np.inf)
    return X


def cost(K, L, sysdata, primal=True):
    X = value(K, L, sysdata)

    # Performance criterion c computed using primal True and False are equal if everything works properly
    if primal:
        c = np.trace(np.dot(Q, X[2]) + np.dot(Q + np.dot(K.T, np.dot(R, K)), X[3]))
    else:
        c = np.trace(np.dot(W, X[0]) + np.dot(W + np.dot(L, np.dot(V, L.T)), X[1]))

    return c


def get_initial_gains(sysdata, method='perturb_are', cost_factor=10.0, scale_factor=1.01):
    A, B, C, a, Aa, b, Bb, c, Cc, Q, R, W, V, n, m, p = unpack(sysdata)

    if method == 'perturb_are':
        # Get initial gains by perturbing the Riccati solution found by value iteration

        # Get optimal gains by value iteration
        X0 = np.stack([np.zeros([n, n]), np.zeros([n, n]), np.zeros([n, n]), np.zeros([n, n])])
        X = value_iteration(X0, sysdata)
        Kopt, Lopt = gain(X)

        # Random perturbation directions with unit Frobenius norm
        Kp = npr.randn(*Kopt.shape)
        Kp /= la.norm(Kp, ord='fro')
        Lp = npr.randn(*Lopt.shape)
        Lp /= la.norm(Lp, ord='fro')

        # Initial perturbation scale
        scale = 0.01

        K = Kopt + scale*Kp
        L = Lopt + scale*Lp

        c_opt = cost(Kopt, Lopt, sysdata)
        c = cost(K, L, sysdata)

        # Increase perturbation scale until cost is worse than the optimum by prescribed cost_factor
        while c < cost_factor*c_opt:
            scale *= scale_factor
            K = Kopt + scale*Kp
            L = Lopt + scale*Lp
            c = cost(K, L, sysdata)
        # Take one step back to ensure cost_factor is an upper bound
        scale /= scale_factor
        K = Kopt + scale*Kp
        L = Lopt + scale*Lp

    elif method == 'zero':
        K = np.zeros([m, n])
        L = np.zeros([n, p])
    else:
        raise ValueError

    return K, L


def value_iteration(X, sysdata, tol=1e-9, max_iters=1000, verbose=False):
    i = 0
    X_prev = np.copy(X)
    while True:
        X = ricc(X_prev, sysdata)
        d = diff(X, X_prev)
        if verbose:
            spacer = '  '
            print('%6d' % i, end=spacer)
            print(d, end=spacer)  # Riccati residual
            # print(np.sort(np.real(la.eig(X_prev[0] - X[0])[0])))  # eigs of X diff from last iter
            print(cost(*gain(X), sysdata))
        if np.all(d < tol):
            return X
        if i >= max_iters:
            return None
        X_prev = np.copy(X)
        i += 1


def f(x):
    X = mat(x)
    return vec(ricc(X) - X)

g = jacobian(f)


def newton(X, tol=1e-9, max_iters=1000, verbose=False):
    x = vec(X)
    i = 0
    while True:
        J = g(x)
        x -= la.solve(J, f(x))
        X = mat(x)
        d = diff(X, ricc(X))
        if verbose:
            spacer = '  '
            print('%6d' % i, end=spacer)
            print(d)  # Riccati residual
            # print(np.sort(np.real(la.eig(X_prev[0]-X[0])[0])))  # eigs of X diff from last iter
        if np.all(d < tol):
            return X
        if i >= max_iters:
            return None
        i += 1


def policy_iteration(X, tol=1e-9, max_iters=1000, verbose=False):
    # TODO check the equations for this, does not seem to be the same as Newton method
    #  (might be that Newton method does not coincide with policy iteration?)
    # Check the policy improvement step, seems like it should be coupled
    i = 0
    while True:
        K, L = gain(X)
        X = value(K, L)
        d = diff(X, ricc(X))
        if verbose:
            spacer = '  '
            print('%6d' % i, end=spacer)
            print(d)  # Riccati residual
            # print(np.sort(np.real(la.eig(X_prev[0]-X[0])[0])))  # eigs of X diff from last iter
        if np.all(d < tol):
            return X
        if i >= max_iters:
            return None
        i += 1


def rollout(K, L, x0, xhat0, T=1000):
    # Preallocate
    x = np.zeros([T+1, n])
    u = np.zeros([T, m])
    y = np.zeros([T, p])
    xhat = np.zeros([T+1, n])

    # Initialize
    x[0] = x0
    xhat[0] = xhat0

    # Simulate
    for t in range(T):
        # Noises
        At = A + np.einsum('i,ijk', np.sqrt(a)*npr.randn(a.size), Aa)
        Bt = B + np.einsum('i,ijk', np.sqrt(b)*npr.randn(b.size), Bb)
        Ct = C + np.einsum('i,ijk', np.sqrt(c)*npr.randn(c.size), Cc)
        w = npr.multivariate_normal(np.zeros(n), W)
        v = npr.multivariate_normal(np.zeros(p), V)

        # Simulation step
        y[t] = np.dot(Ct, x[t]) + v
        u[t] = np.dot(K, xhat[t])

        xhat[t+1] = np.dot(A, xhat[t]) + np.dot(B, u[t]) + np.dot(L, y[t] - np.dot(C, xhat[t]))
        x[t+1] = np.dot(At, x[t]) + np.dot(Bt, u[t]) + w
    return x, u, y, xhat


if __name__ == "__main__":
    plt.close('all')
    # plt.style.use('utility/conlab.mplstyle')

    # This makes policy iteration fail to converge, even though value iteration converges!
    seed = 1
    npr.seed(seed)
    sysdata = make_system_data(n=4, m=3, p=2, r=2.0, s=1.0)

    # # Random problem data
    # seed = 1
    # npr.seed(seed)
    # sysdata = make_system_data(n=4, m=3, p=2, r=1.9, s=0.0)

    # # Example problem data
    # seed = 1
    # npr.seed(seed)
    # sysdata = example_sysdata(beta=0.2)

    A, B, C, a, Aa, b, Bb, c, Cc, Q, R, W, V, n, m, p = unpack(sysdata)

    K0, L0 = get_initial_gains(sysdata, cost_factor=100.0)

    # Initialize value matrix
    # X0 = np.stack([np.eye(n), np.eye(n), np.eye(n), np.eye(n)])
    # X0 = np.zeros((4, n, n))
    X0 = value(K0, L0, sysdata)  # Initialize with value matrices from an initial ms-stabilizing compensator

    # Value iteration
    print('value iteration')
    X = value_iteration(X0, sysdata, verbose=True)
    print('')

    # # Newton method - Policy iteration?
    # print('newton method')
    # Y = newton(X0, verbose=True)
    # print('')

    # print('policy_iteration')
    # Z = policy_iteration(X0, verbose=True)
    # print('')


    # # Plot the value matrices
    # fig, ax = plt.subplots(ncols=4, figsize=(12, 4))
    # for i in range(4):
    #     ax[i].imshow(X[i])

    # Compute gains
    K, L = gain(X)
    BK = np.dot(B, K)
    LC = np.dot(L, C)
    F = A + BK - LC

    # TODO: check solution (also compare with special case of Bernstein + Haddad IJC 1987)
    # TODO: compare to separated solution, find an example where separated solution is ms-unstable?


    print(cost(K0, L0, sysdata))
    print(cost(K, L, sysdata))

    # Simulation

    # Initial state and state estimate
    x0 = 10*(npr.choice([2, -2]) + 2*npr.rand(n) - 1)
    # xhat0 = 10*(2*npr.rand(n) - 1)
    xhat0 = np.copy(x0)

    # Set up plot
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)

    # Closed-loop response w/ optimal gains
    x, u, y, xhat = rollout(K, L, x0, xhat0, T=1000)

    ax[0].plot(x, alpha=0.5)
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('states')

    # Closed-loop response w/ initial gains
    x, u, y, xhat = rollout(K0, L0, x0, xhat0, T=1000)

    ax[1].plot(x, alpha=0.5)
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('states')
    fig.tight_layout()
