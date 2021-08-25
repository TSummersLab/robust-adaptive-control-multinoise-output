import sys
import numpy as np
import numpy.linalg as la
from numpy.linalg import pinv
import cvxpy as cvx
import control


from utility.matrixmath import solveb, mat, vec


def groupdot(A, x):
    """
    Perform dot product over groups of matrices,
    suitable for performing many LTI state transitions in a vectorized fashion
    """
    return np.einsum('...ik,...k', A, x)


def distance_between_lti(A1, A2):
    # Definition following Hsu, Hardt, & Hardt 2019 https://arxiv.org/pdf/1908.01039.pdf
    spectrum1 = la.eig(A1)[0]
    spectrum2 = la.eig(A2)[0]
    return la.norm(spectrum1 - spectrum2)


def make_ss(A, B, C, D=None, Q=None, R=None, S=None, dt=1.0,):
    n, m, p = A.shape[0], B.shape[1], C.shape[0]

    if D is None:
        D = np.zeros([p, m])

    ss = control.ss(A, B, C, D, dt)

    if Q is None:
        Q = np.zeros([n, n])
    if R is None:
        R = np.zeros([m, m])
    if S is None:
        S = np.zeros([n, m])
    ss.Q = Q
    ss.R = R
    ss.S = S
    return ss


def ss_change_coordinates(model_tgt, model_src, method='match'):
    # Find a suitable similarity transform matrix P which transforms coordinates from x (source) to xbar (target)
    # i.e. x = P @ xbar

    A = model_tgt.A
    B = model_tgt.B
    C = model_tgt.C
    D = model_tgt.D

    Abar = model_src.A
    Bbar = model_src.B
    Cbar = model_src.C
    Dbar = model_src.D

    Qbar = model_src.Q
    Rbar = model_src.R
    Sbar = model_src.S

    # Get sizes
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    if method == 'match':
        # Compute by minimizing the error in statespace matrices A, B, C
        P = cvx.Variable((n, n))

        # Express squared Frobenius norm using Forbenius norm
        # cost = cvx.square(cvx.norm(P@Abar - A@P, 'fro'))/A.size \
        #        + cvx.square(cvx.norm(P@Bbar - B, 'fro'))/B.size \
        #        + cvx.square(cvx.norm(Cbar - C@P, 'fro'))/C.size

        # Express squared Frobenius norm directly with sum of squares
        cost = cvx.sum(cvx.square(P@Abar - A@P)) \
               + cvx.sum(cvx.square(P@Bbar - B)) \
               + cvx.sum(cvx.square(Cbar - C@P))

        objective = cvx.Minimize(cost)
        constraints = []
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        P = P.value

        # TODO investigate whether this Jacobi-type algorithm can be used to solve the problem more quickly
        # https://ieeexplore.ieee.org/document/6669166

        # # TODO recheck expression, it is not matching the CVX solution...
        # # Solve the problem in closed form via vectorization, Kronecker products
        # I = np.eye(n)
        # G = 2*np.kron(np.dot(Abar, Abar.T), I) \
        #     - np.kron(A.T, Abar.T) \
        #     - np.kron(Abar.T, A.T) \
        #     - np.kron(A, Abar) \
        #     - np.kron(Abar, A) \
        #     + 2*np.kron(I, np.dot(A.T, A)) \
        #     + 2*np.kron(np.dot(Bbar, Bbar.T), I) \
        #     + 2*np.kron(I, np.dot(Cbar.T, Cbar))
        # H = np.dot(B, Bbar.T) + np.dot(Bbar, B.T) + np.dot(Cbar.T, C) + np.dot(C.T, Cbar)
        # vH = vec(H)
        # vP = la.solve(G, vH)
        # P2 = mat(vP)

    elif method in ['reachable', 'observable', 'modal']:
        ss_model_src = make_ss(model_src.A, model_src.B, model_src.C)
        _, P = control.canonical_form(ss_model_src, form=method)
    else:
        raise ValueError('Invalid coordinate transform method!')

    # Apply the transform to all the system matrices
    Ahat = np.dot(P, solveb(Abar, P))
    Bhat = np.dot(P, Bbar)
    Chat = solveb(Cbar, P)
    Dhat = np.copy(Dbar)

    Qhat = np.dot(P, np.dot(Qbar, P.T))
    Rhat = np.copy(Rbar)
    Shat = np.dot(P, Sbar)

    # Create a new ss object from the transformed system matrices
    model_trans = make_ss(Ahat, Bhat, Chat, Dhat, Qhat, Rhat, Shat)

    return model_trans, P
