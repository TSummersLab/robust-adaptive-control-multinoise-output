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
        R = np.zeros([p, p])
    if S is None:
        S = np.zeros([n, p])
    ss.Q = Q
    ss.R = R
    ss.S = S
    return ss


def ss_change_coordinates(model_tgt, model_src, method='match'):
    # Find a suitable similarity transform matrix P which transforms coordinates from x (source) to xbar (target)
    # i.e. x = P @ xbar

    A = np.asarray(model_tgt.A)
    B = np.asarray(model_tgt.B)
    C = np.asarray(model_tgt.C)
    D = np.asarray(model_tgt.D)

    Abar = np.asarray(model_src.A)
    Bbar = np.asarray(model_src.B)
    Cbar = np.asarray(model_src.C)
    Dbar = np.asarray(model_src.D)

    Qbar = np.asarray(model_src.Q)
    Rbar = np.asarray(model_src.R)
    Sbar = np.asarray(model_src.S)

    # Get sizes
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]

    if method == 'match':
        # Compute by minimizing the error in statespace matrices A, B, C

        weight_A = 1.0
        weight_B = 1.0
        weight_C = 1.0

        # weight_A = 1.0/A.size
        # weight_B = 1.0/B.size
        # weight_C = 1.0/C.size

        # # Solution using CVX
        # P = cvx.Variable((n, n))
        #
        # # Express squared Frobenius norm using Frobenius norm
        # cost =   weight_A*cvx.square(cvx.norm(P@Abar - A@P, 'fro')) \
        #        + weight_B*cvx.square(cvx.norm(P@Bbar - B, 'fro')) \
        #        + weight_C*cvx.square(cvx.norm(Cbar - C@P, 'fro'))
        #
        # # Express squared Frobenius norm directly with sum of squares
        # cost =   weight_A*cvx.sum(cvx.square(P@Abar - A@P)) \
        #        + weight_B*cvx.sum(cvx.square(P@Bbar - B)) \
        #        + weight_C*cvx.sum(cvx.square(Cbar - C@P))
        #
        # objective = cvx.Minimize(cost)
        # constraints = []
        # prob = cvx.Problem(objective, constraints)
        # prob.solve()
        # P = P.value

        # TODO investigate whether this Jacobi-type algorithm can be used to solve the problem more quickly
        #   --seems only valid for systems with full rank C matrix, so we cannot use here
        # https://ieeexplore.ieee.org/document/6669166

        # Solution in closed form via vectorization, Kronecker products (this is a generalized Lyapunov equation)
        I = np.eye(n)
        G = np.kron(weight_A*np.dot(Abar, Abar.T) + weight_B*np.dot(Bbar, Bbar.T), I) \
            + np.kron(I, weight_A*np.dot(A.T, A) + weight_C*np.dot(C.T, C)) \
            - weight_A*np.kron(Abar.T, A.T) - weight_A*np.kron(Abar, A)
        H = weight_B*np.dot(B, Bbar.T) + weight_C*np.dot(C.T, Cbar)
        vH = vec(H)
        vP = la.solve(G, vH)
        P = mat(vP)

        # # DEBUG
        # # Verify solution is a critical point
        # from autograd import grad
        # import autograd.numpy as anp
        #
        # # Manual expression for gradient
        # def g_manual(x):
        #     P = mat(x)
        #
        #     A_term = 2*anp.dot(P, anp.dot(Abar, Abar.T)) - 2*anp.dot(A, anp.dot(P, Abar.T)) - 2*anp.dot(A.T, anp.dot(P, Abar)) + 2*anp.dot(anp.dot(A.T, A), P)
        #     B_term = 2*anp.dot(P, anp.dot(Bbar, Bbar.T)) - 2*anp.dot(B, Bbar.T)
        #     C_term = 2*anp.dot(anp.dot(C.T, C), P) - 2*anp.dot(C.T, Cbar)
        #     return vec(weight_A*A_term + weight_B*B_term + weight_C*C_term)
        #
        # def myobj(x):
        #     P = mat(x)
        #
        #     A_term = anp.sum(anp.square(anp.dot(P, Abar) - anp.dot(A, P)))
        #     B_term = anp.sum(anp.square(anp.dot(P, Bbar) - B))
        #     C_term = anp.sum(anp.square(Cbar - anp.dot(C, P)))
        #     return weight_A*A_term + weight_B*B_term + weight_C*C_term

        # print(myobj(vec(P)))

        # g_auto = grad(myobj)
        #
        # gval1 = g_auto(vec(P))  # should be zero
        # gval2 = g_manual(vec(P))  # should be zero
        # zzz = 0


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
