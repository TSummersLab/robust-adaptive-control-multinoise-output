import numpy as np
import numpy.random as npr

from rocoboom_out.common.signal_gen import SigParam, make_sig
from rocoboom_out.common.ss_tools import groupdot


def make_exploration(m, Ns, T, u_explore_var):
    """
    Generate exploration control signal

    :param m: Number of inputs
    :param Ns: Number of Monte Carlo samples
    :param T: Number of time steps to simulate
    :param u_explore_var: Control input exploration noise variance
    """
    scale = np.sqrt(u_explore_var)
    # u_hist = scale*npr.randn(Ns, T, m)
    params = [SigParam(method='gbn', mean=0.0, scale=scale, ma_length=None)]
    u_hist = np.array([make_sig(T, m, params) for i in range(Ns)])
    return u_hist


def make_disturbance(n, p, Ns, T, W, V):
    """
    Generate process and measurement disturbances/noise

    :param n: Number of states
    :param p: Number of outputs
    :param Ns: Number of Monte Carlo samples
    :param T: Number of time steps to simulate
    :param W: State noise covariance matrix
    :param V: Output noise covariance matrix
    """

    w_hist = npr.multivariate_normal(np.zeros(n), W, size=(Ns, T))
    v_hist = npr.multivariate_normal(np.zeros(p), V, size=(Ns, T))
    return w_hist, v_hist


def lsim(A, B, C, D, Ns, T, u_hist, w_hist, v_hist, x0=None):
    """
    Simulate multiple state-input-output trajectories of a stochastic linear system
    with additive process and measurement disturbances.

    :param A: State matrix
    :param B: Input matrix
    :param C: Output matrix
    :param D: Direct matrix
    :param Ns: Number of Monte Carlo samples
    :param T: Number of time steps to simulate
    :param u_hist: Control input history
    :param w_hist: Additive process noise history
    :param v_hist: Additive measurement noise history
    :param x0: Initial state
    :returns:
    x_hist: State history
    y_hist: Measurement output history
    """

    n, m = B.shape
    p, n = C.shape

    # Preallocate state, output history
    x_hist = np.zeros([Ns, T+1, n])
    y_hist = np.zeros([Ns, T, p])

    # Initial state
    if x0 is None:
        x0 = np.zeros(n)
    x_hist[:, 0] = x0

    # Loop over time
    for t in range(T):
        # Update state
        x_hist[:, t+1] = groupdot(A, x_hist[:, t]) + groupdot(B, u_hist[:, t]) + w_hist[:, t]
        # Update output
        y_hist[:, t] = groupdot(C, x_hist[:, t]) + groupdot(D, u_hist[:, t]) + v_hist[:, t]

    return x_hist, y_hist


def make_offline_data(A, B, C, D, W, V, Ns, T, u_var, x0=None, verbose=False):
    """
    Generate multiple state-input-output trajectories e.g. to be used as training data for sysid.

    :param A: State matrix
    :param B: Input matrix
    :param C: Output matrix
    :param D: Direct matrix
    :param W: State noise covariance matrix
    :param V: Output noise covariance matrix
    :param Ns: Number of Monte Carlo samples
    :param T: Number of time steps to simulate
    :param u_var: Control input exploration noise variance
    :param x0: Initial state
    :param seed: Seed for NumPy random number generator
    :returns:
    x_hist: State history
    u_hist: Control input history
    y_hist: Measurement output history
    w_hist: Additive process noise history
    v_hist: Additive measurement noise history
    """
    if verbose:
        print("Generating offline sample trajectory data...    ", end='')

    n, m = B.shape
    p, n = C.shape

    u_hist = make_exploration(m, Ns, T, u_var)
    w_hist, v_hist = make_disturbance(n, p, Ns, T, W, V)
    x_hist, y_hist = lsim(A, B, C, D, Ns, T, u_hist, w_hist, v_hist, x0)
    if verbose:
        print("...completed!")
    return x_hist, u_hist, y_hist, w_hist, v_hist


def lsim_cl(ss, compensator, x0, w_hist, v_hist, T):
    A, B, C = np.asarray(ss.A), np.asarray(ss.B), np.asarray(ss.C)
    F, K, L = np.asarray(compensator.F), np.asarray(compensator.K), np.asarray(compensator.L)
    n, m, p = A.shape[0], B.shape[1], C.shape[0]

    x_hist = np.zeros([T+1, n])
    xhat_hist = np.zeros([T+1, n])
    u_hist = np.zeros([T, m])
    y_hist = np.zeros([T, p])

    x_hist[0] = x0
    xhat_hist[0] = x0

    for t in range(T):
        x = x_hist[t]
        xhat = xhat_hist[t]
        w = w_hist[t]
        v = v_hist[t]

        u = np.dot(K, xhat)
        y = np.dot(C, x) + v
        x_next = np.dot(A, x) + np.dot(B, u) + w
        xhat_next = np.dot(F, xhat) + np.dot(L, y)

        x_hist[t+1] = x_next
        xhat_hist[t+1] = xhat_next
        u_hist[t] = u
        y_hist[t] = y
    return x_hist, u_hist, y_hist, xhat_hist
