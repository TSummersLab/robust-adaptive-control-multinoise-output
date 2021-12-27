"""
Problem data generation.
Generic outputs are:
:param n: Number of states, integer
:param m: Number of inputs, integer
:param p: Number of outputs, integer
:param A: System state matrix, n x n matrix
:param B: System control input matrix, n x m matrix
:param C: System output matrix, p x n matrix
:param Q: State-dependent quadratic cost, n x n matrix
:param R: Control-dependent quadratic cost, m x m matrix
:param W: Additive process noise covariance, n x n matrix
:param V: Additive output noise covariance, p x p matrix
"""

import numpy as np
import numpy.random as npr
import numpy.linalg as la
import scipy.linalg as sla

from utility.matrixmath import specrad, mdot
from utility.pickle_io import pickle_import, pickle_export


def gen_rand_system(n=4, m=3, p=2, spectral_radius=0.9, noise_scale=0.0001, seed=None):
    """
    Generate a random system
    :param n: Number of states, integer
    :param m: Number of inputs, integer
    :param p: Number of outputs, integer
    :param spectral_radius: Open-loop spectral radius of A, float
    :param noise_scale: Scaling of noise covariance, float
    :param seed: Seed for random number generator, positive integer
    :returns: Number of states, number of inputs, state matrix, input matrix, state cost matrix, input cost matrix,
              additive noise covariance matrix
    """

    npr.seed(seed)

    A = npr.randn(n, n)
    A *= spectral_radius/specrad(A)
    B = npr.randn(n, m)
    C = npr.randn(p, n)
    D = np.zeros([p, m])

    # Might want to do this if we assume we dont know the system up front
    Y = np.eye(p)  # Output penalty
    Q = np.dot(C.T, np.dot(Y, C))  # State penalty
    # Q = np.eye(n)
    R = np.eye(m)  # Control penalty

    # We don't need to do this since SIPPY will estimate the noise covariances for us
    # Z = noise_scale*np.eye(m)  # Control noise covariance
    # W = np.dot(B, np.dot(Z, B.T))  # State noise covariance

    W = noise_scale*np.eye(n)  # State noise covariance
    V = noise_scale*np.eye(p)  # Output noise covariance
    U = np.zeros([n, p])  # State-output noise cross-covariance

    return n, m, p, A, B, C, D, Y, Q, R, W, V, U


def gen_scalar_system(A=1.0, B=1.0, C=1.0, D=0.0, Y=1.0, R=1.0, W=1.0, V=1.0, U=0.0):
    n, m, p = 1, 1, 1
    A, B, C, D, Y, R, W, V, U = [np.atleast_2d(var) for var in [A, B, C, D, Y, R, W, V, U]]
    Q = C*Y*C
    return n, m, p, A, B, C, D, Y, Q, R, W, V, U


def gen_pendulum_system(inverted, mass=10, damp=2, dt=0.1, Y=None, R=None, W=None, V=None, U=None):
    # Pendulum with forward Euler discretization
    # x[0] = angular position
    # x[1] = angular velocity

    n = 2
    m = 1
    p = 1
    if inverted:
        sign = 1
    else:
        sign = -1
    A = np.array([[1.0, dt],
                  [sign*mass*dt, 1.0-damp*dt]])
    B = np.array([[0],
                  [dt]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])

    if Y is None:
        Y = np.eye(p)
    Q = np.dot(C.T, np.dot(Y, C))
    if R is None:
        R = np.eye(m)
    if W is None:
        W = 0.001*np.diag([0.0, 1.0])
    if V is None:
        V = 0.001*np.diag([0.1])
    if U is None:
        U = np.zeros([n, p])
    return n, m, p, A, B, C, D, Y, Q, R, W, V, U


def gen_example_system(idx, noise_scale=1.0):
    """
    Example systems
    :param idx: Selection integer to pick the example system.
    """

    if idx == 1:
        # 2-state shift register from https://www.argmin.net/2020/07/27/discrete-fragility/
        # Possibly interesting lack of robustness under CE control
        n, m, p = 2, 1, 1
        A = np.array([[0.0, 1.0],
                      [0.0, 0.0]])
        B = np.array([[0.0],
                      [1.0]])
        C = np.array([[1.0, -1.0]])
        D = np.array([[0.0]])
        Y = np.eye(p)
        Q = np.dot(C.T, np.dot(Y, C))
        R = 0.01*np.eye(m)
        W = 0.1*np.eye(n)
        V = 0.1*np.eye(p)
        U = np.zeros([n, p])

        # A += 0.1*npr.randn(n, n)
        # B += 0.1*npr.randn(n, m)
        # C += 0.1*npr.randn(p, n)

    elif idx == 2:
        # 3-state system from https://arxiv.org/pdf/1710.01688.pdf
        # Possibly interesting lack of robustness under CE control
        n, m, p = 3, 3, 3
        A = np.array([[1.01, 0.01, 0.00],
                      [0.01, 1.01, 0.01],
                      [0.00, 0.01, 1.01]])
        B = np.eye(3)
        C = np.eye(3)
        D = np.zeros([3, 3])
        Y = np.eye(p)
        Q = np.dot(C.T, np.dot(Y, C))
        R = 0.01*np.eye(m)
        W = 0.1*np.eye(n)
        V = 0.1*np.eye(p)
        U = np.zeros([n, p])

    else:
        raise Exception('Invalid system index chosen, please choose a different one')

    W *= noise_scale
    V *= noise_scale
    U *= noise_scale

    return n, m, p, A, B, C, D, Y, Q, R, W, V, U


def gen_system_omni(system_idx, **kwargs):
    """
    Wrapper for system generation functions.
    """
    if system_idx == 'inverted_pendulum':
        return gen_pendulum_system(inverted=True, **kwargs)
    elif system_idx == 'noninverted_pendulum':
        return gen_pendulum_system(inverted=False, **kwargs)
    elif system_idx == 'scalar':
        return gen_scalar_system(**kwargs)
    elif system_idx == 'rand':
        return gen_rand_system(**kwargs)
    else:
        return gen_example_system(idx=system_idx, **kwargs)


def save_system(n, m, p, A, B, C, D, Y, Q, R, W, V, U, dirname_out, filename_out):
    variables = [n, m, p, A, B, C, D, Y, Q, R, W, V, U]
    variable_names = ['n', 'm', 'p', 'A', 'B', 'C', 'D', 'Y', 'Q', 'R', 'W', 'V', 'U']
    system_data = dict(((variable_name, variable) for variable_name, variable in zip(variable_names, variables)))
    pickle_export(dirname_out, filename_out, system_data)


def load_system(filename_in):
    system_data = pickle_import(filename_in)
    variable_names = ['n', 'm', 'p', 'A', 'B', 'C', 'D', 'Y', 'Q', 'R', 'W', 'V', 'U']
    return [system_data[variable] for variable in variable_names]
