"""
Functions for linear dynamic compensator design.
"""

from time import time
from dataclasses import dataclass

import numpy as np

from utility.matrixmath import mdot, specrad, minsv, lstsqb, dlyap, dare, dare_gain
from utility.printing import create_tag

from rocoboom_out.common.gdare import value_iteration, gain


@dataclass
class Compensator:
    """
    Compensator structure: (F, K, L)
    xhat[t+1] = F @ xhat[t] + L @ y[t]
         u[t] = K @ xhat[t]
     """
    F: np.ndarray
    K: np.ndarray
    L: np.ndarray


def sysmat_cl(A, B, C, K, L):
    # Model matrix used by compensator to propagate state
    return A + np.dot(B, K) - np.dot(L, C)


def aug_sysmat_cl(A, B, C, F, K, L):
    # Closed loop system matrix of the joint system of states and state estimates
    return np.block([[A, np.dot(B, K)],
                     [np.dot(L, C), F]])


def make_ce_compensator(model, Y, R):
    # Model information
    A = model.A
    B = model.B
    C = model.C

    W = model.Q
    V = model.R
    U = model.S

    # Penalty information
    Q = np.dot(C.T, np.dot(Y, C))  # Convert output penalty to state penalty in model coordinates

    # Solve Riccati equations to get control and estimator value matrices, gains
    P, K = dare_gain(A, B, Q, R)
    S, L = dare_gain(A.T, C.T, W, V, E=None, S=U)
    L = -L.T

    # Create the model matrix & compensator
    F = sysmat_cl(A, B, C, K, L)
    compensator = Compensator(F, K, L)
    return compensator


def make_compensator(model, uncertainty, Y, R, noise_pre_scale=1.0, noise_post_scale=1.0,
                     bisection_epsilon=0.01, log_diagnostics=False):
    if log_diagnostics:
        time_start = time()

    # Model information
    A = model.A
    B = model.B
    C = model.C

    n, m, p = A.shape[0], B.shape[1], C.shape[0]

    # TODO make sure the scaling is right for these since docs say they are with respect to outputs with unit variance
    # TODO use the cross-covariance in gdare
    W = model.Q
    V = model.R
    U = model.S

    # Penalty information
    Q = np.dot(C.T, np.dot(Y, C))  # Convert output penalty to state penalty in model coordinates

    tag_list = []

    # TODO account for correlation between A, B, C multiplicative noises in gdare
    # TODO include the (estimated) cross-covariance model.S between additive process & measurement noise in the gdare

    def solve_gdare(sysdata, X0=None, solver=None, solver_kwargs=None):
        if solver is None:
            solver = value_iteration
        if solver_kwargs is None:
            solver_kwargs = dict(tol=1e-6, max_iters=400)
        return solver(sysdata, X0, **solver_kwargs)

    if uncertainty is None or noise_post_scale == 0:
        compensator = make_ce_compensator(model, Y, R)
        scale = 0.0
    else:
        # Uncertainty information
        a = uncertainty.a
        Aa = uncertainty.Aa
        b = uncertainty.b
        Bb = uncertainty.Bb
        c = uncertainty.c
        Cc = uncertainty.Cc

        def make_sysdata(scale=1.0):
            # Prep data for GDARE solver
            return dict(A=A, B=B, C=C,  # Mean
                        a=scale*a, Aa=Aa, b=scale*b, Bb=Bb, c=scale*c, Cc=Cc,  # Variance
                        Q=Q, R=R, W=W, V=V, U=U,  # Penalties
                        n=n, m=m, p=p)  # Dimensions
        cs_lwr = 1.0
        scale = cs_lwr*noise_pre_scale
        sysdata = make_sysdata(scale=scale)

        # Warm-start from noiseless case
        # TODO use dare() instead, find expression for P, Phat, S, Shat = X[0], X[1], X[2], X[3]
        X0 = solve_gdare(make_sysdata(scale=0))
        X = solve_gdare(sysdata, X0)

        if X is None:
            # If assumed multiplicative noise variance is too high to admit solution, decrease noise variance
            # Bisection on the noise variance scaling to find the control
            # when the noise just touches the stability boundary
            cs_upr = 1.0
            cs_lwr = 0.0
            while cs_upr - cs_lwr > bisection_epsilon:
                if log_diagnostics:
                    tag_list.append(create_tag("[bisection_lwr    bisection_upr] = [%.6f    %.6f]" % (cs_lwr, cs_upr)))
                cs_mid = (cs_upr + cs_lwr)/2
                scale = cs_mid*noise_pre_scale
                sysdata = make_sysdata(scale=scale)
                X = solve_gdare(sysdata, X0)
                if X is None:
                    cs_upr = cs_mid
                else:
                    cs_lwr = cs_mid
                    X0 = np.copy(X)

            scale = cs_lwr*noise_pre_scale

            if log_diagnostics:
                tag_list.append(create_tag('Scaled noise variance by %.6f' % scale))

            if scale > 0:
                sysdata = make_sysdata(scale=scale)
                X = solve_gdare(sysdata, X0)
                if X is None:
                    if log_diagnostics:
                        tag_list.append(create_tag('GAIN NOT FOUND BY DARE_MULT, INCREASE SOLVER PRECISION',
                                                   message_type='fail'))
                        tag_list.append(create_tag('Falling back on cert-equiv gain', message_type='fail'))
                    compensator = make_ce_compensator(model, Y, R)
                    scale = 0.0
                    return compensator, scale, tag_list
            else:
                if log_diagnostics:
                    tag_list.append(create_tag('Bisection collapsed to cert-equiv'))
                compensator = make_ce_compensator(model, Y, R)
                scale = 0.0
                return compensator, scale, tag_list

        if noise_post_scale < 1:
            scale = cs_lwr*noise_pre_scale*noise_post_scale
            sysdata = make_sysdata(scale=scale)
            X = solve_gdare(sysdata, X0)

        if X is None:
            raise Exception('MLQG problem did not solve, check ms-compensatability! This should not have happened...')

        # P, Phat, S, Shat = X[0], X[1], X[2], X[3]

        # Get gains
        K, L = gain(X, sysdata)

        # Create the model matrix
        F = sysmat_cl(A, B, C, K, L)

        compensator = Compensator(F, K, L)

    if log_diagnostics:
        time_end = time()
        time_elapsed = time_end - time_start
        tag_list.append(create_tag("time to make compensator: %f" % time_elapsed))
    return compensator, scale, tag_list
