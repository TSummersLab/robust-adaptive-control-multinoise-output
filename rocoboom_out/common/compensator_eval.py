"""
Functions for linear dynamic compensator evaluation.

"""

from dataclasses import dataclass
import numpy as np


from utility.matrixmath import mdot, specrad, minsv, solveb, lstsqb, dlyap, dare, dare_gain

from rocoboom_out.common.compensator_design import aug_sysmat_cl


def specrad_cl(A, B, C, F, K, L):
    Phi = aug_sysmat_cl(A, B, C, F, K, L)
    return specrad(Phi)


@dataclass
class Performance:
    sr: float  # Spectral radius
    ihc: float  # Infinite horizon cost


def compute_performance(A, B, C, Q, R, W, V, F, K, L, primal=True):
    # Compute the closed-loop spectral radius and performance criterion
    n, m, p = A.shape[0], B.shape[1], C.shape[0]

    Phi = aug_sysmat_cl(A, B, C, F, K, L)
    sr = specrad(Phi)

    if sr > 1:
        ihc = np.inf
    else:
        KRK = np.dot(K.T, np.dot(R, K))
        Qprime = np.block([[Q, np.zeros([n, n])],
                           [np.zeros([n, n]), KRK]])
        LVL = np.dot(L, np.dot(V, L.T))
        Wprime = np.block([[W, np.zeros([n, n])],
                           [np.zeros([n, n]), LVL]])

        # Theoretically these two quantities should be equal if the problem is well-posed.
        if primal:
            # Primal i.e. compute performance criterion from steady-state value matrix
            Pprime = dlyap(Phi.T, Qprime)
            ihc = np.trace(np.dot(Wprime, Pprime))
        else:
            # Dual i.e. compute performance criterion from steady-state covariance matrix
            Sprime = dlyap(Phi, Wprime)
            ihc = np.trace(np.dot(Qprime, Sprime))

    return Performance(sr, ihc)
