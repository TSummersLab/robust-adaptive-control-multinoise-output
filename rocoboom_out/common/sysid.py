# Copied & modified from SIPPY package


import sys
import numpy as np

from sippy.OLSims_methods import SS_model, check_types, check_inputs, extracting_matrices, forcing_A_stability, \
    SVD_weighted, algorithm_1, SS_lsim_process_form
from sippy.functionsetSIM import rescale, old_div, Vn_mat, K_calc


def system_identification(y, u, id_method='N4SID',
                          tsample=1.0, SS_f=None,  SS_threshold=0.1,
                          SS_max_order=np.NaN, SS_fixed_order=np.NaN,
                          SS_D_required=False, SS_A_stability=False,
                          return_residuals=False):

    y = 1.0 * np.atleast_2d(y)
    u = 1.0 * np.atleast_2d(u)
    [n1, n2] = y.shape
    ydim = min(n1, n2)
    ylength = max(n1, n2)
    if ylength == n1:
        y = y.T
    [n1, n2] = u.shape
    ulength = max(n1, n2)
    udim = min(n1, n2)
    if ulength == n1:
        u = u.T

    # Checking data consistency
    if ulength != ylength:
        sys.stdout.write("\033[0;35m")
        print("Warning! y and u lengths are not the same. The minor value between the two lengths has been chosen. The performed indentification may be not correct, be sure to check your input and output data alignement")
        sys.stdout.write(" ")
        # Recasting data cutting out the over numbered data
        minlength = min(ulength, ylength)
        y = y[:, :minlength]
        u = u[:, :minlength]

    if SS_f is None:
        if np.isfinite(SS_fixed_order):
            SS_f = SS_fixed_order

    # if np.isfinite(SS_fixed_order):
    #     if SS_f < SS_fixed_order:
    #         print("Warning! The horizon length has been chosen as less than the system order n. "
    #               "Recommend increasing so SS_f >= n!")

    A, B, C, D, Vn, Q, R, S, K, res = OLSims(y, u, SS_f, id_method, SS_threshold,
                                                               SS_max_order, SS_fixed_order,
                                                               SS_D_required, SS_A_stability)
    model = SS_model(A, B, C, D, K, Q, R, S, tsample, Vn)

    if return_residuals:
        return model, res
    else:
        return model


def OLSims(y, u, f, weights='N4SID', threshold=0.1, max_order=np.NaN, fixed_order=np.NaN,
           D_required=False, A_stability=False):
    y = 1. * np.atleast_2d(y)
    u = 1. * np.atleast_2d(u)
    l, L = y.shape
    m = u[:, 0].size
    if not check_types(threshold, max_order, fixed_order, f):
        return np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]]), np.array(
            [[0.0]]), np.inf, [], [], [], []
    else:
        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)  # threshold, max_order = 0, fixed_order
        N = L - 2*f + 1

        # # This chunk enables standardization of the input and output data
        # Ustd = np.zeros(m)
        # Ystd = np.zeros(l)
        # for j in range(m):
        #     Ustd[j], u[j] = rescale(u[j])
        # for j in range(l):
        #     Ystd[j], y[j] = rescale(y[j])

        # This chunk disables standardization of the input and output data
        Ustd = np.ones(m)
        Ystd = np.ones(l)
        # for j in range(m):
        #     Ustd[j], u[j] = rescale(u[j])
        # for j in range(l):
        #     Ystd[j], y[j] = rescale(y[j])

        U_n, S_n, V_n, W1, O_i = SVD_weighted(y, u, f, l, weights)
        Ob, X_fd, M, n, residuals = algorithm_1(y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, threshold,
                                                max_order, D_required)
        if A_stability:
            M, residuals[0:n, :], useless = forcing_A_stability(M, n, Ob, l, X_fd, N, u, f)
        A, B, C, D = extracting_matrices(M, n)
        Covariances = old_div(np.dot(residuals, residuals.T), (N - 1))
        Q = Covariances[0:n, 0:n]
        R = Covariances[n:, n:]
        S = Covariances[0:n, n:]
        X_states, Y_estimate = SS_lsim_process_form(A, B, C, D, u)

        Vn = Vn_mat(y, Y_estimate)

        K, K_calculated = K_calc(A, C, Q, R, S)
        for j in range(m):
            B[:, j] = old_div(B[:, j], Ustd[j])
            D[:, j] = old_div(D[:, j], Ustd[j])
        for j in range(l):
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
            if K_calculated:
                K[:, j] = old_div(K[:, j], Ystd[j])
        return A, B, C, D, Vn, Q, R, S, K, residuals
