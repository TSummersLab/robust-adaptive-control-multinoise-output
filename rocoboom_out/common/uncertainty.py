import control
import numpy as np
import numpy.linalg as la
import numpy.random as npr

from utility.matrixmath import mdot

from rocoboom_out.common.sysid import system_identification
from rocoboom_out.common.ss_tools import ss_change_coordinates, groupdot


class Uncertainty:
    def __init__(self, a=None, Aa=None, b=None, Bb=None, c=None, Cc=None):
        self.a = a
        self.Aa = Aa
        self.b = b
        self.Bb = Bb
        self.c = c
        self.Cc = Cc


def block_bootstrap(u_hist, y_hist, t=None, Nb=None, blocksize=10):
    # Get sizes
    p = y_hist.shape[1]
    m = u_hist.shape[1]

    # Get time horizon
    if t is None:
        t = u_hist.shape[0]

    # Preallocate bootstrap sample arrays
    u_boot_hist = np.zeros([Nb, t, m])
    y_boot_hist = np.zeros([Nb, t, p])

    if blocksize > t:
        raise ValueError('Blocksize exceeds data length, reduce blocksize!')
    for i in range(Nb):
        # Sample blocks of i/o data iid with replacement until the buffer is filled out
        start = 0
        end = start + blocksize
        while end < t:
            if start + blocksize < t:
                end = start + blocksize
            else:
                end = t
            idx = npr.randint(t-blocksize)+np.arange(blocksize)
            u_boot_hist[i, start:end] = u_hist[idx]
            y_boot_hist[i, start:end] = y_hist[idx]

            start = end
    return u_boot_hist, y_boot_hist


def semiparametric_bootstrap(model, u_hist, y_hist, w_hist, v_hist, t=None, Nb=None):
    """
    Compute estimate of model uncertainty (covariance) via semiparametric bootstrap
    :param model: object, nominal model with attributes A matrix, B matrix, C matrix
    :param t: int, time up to which to use the available data.
    :param Nb: int, number of bootstrap samples
    """
    A = model.A
    B = model.B
    C = model.C
    # Get sizes
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    # Get time horizon
    if t is None:
        t = u_hist.shape[0]

    # Preallocate bootstrap sample arrays
    x_boot_hist = np.zeros([Nb, t+1, n])
    u_boot_hist = np.zeros([Nb, t, m])
    y_boot_hist = np.zeros([Nb, t, p])
    w_boot_hist = np.zeros([Nb, t, n])
    v_boot_hist = np.zeros([Nb, t, p])

    # Initialize bootstrap training data
    for i in range(Nb):
        # Initialize state
        x_boot_hist[i, 0] = np.zeros(n)

        # Copy input sequence
        u_boot_hist[i] = u_hist[0:t]

        # Sample residuals iid with replacement
        # TODO should we use different idx for process and sensor residuals/noises?
        idx = npr.randint(w_hist.shape[0], size=t)
        w_boot_hist[i] = w_hist[idx]
        v_boot_hist[i] = v_hist[idx]

    # Form bootstrap training data
    for t_samp in range(t):
        # Update state
        x_boot_hist[:, t_samp+1] = (groupdot(A, x_boot_hist[:, t_samp])
                                    + groupdot(B, u_boot_hist[:, t_samp])
                                    + w_boot_hist[:, t_samp])
        # Update output
        y_boot_hist[:, t_samp] = (groupdot(C, x_boot_hist[:, t_samp])
                                  + v_boot_hist[:, t_samp])
    return u_boot_hist, y_boot_hist


def check_diff(model1, model2):
    A1, B1, C1 = model1.A, model1.B, model1.C
    A2, B2, C2 = model2.A, model2.B, model2.C
    print(A1)
    print(A2)
    print('')
    print(B1)
    print(B2)
    print('')
    print(C1)
    print(C2)
    print('')


def ensemble2multnoise(model, u_boot_hist, y_boot_hist, return_models=False, verbose=False):
    """Convert an ensemble of data histories to a multiplicative noise representation."""
    A = model.A
    B = model.B
    C = model.C
    # Get sizes
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    Nb = u_boot_hist.shape[0]

    # Form bootstrap model estimates
    Ahat_boot = np.zeros([Nb, n, n])
    Bhat_boot = np.zeros([Nb, n, m])
    Chat_boot = np.zeros([Nb, p, n])
    if return_models:
        models_boot = []
    for i in range(Nb):
        model_boot = system_identification(y_boot_hist[i], u_boot_hist[i], SS_fixed_order=n)
        # check_diff(model, model_boot)

        # Transform to align coordinate systems as much as possible
        model_trans, P = ss_change_coordinates(model, model_boot, method='match')

        if return_models:
            models_boot.append(model_trans)

        Ahat_boot[i] = model_trans.A
        Bhat_boot[i] = model_trans.B
        Chat_boot[i] = model_trans.C

        if verbose:
            print('created bootstrap model %6d / %6d' % (i+1, Nb))

    # Sample variance of bootstrap estimates
    Ahat_boot_reshaped = np.reshape(Ahat_boot, [Nb, n*n], order='F')
    Bhat_boot_reshaped = np.reshape(Bhat_boot, [Nb, n*m], order='F')
    Chat_boot_reshaped = np.reshape(Chat_boot, [Nb, p*n], order='F')

    Ahat_boot_mean_reshaped = np.mean(Ahat_boot_reshaped, axis=0)
    Bhat_boot_mean_reshaped = np.mean(Bhat_boot_reshaped, axis=0)
    Chat_boot_mean_reshaped = np.mean(Chat_boot_reshaped, axis=0)


    # TODO account for correlation between A, B, C
    # can we do it easily with a decorrelation scheme /coordinate change?
    # or do we need to re-derive the full gdare w/ terms?

    Abar = Ahat_boot_reshaped - Ahat_boot_mean_reshaped
    Bbar = Bhat_boot_reshaped - Bhat_boot_mean_reshaped
    Cbar = Chat_boot_reshaped - Chat_boot_mean_reshaped

    SigmaA = np.dot(Abar.T, Abar)/(Nb-1)
    SigmaB = np.dot(Bbar.T, Bbar)/(Nb-1)
    SigmaC = np.dot(Cbar.T, Cbar)/(Nb-1)

    SigmaAeigvals, SigmaAeigvecs = la.eigh(SigmaA)
    SigmaBeigvals, SigmaBeigvecs = la.eigh(SigmaB)
    SigmaCeigvals, SigmaCeigvecs = la.eigh(SigmaC)

    a = np.real(SigmaAeigvals)
    b = np.real(SigmaBeigvals)
    c = np.real(SigmaCeigvals)

    Aa = np.reshape(SigmaAeigvecs, [n*n, n, n], order='F')  # These uncertainty directions have unit Frobenius norm
    Bb = np.reshape(SigmaBeigvecs, [n*m, n, m], order='F')  # These uncertainty directions have unit Frobenius norm
    Cc = np.reshape(SigmaCeigvecs, [p*n, p, n], order='F')  # These uncertainty directions have unit Frobenius norm

    uncertainty = Uncertainty(a, Aa, b, Bb, c, Cc)

    if return_models:
        return uncertainty, models_boot
    else:
        return uncertainty


def estimate_model_uncertainty(model, u_hist, y_hist, w_est, v_est, t, Nb,
                               uncertainty_estimator=None, return_models=False):
    if uncertainty_estimator is None:
        return None, None, None, None, None, None

    if uncertainty_estimator == 'exact':
        raise NotImplementedError
        # # TODO
        # # "Cheat" by using the true error as the multiplicative noise
        # Aa = np.zeros([n*n, n, n])
        # Bb = np.zeros([n*m, n, m])
        # Cc = np.zeros([p*n, p, n])
        # Aa[0] = Adiff/Aerr
        # Bb[0] = Bdiff/Berr
        # Cc[0] = Cdiff/Cerr
        # a = np.zeros(n*n)
        # b = np.zeros(n*m)
        # c = np.zeros(p*n)
        # a[0] = Aerr**2
        # b[0] = Berr**2
        # c[0] = Cerr**2
    else:
        if uncertainty_estimator == 'block_bootstrap':
            u_boot_hist, y_boot_hist = block_bootstrap(u_hist, y_hist, t, Nb)
        elif uncertainty_estimator == 'semiparametric_bootstrap':
            u_boot_hist, y_boot_hist = semiparametric_bootstrap(model, u_hist, y_hist, w_est, v_est, t, Nb)
        else:
            raise ValueError('Invalid uncertainty estimator method!')


        # TEST/DEBUG ONLY
        # # this is OK
        # # CHEAT by passing the true model and true residuals in
        # # This gives the most accurate assessment of the uncertainty in the model estimate
        # # because it is almost like getting a brand new dataset for each model sample
        # u_boot_hist, y_boot_hist = semiparametric_bootstrap(model_true, u_hist, y_hist, w_hist, v_hist, Nb=Nb)

        # this is OK
        # # Transform process noise approximately into model coordinates
        # w_hat_hist = np.zeros_like(w_hist)
        # v_hat_hist = np.copy(v_hist)
        # for i in range(T):
        #     # w_hat_hist[i] = np.dot(P, w_hist[i])
        #     w_hat_hist[i] = np.dot(la.inv(P), w_hist[i])
        #
        # u_boot_hist, y_boot_hist = semiparametric_bootstrap(model, u_hist, y_hist, w_hat_hist, v_hat_hist, Nb=Nb)

        # # This is what we intend to do - true semiparametric bootstrap
        # # TODO figure out why w_est and v_est are so much smaller in magnitude than w_hist and v_hist
        # #    can be verified by comparing model.Q vs W and model.R vs V which are the process & sensor noise covariances
        # u_boot_hist, y_boot_hist = semiparametric_bootstrap(model, u_hist, y_hist, w_est, v_est, Nb=Nb)

        # u_boot_hist, y_boot_hist = block_bootstrap(u_hist, y_hist, Nb=100, blocksize=100)



        # Form bootstrap model estimates from bootstrap datasets
        # and convert covariances into multiplicative noises
        return ensemble2multnoise(model, u_boot_hist, y_boot_hist, return_models=return_models)
