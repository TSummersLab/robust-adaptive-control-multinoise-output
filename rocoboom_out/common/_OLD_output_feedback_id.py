"""Initial scratch file for (adaptive) data-based control with only input-output data (i.e., not state data)"""

import numpy as np
import numpy.random as npr
import scipy as sc
import numpy.linalg as la
import scipy.linalg as sla

from utility.matrixmath import mdot, specrad, minsv, solveb, lstsqb, dlyap, dare, dare_gain
from utility.ltimult import dare_mult
from utility.lti import ctrb, dctg


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import get_cmap

import cvxpy as cvx
import control

# from SIPPY.sippy import system_identification
from SIPPY.sippy import functionset as fset
from SIPPY.sippy import functionsetSIM as fsetSIM

from problem_data_gen import gen_system_omni
from sysid import system_identification
from ss_tools import make_ss
from signal_gen import SigParam, make_sig
from uncertainty import block_bootstrap, semiparametric_bootstrap, ensemble2multnoise




# Estimate state space model from input-output data via subspace ID using the SIPPY package
model, res = system_identification(y_hist, u_hist, id_method='N4SID', SS_fixed_order=n, return_residuals=True)
w_est = res[0:n].T
v_est = res[n:].T

Abar = model.A
Bbar = model.B
Cbar = model.C
Dbar = model.D

# We cannot just use Q, W because the coordinate systems are not the same...
# ... R and V are OK because the input and output spaces are known/shared

Qbar = np.dot(Cbar.T, np.dot(Y, Cbar))
Rbar = np.copy(R)

# Don't do this since SIPPY is estimating covariances from data
# Wbar = np.dot(Bbar, np.dot(Z, Bbar.T))
# Vbar = np.copy(V)

# TODO make sure the scaling is right for these since docs say they are with respect to outputs with unit variance
Wbar = model.Q
Vbar = model.R
Ubar = model.S

# Compute transform by minimizing the error in statespace matrices A, B, C
# i.e. x = P @ xbar
P = cvx.Variable((n, n))
cost = cvx.square(cvx.norm(P@Abar - A@P, 'fro'))/A.size \
       + cvx.square(cvx.norm(P@Bbar - B, 'fro'))/B.size \
       + cvx.square(cvx.norm(Cbar - C@P, 'fro'))/C.size
prob = cvx.Problem(cvx.Minimize(cost))
prob.solve()

P = P.value
Ahat = np.dot(P, solveb(Abar, P))
Bhat = np.dot(P, Bbar)
Chat = solveb(Cbar, P)
Dhat = np.copy(Dbar)

ss_model_hat = make_ss(Ahat, Bhat, Chat, Dhat)

# These should match W, V, (U=0) closely with lots of data
What = np.dot(P, np.dot(Wbar, P.T))
Vhat = np.copy(Vbar)
Uhat = np.dot(P, Ubar)


ss_model = make_ss(model.A, model.B, model.C, model.D)
ss_model_modal, _ = control.canonical_form(ss_model, form='modal')


# Find robust compensator

# TODO account for correlation between A, B, C in gdare

# TODO include the (estimated) cross-covariance model.S in the gdare





# Compute certainty equivalent compensator
# TODO include the (estimated) cross-covariance model.S in the dare
Pceq, Kceq = dare_gain(Abar, Bbar, Qbar, Rbar)
Sceq, Lceq = dare_gain(Abar.T, Cbar.T, Wbar, Vbar)
Lceq = -Lceq.T


# Compute true optimal compensator when the true system is known
# IMPORTANT: cannot compare gains directly because the internal state representation is different
# Only compare closed-loop transfer functions or closed-loop performance cost
Popt, Kopt = dare_gain(A, B, Q, R)
Sopt, Lopt = dare_gain(A.T, C.T, W, V)
Lopt = -Lopt.T



# print(specrad(A + B@Kopt))
# print(specrad(Abar + Bbar@K))

# print(specrad(Abar + Bbar@Krob - Lrob@Cbar))  # spectral radius of nominal system under robust gains
# print(specrad(Abar + Bbar@Kceq - Lceq@Cbar))  # spectral radius of nominal system under c.e. gains
# print(specrad(A + B@Kopt - Lopt@C))  # spectral radius of true system under optimal gains



from compensator_eval import aug_sysmat_cl, compute_perf


z_hist = np.vstack([make_sig(T//2, n, [SigParam(method='gbn', mean=0, scale=1.0, ma_length=None)]),
                    make_sig(T//2, n, [SigParam(method='zeros', mean=0, scale=1.0, ma_length=None)])])

# fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots(ncols=p)
if p == 1:
    ax2 = [ax2]
for mod, K, L, label in zip([model, model, ss_model_true], [Krob, Kceq, Kopt], [Lrob, Lceq, Lopt], ['robust', 'cert. equiv.', 'optimal']):
    x_hist, u_hist, y_hist, xhat_hist = lsim_cl(ss_model_true, mod, K, L, z_hist, v_hist, T)
    # ax1.plot(x_hist)
    for i in range(p):
        ax2[i].plot(y_hist[:, i], label=label)
for i in range(p):
    ax2[i].legend()








# print(la.norm(P@Abar - A@P, 'fro'))  # error in A
# print(la.norm(P@Bbar - B, 'fro'))  # error in B
# print(la.norm(Cbar - C@P, 'fro'))  # error in C
# print('')
# # print(prob.value/n**2)
#
# # Simulate with estimated model
# xhat_hist = np.zeros([T+1, n])
# xhat_hist[0] = la.pinv(P)@x0
# yhat_hist = np.zeros([T, p])
#
# for t in range(T):
#     # Update state and generate output
#     xhat = xhat_hist[t]
#     u = u_hist[t]
#     xhat_hist[t+1] = np.dot(Abar, xhat) + np.dot(Bbar, u)
#     yhat_hist[t] = np.dot(Cbar, xhat)
#
# # print(la.norm(y_hist - yhat_hist)/T)  # error in outputs over all time
#










# # OLD
# # semiparametric bootstrap
# # compute residuals
# # generate boostrap datasets and compute bootstrap estimates
# # find common basis to compare bootstrap estimates
# # compute sample covariance of state space parameters
#
#
# # compute controller based on nominal state space model and model uncertainty estimate
#
#
# # plotting
# plt.close('all')
#
#
# # plot system matrices
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(A)
# ax[1].imshow(Ahat)
# ax[0].set_title(r'$A$')
# ax[1].set_title(r'$\hat{A}$')
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(B)
# ax[1].imshow(Bhat)
# ax[0].set_title(r'$B$')
# ax[1].set_title(r'$\hat{B}$')
#
# fig, ax = plt.subplots(ncols=2)
# ax[0].imshow(C)
# ax[1].imshow(Chat)
# ax[0].set_title(r'$C$')
# ax[1].set_title(r'$\hat{C}$')
#
#
#
#
#
#
# # plot responses
# t_hist = np.arange(T+1)
#
#
# plt.figure()
# plt.plot(t_hist, x_hist)
# plt.xlabel('time')
# plt.ylabel('states')
#
# plt.figure()
# plt.plot(t_hist[0:T], u_hist)
# plt.xlabel('time')
# plt.ylabel('inputs')
#
# plt.figure()
# plt.plot(t_hist[0:T], y_hist)
# # plt.plot(t_hist[0:T], yhat_hist)
# plt.xlabel('time')
# plt.ylabel('outputs')
# plt.show()
#
#
#
# plt.figure()
# plt.plot(t_hist[0:T], y_hist)
# for i in range(Nb):
#     plt.plot(t_hist[0:T], y_boot_hist[i], color='k', alpha=0.1)
# plt.xlabel('time')
# plt.ylabel('outputs')
# plt.show()
