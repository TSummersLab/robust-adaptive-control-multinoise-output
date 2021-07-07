"""Initial scratch file for (adaptive) data-based control with only input-output data (i.e., not state data)"""

import numpy as np
import numpy.random as npr
import scipy as sc
import numpy.linalg as la
import scipy.linalg as sla

from utility.matrixmath import mdot, specrad, minsv, lstsqb, dlyap, dare, dare_gain
from utility.ltimult import dare_mult
from utility.lti import ctrb, dctg
from monte_carlo_comparison import groupdot

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import get_cmap

import cvxpy as cvx

from SIPPY.sippy import *
from SIPPY.sippy import functionset as fset
from SIPPY.sippy import functionsetSIM as fsetSIM


# problem data
n = 4
m = 2
p = 1
spectral_radius = 0.98

A = npr.randn(n, n)
A *= spectral_radius/specrad(A)
B = npr.randn(n, m)
C = npr.randn(p, n)

Q = np.identity(n)
R = np.identity(m)

U = npr.randn(n, n)
Y = npr.randn(p, p)
W = 0.005*np.dot(U, U.T)
V = 0.005*np.dot(Y, Y.T)

# generate trajectory data
T = 300
x0 = npr.randn(n)

u_explore_var = 2
# u_hist = np.sqrt(u_explore_var)*npr.randn(m, T)
u_hist = np.zeros((m, T))
u_hist[0, :] = fset.GBN_seq(T, 0.05)
u_hist[1, :] = fset.GBN_seq(T, 0.05)

# Generate noise
w_hist = npr.multivariate_normal(np.zeros(n), W, size=(1, T))
v_hist = npr.multivariate_normal(np.zeros(p), V, size=(1, T))

# Preallocate state and output history
x_hist = np.zeros([n, T+1])
y_hist = np.zeros([p, T])

# Initial state
x_hist[:, 0] = x0

# Loop over time
for t in range(T):
    # Update state and generate output
    x_hist[:, t+1] = groupdot(A, x_hist[:, t]) + groupdot(B, u_hist[:, t]) + w_hist[:, t]
    y_hist[:, t] = groupdot(C, x_hist[:,t]) + v_hist[:, t]


# estimate state space model from input-output data via subspace ID (for now we use the SIPPY package)
method = 'N4SID'
sys_id = system_identification(y_hist, u_hist, method, SS_fixed_order=n)

Abar = sys_id.A
Bbar = sys_id.B
Cbar = sys_id.C
P = cvx.Variable((n, n))
cost = cvx.atoms.norm(P@Abar - A@P, 'fro') + cvx.atoms.norm(P@Bbar - B, 'fro') + cvx.atoms.norm(Cbar - C@P, 'fro')
prob = cvx.Problem(cvx.Minimize(cost))
prob.solve()

print(A)
print(P.value@Abar@(la.pinv(P.value)))
print(prob.value/n**2)

# Simulate with estimated model
xhat_hist = np.zeros([n, T+1])
xhat_hist[:, 0] = la.pinv(P.value)@x0
yhat_hist = np.zeros([p, T])

for t in range(T):
    # Update state and generate output
    xhat_hist[:, t+1] = groupdot(Abar, xhat_hist[:, t]) + groupdot(Bbar, u_hist[:, t])
    yhat_hist[:, t] = groupdot(Cbar, xhat_hist[:, t])

plt.figure()
plt.step(range(T), yhat_hist.T)
plt.xlabel('time')
plt.ylabel('outputs')

print(la.norm(y_hist - yhat_hist)/T)

# semiparametric bootstrap
# compute residuals
# generate boostrap datasets and compute bootstrap estimates
# find common basis to compare bootstrap estimates
# compute sample covariance of state space parameters


# compute controller based on nominal state space model and model uncertainty estimate


# plotting
plt.figure()
plt.step(range(T+1), x_hist.T)
plt.xlabel('time')
plt.ylabel('states')

plt.figure()
plt.step(range(T), u_hist.T)
plt.xlabel('time')
plt.ylabel('inputs')

plt.figure()
plt.step(range(T), y_hist.T)
plt.step(range(T), yhat_hist.T)
plt.xlabel('time')
plt.ylabel('outputs')
