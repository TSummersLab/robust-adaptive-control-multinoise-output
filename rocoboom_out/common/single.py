from dataclasses import dataclass

import numpy as np
import numpy.random as npr
import scipy as sc
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt

import control

from rocoboom_out.common.problem_data_gen import gen_system_omni
from rocoboom_out.common.signal_gen import SigParam, make_sig
from rocoboom_out.common.sim import make_offline_data, lsim_cl
from rocoboom_out.common.sysid import system_identification
from rocoboom_out.common.ss_tools import make_ss, ss_change_coordinates
from rocoboom_out.common.uncertainty import estimate_model_uncertainty
from rocoboom_out.common.compensator_design import make_compensator
from rocoboom_out.common.compensator_eval import compute_performance
from rocoboom_out.common.misc import get_entry


@dataclass
class Result:
    model: None
    uncertainty: None
    ss_models_boot: None
    compensator: None
    performance: None
    design_info: dict


def get_result(method):
    ss_models_boot = None
    if method == 'opt':
        uncertainty = None
        my_model = ss_true
    elif method == 'ceq':
        uncertainty = None
        my_model = ss_model
    elif method == 'rob':
        uncertainty_estimator = 'semiparametric_bootstrap'
        uncertainty, ss_models_boot = estimate_model_uncertainty(model, u_train_hist, y_train_hist, w_est, v_est, t, Nb,
                                                                 uncertainty_estimator, return_models=True)
        my_model = ss_model
    else:
        raise ValueError

    compensator, noise_scale, tag_str_list_cg = make_compensator(my_model, uncertainty, Y, R,
                                                                 noise_pre_scale, noise_post_scale, bisection_epsilon)
    F, K, L = compensator.F, compensator.K, compensator.L

    performance = compute_performance(A, B, C, Q, R, W, V, F, K, L)
    design_info = dict(noise_scale=noise_scale)

    return Result(model, uncertainty, ss_models_boot, compensator, performance, design_info)


# Options
plt.close('all')

seed = 1
npr.seed(seed)

# Number of Monte Carlo samples
Ns = 1

# Number of bootstrap samples
Nb = 50

# Simulation time
t = 1000  # This is the amount of data that will be used by sysid
T = t + 1  # This is the amount of data that will be simulated

# u_explore_var = np.max(la.eig(W)[0])
u_explore_var = 1.0

noise_pre_scale = 1.0
noise_post_scale = 1.0
bisection_epsilon = 0.01


# Problem data
system_kwargs = dict(n=4, m=2, p=2, spectral_radius=0.9, noise_scale=0.1, seed=1)
n, m, p, A, B, C, D, Y, Q, R, W, V = gen_system_omni('rand', **system_kwargs)
ss_true = make_ss(A, B, C, D, W, V)


# Make training data
x_train_hist, u_train_hist, y_train_hist, w_hist, v_hist = make_offline_data(A, B, C, D, W, V, Ns, T, u_explore_var)
x_train_hist = x_train_hist[0]
u_train_hist = u_train_hist[0]
y_train_hist = y_train_hist[0]
w_hist = w_hist[0]
v_hist = v_hist[0]

# Estimate the system model and residuals
model, res = system_identification(y_train_hist[0:t], u_train_hist[0:t], id_method='N4SID',
                                   SS_fixed_order=n, return_residuals=True)
w_est = res[0:n].T
v_est = res[n:].T

Ahat = model.A
Bhat = model.B
Chat = model.C
Dhat = model.D

# TODO make sure the scaling is right for these since docs say they are with respect to outputs with unit variance
What = model.Q
Vhat = model.R
Uhat = model.S

ss_model = make_ss(Ahat, Bhat, Chat, Dhat, What, Vhat, Uhat)


########################################################################################################################
# # TODO change basis for A, B, C of true system by putting it into e.g. modal form
# #   That way, the estimated parameters will approach the true parameters after using the same standardizing transform
# #   needs more investigation, control.canonical_form with form='modal' does not make them match at all
# ss_true_modal, true_modal_transform = control.canonical_form(ss_true, form='modal')
# ss_model_modal, model_modal_transform = control.canonical_form(ss_model, form='modal')

# ss_true_reachable, true_reachable_transform = control.canonical_form(ss_true, form='reachable')
# ss_model_reachable, model_reachable_transform = control.canonical_form(ss_model, form='reachable')
########################################################################################################################


########################################################################################################################
# This code chunk is for debug/test only
# Cannot use this practically since ss_true is not accessible
ss_trans, P = ss_change_coordinates(ss_true, model, method='match')

# With lots of data, ss_trans should match ss_true very closely in A, B, C, D, Q, R, S
# TODO troubleshoot why the noise covariances do not seem to converge to the true values
########################################################################################################################


########################################################################################################################
# Get the results for each control synthesis method
methods = ['opt', 'ceq', 'rob']
result_dict = {method: get_result(method) for method in methods}

for method in methods:
    print("%s    cost = %.6f" % (method, result_dict[method].performance.ihc/result_dict['opt'].performance.ihc))

print(result_dict['rob'].design_info)
# print(result_dict['rob'].ss_models_boot)
print(result_dict['rob'].uncertainty.a)
print(result_dict['rob'].uncertainty.b)
print(result_dict['rob'].uncertainty.c)
########################################################################################################################


########################################################################################################################
# Evaluate closed-loop responses
def response_plot(ss, T=None, fig=None, axs=None, response_type='impulse', *args, **kwargs):
    # Choose the response function
    if response_type == 'impulse':
        response_fun = control.impulse_response
    elif response_type == 'step':
        response_fun = control.step_response
    else:
        raise ValueError

    # Get the response data
    t, ys = response_fun(ss, T)

    # Create plot
    if fig is None:
        fig, axs = plt.subplots(nrows=p, ncols=m, sharex=True, figsize=(8, 6))
    for i in range(p):
        for j in range(m):
            ax = get_entry(axs, i, j, p, m)
            y = get_entry(ys, i, j, p, m)
            ax.plot(t, y, *args, **kwargs)
    fig.tight_layout()
    return fig, axs


def comparison_response_plot(ss_true, ss_model, ss_models_boot, T_resp=50, num_models_boot_to_plot=100):
    fig, ax = response_plot(ss_true, T_resp, alpha=0.7, zorder=10, label='true')
    fig, ax = response_plot(ss_model, T_resp, fig, ax, alpha=0.7, zorder=11, label='nominal')

    for i, ss_model_boot in enumerate(ss_models_boot):
        if i > num_models_boot_to_plot:
            continue
        if i == 0:
            label = 'boot'
        else:
            label = None
        fig, ax = response_plot(ss_model_boot, T_resp, fig, ax, color='k', alpha=0.1, zorder=1, label=label)

    for i in range(p):
        for j in range(m):
            my_ax = get_entry(ax, i, j, p, m)
            my_ax.legend()


def comparison_closed_loop_plot(result_dict, disturb_method='zeros'):
    # Make initial state
    x0 = np.ones(n)

    # Make disturbance histories
    w_play_hist = make_sig(T, n, [SigParam(method=disturb_method, mean=0, scale=1.0, ma_length=None)])
    v_play_hist = make_sig(T, p, [SigParam(method=disturb_method, mean=0, scale=1.0, ma_length=None)])

    fig, ax = plt.subplots(ncols=p, sharex=True, figsize=(8, 6))
    if p == 1:
        ax = [ax]
    for key in ['rob', 'ceq', 'opt']:
        result = result_dict[key]
        x_hist, u_hist, y_hist, xhat_hist = lsim_cl(ss_true, result.compensator, x0, w_play_hist, v_play_hist, T)
        for i in range(p):
            ax[i].plot(y_hist[:, i], label=key)
    for i in range(p):
        ax[i].legend()


ss_models_boot = result_dict['rob'].ss_models_boot
comparison_response_plot(ss_true, ss_model, ss_models_boot)
comparison_closed_loop_plot(result_dict)
########################################################################################################################




# Redesign from result data
result = result_dict['rob']
my_model = result.model
uncertainty = result.uncertainty
compensator, noise_scale, tag_str_list_cg = make_compensator(my_model, uncertainty, Y, R,
                                                             noise_pre_scale=1.0,
                                                             noise_post_scale=0.5,
                                                             bisection_epsilon=bisection_epsilon)
F, K, L = compensator.F, compensator.K, compensator.L
performance = compute_performance(A, B, C, Q, R, W, V, F, K, L)
for tag in tag_str_list_cg:
    print(tag)
print(performance.ihc/result_dict['opt'].performance.ihc)
