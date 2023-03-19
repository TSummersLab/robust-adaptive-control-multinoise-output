from dataclasses import dataclass

import numpy as np
import numpy.random as npr
import scipy as sc
import numpy.linalg as la
import scipy.linalg as sla
import matplotlib.pyplot as plt

import control

from utility.lti import ctrb, obsv
from utility.printing import create_tag

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
        # uncertainty_estimator = 'block_bootstrap'
        uncertainty_dict = estimate_model_uncertainty(model, u_train_hist, y_train_hist, w_est, v_est, t, Nb,
                                                      uncertainty_estimator, return_models=True, log_diagnostics=True)

        # TODO DEBUG ONLY
        # CHEAT by using the true model and true residuals in the semiparametric bootstrap
        # uncertainty_estimator = 'semiparametric_bootstrap'
        # uncertainty_dict = estimate_model_uncertainty(ss_true, u_train_hist, y_train_hist,
        #                                                                           w_hist, v_hist, t, Nb,
        #                                                                           uncertainty_estimator,
        #                                                                           return_models=True,
        #                                                                           log_diagnostics=True)

        uncertainty = uncertainty_dict['uncertainty']
        ss_models_boot = uncertainty_dict['models_boot']
        tag_list_uc = uncertainty_dict['tag_list']

        for tag in tag_list_uc:
            print(tag)
        my_model = ss_model
    else:
        raise ValueError

    compensator, noise_scale, tag_str_list_cg = make_compensator(my_model, uncertainty, Y, R,
                                                                 noise_pre_scale, noise_post_scale, bisection_epsilon, log_diagnostics=True)
    for tag in tag_str_list_cg:
        print(tag)
    F, K, L = compensator.F, compensator.K, compensator.L

    performance = compute_performance(A, B, C, Q, R, W, V, F, K, L)
    design_info = dict(noise_scale=noise_scale)

    return Result(model, uncertainty, ss_models_boot, compensator, performance, design_info)


def response_plot(ss, T=None, y_ref=None, fig=None, axs=None, response_type='impulse', *args, **kwargs):
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
        fig, axs = plt.subplots(nrows=p, ncols=m, sharex=True, figsize=(4, 3))
    for i in range(p):
        for j in range(m):
            ax = get_entry(axs, i, j, p, m)
            y = get_entry(ys, i, j, p, m)
            if y_ref is not None:
                yr = get_entry(y_ref, i, j, p, m)
                # y -= yr  # Remove the true response so that the plotted quantity is the deviation response
            ax.plot(t, y, *args, **kwargs)
    fig.tight_layout()
    return t, ys, fig, axs


def comparison_response_plot(ss_true, ss_model, ss_models_boot, t_sim=None, num_models_boot_to_plot=100):
    # Simulation time
    if t_sim is None:
        t_sim = T

    t, ys_true = control.impulse_response(ss_true, t_sim)
    t, ys_true, fig, axs = response_plot(ss_true, t_sim, ys_true, marker='o', alpha=0.7, zorder=10, label='True')
    t, ys_model, fig, axs = response_plot(ss_model, t_sim, ys_true, fig, axs, marker='d', alpha=0.7, zorder=11, label='Nominal')

    for i, ss_model_boot in enumerate(ss_models_boot):
        if i > num_models_boot_to_plot:
            break
        if i == 0:
            label = 'Bootstrap samples'
        else:
            label = None
        t, ys_boot, fig, axs = response_plot(ss_model_boot, t_sim, ys_true, fig, axs, color='k', alpha=0.05, zorder=1, label=label)

    for i in range(p):
        for j in range(m):
            my_ax = get_entry(axs, i, j, p, m)
            my_ax.legend()
    return fig, axs


def comparison_closed_loop_plot(result_dict, t_sim=None, disturb_method='zeros', disturb_scale=1.0):
    # Make initial state
    x0 = np.ones(n)

    # Simulation time
    if t_sim is None:
        t_sim = T

    # Make disturbance histories
    w_play_hist = make_sig(t_sim, n, [SigParam(method=disturb_method, mean=0, scale=disturb_scale)])
    v_play_hist = make_sig(t_sim, p, [SigParam(method=disturb_method, mean=0, scale=disturb_scale)])

    fig, axs = plt.subplots(ncols=p, sharex=True, figsize=(8, 6))
    if p == 1:
        axs = [axs]
    for key in methods:
        result = result_dict[key]
        x_hist, u_hist, y_hist, xhat_hist = lsim_cl(ss_true, result.compensator, x0, w_play_hist, v_play_hist, t_sim)
        for i in range(p):
            axs[i].plot(y_hist[:, i], label=key)
    for i in range(p):
        axs[i].legend()

    fig.suptitle('Comparison of closed-loop response to initial state')
    return fig, axs


def matrix_distplot(M_true, M_model, M_models_boot, title_str=None, plot_type='violin',
                    show_true=True, show_model=True, show_boot_mean=False, show_boot_median=False,
                    show_samples=True, show_dist=True, show_legend=True):
    d1, d2 = M_true.shape

    fig, axs = plt.subplots(nrows=d1, ncols=d2, figsize=(d2*3, d1*3))
    for i in range(d1):
        for j in range(d2):
            ax = get_entry(axs, i, j, d1, d2)

            # Data extraction
            x_true = M_true[i, j]
            x_model = M_model[i, j]
            x_models_boot = np.array([M_model_boot[i, j] for M_model_boot in M_models_boot])
            x_models_boot_mean = np.mean(x_models_boot)
            x_models_boot_median = np.median(x_models_boot)

            if show_dist:
                if plot_type == 'violin':
                    violinplot_parts = ax.violinplot(x_models_boot, positions=[0], showextrema=False, points=400)
                    for part in violinplot_parts['bodies']:
                        part.set_facecolor('k')
                elif plot_type == 'box':
                    ax.boxplot(x_models_boot, positions=[0], medianprops=dict(color='C2'))
                else:
                    raise ValueError

            linex = [-0.1, 0.1]
            if show_true:
                ax.plot(linex, x_true*np.ones(2), c='C0', linestyle='--', lw=3, label='True')
            if show_model:
                ax.plot(linex, x_model*np.ones(2), c='C1', linestyle='-', lw=3, label='Nominal')
            if show_boot_mean:
                ax.plot(linex, x_models_boot_mean*np.ones(2), c='k', label='Bootstrap mean')
            if show_boot_median:
                ax.plot(linex, x_models_boot_median*np.ones(2), c='C2', label='Bootstrap median')
            if show_samples:
                ax.scatter(np.zeros_like(x_models_boot), x_models_boot, s=40, c='k', marker='o', alpha=0.2, label='Bootstrap samples')

            # Set the ylimit
            p = 1  # percentile
            x_models_boot_pct_lwr = np.percentile(x_models_boot, p)
            x_models_boot_pct_upr = np.percentile(x_models_boot, 100-p)
            xpct_diff = x_models_boot_pct_upr - x_models_boot_pct_lwr
            xlim_lwr = np.min([x_models_boot_pct_lwr, x_true-0.1*xpct_diff])
            xlim_upr = np.max([x_models_boot_pct_upr, x_true+0.1*xpct_diff])
            ax.set_ylim([xlim_lwr, xlim_upr])

            if show_legend:
                ax.legend()
    fig.suptitle(title_str)
    fig.tight_layout()
    return fig, axs


if __name__ == "__main__":
    # Options
    plt.close('all')

    seed = 1
    npr.seed(seed)

    # Number of Monte Carlo samples
    Ns = 1

    # Number of bootstrap samples
    Nb = 100

    # Simulation time
    t = 20  # This is the amount of data that will be used by sysid
    T = t + 1  # This is the amount of data that will be simulated


    # Problem data
    system_kwargs = dict()
    # n, m, p, A, B, C, D, Y, Q, R, W, V, U = gen_system_omni('inverted_pendulum', **system_kwargs)
    # n, m, p, A, B, C, D, Y, Q, R, W, V, U = gen_system_omni('noninverted_pendulum', **system_kwargs)

    # system_kwargs = dict(n=4, m=2, p=2, spectral_radius=0.9, noise_scale=0.1, seed=1)
    # n, m, p, A, B, C, D, Y, Q, R, W, V, U = gen_system_omni('rand', **system_kwargs)

    # n, m, p, A, B, C, D, Y, Q, R, W, V, U = gen_system_omni(system_idx=1)
    n, m, p, A, B, C, D, Y, Q, R, W, V, U = gen_system_omni(system_idx=3)

    ss_true = make_ss(A, B, C, D, W, V, U)


    # Exploration control signal
    # u_explore_var = 10*(np.max(la.eig(W)[0]) + np.max(la.eig(V)[0]))  # for 2-state shift register
    u_explore_var = 1000*(np.max(la.eig(W)[0]) + np.max(la.eig(V)[0]))
    # u_explore_var = 10.0

    noise_pre_scale = 1.0
    noise_post_scale = 1.0
    bisection_epsilon = 0.01


    # Check controllability and observability of the true system
    check_tags = []
    if la.matrix_rank(ctrb(A, B)) < n:
        check_tags.append(create_tag("The pair (A, B) is NOT controllable.", message_type='warn'))
    if la.matrix_rank(obsv(Q, A)) < n:
        check_tags.append(create_tag("The pair (Q, A) is NOT observable.", message_type='warn'))
    if la.matrix_rank(obsv(A, C)) < n:
        check_tags.append(create_tag("The pair (A, C) is NOT observable.", message_type='warn'))
    if la.matrix_rank(ctrb(A - np.dot(U, la.solve(V, C)), B)) < n:
        check_tags.append(create_tag("The pair (A - U V^-1 C, B) is NOT controllable.", message_type='warn'))
    for tag in check_tags:
        print(tag)


    # Make training data
    x_train_hist, u_train_hist, y_train_hist, w_hist, v_hist = make_offline_data(A, B, C, D, W, V, Ns, T, u_explore_var)
    x_train_hist = x_train_hist[0]
    u_train_hist = u_train_hist[0]
    y_train_hist = y_train_hist[0]
    w_hist = w_hist[0]
    v_hist = v_hist[0]

    t_train_hist = np.arange(T)

    # # Plot
    # fig, ax = plt.subplots(nrows=2)
    # ax[0].step(t_train_hist, u_train_hist)
    # ax[1].step(t_train_hist, y_train_hist)
    # fig.suptitle('training data')
    # ax[0].set_ylabel('input')
    # ax[1].set_ylabel('output')

    # Estimate the system model and residuals
    model, res = system_identification(y_train_hist[0:t], u_train_hist[0:t], id_method='N4SID',
                                       SS_fixed_order=n, return_residuals=True)
    w_est = res[0:n].T
    v_est = res[n:].T

    Ahat = model.A
    Bhat = model.B
    Chat = model.C
    Dhat = model.D

    What = model.Q
    Vhat = model.R
    Uhat = model.S

    ss_model = make_ss(Ahat, Bhat, Chat, Dhat, What, Vhat, Uhat)



    ####################################################################################################################
    # # change basis for A, B, C of true system by putting it into e.g. modal form
    # # That way, the estimated parameters will approach the true parameters after using the same standardizing transform
    # # needs more investigation, control.canonical_form with form='modal' does not make them match at all
    # ss_true_modal, true_modal_transform = control.canonical_form(ss_true, form='modal')
    # ss_model_modal, model_modal_transform = control.canonical_form(ss_model, form='modal')

    # ss_true_reachable, true_reachable_transform = control.canonical_form(ss_true, form='reachable')
    # ss_model_reachable, model_reachable_transform = control.canonical_form(ss_model, form='reachable')
    ####################################################################################################################


    ####################################################################################################################
    # This code chunk is for debug/test only
    # Cannot use this practically since ss_true is not accessible
    ss_model_trans, P = ss_change_coordinates(ss_true, model, method='match')

    # With lots of data, ss_model_trans should match ss_true very closely in A, B, C, D
    # but Q, R, S cannot be identified uniquely - see subspace ID book pg 66
    print('True noise covariances')
    print(ss_true.Q)
    print(ss_true.R)
    print(ss_true.S)
    print('')
    print('Estimated noise covariances')
    print(ss_model_trans.Q)
    print(ss_model_trans.R)
    print(ss_model_trans.S)
    print('')

    # # Check that model transformation has no impact on the performance of the ce compensator
    # compensator = make_compensator(ss_model, None, Y, R)[0]
    # print(compensator)
    # print(compute_performance(A, B, C, Q, R, W, V, compensator.F, compensator.K, compensator.L))
    #
    # compensator = make_compensator(ss_model_trans, None, Y, R)[0]
    # print(compensator)
    # print(compute_performance(A, B, C, Q, R, W, V, compensator.F, compensator.K, compensator.L))

    ####################################################################################################################

    # DEBUG ONLY
    # CHEAT by matching model coordinates with the true
    # ss_model = ss_model_trans

    # # CHEAT by using true noise covariance, appropriately transformed
    # ss_model.Q = ss_true.Q
    # ss_model.R = ss_true.R
    # ss_model.S = ss_true.S



    ####################################################################################################################
    # Get the results for each control synthesis method
    methods = ['opt', 'ceq', 'rob']
    result_dict = {method: get_result(method) for method in methods}

    for method in methods:
        print("%s    cost = %.6f" % (method, result_dict[method].performance.ihc/result_dict['opt'].performance.ihc))

    # print(result_dict['rob'].design_info)
    # print(result_dict['rob'].uncertainty.a)
    # print(result_dict['rob'].uncertainty.b)
    # print(result_dict['rob'].uncertainty.c)
    # print(result_dict['rob'].ss_models_boot)
    ####################################################################################################################


    ####################################################################################################################
    ss_models_boot = result_dict['rob'].ss_models_boot
    comparison_response_plot(ss_true, ss_model, ss_models_boot, t_sim=40)
    # comparison_closed_loop_plot(result_dict, disturb_method='wgn', disturb_scale=0.1, t_sim=20)
    ####################################################################################################################


    # ####################################################################################################################
    # # Redesign from result data
    # result = result_dict['rob']
    # my_model = result.model
    # uncertainty = result.uncertainty
    # compensator, noise_scale, tag_str_list_cg = make_compensator(my_model, uncertainty, Y, R,
    #                                                              noise_pre_scale=1.0,
    #                                                              noise_post_scale=0.5,
    #                                                              bisection_epsilon=bisection_epsilon)
    # F, K, L = compensator.F, compensator.K, compensator.L
    # performance = compute_performance(A, B, C, Q, R, W, V, F, K, L)
    # for tag in tag_str_list_cg:
    #     print(tag)
    # print(performance.ihc/result_dict['opt'].performance.ihc)
    # ####################################################################################################################


    ####################################################################################################################
    # Bootstrap model state space matrices vs true system

    # Match all systems coordinates with the true system
    ss_models_boot_trans = [ss_change_coordinates(ss_true, ss_model_boot, method='match')[0] for ss_model_boot in ss_models_boot]

    matrix_distplot(ss_true.A, ss_model_trans.A, [ss_model_boot_trans.A for ss_model_boot_trans in ss_models_boot_trans], title_str='A')
    matrix_distplot(ss_true.B, ss_model_trans.B, [ss_model_boot_trans.B for ss_model_boot_trans in ss_models_boot_trans], title_str='B')
    matrix_distplot(ss_true.C, ss_model_trans.C, [ss_model_boot_trans.C for ss_model_boot_trans in ss_models_boot_trans], title_str='C')
    ####################################################################################################################
