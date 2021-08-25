"""
Robust adaptive control from output measurements via multiplicative noise from bootstrapped uncertainty estimates.
"""


from time import time
import os
import multiprocessing as mp

import numpy as np
import numpy.linalg as la
import numpy.random as npr

from utility.matrixmath import mdot, specrad, minsv, lstsqb, dlyap, dare, dare_gain

from utility.pickle_io import pickle_export
from utility.path_utility import create_directory
from utility.user_input import yes_or_no
from utility.printing import printcolors, create_tag

from rocoboom_out.common.problem_data_gen import gen_system_omni, save_system
from rocoboom_out.common.ss_tools import make_ss
from rocoboom_out.common.sim import make_offline_data
from rocoboom_out.common.sysid import system_identification
from rocoboom_out.common.uncertainty import estimate_model_uncertainty
from rocoboom_out.common.compensator_design import make_compensator, sysmat_cl
from rocoboom_out.common.compensator_eval import compute_performance


def monte_carlo_sample(control_scheme, uncertainty_estimator, required_args,
                       x_train_hist, u_train_hist, y_train_hist, w_hist, v_hist,
                       monte_carlo_idx, print_diagnostics=False, log_diagnostics=True):
    log_str = ''
    if log_diagnostics:
        code_start_time = time()
        log_str += 'Monte Carlo sample %d \n' % (monte_carlo_idx+1)

    # Unpack arguments from dictionary
    n = required_args['n']
    m = required_args['m']
    p = required_args['p']
    A = required_args['A']
    B = required_args['B']
    C = required_args['C']
    D = required_args['D']
    Y = required_args['Y']
    Q = required_args['Q']
    R = required_args['R']
    W = required_args['W']
    V = required_args['V']
    Ns = required_args['Ns']
    Nb = required_args['Nb']
    T = required_args['T']
    x0 = required_args['x0']
    bisection_epsilon = required_args['bisection_epsilon']
    t_start_estimate = required_args['t_start_estimate']
    t_explore = required_args['t_explore']
    t_cost_fh = required_args['t_cost_fh']
    cost_horizon = required_args['cost_horizon']
    Kare_true = required_args['Kopt']
    Lare_true = required_args['Lopt']
    u_explore_var = required_args['u_explore_var']
    u_exploit_var = required_args['u_exploit_var']
    noise_pre_scale = required_args['noise_pre_scale']
    noise_post_scale = required_args['noise_post_scale']

    if control_scheme == 'certainty_equivalent':
        noise_post_scale = 0

    # Preallocate history arrays
    # State and input
    x_test_hist = np.zeros([T+1, n])
    u_test_hist = np.zeros([T, m])
    y_test_hist = np.zeros([T, p])

    x_opt_test_hist = np.zeros([T+1, n])
    u_opt_test_hist = np.zeros([T, m])
    y_opt_test_hist = np.zeros([T, p])

    # Gain
    K_hist = np.zeros([T, m, n])
    L_hist = np.zeros([T, n, p])

    # Nominal model
    Ahat_hist = np.full([T, n, n], np.inf)
    Bhat_hist = np.full([T, n, m], np.inf)
    Chat_hist = np.full([T, p, n], np.inf)

    # Model uncertainty
    a_hist = np.full([T, n*n], np.inf)
    b_hist = np.full([T, n*m], np.inf)
    c_hist = np.full([T, p*n], np.inf)

    Aahist = np.full([T, n*n, n, n], np.inf)
    Bbhist = np.full([T, n*m, n, m], np.inf)
    Cchist = np.full([T, p*n, p, n], np.inf)

    gamma_reduction_hist = np.ones(T)

    # Spectral radius
    specrad_hist = np.full(T, np.inf)

    # Cost
    cost_future_hist = np.full(T, np.inf)
    cost_adaptive_hist = np.full(T, np.inf)
    cost_optimal_hist = np.full(T, np.inf)

    # Model error
    Aerr_hist = np.full(T, np.inf)
    Berr_hist = np.full(T, np.inf)
    Cerr_hist = np.full(T, np.inf)



    # Loop over time
    for t in range(T):
        if log_diagnostics:
            stable_str = printcolors.Green+'Stabilized'+printcolors.Default
            tag_str_list = []

        # Only use the training data we have observed up until now (do not cheat by looking ahead into the full history)

        if t < t_start_estimate:
            u_str = "Explore"
        else:
            if not t in t_evals:
                cost_future_hist[t] = -1
                continue

            print(t)
            # Start generating model and uncertainty estimates once there is enough data to get non-degenerate estimates
            # Estimate state space model from input-output data via subspace ID using the SIPPY package
            model, res = system_identification(y_train_hist[0:t], u_train_hist[0:t], id_method='N4SID',
                                               SS_fixed_order=n, return_residuals=True)
            w_est = res[0:n].T
            v_est = res[n:].T

            Ahat = model.A
            Bhat = model.B
            Chat = model.C

            # Record model estimates and errors
            Ahat_hist[t] = Ahat
            Bhat_hist[t] = Bhat
            Chat_hist[t] = Chat

            # Estimate model uncertainty
            if control_scheme == 'robust':
                uncertainty = estimate_model_uncertainty(model, u_train_hist, y_train_hist, w_est, v_est, t, Nb,
                                                                 uncertainty_estimator)
                # Record multiplicative noise history
                a_hist[t] = uncertainty.a
                b_hist[t] = uncertainty.b
                c_hist[t] = uncertainty.c
                Aahist[t] = uncertainty.Aa
                Bbhist[t] = uncertainty.Bb
                Cchist[t] = uncertainty.Cc
            else:
                uncertainty = None

            # Exploit estimated model
            if log_diagnostics:
                u_str = 'Exploit'


            # # TODO do the same thing with the C matrix, observable
            # # Check if estimated system is controllable within a tolerance
            # # If not, perturb the estimated B matrix until it is
            # ctrb_tol = 1e-4
            # while minsv(ctrb(Ahat, Bhat)) < ctrb_tol:
            #     if log_diagnostics:
            #         tag_str_list.append(create_tag('Estimated system uncontrollable, adjusted Bhat'))
            #     Bhat += 0.00001*npr.randn(n, m)

            compensator, gamma_reduction, tag_str_list_cg = make_compensator(model, uncertainty, Y, R,
                                                                         noise_pre_scale, noise_post_scale,
                                                                         bisection_epsilon, log_diagnostics)
            F, K, L = compensator.F, compensator.K, compensator.L
            gamma_reduction_hist[t] = gamma_reduction
            if log_diagnostics:
                tag_str_list += tag_str_list_cg


            # # Compute exploration control component
            # if training_type == 'online':
            #     if control_scheme == 'robust':
            #         u_explore_scale = np.sqrt(np.max(a)) + np.sqrt(np.max(b))
            #         u_explore = u_explore_scale*np.sqrt(u_exploit_var)*u_train_hist[t]
            #     else:
            #         u_explore = np.sqrt(u_exploit_var)*npr.randn(m)
            # else:
            #     u_explore = np.zeros(m)

            # # Compute control using optimal control using estimate of system
            # u_optimal_estimated = mdot(K, x_test)

            # # Apply the sum of optimal and exploration controls
            # u = u_optimal_estimated + u_explore
            #
            # # Compute control using optimal control given knowledge of true system
            # u_opt = mdot(Kare_true, x_opt_test)

            # Evaluate spectral radius of true closed-loop system with current compensator
            performance = compute_performance(A, B, C, Q, R, W, V, F, K, L)
            specrad_hist[t], cost_future_hist[t] = performance.sr, performance.ihc
            if log_diagnostics:
                if specrad_hist[t] > 1:
                    stable_str = printcolors.Red+'Unstable'+printcolors.Default

            # Record compensator
            # TODO Ahat, Bhat, Chat, What, Vhat, maybe F
            K_hist[t] = K
            L_hist[t] = L


        # if log_diagnostics:
        #     if la.norm(x_test) > 1e4:
        #         tag_str_list.append(create_tag('x_test = %e > 1e3' % (la.norm(x_test)), message_type='fail'))

        # # Accumulate cost
        # if testing_type == 'online':
        #     cost_adaptive_hist[t] = mdot(x_test.T, Q, x_test) + mdot(u.T, R, u)
        #     cost_optimal_hist[t] = mdot(x_opt_test.T, Q, x_opt_test) + mdot(u_opt.T, R, u_opt)

        # # Look up noise
        # w = w_hist[t]
        # v = v_hist[t]

        # # Update test-time state
        # if testing_type == 'online':
        #     if training_type == 'online':
        #         x_test = np.copy(x_train)
        #     else:
        #         # Update state under adaptive control
        #         x_test = np.dot(A, x_test) + np.dot(B, u) + w
        #
        #     # Update the test-time state under optimal control
        #     x_opt_test = np.dot(A, x_opt_test) + np.dot(B, u_opt) + w
        #
        #     # Record test-time state and control history
        #     x_test_hist[t+1] = x_test
        #     x_opt_test_hist[t+1] = x_opt_test
        #     u_test_hist[t] = u
        #     u_opt_test_hist[t] = u_opt


        # Print and log diagnostic messages
        if log_diagnostics:
            time_whole_str = ''
            time_header_str = "Time = %4d  %s. %s." % (t, u_str, stable_str)
            time_whole_str += time_header_str + '\n'
            for tag_str in tag_str_list:
                time_whole_str += tag_str + '\n'
            log_str += time_whole_str
        if print_diagnostics:
            print(time_whole_str)
    if log_diagnostics:
        code_end_time = time()
        code_elapsed_time = code_end_time - code_start_time
        time_elapsed_str = '%12.6f' % code_elapsed_time
        log_str += "Completed Monte Carlo sample %6d / %6d in %s seconds\n" % (monte_carlo_idx+1, Ns, time_elapsed_str)
    else:
        time_elapsed_str = '?'

    return {'cost_adaptive_hist': cost_adaptive_hist,
            'cost_optimal_hist': cost_optimal_hist,
            'cost_future_hist': cost_future_hist,
            'specrad_hist': specrad_hist,
            'Ahat_hist': Ahat_hist,
            'Bhat_hist': Bhat_hist,
            'Chat_hist': Chat_hist,
            'Aerr_hist': Aerr_hist,
            'Berr_hist': Berr_hist,
            'Cerr_hist': Cerr_hist,
            'a_hist': a_hist,
            'b_hist': b_hist,
            'c_hist': c_hist,
            'Aahist': Aahist,
            'Bbhist': Bbhist,
            'Cchist': Cchist,
            'gamma_reduction_hist': gamma_reduction_hist,
            'x_train_hist': x_train_hist,
            'u_train_hist': u_train_hist,
            'y_train_hist': u_train_hist,
            'x_test_hist': x_test_hist,
            'u_test_hist': u_test_hist,
            'y_test_hist': y_test_hist,
            'x_opt_test_hist': x_opt_test_hist,
            'u_opt_test_hist': u_opt_test_hist,
            'y_opt_test_hist': y_opt_test_hist,
            'K_hist': K_hist,
            'L_hist': L_hist,
            'monte_carlo_idx': np.array(monte_carlo_idx),
            'log_str': log_str,
            'time_elapsed_str': time_elapsed_str}


def monte_carlo_group(control_scheme, uncertainty_estimator, required_args,
                      conditional_args, w_hist, v_hist, parallel_option='serial'):
    print("Evaluating control scheme: "+control_scheme)

    # Unpack arguments from dictionaries
    n = required_args['n']
    m = required_args['m']
    p = required_args['p']
    Ns = required_args['Ns']
    T = required_args['T']

    # History arrays
    x_train_hist = conditional_args['x_train_hist']
    u_train_hist = conditional_args['u_train_hist']
    y_train_hist = conditional_args['y_train_hist']

    # Simulate each Monte Carlo trial
    shape_dict = {'x_train_hist': [T+1, n],
                  'u_train_hist': [T, m],
                  'y_train_hist': [T, p],
                  'x_test_hist': [T+1, n],
                  'x_opt_test_hist': [T+1, n],
                  'u_test_hist': [T, m],
                  'u_opt_test_hist': [T, m],
                  'y_test_hist': [T, p],
                  'y_opt_test_hist': [T, p],
                  'K_hist': [T, m, n],
                  'L_hist': [T, n, p],
                  'Ahat_hist': [T, n, n],
                  'Bhat_hist': [T, n, m],
                  'Chat_hist': [T, p, n],
                  'a_hist': [T, n*n],
                  'b_hist': [T, n*m],
                  'c_hist': [T, p*n],
                  'Aahist': [T, n*n, n, n],
                  'Bbhist': [T, n*m, n, m],
                  'Cchist': [T, p*n, p, n],
                  'gamma_reduction_hist': [T],
                  'specrad_hist': [T],
                  'cost_future_hist': [T],
                  'cost_adaptive_hist': [T],
                  'cost_optimal_hist': [T],
                  'Aerr_hist': [T],
                  'Berr_hist': [T],
                  'Cerr_hist': [T],
                  'monte_carlo_idx': [1],
                  'log_str': None,
                  'time_elapsed_str': None}
    fields = shape_dict.keys()
    output_dict = {}
    for field in fields:
        if field == 'log_str' or field == 'time_elapsed_str':
            output_dict[field] = [None]*Ns
        else:
            output_field_shape = [Ns] + shape_dict[field]
            output_dict[field] = np.zeros(output_field_shape)

    def collect_result(sample_dict):
        k = sample_dict['monte_carlo_idx']
        time_elapsed_str = sample_dict['time_elapsed_str']
        for field in fields:
            output_dict[field][k] = sample_dict[field]
        print("Completed Monte Carlo sample %6d / %6d in %s seconds" % (k+1, Ns, time_elapsed_str))

    if parallel_option == 'parallel':
        # Start the parallel process pool
        num_cpus_to_use = mp.cpu_count() - 1
        pool = mp.Pool(num_cpus_to_use)

    for k in range(Ns):
        sample_args = (control_scheme, uncertainty_estimator, required_args,
                       x_train_hist[k], u_train_hist[k], y_train_hist[k], w_hist[k], v_hist[k], k)

        if parallel_option == 'serial':
            # Serial single-threaded processing
            sample_dict = monte_carlo_sample(*sample_args)
            collect_result(sample_dict)

        elif parallel_option == 'parallel':
            # Asynchronous parallel CPU processing
            pool.apply_async(monte_carlo_sample, args=sample_args, callback=collect_result)

    if parallel_option == 'parallel':
        # Close and join the parallel process pool
        pool.close()
        pool.join()

    print('')
    return output_dict


def compute_derived_data(output_dict, receding_horizon=5):
    """
    Compute derived cost data quantities from the results.
    These derived data can be computed and stored for faster loading/plotting,
    or can be calculated after loading to reduce data storage requirements.

    output_dict is modified/mutated
    """

    for control_scheme in output_dict.keys():
        cost_adaptive_hist = output_dict[control_scheme]['cost_adaptive_hist']
        cost_optimal_hist = output_dict[control_scheme]['cost_optimal_hist']

        # Compute receding horizon data
        Ns, T = cost_adaptive_hist.shape
        cost_adaptive_hist_receding = np.full([Ns, T], np.inf)
        cost_optimal_hist_receding = np.full([Ns, T], np.inf)
        for k in range(Ns):
            for t in range(T):
                if t > receding_horizon:
                    cost_adaptive_hist_receding[k, t] = np.mean(cost_adaptive_hist[k, t-receding_horizon:t])
                    cost_optimal_hist_receding[k, t] = np.mean(cost_optimal_hist[k, t-receding_horizon:t])

        # Compute accumulated cost
        cost_adaptive_hist_accum = np.full([Ns, T], np.inf)
        cost_optimal_hist_accum = np.full([Ns, T], np.inf)
        for k in range(Ns):
            for t in range(T):
                cost_adaptive_hist_accum[k, t] = np.sum(cost_adaptive_hist[k, 0:t])
                cost_optimal_hist_accum[k, t] = np.sum(cost_optimal_hist[k, 0:t])

        # Compute regret and regret_ratio
        regret_hist = cost_adaptive_hist - np.mean(cost_optimal_hist, axis=0)
        regret_hist_receding = cost_adaptive_hist_receding - np.mean(cost_optimal_hist_receding, axis=0)
        regret_hist_accum = cost_adaptive_hist_accum - np.mean(cost_optimal_hist_accum, axis=0)

        regret_ratio_hist = cost_adaptive_hist / np.mean(cost_optimal_hist, axis=0)
        regret_ratio_hist_receding = cost_adaptive_hist_receding / np.mean(cost_optimal_hist_receding, axis=0)
        regret_ratio_hist_accum = cost_adaptive_hist_accum / np.mean(cost_optimal_hist_accum, axis=0)

        output_dict[control_scheme]['cost_adaptive_hist_receding'] = cost_adaptive_hist_receding
        output_dict[control_scheme]['cost_optimal_hist_receding'] = cost_optimal_hist_receding
        output_dict[control_scheme]['cost_adaptive_hist_accum'] = cost_adaptive_hist_accum
        output_dict[control_scheme]['cost_optimal_hist_accum'] = cost_optimal_hist_accum
        output_dict[control_scheme]['regret_hist'] = regret_hist
        output_dict[control_scheme]['regret_hist_receding'] = regret_hist_receding
        output_dict[control_scheme]['regret_hist_accum'] = regret_hist_accum
        output_dict[control_scheme]['regret_ratio_hist'] = regret_ratio_hist
        output_dict[control_scheme]['regret_ratio_hist_receding'] = regret_ratio_hist_receding
        output_dict[control_scheme]['regret_ratio_hist_accum'] = regret_ratio_hist_accum


def mainfun(uncertainty_estimator, Ns, Nb, T, noise_pre_scale, noise_post_scale,
            cost_horizon, horizon_method, t_cost_fh, system_idx, system_kwargs, seed, parallel_option):
    # Set up output directory
    timestr = str(time()).replace('.', 'p')
    dirname_out = timestr+'_Ns_'+str(Ns)+'_T_'+str(T)+'_system_'+str(system_idx)+'_seed_'+str(seed)
    dirname_out = os.path.join('..', 'experiments', dirname_out)
    create_directory(dirname_out)

    # Seed the random number generator
    npr.seed(seed)

    # Problem data
    n, m, p, A, B, C, D, Y, Q, R, W, V = gen_system_omni(system_idx, **system_kwargs)
    filename_out = 'system_dict.pickle'
    save_system(n, m, p, A, B, C, D, Y, Q, R, W, V, dirname_out, filename_out)
    model_true = make_ss(A, B, C, D)

    # Catch numerical error-prone case when system is open-loop unstable and not using control during training
    if specrad(A) > 1:
        response = yes_or_no("System is open-loop unstable, offline trajectories may cause numerical issues. Continue?")
        if not response:
            return

    # Initial state
    x0 = np.zeros(n)

    # Compare with the true optimal gains given perfect information of the system
    # IMPORTANT: cannot compare gains directly because the internal state representation is different
    # Only compare closed-loop transfer functions or closed-loop performance cost
    Popt, Kopt = dare_gain(A, B, Q, R)
    Sopt, Lopt = dare_gain(A.T, C.T, W, V)
    Lopt = -Lopt.T

    Fopt = sysmat_cl(A, B, C, Kopt, Lopt)
    performance_true = compute_performance(A, B, C, Q, R, W, V, Fopt, Kopt, Lopt)
    specrad_true, cost_are_true = performance_true.sr, performance_true.ihc



    # Time history
    t_hist = np.arange(T)

    # Time to begin forming model estimates
    t_start_estimate_lwr = int(n*(n+m+p)/p)

    # t_start_estimate = 2*t_start_estimate_lwr
    # TODO use the auto value instead of this hardcode value
    # TODO Figure out why SIPPY fails with t_start_estimate = 2*t_start_estimate_lwr - maybe the horizon length SS_f in system_identification?
    t_start_estimate = 100

    if t_start_estimate < t_start_estimate_lwr:
        response = yes_or_no("t_start_estimate chosen < int(n*(n+m+p)/p). Continue?")
        if not response:
            return

    # TODO remove this if unused
    # Time to switch from exploration to exploitation
    t_explore = t_start_estimate+1
    if t_explore < t_start_estimate_lwr+1:
        response = yes_or_no("t_explore chosen < int(n*(n+m+p)/p) + 1. Continue?")
        if not response:
            return

    # TODO make this choice explicit in the paper! practically we do not have knowledge of W so this is not exactly fair
    # Input exploration noise during explore and exploit phases
    u_explore_var = np.max(la.eig(W)[0])
    u_exploit_var = np.max(la.eig(W)[0])


    # Bisection tolerance
    bisection_epsilon = 0.01

    # Export the simulation options for later reference
    sim_options = {'uncertainty_estimator': uncertainty_estimator,
                   'Ns': Ns,
                   'Nb': Nb,
                   'T': T,
                   'system_idx': system_idx,
                   'seed': seed,
                   'bisection_epsilon': bisection_epsilon,
                   't_start_estimate': t_start_estimate,
                   't_explore': t_explore,
                   'u_explore_var': u_explore_var,
                   'u_exploit_var': u_exploit_var}
    filename_out = 'sim_options.pickle'
    pickle_export(dirname_out, filename_out, sim_options)

    # control_schemes = ['certainty_equivalent']
    control_schemes = ['certainty_equivalent', 'robust']
    output_dict = {}

    # Generate sample trajectory data (pure exploration)
    x_train_hist, u_train_hist, y_train_hist, w_hist, v_hist = make_offline_data(A, B, C, D, W, V, Ns, T, u_explore_var, x0)

    # Evaluate control schemes
    required_args = {'n': n,
                     'm': m,
                     'p': p,
                     'A': A,
                     'B': B,
                     'C': C,
                     'D': D,
                     'Y': Y,
                     'Q': Q,
                     'R': R,
                     'W': W,
                     'V': V,
                     'Ns': Ns,
                     'Nb': Nb,
                     'T': T,
                     'x0': x0,
                     'bisection_epsilon': bisection_epsilon,
                     't_start_estimate': t_start_estimate,
                     't_explore': t_explore,
                     't_cost_fh': t_cost_fh,
                     'cost_horizon': cost_horizon,
                     'Kopt': Kopt,
                     'Lopt': Lopt,
                     'u_explore_var': u_explore_var,
                     'u_exploit_var': u_exploit_var,
                     'noise_pre_scale': noise_pre_scale,
                     'noise_post_scale': noise_post_scale}

    conditional_args = {'x_train_hist': x_train_hist,
                        'u_train_hist': u_train_hist,
                        'y_train_hist': y_train_hist}

    for control_scheme in control_schemes:
        output_dict[control_scheme] = monte_carlo_group(control_scheme=control_scheme,
                                                        uncertainty_estimator=uncertainty_estimator,
                                                        required_args=required_args,
                                                        conditional_args=conditional_args,
                                                        w_hist=w_hist,
                                                        v_hist=v_hist,
                                                        parallel_option=parallel_option)

    compute_derived_data(output_dict)

    # # Export relevant data
    # filename_out = training_type+'_training_'+testing_type+'_testing_'+'comparison_results'+'.pickle'
    # data_out = [output_dict, cost_are_true, t_hist, t_explore]
    # pickle_export(dirname_out, filename_out, data_out)

    return output_dict, cost_are_true, t_hist, t_start_estimate, t_evals


if __name__ == "__main__":
    # # Choose between offline and online training data
    # training_type = 'offline'
    #
    # # Choose between offline and online testing
    # testing_type = 'online'

    # Choose the uncertainty estimation scheme
    # uncertainty_estimator = 'exact'
    # uncertainty_estimator = 'sample_transition_bootstrap'
    uncertainty_estimator = 'semiparametric_bootstrap'

    # Number of Monte Carlo samples
    # Ns = 100000
    # Ns = 10
    Ns = 2

    # Number of bootstrap samples
    Nb = 50

    # Simulation time
    t_evals = np.arange(200, 300+1, 50)
    # t_evals = np.arange(200, 600+1, 50)
    T = np.max(t_evals)+1

    # Choose noise_pre_scale (AKA gamma), the pre-limit multiplicative noise scaling parameter, should be >= 1
    # "How much mult noise do you want?"
    noise_pre_scale = 1.0
    # noise_pre_scale = 0.001

    # Choose the post-limit multiplicative noise scaling parameter, must be between 0 and 1
    # "How much of the max possible mult noise do you want?"
    noise_post_scale = 1.0
    # noise_post_scale = 1 / noise_pre_scale

    # Choose cost horizon
    cost_horizon = 'infinite'
    horizon_method = None
    t_cost_fh = None


    # Random number generator seed
    seed = 1
    # seed = npr.randint(1000)

    # System to choose
    # system_idx = 'scalar'
    system_idx = 'rand'

    if system_idx == 'scalar':
        system_kwargs = dict(A=1, B=1, Q=1, R=0, W=1, V=0.1)
    elif system_idx == 'rand':
        system_kwargs = dict(n=4, m=2, p=2, spectral_radius=0.9, noise_scale=0.00001, seed=1)
        # system_kwargs = dict(n=2, m=1, p=1, spectral_radius=0.9, noise_scale=0.0001, seed=1)
    else:
        system_kwargs = dict()

    # Parallel computation option
    parallel_option = 'serial'
    # parallel_option = 'parallel'

    # Run main
    output_dict, cost_are_true, t_hist, t_start_estimate, t_evals = mainfun(uncertainty_estimator, Ns, Nb, T, noise_pre_scale, noise_post_scale,
                          cost_horizon, horizon_method, t_cost_fh, system_idx, system_kwargs, seed, parallel_option)


    # Plotting
    import matplotlib.pyplot as plt
    from plotting import multi_plot, multi_plot_paper
    plt.close('all')

    # multi_plot(output_dict, cost_are_true, t_hist, t_start_estimate)
    multi_plot_paper(output_dict, cost_are_true, t_hist, t_start_estimate, t_evals)