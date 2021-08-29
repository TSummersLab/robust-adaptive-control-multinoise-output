"""
Robust adaptive control via multiplicative noise from bootstrapped uncertainty estimates.
"""
# Authors: Ben Gravell and Tyler Summers

import os

import numpy as np

from utility.pickle_io import pickle_import

from rocoboom_out.common.config import EXPERIMENT_FOLDER
from rocoboom_out.common.problem_data_gen import load_system
from rocoboom_out.common.plotting import multi_plot, multi_plot_paper


def load_results(dirname_in):
    # filename_in = training_type+'_training_'+testing_type+'_testing_'+'comparison_results'+'.pickle'
    filename_in = 'comparison_results' + '.pickle'
    path_in = os.path.join(dirname_in, filename_in)
    data_in = pickle_import(path_in)
    output_dict, cost_are_true, t_hist, t_start_estimate, t_evals = data_in
    return output_dict, cost_are_true, t_hist, t_start_estimate, t_evals


def load_problem_data(dirname_in):
    # Load simulation options
    filename_in = 'sim_options.pickle'
    path_in = os.path.join(dirname_in, filename_in)
    sim_options = pickle_import(path_in)

    # training_type = sim_options['training_type']
    # testing_type = sim_options['testing_type']
    Ns = sim_options['Ns']
    Nb = sim_options['Nb']
    T = sim_options['T']
    system_idx = sim_options['system_idx']
    seed = sim_options['seed']
    bisection_epsilon = sim_options['bisection_epsilon']
    t_start_estimate = sim_options['t_start_estimate']
    t_explore = sim_options['t_explore']
    u_explore_var = sim_options['u_explore_var']
    u_exploit_var = sim_options['u_exploit_var']

    # Load system definition
    filename_in = 'system_dict.pickle'
    path_in = os.path.join(dirname_in, filename_in)
    n, m, p, A, B, C, D, Y, Q, R, W, V, U = load_system(path_in)

    return [Ns, Nb, T, system_idx, seed,
            bisection_epsilon, t_start_estimate, t_explore, u_explore_var, u_exploit_var,
            n, m, p, A, B, C, D, Y, Q, R, W, V, U]


def get_last_dir(parent_folder):
    num_max = 0
    name_max = None
    for name in next(os.walk(parent_folder))[1]:
        try:
            num = float(name.split('_')[0].replace('p', '.'))
            if num > num_max:
                num_max = num
                name_max = name
        except:
            pass
    return name_max


if __name__ == "__main__":
    experiment_folder = EXPERIMENT_FOLDER
    experiment = 'last'

    if experiment == 'last':
        experiment = get_last_dir(experiment_folder)

    dirname_in = os.path.join(experiment_folder, experiment)

    # training_type = 'offline'
    # testing_type = 'online'

    # Load the problem data into the main workspace (not needed for plotting)
    (Ns, Nb, T, system_idx, seed,
    bisection_epsilon, t_start_estimate, t_explore, u_explore_var, u_exploit_var,
    n, m, p, A, B, C, D, Y, Q, R, W, V, U) = load_problem_data(dirname_in)

    # Load results
    output_dict, cost_are_true, t_hist, t_start_estimate, t_evals = load_results(dirname_in)

    # # Print the log data, if it exists
    # try:
    #     for log_str in output_dict['robust']['log_str']:
    #         print(log_str)
    # except:
    #     pass

    # Plotting
    multi_plot_paper(output_dict, cost_are_true, t_hist, t_start_estimate, t_evals, save_plots=True, dirname_out=dirname_in)
