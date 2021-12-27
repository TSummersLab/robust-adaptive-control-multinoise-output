"""
Robust adaptive control via multiplicative noise from bootstrapped uncertainty estimates.
"""
# Authors: Ben Gravell and Tyler Summers

import os
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from utility.pickle_io import pickle_import

from rocoboom_out.common.config import EXPERIMENT_FOLDER
from rocoboom_out.common.problem_data_gen import load_system
from rocoboom_out.common.plotting import multi_plot_paper
# from rocoboom_out.common.plotting import multi_plot


def cat_dict(base, layer):
    for key in base.keys():
        if type(base[key]) is dict and type(layer[key]) is dict:
            cat_dict(base[key], layer[key])
        elif type(base[key]) is np.ndarray and type(layer[key]) is np.ndarray:
            base[key] = np.vstack([base[key], layer[key]])
        elif type(base[key]) is list and type(layer[key]) is list:
            base[key].append(layer[key])
        else:
            pass
    return


def load_results(experiment):
    dirname_in = os.path.join(experiment_folder, experiment)
    filename_in = 'comparison_results' + '.pickle'
    path_in = os.path.join(dirname_in, filename_in)
    data_in = pickle_import(path_in)
    output_dict, cost_are_true, t_hist, t_start_estimate, t_evals = data_in
    return output_dict, cost_are_true, t_hist, t_start_estimate, t_evals


def aggregate_experiment_results(experiments):
    output_dict = None

    for i, experiment in enumerate(experiments):
        output_dict_i, cost_are_true, t_hist, t_start_estimate, t_evals = load_results(experiment)

        if output_dict is None:
            output_dict = copy(output_dict_i)
        else:
            cat_dict(output_dict, output_dict_i)
    return output_dict, cost_are_true, t_hist, t_start_estimate, t_evals


def load_problem_data(experiment):
    # Load simulation options
    dirname_in = os.path.join(experiment_folder, experiment)
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


def get_dirs_by_time(parent_folder, time_min, time_max):
    dirs = []
    for name in next(os.walk(parent_folder))[1]:
        try:
            num = float(name.split('_')[0].replace('p', '.'))
            if time_min <= num <= time_max+1:
                dirs.append(name)
        except:
            pass
    return dirs
    

if __name__ == "__main__":
    experiment_folder = EXPERIMENT_FOLDER

    # experiment = get_last_dir(experiment_folder)
    # experiment = '1637467652p5037403_Ns_10000_T_321_system_1_seed_1'
    # experiment = '1637595430p9019032_Ns_10000_T_321_system_1_seed_2'
    # experiment = '1637679639p3378303_Ns_10000_T_321_system_1_seed_10'
    # experiment = '1637770954p6599007_Ns_10_T_321_system_1_seed_1'

    # # Load the problem data into the main workspace
    # # (not needed for plotting, just for convenience of data inspection using console)
    # (Ns, Nb, T, system_idx, seed,
    # bisection_epsilon, t_start_estimate, t_explore, u_explore_var, u_exploit_var,
    # n, m, p, A, B, C, D, Y, Q, R, W, V, U) = load_problem_data(experiment)

    # Load results
    # output_dict, cost_are_true, t_hist, t_start_estimate, t_evals = load_results(experiment)

    # Load results from multiple experiments
    # experiments = ['1637770954p6599007_Ns_10_T_321_system_1_seed_1',
    #                '1637770969p3941662_Ns_10_T_321_system_1_seed_2',
    #                '1637770985p5349584_Ns_10_T_321_system_1_seed_3']
    # experiments = get_dirs_by_time(experiment_folder, 1637770954, 1637770985)  # 10
    experiments = get_dirs_by_time(experiment_folder, 1637467652, 1637679639)  # 10000
    experiments.reverse()  # This is only because seed=1 has extra fields that should not be catted, redo seed=1 with current code to eliminate this line
    #
    output_dict, cost_are_true, t_hist, t_start_estimate, t_evals = aggregate_experiment_results(experiments)

    # # Print the log data, if it exists
    # show_diagnostics = False
    # if show_diagnostics:
    #     try:
    #         for log_str in output_dict['robust']['log_str']:
    #             print(log_str)
    #     except:
    #         pass

    dirname_out = experiment_folder

    # Plotting
    plt.close('all')
    multi_plot_paper(output_dict, cost_are_true, t_hist, t_start_estimate, t_evals,
                     plotfun_str='plot', show_print=True, show_plot=True,
                     save_plots=False, dirname_out=dirname_out)


    # multi_plot_paper(output_dict, cost_are_true, t_hist, t_start_estimate, t_evals,
    #                  plotfun_str='plot', show_print=True, show_plot=True,
    #                  show_legend=False, show_grid=True,
    #                  save_plots=False, dirname_out=dirname_out)
    # multi_plot_paper(output_dict, cost_are_true, t_hist, t_start_estimate, t_evals,
    #                  plotfun_str='plot', show_print=True, show_plot=True,
    #                  show_legend=True, show_grid=False,
    #                  save_plots=False, dirname_out=dirname_out)
    plt.show()