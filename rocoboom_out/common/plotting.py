"""
Plotting functions
"""


from copy import copy
from warnings import warn
from time import sleep
import os

import numpy as np
import numpy.linalg as la
from scipy.stats import trim_mean
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.cm import get_cmap

from utility.matrixmath import specrad


BIG = 1e100  # A big number to replace np.inf for plotting purposes


def compute_transparencies(quantiles, quantile_fill_alpha):
    # Manually compute alphas of overlapping regions for legend patches
    quantile_alphas = []
    for j, quantile in enumerate(quantiles):
        if j > 0:
            quantile_alpha_old = quantile_alphas[j - 1]
            quantile_alpha_new = quantile_fill_alpha + (1 - quantile_fill_alpha)*quantile_alpha_old
        else:
            quantile_alpha_new = quantile_fill_alpha
        quantile_alphas.append(quantile_alpha_new)
    return quantile_alphas


def multi_plot_paper(output_dict, cost_are_true, t_hist, t_start_estimate, t_evals,
                     show_print=True, show_plot=True,
                     plotfun_str='plot', xscale='linear', yscale='symlog',
                     show_mean=True, show_median=True, show_trimmed_mean=False, show_quantiles=True,
                     trim_mean_quantile=None, quantile_fill_alpha=0.2, quantile_color='tab:blue',
                     quantile_region='upper', quantile_style='fill', quantile_legend=True,
                     stat_diff_type='stat_of_diff',
                     show_xlabel=True, show_ylabel=True, show_title=True, show_legend=True,
                     show_grid=True, show_guideline=True, zoom=False,
                     figsize=(4.5, 3), save_plots=False, dirname_out=None):

    # from matplotlib import rc
    # rc('text', usetex=True)

    control_schemes = list(output_dict.keys())
    diff_scheme = control_schemes[0]+' minus '+control_schemes[1]

    plot_fields = ['cost_future_hist',
                   'cost_future_hist',
                   'cost_future_hist',
                   'specrad_hist',
                   'specrad_hist',
                   'specrad_hist',
                   'Aerr_hist',
                   'Berr_hist',
                   'Cerr_hist',
                   'a_hist',
                   'b_hist',
                   'c_hist',
                   'gamma_reduction_hist']
    plot_control_schemes = ['certainty_equivalent',
                            'robust',
                            diff_scheme,
                            'certainty_equivalent',
                            'robust',
                            diff_scheme,
                            'robust',
                            'robust',
                            'robust',
                            'robust',
                            'robust',
                            'robust',
                            'robust']
    ylabels = ['Inf.-horz. perf.',
               'Inf.-horz. perf.',
               'Inf.-horz. perf. diff.',
               'Spec. rad.',
               'Spec. rad.',
               'Spec. rad. diff.',
               r'$\Vert \hat{A}-A \Vert$',
               r'$\Vert \hat{B}-B \Vert$',
               r'$\Vert \hat{C}-C \Vert$',
               r'$a$',
               r'$b$',
               r'$c$',
               r'$c_\gamma$']
    filenames = ['cost_future_ce',
                 'cost_future_rmn',
                 'cost_future_diff',
                 'specrad_ce',
                 'specrad_rmn',
                 'specrad_diff',
                 'Aerr',
                 'Berr',
                 'Cerr',
                 'a',
                 'b',
                 'c',
                 'gamma_scale']

    quantiles = [1.00, 0.999, 0.99, 0.95, 0.75]
    quantiles = np.array(quantiles)

    # Set quantile level for trimmed mean
    if trim_mean_quantile is None:
        trim_mean_quantile = np.max(quantiles[quantiles < 1])

    quantile_alphas = compute_transparencies(quantiles, quantile_fill_alpha)

    # Process history data for plotting
    fields_to_normalize_by_cost_are_true = ['cost_future_hist']
    fields_to_mean = []
    fields_to_absmax = ['a_hist', 'b_hist', 'c_hist']
    fields_to_vecnorm = ['x_train_hist', 'x_test_hist', 'u_train_hist', 'u_test_hist', 'x_opt_test_hist']
    fields_to_fronorm = ['K_hist']
    fields_to_squeeze = []
    fields_to_truncate = ['x_train_hist', 'x_test_hist', 'x_opt_test_hist']

    # Make the list of statistic names
    statistics = ['ydata', 'mean', 'trimmed_mean', 'median']
    for quantile in quantiles:
        statistics.append('quantile_'+str(quantile))
        statistics.append('quantile_'+str(1-quantile))

    # Build the ydata dictionary from output_dict
    ydata_dict = {}
    for control_scheme in control_schemes:
        ydata_dict[control_scheme] = {}
        for field in plot_fields:
            ydata_dict[control_scheme][field] = {}
            # Preprocessing
            if field in fields_to_normalize_by_cost_are_true:
                ydata = (output_dict[control_scheme][field] / cost_are_true)
            elif field in fields_to_mean:
                ydata = np.mean(output_dict[control_scheme][field], axis=2)
            elif field in fields_to_absmax:
                ydata = np.max(np.abs(output_dict[control_scheme][field]), axis=2)
            elif field in fields_to_vecnorm:
                ydata = la.norm(output_dict[control_scheme][field], axis=2)
            elif field in fields_to_fronorm:
                ydata = la.norm(output_dict[control_scheme][field], ord='fro', axis=(2, 3))
            else:
                ydata = output_dict[control_scheme][field]
            if field in fields_to_squeeze:
                ydata = np.squeeze(ydata)
            if field in fields_to_truncate:
                ydata = ydata[:, :-1]
            # Convert nan to inf
            ydata[np.isnan(ydata)] = np.inf
            # Convert inf to big number (needed for plotting so that inf values are not neglected)
            ydata[np.isinf(ydata)] = BIG
            # Store processed data
            ydata_dict[control_scheme][field]['ydata'] = ydata
            # Compute statistics
            ydata_dict[control_scheme][field]['mean'] = np.mean(ydata, axis=0)
            ydata_dict[control_scheme][field]['trimmed_mean'] = trim_mean(ydata, proportiontocut=1-trim_mean_quantile, axis=0)
            ydata_dict[control_scheme][field]['median'] = np.median(ydata, axis=0)
            for quantile in quantiles:
                ydata_dict[control_scheme][field]['quantile_'+str(quantile)] = np.quantile(ydata, quantile, axis=0)
                ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)] = np.quantile(ydata, 1-quantile, axis=0)

    # Compute statistic differences
    control_scheme = control_schemes[0] + ' minus ' + control_schemes[1]
    control_schemes.append(control_scheme)
    ydata_dict[control_scheme] = {}
    for field in plot_fields:
        ydata_dict[control_scheme][field] = {}
        for statistic in statistics:
            # Choose whether to calculate statistics before or after taking the difference
            if stat_diff_type=='diff_of_stat':
                stat1 = ydata_dict[control_schemes[0]][field][statistic]
                stat2 = ydata_dict[control_schemes[1]][field][statistic]
                ydata_dict[control_scheme][field][statistic] = stat1 - stat2
            elif stat_diff_type=='stat_of_diff':
                #TODO: reuse statistic computation code above
                ydata1 = ydata_dict[control_schemes[0]][field]['ydata']
                ydata2 = ydata_dict[control_schemes[1]][field]['ydata']
                ydata_diff = ydata1 - ydata2
                ydata_dict[control_scheme][field]['ydata'] = ydata_diff
                # Compute statistics
                ydata_dict[control_scheme][field]['median'] = np.median(ydata_diff, axis=0)
                ydata_dict[control_scheme][field]['mean'] = np.mean(ydata_diff, axis=0)
                ydata_dict[control_scheme][field]['trimmed_mean'] = trim_mean(ydata_diff, proportiontocut=1-trim_mean_quantile, axis=0)
                for quantile in quantiles:
                    ydata_dict[control_scheme][field]['quantile_'+str(quantile)] = np.quantile(ydata_diff, quantile, axis=0)
                    ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)] = np.quantile(ydata_diff, 1-quantile, axis=0)

    # x start index
    x_start_idx = t_start_estimate

    for control_scheme, field, ylabel_str, filename in zip(plot_control_schemes, plot_fields, ylabels, filenames):
        try:
            if show_plot:
                # Initialize figure and axes
                fig, ax = plt.subplots(figsize=figsize)

                # Choose the plotting function
                if plotfun_str == 'plot':
                    plotfun = ax.plot
                    fbstep = None
                elif plotfun_str == 'step':
                    plotfun = ax.step
                    fbstep = 'pre'

            # Choose the quantiles
            if field == 'gamma_reduction_hist':
                quantiles = [1.00, 0.95, 0.75]
                quantile_regions = ['middle']
            else:
                quantiles = [0.999, 0.99, 0.95]
                quantile_regions = ['upper', 'lower']
            quantiles = np.array(quantiles)

            legend_handles = []
            legend_labels = []

            if show_print:
                print(control_scheme)
                print(field)

            t_idxs = []
            for t in t_evals:
                t_idxs.append(np.where(t_hist == t)[0][0])
            xdata = t_hist[t_idxs]

            if show_print:
                print('Time')
                print(xdata)

            # Plot mean
            if show_mean:
                ydata = ydata_dict[control_scheme][field]['mean'][t_idxs]
                if show_plot:
                    artist, = plotfun(xdata, ydata, color='k', lw=3, marker='d', markersize=8, zorder=120)
                    legend_handles.append(artist)
                    legend_labels.append("Mean")
                if show_print:
                    print('Mean')
                    print(ydata)

            # Plot trimmed mean
            if show_trimmed_mean:
                ydata = ydata_dict[control_scheme][field]['trimmed_mean'][t_idxs]
                if show_plot:
                    artist, = plotfun(xdata, ydata, color='tab:grey', lw=3, zorder=130)
                    legend_handles.append(artist)
                    legend_labels.append("Trimmed mean, middle %.0f%%" % (100*(1-((1-trim_mean_quantile)*2))))
                if show_print:
                    print('Trimmed mean')
                    print(ydata)

            # Plot median
            if show_median:
                ydata = ydata_dict[control_scheme][field]['median'][t_idxs]
                if show_plot:
                    artist, = plotfun(xdata, ydata, color='b', lw=3, zorder=110)
                    legend_handles.append(artist)
                    legend_labels.append("Median")
                if show_print:
                    print('Median')
                    print(ydata)

            # Plot quantiles
            if show_quantiles:
                if show_plot:
                    def plot_quantiles(quantile_region, quantile_color):
                        qi = 0
                        my_quantiles = reversed(quantiles) if quantile_region == 'lower' else quantiles
                        my_quantile_alphas = reversed(quantile_alphas) if quantile_region == 'lower' else quantile_alphas
                        for quantile, quantile_alpha in zip(my_quantiles, my_quantile_alphas):
                            if quantile_region == 'upper':
                                y_lwr = ydata_dict[control_scheme][field]['median'][t_idxs]
                            else:
                                y_lwr = ydata_dict[control_scheme][field]['quantile_'+str(1-quantile)][t_idxs]
                            if quantile_region == 'lower':
                                y_upr = ydata_dict[control_scheme][field]['median'][t_idxs]
                            else:
                                y_upr = ydata_dict[control_scheme][field]['quantile_'+str(quantile)][t_idxs]

                            ax.fill_between(xdata, y_lwr, y_upr, step=fbstep,
                                            color=quantile_color, alpha=quantile_fill_alpha, zorder=qi)
                            qi += 1

                            if quantile_legend:
                                legend_handles.append(mpatches.Patch(color=quantile_color, alpha=quantile_alpha))
                                if quantile_region == 'middle':
                                    legend_label_str = "Middle %.1f%%" % (100*(1-((1-quantile)*2)))
                                elif quantile_region == 'upper':
                                    legend_label_str = "Upper %.1f%%" % (50*(1-((1-quantile)*2)))
                                elif quantile_region == 'lower':
                                    legend_label_str = "Lower %.1f%%" % (50*(1-((1-quantile)*2)))
                                legend_labels.append(legend_label_str)

                    for quantile_region in quantile_regions:
                        if quantile_region == 'upper' or quantile_region == 'middle':
                            quantile_color = 'tab:blue'
                        elif quantile_region == 'lower':
                            if 'minus' in control_scheme:
                                quantile_color = 'tab:red'
                            else:
                                quantile_color = 'tab:green'
                        plot_quantiles(quantile_region, quantile_color)

                # Print quantiles
                if show_print:
                    for quantile in quantiles:
                        print('Quantile % .1f%%'%(100*(1 - quantile)))
                        y_lwr = ydata_dict[control_scheme][field]['quantile_' + str(1 - quantile)][t_idxs]
                        print(y_lwr)
                    for quantile in reversed(quantiles):
                        print('Quantile % .1f%%'%(100*quantile))
                        y_upr = ydata_dict[control_scheme][field]['quantile_' + str(quantile)][t_idxs]
                        print(y_upr)

            # ax.set_xlim(xl)

            # Plot guidelines
            if show_plot:
                if show_guideline:
                    y_guide = np.zeros(ydata_dict[control_scheme][field]['ydata'].shape[1])[t_idxs]
                    if field in ['specrad_hist', 'gamma_reduction_hist', 'cost_future_hist'] and not 'minus' in control_scheme:
                        y_guide = np.ones(ydata_dict[control_scheme][field]['ydata'].shape[1])[t_idxs]
                    plotfun(xdata, y_guide, color='tab:grey', lw=2, linestyle='--', zorder=20)

                yscale = 'symlog'
                if field == 'gamma_reduction_hist':
                    yscale = 'linear'
                elif field in ['a_hist', 'b_hist', 'c_hist', 'Aerr_hist', 'Berr_hist', 'Cerr_hist']:
                    yscale = 'log'

                # Set axes options
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)

                if show_legend:
                    loc = 'best'
                    if field == 'regret_hist':
                        loc = 'center right'
                    elif field == 'gamma_reduction_hist':
                        loc = 'lower right'

                    leg = ax.legend(handles=legend_handles, labels=legend_labels, loc=loc)
                    leg.set_zorder(1000)


                # if field == 'regret_hist' and not control_scheme == diff_scheme:
                #     yl = [-0.1, 1e27]
                # else:
                #     ydata_lim_lwr = ydata_dict[control_scheme][field]['median'][x_start_idx:]
                #     ydata_lim_upr = ydata_dict[control_scheme][field]['quantile_'+str(max(quantiles))][x_start_idx:]
                #     ydata_lim_lwr = ydata_lim_lwr[np.isfinite(ydata_lim_lwr)]
                #     ydata_lim_upr = ydata_lim_upr[np.isfinite(ydata_lim_upr)]
                #     yl_lwr = np.min(ydata_lim_lwr)
                #     yl_upr = np.max(ydata_lim_upr)
                #     yl = [yl_lwr, yl_upr]
                # ax.set_ylim(yl)

                # Hardcode axis limits and ticks
                if not 'minus' in control_scheme:
                    if field == 'cost_future_hist':
                        yl = [0.98, 1.22]
                        ax.set_ylim(yl)
                        plt.yticks([1.0, 1.1, 1.2])
                        # pass
                    elif field == 'regret_hist':
                        # plt.locator_params(axis='y', numticks=6)
                        plt.yticks([0, 1e10, 1e20, 1e30, 1e40, 1e50, 1e60, 1e70, 1e80])
                        # pass
                    elif field == 'specrad_hist':
                        plt.yticks([0.0, 0.5, 1.0, 1.5])
                        # pass
                    elif field in ['a_hist', 'b_hist', 'c_hist']:
                        # plt.yticks([0, 0.5, 1, 1.5, 2])
                        pass
                    elif field == 'gamma_reduction_hist':
                        plt.yticks([0, 0.25, 0.5, 0.75, 1])
                else:
                    if field == 'cost_future_hist':
                        yl = [-0.02, 0.02]
                        ax.set_ylim(yl)
                        plt.yticks([-0.02, -0.01, 0.0, 0.01, 0.02])
                    elif field == 'specrad_hist':
                        yl = [-0.6, 0.6]
                        ax.set_ylim(yl)
                        plt.yticks([-0.6, -0.3, 0.0, 0.3, 0.6])

                # xl = [t_hist[x_start_idx], t_hist[-1]*1.25]
                # ax.set_xlim(xl)
                # if field == 'a_hist' or field == 'b_hist' or field == 'gamma_reduction_hist':
                #     xl = [t_hist[x_start_idx], 20]

                # Plot options
                if show_grid:
                    ax.grid('on')

                ax.set_axisbelow(True)

                if show_xlabel:
                    xlabel_str = 'Time'
                    ax.set_xlabel(xlabel_str, fontsize=12)
                if show_ylabel:
                    # rot = None
                    # ax.set_ylabel(ylabel_str, fontsize=12, rotation=rot)
                    ax.set_ylabel(ylabel_str, fontsize=12)

                if show_title:
                    # title_str = ylabel_str + '_' + control_scheme
                    title_str = control_scheme
                    title_str = title_str.replace('_', ' ').title()
                    ax.set_title(title_str)

                fig.tight_layout()

                if save_plots:
                   filename_out = 'plot_' + filename + '.png'
                   path_out = os.path.join(dirname_out, filename_out)
                   plt.savefig(path_out, dpi=600, bbox_inches="tight")

            if show_print:
                print()

        except:
            pass

    if show_plot:
        plt.show()
