# Randomized Optimization script

import numpy as np
import matplotlib.pyplot as plt
from .utils import *
import datetime
from mlrose.algorithms import random_hill_climb, simulated_annealing, genetic_alg, mimic
from mlrose.decay import ExpDecay

IMAGE_DIR = 'images/'


def plot_ga_mimic_optimization(problem, param_name, random_seeds, **kwargs):
    ga_curve, mimic_curve = [], []  # curves to plot for GA and MIMIC

    # Initialize, for GA and MIMIC, the parameter we don't have to loop through and label for plotting
    if param_name == 'keep_pct':
        ga_pop_size = kwargs['ga_pop_size']
        mimic_pop_size = kwargs['mimic_pop_size']
        label = 'Percentage to keep'
    elif param_name == 'pop_size':
        ga_keep_pct = kwargs['ga_keep_pct']
        mimic_keep_pct = kwargs['mimic_keep_pct']
        label = 'Population size'
    else:
        raise Exception('Param name has to be either pop_size or keep_pct')  # raise exception if invalid entry

    # For all parameters
    for param in kwargs[param_name + 's']:

        print('\nGA & MIMIC: {} = {:.3f}'.format(param_name, param))
        ga_objectives, mimic_objectives = [], []  # list of results from multiple random runs for current parameter

        # Initialize, for GA and MIMIC, the parameter we have to loop through
        if param_name == 'keep_pct':
            ga_keep_pct = param
            mimic_keep_pct = param
        elif param_name == 'pop_size':
            ga_pop_size = int(param)
            mimic_pop_size = int(param)

        # For multiple random runs
        for random_seed in random_seeds:

            # Run GA and get best state and objective found
            best_state, best_objective, _, _ = genetic_alg(problem,
                                                        pop_size=ga_pop_size,  # population size
                                                        mutation_prob=1.0 - ga_keep_pct,  # percentage to breed
                                                        max_attempts=kwargs['ga_max_iters'],  # unlimited attempts
                                                        max_iters=kwargs['ga_max_iters'],
                                                        curve=True, random_state=random_seed)
            ga_objectives.append(best_objective)
            print('GA: best_objective = {:.3f}'.format(best_objective))

            # Run MIMIC and get best state and objective found
            best_state, best_objective, _, _ = mimic(problem,
                                                  pop_size=mimic_pop_size,  # population size
                                                  keep_pct=mimic_keep_pct,  # percentage to keep
                                                  max_attempts=kwargs['mimic_max_iters'],  # unlimited attempts
                                                  max_iters=kwargs['mimic_max_iters'],
                                                  curve=True, random_state=random_seed)
            mimic_objectives.append(best_objective)
            print('MIMIC: best_objective = {:.3f}'.format(best_objective))

        # Append random run to GA and MIMIC curves
        ga_curve.append(ga_objectives)
        mimic_curve.append(mimic_objectives)

    # Plot, set title and labels
    plt.figure()
    plot_helper(x_axis=kwargs[param_name + 's'], y_axis=np.array(ga_curve).T, label='GA')
    plot_helper(x_axis=kwargs[param_name + 's'], y_axis=np.array(mimic_curve).T, label='MIMIC')
    set_plot_title_labels(title='{} - Objective vs. {}'.format(kwargs['plot_name'], label),
                                x_label=label,
                                y_label=kwargs['plot_ylabel'])
    plt.savefig(IMAGE_DIR + '{}_objective_vs_{}'.format(kwargs['plot_name'], param_name))


def plot_sa_optimization(problem, random_seeds, **kwargs):
    sa_curve = []  # curve to plot for SA

    # For all temperature exponential decay rate
    for exp_decay_rate in kwargs['sa_decay_rates']:

        # Define exponential decay schedule
        print('\nSA: exp decay rate = {:.3f}'.format(exp_decay_rate))
        exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                             exp_const=exp_decay_rate,
                             min_temp=kwargs['sa_min_temp'])

        sa_objectives = []  # list of results from multiple random runs for current rate

        # For multiple random runs
        for random_seed in random_seeds:

            # Run SA and get best state and objective found
            best_state, best_objective, _, _ = simulated_annealing(problem,
                                                                schedule=exp_decay,
                                                                max_attempts=kwargs['sa_max_iters'],
                                                                max_iters=kwargs['sa_max_iters'],
                                                                init_state=None,
                                                                curve=True, random_state=random_seed)
            sa_objectives.append(best_objective)
            print('SA: best_fitness = {:.3f}'.format(best_objective))

        sa_curve.append(sa_objectives)  # append random run to SA curve

    # Plot, set title and labels
    plt.figure()
    plot_helper(x_axis=kwargs['sa_decay_rates'], y_axis=np.array(sa_curve).T, label='SA')
    set_plot_title_labels(title='{} - Objective vs. temperature exponential decay rate'.format(kwargs['plot_name']),
                                x_label='Exponential decay rate',
                                y_label=kwargs['plot_ylabel'])

    # Save figure
    plt.savefig(IMAGE_DIR + '{}_objective_vs_temp'.format(kwargs['plot_name']))


def plot_optimizations(problem, random_seeds, **kwargs):
    plot_sa_optimization(problem, random_seeds, **kwargs)
    plot_ga_mimic_optimization(problem, 'pop_size', random_seeds, **kwargs)
    plot_ga_mimic_optimization(problem, 'keep_pct', random_seeds, **kwargs)


def plot_performances(problem, random_seeds, **kwargs):
    # Initialize lists of objectives curves and time curves
    rhc_objectives, sa_objectives, ga_objectives, mimic_objectives = [], [], [], []
    rhc_times, sa_times, ga_times, mimic_times = [], [], [], []

    # Set an exponential decay schedule for SA
    exp_decay = ExpDecay(init_temp=kwargs['sa_init_temp'],
                         exp_const=kwargs['sa_exp_decay_rate'],
                         min_temp=kwargs['sa_min_temp'])

    # For multiple random runs
    for random_seed in random_seeds:

        # Run RHC and get best state and objective found
        _, best_objective, objective_curve, time_per_iteration = random_hill_climb(problem,
                                                               max_attempts=kwargs['rhc_max_iters'],
                                                               max_iters=kwargs['rhc_max_iters'],
                                                               curve=True, random_state=random_seed)
        rhc_objectives.append(objective_curve)
        rhc_times.append(time_per_iteration)
        print('\nRHC: best_objective = {:.3f}'.format(best_objective))

        # Run SA and get best state and objective found
        _, best_objective, objective_curve, time_per_iteration = simulated_annealing(problem,
                                                                 schedule=exp_decay,
                                                                 max_attempts=kwargs['sa_max_iters'],
                                                                 max_iters=kwargs['sa_max_iters'],
                                                                 curve=True, random_state=random_seed)
        sa_objectives.append(objective_curve)
        sa_times.append(time_per_iteration)
        print('SA: best_objective = {:.3f}'.format(best_objective))

        # Run GA and get best state and objective found
        _, best_objective, objective_curve, time_per_iteration = genetic_alg(problem,
                                                         pop_size=kwargs['ga_pop_size'],
                                                         max_attempts=kwargs['ga_max_iters'],
                                                         max_iters=kwargs['ga_max_iters'],
                                                         curve=True, random_state=random_seed)

        ga_objectives.append(objective_curve)
        ga_times.append(time_per_iteration)
        print('GA: best_objective = {:.3f}'.format(best_objective))

        # Run MIMIC and get best state and objective found
        _, best_objective, objective_curve, time_per_iteration = mimic(problem,
                                                   pop_size=kwargs['mimic_pop_size'],
                                                   keep_pct=kwargs['mimic_keep_pct'],
                                                   max_attempts=kwargs['mimic_max_iters'],
                                                   max_iters=kwargs['mimic_max_iters'],
                                                   curve=True, random_state=random_seed)

        mimic_objectives.append(objective_curve)
        mimic_times.append(time_per_iteration)
        print('MIMIC: best_objective = {:.3f}'.format(best_objective))

    # Array of iterations to plot objectives vs. for RHC, SA, GA and MIMIC
    rhc_iterations = np.arange(1, kwargs['rhc_max_iters']+1)
    sa_iterations = np.arange(1, kwargs['sa_max_iters']+1)
    ga_iterations = np.arange(1, kwargs['ga_max_iters']+1)
    mimic_iterations = np.arange(1, kwargs['mimic_max_iters']+1)

    # Plot objective curves, set title and labels
    plt.figure()
    plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_objectives), label='RHC')
    plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_objectives), label='SA')
    plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_objectives), label='GA')
    plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_objectives), label='MIMIC')
    set_plot_title_labels(title='{} - Objective versus iterations'.format(kwargs['plot_name']),
                                x_label='Iterations',
                                y_label=kwargs['plot_ylabel'])

    # Save figure
    plt.savefig(IMAGE_DIR + '{}_objective_vs_iterations'.format(kwargs['plot_name']))

    # Plot times, set title and labels
    plt.figure()
    plot_helper(x_axis=rhc_iterations, y_axis=np.array(rhc_times), label='RHC')
    plot_helper(x_axis=sa_iterations, y_axis=np.array(sa_times), label='SA')
    plot_helper(x_axis=ga_iterations, y_axis=np.array(ga_times), label='GA')
    plot_helper(x_axis=mimic_iterations, y_axis=np.array(mimic_times), label='MIMIC')
    set_plot_title_labels(title='{} - Time versus iterations'.format(kwargs['plot_name']),
                                x_label='Iterations',
                                y_label='Time (milliseconds)')

    # Save figure
    plt.savefig(IMAGE_DIR + '{}_time_vs_iterations'.format(kwargs['plot_name']))
