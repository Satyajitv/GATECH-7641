# Neural Networks

import numpy as np
import matplotlib.pyplot as plt
import time

from .utils import plot_helper, set_plot_title_labels

from mlrose.decay import ExpDecay
from mlrose.neural import NeuralNetwork

from sklearn.metrics import log_loss, classification_report
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

IMAGE_DIR = 'images/'


def plot_nn_performances(x_train, y_train, random_seeds, **kwargs):
    #algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']
    algorithms = ['genetic_alg']
    #acronyms = ['RHC', 'SA', 'GA', 'GD']
    acronyms = ['GA']
    #max_iters = ['rhc_max_iters', 'sa_max_iters', 'ga_max_iters', 'gd_max_iters']
    max_iters = ['ga_max_iters']

    # Initialize lists of training curves, validation curves and training times curves
    train_curves, val_curves, train_time_curves = [], [], []

    # Define SA exponential decay schedule
    exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
                         exp_const=kwargs['exp_decay_rate'],
                         min_temp=kwargs['min_temp'])

    # Create one figure for training and validation losses, the second for training time
    plt.figure()
    train_val_figure = plt.gcf().number
    plt.figure()
    train_times_figure = plt.gcf().number

    # For each of the optimization algorithms to test the Neural Network with
    for i, algorithm in enumerate(algorithms):
        print('\nAlgorithm = {}'.format(algorithm))

        for random_seed in random_seeds:

            train_losses, val_losses, train_times = [], [], []

            x_train_fold, x_val_fold, y_train_fold, y_val_fold = train_test_split(x_train, y_train,
                                                                                  test_size=0.2, shuffle=True,
                                                                                  random_state=random_seed,
                                                                                  stratify=y_train)
            for max_iter in kwargs[max_iters[i]]:
                nn = NeuralNetwork(hidden_nodes=[100, 50], activation='relu',
                                   algorithm=algorithm, max_iters=int(max_iter),
                                   bias=True, is_classifier=True, learning_rate=0.001,
                                   early_stopping=True, clip_max=1e10, schedule=exp_decay,
                                   pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
                                   max_attempts=int(max_iter), random_state=random_seed, curve=False, restarts=100)
                start_time = time.time()
                nn.fit(x_train_fold, y_train_fold)
                train_times.append(time.time() - start_time)

                train_loss = log_loss(y_train_fold, nn.predict(x_train_fold))
                val_loss = log_loss(y_val_fold, nn.predict(x_val_fold))
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print('{} - train loss = {:.3f}, val loss = {:.3f}'.format(max_iter, train_loss, val_loss))

            train_curves.append(train_losses)
            val_curves.append(val_losses)
            train_time_curves.append(train_times)

        plt.figure(train_val_figure)
        plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(train_curves), label='{} train'.format(acronyms[i]))
        plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(val_curves), label='{} val'.format(acronyms[i]))
        plt.figure(train_times_figure)
        plot_helper(x_axis=kwargs[max_iters[i]], y_axis=np.array(train_time_curves), label=acronyms[i])
    plt.figure(train_val_figure)
    set_plot_title_labels(title='Neural Network - Loss vs. iterations',
                                x_label='Iterations',
                                y_label='Loss')

    # Save figure
    plt.savefig(IMAGE_DIR + 'nn_objective_vs_iterations')

    # Set title and labels to training time figure
    plt.figure(train_times_figure)
    set_plot_title_labels(title='Neural Network - Time vs. iterations',
                                x_label='Iterations',
                                y_label='Time (milliseconds)')

    # Save figure
    plt.savefig(IMAGE_DIR + 'nn_time_vs_iterations')


def test_nn_performances(x_train, x_test, y_train, y_test, random_seed, **kwargs):
    exp_decay = ExpDecay(init_temp=kwargs['init_temp'],
                         exp_const=kwargs['exp_decay_rate'],
                         min_temp=kwargs['min_temp'])
    rhc_nn = NeuralNetwork(hidden_nodes=[100, 50], activation='relu',
                           algorithm='random_hill_climb', max_iters=kwargs['max_iters'],
                           bias=True, is_classifier=True, learning_rate=0.001,
                           early_stopping=True, clip_max=1e10,
                           max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False, restarts=5)
    sa_nn = NeuralNetwork(hidden_nodes=[100, 50], activation='relu',
                          algorithm='simulated_annealing', max_iters=kwargs['max_iters'],
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10, schedule=exp_decay,
                          max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)
    ga_nn = NeuralNetwork(hidden_nodes=[100, 50], activation='relu',
                          algorithm='genetic_alg', max_iters=kwargs['max_iters'],
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10,
                          pop_size=kwargs['pop_size'], mutation_prob=kwargs['mutation_prob'],
                          max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)
    gd_nn = NeuralNetwork(hidden_nodes=[100, 50], activation='relu',
                          algorithm='gradient_descent', max_iters=kwargs['max_iters'],
                          bias=True, is_classifier=True, learning_rate=0.001,
                          early_stopping=False, clip_max=1e10,
                          max_attempts=kwargs['max_iters'], random_state=random_seed, curve=False)
    rhc_nn.fit(x_train, y_train)
    sa_nn.fit(x_train, y_train)
    ga_nn.fit(x_train, y_train)
    gd_nn.fit(x_train, y_train)
    print('RHC test classification report = \n {}'.format(classification_report(y_test, rhc_nn.predict(x_test))))
    print('SA test classification report = \n {}'.format(classification_report(y_test, sa_nn.predict(x_test))))
    print('GA test classification report = \n {}'.format(classification_report(y_test, ga_nn.predict(x_test))))
    print('GD test classification report = \n {}'.format(classification_report(y_test, gd_nn.predict(x_test))))
