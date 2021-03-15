# Script to run experiments

from Randomized_Optimization.utils.neural_networks import plot_nn_performances, test_nn_performances
import numpy as np
import matplotlib.pyplot as plt
import Randomized_Optimization.utils.randomized_optimization as ro
import pandas as pd
import mlrose as mlrose

from mlrose.opt_probs import TSPOpt, DiscreteOpt
from mlrose.fitness import TravellingSales, FlipFlop, FourPeaks

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def get_data_train_test(dataset_path="letter"):
    if "letter" in dataset_path:
        df = pd.read_csv('/Users/balu/dev/GATech/Machine_Learning/Randomized_Optimization/data/letter-recognition.data')
        names = ['class',
                 'x-box',
                 'y-box',
                 'width',
                 'high',
                 'onpix',
                 'x-bar',
                 'y-bar',
                 'x2bar',
                 'y2bar',
                 'xybar',
                 'x2ybr',
                 'xy2br',
                 'x-ege',
                 'xegvy',
                 'y-ege',
                 'yegvx']
        dataset = pd.read_csv('/Users/balu/dev/GATech/Machine_Learning/Randomized_Optimization/data/letter-recognition.data', names=names)
        array = dataset.values

        X = array[:, 1:17]
        Y = array[:, 0]
        le = preprocessing.LabelEncoder()
        Y = le.fit_transform(Y)
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20,
                                                                            random_state=10, stratify=Y)
    else:
        df = pd.read_csv('./data/wifi_localization.txt')
        df = df.iloc[1:]
        data = df.values.astype(np.float32)
        np.random.shuffle(data)
        X = data[:, :-1]
        Y = data[:, -1]

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20,
                                                                            random_state=10, stratify=Y)

    return X_train, X_test, Y_train, Y_test

def get_normalized_data(dataset_path):
    print("Reading in and transforming data...")
    Xtrain, Xtest, Ytrain, Ytest = get_data_train_test(dataset_path)
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(Xtrain)
    X_test_scaled = scaler.transform(Xtest)

    one_hot = OneHotEncoder()

    y_train_hot = one_hot.fit_transform(Ytrain.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(Ytest.reshape(-1, 1)).todense()

    return X_train_scaled, X_test_scaled, y_train_hot, y_test_hot

def load_dataset(split_percentage=0.2):
    return get_normalized_data("")


def flip_plop(length, random_seeds):
    flip_flop_objective = FlipFlop()
    problem = DiscreteOpt(length=length, fitness_fn=flip_flop_objective, maximize=True, max_val=2)
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=200, sa_max_iters=200, ga_max_iters=200, mimic_max_iters=200,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.05, 2.01, 0.05), sa_min_temp=0.001,
                          ga_pop_size=300, mimic_pop_size=1500, ga_keep_pct=0.2, mimic_keep_pct=0.4,
                          pop_sizes=np.arange(100, 2001, 200), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='Flip Flop', plot_ylabel='Fitness')
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=200, sa_max_iters=200, ga_max_iters=200, mimic_max_iters=200,
                         sa_init_temp=100, sa_exp_decay_rate=0.3, sa_min_temp=0.001,
                         ga_pop_size=300, ga_keep_pct=0.2,
                         mimic_pop_size=1500, mimic_keep_pct=0.4,
                         plot_name='Flip Flop', plot_ylabel='Fitness')


def four_peaks(length, random_seeds):
    four_fitness = FourPeaks(t_pct=0.1)
    problem = DiscreteOpt(length=length, fitness_fn=four_fitness, maximize=True, max_val=2)
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=10000, sa_max_iters=10000, ga_max_iters=250, mimic_max_iters=250,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.002, 0.1, 0.002), sa_min_temp=0.001,
                          ga_pop_size=1000, mimic_pop_size=1000, ga_keep_pct=0.1, mimic_keep_pct=0.2,
                          pop_sizes=np.arange(100, 1001, 100), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='Four Peaks', plot_ylabel='Fitness')
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=7000, sa_max_iters=7000, ga_max_iters=250, mimic_max_iters=250,
                         sa_init_temp=100, sa_exp_decay_rate=0.02, sa_min_temp=0.001,
                         ga_pop_size=1000, ga_keep_pct=0.1,
                         mimic_pop_size=1000, mimic_keep_pct=0.2,
                         plot_name='Four Peaks', plot_ylabel='Fitness')


def neural_network(x_train, x_test, y_train, y_test, random_seeds):
    iterations = np.array([i for i in range(1, 10)] + [10 * i for i in range(1, 20, 2)])
    plot_nn_performances(x_train, y_train,
                            random_seeds=random_seeds,
                            rhc_max_iters=iterations, sa_max_iters=iterations,
                            ga_max_iters=iterations, gd_max_iters=iterations,
                            init_temp=100, exp_decay_rate=0.1, min_temp=0.001,
                            pop_size=100, mutation_prob=0.2)
    test_nn_performances(x_train, x_test, y_train, y_test,
                            random_seed=random_seeds[0], max_iters=200,
                            init_temp=100, exp_decay_rate=0.1, min_temp=0.001,
                            pop_size=100, mutation_prob=0.2)


def travel_salesman(length, distances, random_seeds):
    tsp_objective = TravellingSales(distances=distances)
    problem = mlrose.TSPOpt(length=length, fitness_fn=tsp_objective, maximize=True)
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=200, sa_max_iters=200, ga_max_iters=200, mimic_max_iters=200,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.005, 0.05, 0.005), sa_min_temp=0.001,
                          ga_pop_size=100, mimic_pop_size=700, ga_keep_pct=0.2, mimic_keep_pct=0.2,
                          pop_sizes=np.arange(100, 1001, 100), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='TSP', plot_ylabel='Cost')

    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=200, sa_max_iters=200, ga_max_iters=200, mimic_max_iters=200,
                         sa_init_temp=100, sa_exp_decay_rate=0.03, sa_min_temp=0.001,
                         ga_pop_size=100, ga_keep_pct=0.2,
                         mimic_pop_size=700, mimic_keep_pct=0.2,
                         plot_name='TSP', plot_ylabel='Cost')


if __name__ == "__main__":

    random_seeds = [5 + 5 * i for i in range(2)]  # random seeds for get performances over multiple random runs
    cities_distances = [(0, 1, 0.274), (0, 2, 1.367), (1, 2, 1.091), (0, 3, 1.422), (1, 3, 1.153), (2, 3, 1.038),
                        (0, 4, 1.870), (1, 4, 1.602), (2, 4, 1.495), (3, 4, 0.475), (0, 5, 1.652), (1, 5, 1.381),
                        (2, 5, 1.537), (3, 5, 0.515), (4, 5, 0.539), (0, 6, 1.504), (1, 6, 1.324), (2, 6, 1.862),
                        (3, 6, 1.060), (4, 6, 1.097), (5, 6, 0.664), (0, 7, 1.301), (1, 7, 1.031), (2, 7, 1.712),
                        (3, 7, 1.031), (4, 7, 1.261), (5, 7, 0.893), (6, 7, 0.350), (0, 8, 1.219), (1, 8, 0.948),
                        (2, 8, 1.923), (3, 8, 1.484), (4, 8, 1.723), (5, 8, 1.396), (6, 8, 0.872), (7, 8, 0.526),
                        (0, 9, 0.529), (1, 9, 0.258), (2, 9, 1.233), (3, 9, 1.137), (4, 9, 1.560), (5, 9, 1.343),
                        (6, 9, 1.131), (7, 9, 0.816), (8, 9, 0.704)]

    travel_salesman(length=10, distances=cities_distances, random_seeds=random_seeds)
    flip_plop(length=100, random_seeds=random_seeds)
    four_peaks(length=100, random_seeds=random_seeds)

    # Experiment Neural Networks optimization with RHC, SA, GA and GD on the WDBC dataset
    x_train, x_test, y_train, y_test = load_dataset(split_percentage=0.2)
    neural_network(x_train, x_test, y_train, y_test, random_seeds=random_seeds)
