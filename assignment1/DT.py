# Load libraries

# Load libraries
import datetime as dt

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import tree
import matplotlib.pyplot as plt
import util as lutil
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


def DT(dataset_path, optiimal_ccp_alphas):
    X_train, X_test, y_train, y_test = lutil.get_data_train_test(dataset_path)

    clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc_score = accuracy_score(y_test, pred)

    print("Score of the model without any pruning..: "+str(acc_score))

    from sklearn import tree
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, filled=True)
    plt.savefig('Complete tree without prunning!', dpi=100)

    #doing grid search using sklearn techniques

    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha, criterion='entropy')
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.show()

    # pick the ccp_alpha value based on above graph
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=optiimal_ccp_alphas, criterion='entropy')
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, pred)

    from sklearn import tree
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, filled=True)




