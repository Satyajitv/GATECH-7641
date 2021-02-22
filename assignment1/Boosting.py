# Load libraries
import datetime as dt

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
import util as lutil

def Boosting(dataset_path):
    X_train, X_test, y_train, y_test = lutil.get_data_train_test(dataset_path)
    # Fit regression model
    clf_1 = DecisionTreeClassifier(random_state=0, ccp_alpha=0.0008, criterion='entropy')

    estimators = [10, 50, 100, 500]
    clf_2_list = []
    for est in estimators:
        clf_2_list.append(
            AdaBoostClassifier(DecisionTreeClassifier(random_state=0, ccp_alpha=0.0008, criterion='entropy'),
                               n_estimators=est))

    clf_1.fit(X_train, y_train)
    for i in clf_2_list:
        i.fit(X_train, y_train)

    y_2_list_test = np.empty(len(estimators))
    y_2_list_train = np.empty(len(estimators))
    for k, i in enumerate(clf_2_list):
        y_1 = i.predict(X_train)
        y_2 = i.predict(X_test)
        from sklearn.metrics import accuracy_score
        y_2_list_train[k] = accuracy_score(y_train, y_1)
        y_2_list_test[k] = accuracy_score(y_test, y_2)

    plt.title('Boosting performance based on no of estimators!')
    plt.plot(estimators, y_2_list_test, label='Testing Accuracy')
    plt.plot(estimators, y_2_list_train, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()