# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import util as lutil
from sklearn.model_selection import train_test_split

def KNN(dataset_path):

    X_train, X_test, y_train, y_test = lutil.get_data_train_test(dataset_path)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train, y_train)

    print("Preliminary model score:")
    print(knn.score(X_test, y_test))

    no_neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(no_neighbors))
    test_accuracy = np.empty(len(no_neighbors))

    for i, k in enumerate(no_neighbors):
        # We instantiate the classifier
        knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        # Fit the classifier to the training data
        knn.fit(X_train, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        # Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)

    # Visualization of k values vs accuracy

    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(no_neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(no_neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()