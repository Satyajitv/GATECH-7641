import argparse

import DT as dt
import ANN as ann
import KNN as knn
import Boosting as boosting
import SVM as svm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS-7641 Analaysis')
    parser.add_argument('--algo', help='Select ALGO from List[DT, ANN, KNN, BOOSTING, SVM]')
    parser.add_argument('--dataset', type=int, help='Dataset selection 1 or 2')
    args = parser.parse_args()
    algo = args.algo
    dataset = args.dataset

    dataset_path = ""
    if dataset==1:
        dataset_path = "/Users/balu/PycharmProjects/GATech-CS7641/data/letter-recognition.data"
    elif dataset==2:
        dataset_path = "/Users/balu/PycharmProjects/GATech-CS7641/data/wifi_localization.txt"

    if algo=="DT":
        if dataset == 1:
            ccp_value = 0.0001
        else:
            ccp_value = 0.008
        dt.DT(dataset_path, ccp_value)
    elif algo=="ANN":
        ann.ANN(dataset_path)
    elif algo=="KNN":
        knn.KNN(dataset_path)
    elif algo=="BOOSTING":
        boosting.Boosting(dataset_path)
    elif algo=="SVM":
        svm.SVM(dataset_path, 'sigmoid')
        svm.SVM(dataset_path, 'linear')
        svm.SVM(dataset_path, 'rbf')
