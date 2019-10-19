import json
import os

import numpy as np
from sklearn import svm, dummy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import definitions
from data_prep import data_preparation as dp


def result_in_csv(STOCK, Algo, Estimator=0, Depth=0, Distance_function='0', No_of_features=0, Model_Score=0,
                  Future_day=0, C=-1, degree='0', No_of_neighbors=-1,
                  Our_test_score=-1.0):
    return f"{STOCK},{Algo},{Estimator},{Depth},{Distance_function},{No_of_features},{Model_Score},{Future_day},{C},{degree},{No_of_neighbors},{Our_test_score}\n "


def get_splits(X):
    splits = []
    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(X):
        splits.append((train_index, test_index))
    return splits


def get_params_rf(estimator_start, estimator_stop, depth_start, depth_stop):
    estimator_options = [x for x in range(estimator_start, estimator_stop, 10)]
    depth_options = [x for x in range(depth_start, depth_stop, 10)]
    parameters = dict(n_estimators=estimator_options, max_depth=depth_options)
    return parameters


def testRandomForests(estimator_start, estimator_stop,
                      depth_start, depth_stop):
    test_size = get_test_size()
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    parameters = get_params_rf(estimator_start, estimator_stop, depth_start, depth_stop)
    estimator = RandomForestClassifier()
    # selector = RFECV(estimator, step=1, cv=TimeSeriesSplit(n_splits=3), scoring='accuracy')
    return perform_grid_search_and_get_result('RF', estimator, parameters, X, y, test_size)


def testSVM(C):
    test_size = get_test_size()
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    parameters = {'kernel': ('linear', 'poly', 'rbf'), 'C': C}
    estimator = svm.SVC(gamma="scale")
    return perform_grid_search_and_get_result('SVM', estimator, parameters, X, y, test_size)


def testKNN():
    test_size = get_test_size()
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    parameters = {'n_neighbors': [3, 5, 7, 9, 11],
                  'metric': ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']}
    estimator = KNeighborsClassifier()
    return perform_grid_search_and_get_result('KNN', estimator, parameters, X, y, test_size)


def testZeroHour():
    test_size = get_test_size()
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    try:
        model = dummy.DummyClassifier(strategy="most_frequent")
        model.fit(X[:-test_size], y[:-test_size])
    except:
        return result_in_csv(STOCK, 'ZR', Future_day=future_day, Our_test_score=-1)
    our_score = model.score(X[-test_size:], y[-test_size:])
    return result_in_csv(STOCK, 'ZR', Future_day=future_day, Our_test_score=our_score)


def get_test_size():
    return min(int(data_for_algos.shape[0] * 0.1), future_day)


def perform_grid_search_and_get_result(algo, estimator, parameters, X, y, test_size):
    clf = GridSearchCV(estimator, param_grid=parameters, cv=TimeSeriesSplit(n_splits=5), n_jobs=-1, scoring='accuracy')
    try:
        clf.fit(X[:-test_size], y[:-test_size])
    except:
        return result_in_csv(STOCK, algo, Future_day=future_day)
    model_val_score = clf.best_score_
    our_score = clf.score(X[-test_size:], y[-test_size:])
    if algo == 'RF':
        return get_rf_result(clf, algo, model_val_score, our_score)
    elif algo == 'SVM':
        return get_svm_result(clf, algo, model_val_score, our_score)
    elif algo == 'KNN':
        return get_knn_result(clf, algo, model_val_score, our_score)


def get_rf_result(clf, algo, model_val_score, our_score):
    max_depth, n_estimators, n_features = get_rf_clf_params(clf)
    print(max_depth, n_estimators, n_features, future_day)
    print(clf.predict(actual_data_to_predict))
    return result_in_csv(STOCK, algo, n_estimators, max_depth, No_of_features=n_features, Future_day=future_day,
                         Our_test_score=our_score, Model_Score=model_val_score)


def get_svm_result(clf, algo, model_val_score, our_score):
    print(clf.predict(actual_data_to_predict))
    kernel = clf.best_estimator_.kernel
    c_val = clf.best_estimator_.C
    if kernel == 'linear' or kernel == 'rbf':
        return result_in_csv(STOCK, algo, Future_day=future_day, C=c_val, Distance_function=kernel,
                             Our_test_score=our_score, Model_Score=model_val_score)
    elif kernel == 'poly':
        degree = clf.best_estimator_.degree
        return result_in_csv(STOCK, algo, Future_day=future_day, C=c_val, Distance_function=kernel,
                             Our_test_score=our_score, Model_Score=model_val_score, degree=degree)


def get_knn_result(clf, algo, model_val_score, our_score):
    print(clf.predict(actual_data_to_predict))
    metric = clf.best_estimator_.metric
    neighbors = clf.best_estimator_.n_neighbors
    return result_in_csv(STOCK, algo, Future_day=future_day, Distance_function=metric,
                         Our_test_score=our_score, Model_Score=model_val_score, No_of_neighbors=neighbors)


def get_rf_clf_params(clf):
    return clf.best_estimator_.max_depth, clf.best_estimator_.n_estimators, clf.best_estimator_.n_features_


def get_prepared_data(STOCK_FILE, window_size, feature_window_size, discretize):
    df, actual_data_to_predict = dp.data_preparation(os.path.join(f"{definitions.ROOT_DIR}", f"data/{STOCK_FILE}"),
                                                     window_size=window_size,
                                                     feature_window_size=feature_window_size,
                                                     discretize=discretize).data_frame_with_features()
    # df.drop(columns=['open', 'high', 'low', 'close'], inplace=True)
    data_for_algos = df.to_numpy()
    return data_for_algos, actual_data_to_predict.to_numpy()


def write_result_to_file(lock, RESULT_FILE, result):
    # lock.acquire()
    try:
        open(RESULT_FILE, 'a').write(result)
    except:
        print("Error while writing to a file.")
    # lock.release()


def make_missing_dirs(path):
    head, tail = os.path.split(path)
    if not os.path.exists(head):
        os.makedirs(head)


def add_headers(RESULT_FILE):
    with open(RESULT_FILE, 'w') as f:
        f.write("Stock,Algorithm,Estimators,Depth,Distance_function,No_of_features,Model_Score,Future_day,"
                "C,degree,No_of_neighbors,Our_test_score\n")


def get_config_from_dict(dictionary):
    RESULT_FILE = dictionary["RESULT_FILE"]
    COMPLETED_FILE = dictionary["COMPLETED_FILE"]
    algos = dictionary["algos"]
    future_day_start = dictionary["future_day_start"]
    future_day_stop = dictionary["future_day_stop"]
    estimator_start = dictionary["estimator_start"]
    estimator_stop = dictionary["estimator_stop"]
    depth_start = dictionary["depth_start"]
    depth_stop = dictionary["depth_stop"]
    initial_no_of_features = dictionary["initial_no_of_features"]
    max_features = dictionary["max_features"]
    feature_window_size = dictionary["feature_window_size"]
    discretize = True if dictionary["discretize"] == 1 else False
    C = dictionary["C"]

    return RESULT_FILE, COMPLETED_FILE, algos, future_day_start, future_day_stop, estimator_start, \
           estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features, \
           feature_window_size, discretize, C


def pre_run_tasks(RESULT_FILE, COMPLETED_FILE):
    make_missing_dirs(RESULT_FILE)
    make_missing_dirs(COMPLETED_FILE)
    add_headers(RESULT_FILE)


file = open('configs/config_all_disc.json')
configs = json.load(file)

for dictionary in configs:
    RESULT_FILE, COMPLETED_FILE, algos, future_day_start, future_day_stop, estimator_start, \
    estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features, \
    feature_window_size, discretize, C = get_config_from_dict(dictionary)
    pre_run_tasks(RESULT_FILE, COMPLETED_FILE)

    files = list(map(lambda x: x.replace("\n", ""), open('10stocks.txt', 'r').readlines()))
    files.reverse()
    print(files)
    for filename in files:
        STOCK = filename.split(".csv")[0]
        result = ""
        for future_day in range(future_day_start, future_day_stop, 10):
            try:
                data_for_algos, actual_data_to_predict = get_prepared_data(
                    filename, future_day, feature_window_size, discretize)
                # print(data_for_algos.shape[1])
            except:
                continue

            if 'RF' in algos:
                print('Starting RF testing')
                if __name__ == '__main__':
                    result += testRandomForests(estimator_start,
                                                estimator_stop, depth_start, depth_stop)
                print(result)
                print('Finished RF testing')

            if 'SVM' in algos:
                print('Starting SVM testing')
                if __name__ == '__main__':
                    result += testSVM(C)
                print(result)
                print('Finished SVM testing')

            if 'KNN' in algos:
                print('Starting KNN testing')
                if __name__ == '__main__':
                    result += testKNN()
                print(result)
                print('Finished KNN testing')

            if 'ZR' in algos:
                print('Starting ZR testing')
                result += testZeroHour()
                print(result)
                print('Finished ZR testing')

        write_result_to_file(0, RESULT_FILE, result)

# def runExperiment(lock, STOCK_FILE, RESULT_FILE, algos, future_day_start, future_day_stop, estimator_start,
#                   estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features, feature_window_size,
#                   discretize, C):
#     STOCK = STOCK_FILE.split(".csv")[0]
#     print(STOCK)
#
#     result = ""
#     for future_day in range(future_day_start, future_day_stop, 10):
#         try:
#             data_for_algos, actual_data_to_predict = get_prepared_data(
#                 STOCK_FILE, future_day, feature_window_size, discretize)
#             # print(data_for_algos.shape[1])
#         except:
#             continue
#
#         if 'RF' in algos:
#             result += testRandomForests(STOCK, future_day, data_for_algos, estimator_start,
#                                         estimator_stop,
#                                         depth_start, depth_stop, initial_no_of_features, max_features)
#
#     write_result_to_file(lock, RESULT_FILE, result)
