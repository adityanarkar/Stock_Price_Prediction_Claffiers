import json
import os
from threading import Lock

import numpy as np
from sklearn import svm, dummy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif, RFE
from sklearn.feature_selection import SelectKBest
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
    parameters = dict(estimator__n_estimators=estimator_options, estimator__max_depth=depth_options)
    return parameters


def testRandomForests(STOCK, future_day, actual_data_to_predict, estimator_start, estimator_stop,
                      depth_start, depth_stop, data_for_algos):
    test_size = get_test_size(data_for_algos, future_day)
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    parameters = get_params_rf(estimator_start, estimator_stop, depth_start, depth_stop)
    estimator = RandomForestClassifier()
    selector = RFECV(estimator, step=1, cv=TimeSeriesSplit(n_splits=3), scoring='accuracy')
    return perform_grid_search_and_get_result(STOCK, future_day, actual_data_to_predict, 'RF', selector, parameters, X,
                                              y, test_size)


def get_rfe_rf_selectors(max_features):
    estimator = RandomForestClassifier()
    list_of_selectors = []
    for i in range(1, max_features):
        list_of_selectors.append(RFE(estimator, n_features_to_select=i))
    return list_of_selectors


def testSVM(STOCK, future_day, actual_data_to_predict, C, data_for_algos):
    test_size = get_test_size(data_for_algos, future_day)
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    parameters = {'estimator__kernel': ['poly'], 'estimator__C': C, 'estimator__degree': [1]}
    estimator = svm.SVC()
    return perform_grid_search_and_get_result_for_svm(STOCK, future_day, actual_data_to_predict, 'SVM', estimator,
                                                      parameters, X,
                                                      y, test_size)


def testKNN(STOCK, future_day, actual_data_to_predict, data_for_algos):
    test_size = get_test_size(data_for_algos, future_day)
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    parameters = {'estimator__n_neighbors': [3, 5, 7, 9, 11],
                  'estimator__metric': ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']}
    estimator = KNeighborsClassifier()
    return perform_grid_search_and_get_result_for_knn(STOCK, future_day, actual_data_to_predict, 'KNN', estimator,
                                                      parameters,
                                                      X, y, test_size)


def testZeroHour(STOCK, future_day, data_for_algos):
    test_size = get_test_size(data_for_algos, future_day)
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    try:
        model = dummy.DummyClassifier(strategy="most_frequent")
        model.fit(X[:-test_size], y[:-test_size])
    except:
        return result_in_csv(STOCK, 'ZR', Future_day=future_day, Our_test_score=-1)
    our_score = model.score(X[-test_size:], y[-test_size:])
    return result_in_csv(STOCK, 'ZR', Future_day=future_day, Our_test_score=our_score)


def get_test_size(data_for_algos, future_day):
    return min(int(data_for_algos.shape[0] * 0.1), future_day)


def perform_grid_search_and_get_result_for_knn(STOCK, future_day, actual_data_to_predict, algo, estimator, parameters,
                                               X, y,
                                               test_size):
    best_clf = 0
    max_score = 0
    model_val_score = -1
    no_of_features = -1
    for i in range(10, actual_data_to_predict.shape[1] + 1):
        print(i)
        X_new = SelectKBest(mutual_info_classif, k=i).fit_transform(X, y)
        clf = GridSearchCV(estimator, param_grid=parameters, cv=TimeSeriesSplit(n_splits=5), n_jobs=-1,
                           scoring='accuracy')
        try:
            clf.fit(X_new[:-test_size], y[:-test_size])
        except:
            continue
        model_val_score = clf.best_score_
        if model_val_score > max_score:
            best_clf = clf
            max_score = model_val_score
            no_of_features = i
    if best_clf == 0:
        return result_in_csv(STOCK, algo, Future_day=future_day)
    selector = SelectKBest(mutual_info_classif, k=no_of_features)
    X_new = selector.fit_transform(X, y)
    our_score = best_clf.score(X_new[-test_size:], y[-test_size:])
    return get_knn_result(STOCK, best_clf, algo, model_val_score, our_score, future_day, actual_data_to_predict,
                          no_of_features, X, y)


def perform_grid_search_and_get_result_for_svm(STOCK, future_day, actual_data_to_predict, algo, estimator, parameters,
                                               X, y,
                                               test_size):
    best_clf = 0
    max_score = 0
    model_val_score = -1
    no_of_features = -1
    for i in range(10, actual_data_to_predict.shape[1] + 1):
        print(i)
        X_new = SelectKBest(mutual_info_classif, k=i).fit_transform(X, y)
        clf = GridSearchCV(estimator, param_grid=parameters, cv=TimeSeriesSplit(n_splits=5), n_jobs=-1,
                           scoring='accuracy')
        try:
            clf.fit(X_new[:-test_size], y[:-test_size])
        except:
            continue
        model_val_score = clf.best_score_
        if model_val_score > max_score:
            best_clf = clf
            max_score = model_val_score
            no_of_features = i
    if best_clf == 0:
        return result_in_csv(STOCK, algo, Future_day=future_day)
    selector = SelectKBest(mutual_info_classif, k=no_of_features)
    X_new = selector.fit_transform(X, y)
    our_score = best_clf.score(X_new[-test_size:], y[-test_size:])
    return get_svm_result(STOCK, best_clf, algo, model_val_score, our_score, future_day, actual_data_to_predict,
                          no_of_features, X, y)


def perform_grid_search_and_get_result(STOCK, future_day, actual_data_to_predict, algo, estimator, parameters, X, y,
                                       test_size):
    clf = GridSearchCV(estimator, param_grid=parameters, cv=TimeSeriesSplit(n_splits=5), n_jobs=-1, scoring='accuracy')
    try:
        clf.fit(X[:-test_size], y[:-test_size])
    except:
        return result_in_csv(STOCK, algo, Future_day=future_day)
    model_val_score = clf.best_score_
    our_score = clf.score(X[-test_size:], y[-test_size:])
    return get_rf_result(STOCK, clf, algo, model_val_score, our_score, future_day, actual_data_to_predict)


def get_rf_result(STOCK, clf, algo, model_val_score, our_score, future_day, actual_data_to_predict):
    max_depth, n_estimators, n_features = get_rf_clf_params(clf)
    print(max_depth, n_estimators, n_features, future_day)
    print(clf.predict(actual_data_to_predict))
    return result_in_csv(STOCK, algo, n_estimators, max_depth, No_of_features=n_features, Future_day=future_day,
                         Our_test_score=our_score, Model_Score=model_val_score)


def get_svm_result(STOCK, clf, algo, model_val_score, our_score, future_day, actual_data_to_predict, no_of_features, X,
                   y):
    print(f"No of features {no_of_features}")
    selector = SelectKBest(mutual_info_classif, k=no_of_features)
    selector.fit(X, y)
    actual_data_to_predict = selector.transform(actual_data_to_predict)
    print(clf.predict(actual_data_to_predict))
    kernel = clf.best_params_['estimator__kernel']
    c_val = clf.best_params_['estimator__C']
    n_features = clf.best_estimator_.n_features_
    if kernel == 'linear' or kernel == 'rbf':
        return result_in_csv(STOCK, algo, Future_day=future_day, C=c_val, Distance_function=kernel,
                             Our_test_score=our_score, Model_Score=model_val_score, No_of_features=n_features)
    elif kernel == 'poly':
        degree = clf.best_params_['estimator__degree']
        return result_in_csv(STOCK, algo, Future_day=future_day, C=c_val, Distance_function=kernel,
                             Our_test_score=our_score, Model_Score=model_val_score, degree=degree,
                             No_of_features=n_features)


def get_knn_result(STOCK, clf, algo, model_val_score, our_score, future_day, actual_data_to_predict, no_of_features, X,
                   y):
    print(f"No of features {no_of_features}")
    selector = SelectKBest(mutual_info_classif, k=no_of_features)
    selector.fit(X, y)
    actual_data_to_predict = selector.transform(actual_data_to_predict)
    print(clf.predict(actual_data_to_predict))
    metric = clf.best_estimator_.metric
    neighbors = clf.best_estimator_.n_neighbors
    return result_in_csv(STOCK, algo, Future_day=future_day, Distance_function=metric,
                         Our_test_score=our_score, Model_Score=model_val_score, No_of_neighbors=neighbors,
                         No_of_features=no_of_features)


def get_rf_clf_params(clf):
    return clf.best_params_['estimator__max_depth'], clf.best_params_[
        'estimator__n_estimators'], clf.best_estimator_.n_features_


def get_prepared_data(STOCK_FILE, window_size, feature_window_size, discretize):
    df, actual_data_to_predict = dp.data_preparation(os.path.join(f"{definitions.ROOT_DIR}", f"data/{STOCK_FILE}"),
                                                     window_size=window_size,
                                                     feature_window_size=feature_window_size,
                                                     discretize=discretize).data_frame_with_features()
    data_for_algos = df.to_numpy()
    return data_for_algos, actual_data_to_predict.to_numpy()


def write_result_to_file(lock, RESULT_FILE, result):
    lock.acquire()
    try:
        open(RESULT_FILE, 'a').write(result)
    except:
        print("Error while writing to a file.")
    lock.release()


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


def run_tests_for_a_stock(filename):
    STOCK = filename.split(".csv")[0]
    result = ""
    for future_day in range(future_day_start, future_day_stop, 10):
        try:
            data_for_algos, actual_data_to_predict = get_prepared_data(
                filename, future_day, feature_window_size, discretize)
        except:
            continue

        if 'RF' in algos:
            print('Starting RF testing')
            result += testRandomForests(STOCK, future_day, actual_data_to_predict, estimator_start,
                                        estimator_stop, depth_start, depth_stop, data_for_algos)
            print(result)
            print('Finished RF testing')

        if 'SVM' in algos:
            print('Starting SVM testing')
            result += testSVM(STOCK, future_day, actual_data_to_predict, C, data_for_algos)
            print(result)
            print('Finished SVM testing')

        if 'KNN' in algos:
            print('Starting KNN testing')
            result += testKNN(STOCK, future_day, actual_data_to_predict, data_for_algos)
            print(result)
            print('Finished KNN testing')

        if 'ZR' in algos:
            print('Starting ZR testing')
            result += testZeroHour(STOCK, future_day, data_for_algos)
            print(result)
            print('Finished ZR testing')

    write_result_to_file(Lock(), RESULT_FILE, result)
    return filename


file = open('configs/config_all_fs.json')
configs = json.load(file)

for dictionary in configs:
    RESULT_FILE, COMPLETED_FILE, algos, future_day_start, future_day_stop, estimator_start, \
    estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features, \
    feature_window_size, discretize, C = get_config_from_dict(dictionary)
    pre_run_tasks(RESULT_FILE, COMPLETED_FILE)

    files = list(map(lambda x: x.replace("\n", ""), open('10stocks.txt', 'r').readlines()))
    files.reverse()
    print(files)

    if __name__ == '__main__':
        for filename in files:
            run_tests_for_a_stock(filename)
