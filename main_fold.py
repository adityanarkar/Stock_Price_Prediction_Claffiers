import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import definitions
from data_prep import data_preparation as dp


def getInitial():
    initial = []
    for i in range(3):
        initial.append({"our_test_score": 0})
    return initial


def result_in_csv(STOCK, Algo, Estimator=0, Depth=0, Distance_function='0', No_of_features=0, Model_Score=0,
                  Future_day=0, C=-1,
                  Our_test_score=-1):
    return f"{STOCK},{Algo},{Estimator},{Depth},{Distance_function},{No_of_features},{Model_Score},{Future_day},{C},{Our_test_score}\n "


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


def testRandomForests(STOCK, future_day, data_for_algos, estimator_start, estimator_stop,
                      depth_start, depth_stop,
                      initial_no_of_features, max_features):
    result = ""
    train_size = int(data_for_algos.shape[0] * 0.9)
    print(train_size)
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    parameters = get_params_rf(estimator_start, estimator_stop, depth_start, depth_stop)
    print(parameters)
    splits = get_splits(X)
    print(splits)
    estimator = RandomForestClassifier()
    selector = RFECV(estimator, step=1, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
    clf = GridSearchCV(selector, param_grid=parameters, cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
    print(X.shape)
    print(y.shape)
    clf.fit(X[:train_size], y[:train_size])
    print(clf.best_estimator_.estimator_)
    print(clf.best_estimator_.grid_scores_)
    print(clf.best_estimator_.ranking_)
    print(clf.best_params_)
    print(clf.score(X[train_size:], y[train_size:]))
    return result


def get_top_rf(estimators, max_depth, model_score, future_day, no_of_features, last_test_score):
    return {'estimators': estimators, 'max_depth': max_depth, 'model_score': model_score, 'future_day': future_day,
            'no_of_features': no_of_features, 'our_test_score': last_test_score}


def get_top_rf_result_csv_format(STOCK, top):
    top_estimator = top["estimators"]
    top_depth = top['max_depth']
    top_model_score = top['model_score']
    top_future_day = top['future_day']
    top_no_of_features = top['no_of_features']
    top_our_test_score = top['our_test_score']
    result = result_in_csv(STOCK, 'RF', top_estimator, top_depth, No_of_features=top_no_of_features,
                           Model_Score=top_model_score, Future_day=top_future_day, Our_test_score=top_our_test_score)
    return result


def get_prepared_data(STOCK_FILE, window_size, feature_window_size, discretize):
    df, actual_data_to_predict = dp.data_preparation(os.path.join(f"{definitions.ROOT_DIR}", f"data/{STOCK_FILE}"),
                                                     window_size=window_size,
                                                     feature_window_size=feature_window_size,
                                                     discretize=discretize).data_frame_with_features()
    # df.drop(columns=['open', 'high', 'low', 'close'], inplace=True)
    data_for_algos = df.to_numpy()
    return data_for_algos, actual_data_to_predict


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
                "C,Our_test_score\n")


def get_csv_result_knn(STOCK, top, future_day):
    return result_in_csv(STOCK, 'KNN', Distance_function=top["distance_function"], Model_Score=top['score'],
                         Future_day=future_day,
                         Our_test_score=top['our_test_score'],
                         No_of_features=top['no_of_features'])


def runExperiment(lock, STOCK_FILE, RESULT_FILE, algos, future_day_start, future_day_stop, estimator_start,
                  estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features, feature_window_size,
                  discretize, C):
    STOCK = STOCK_FILE.split(".csv")[0]
    print(STOCK)

    result = ""
    for future_day in range(future_day_start, future_day_stop, 10):
        try:
            data_for_algos, actual_data_to_predict = get_prepared_data(
                STOCK_FILE, future_day, feature_window_size, discretize)
            # print(data_for_algos.shape[1])
        except:
            continue

        if 'RF' in algos:
            result += testRandomForests(STOCK, future_day, data_for_algos, estimator_start,
                                        estimator_stop,
                                        depth_start, depth_stop, initial_no_of_features, max_features)

    write_result_to_file(lock, RESULT_FILE, result)


def collect_data(no_of_symbols: int, filepath: str):
    dc.sample_data(no_of_symbols, filepath)
    dc.collect_data(filepath)


def get_requested_tickrs(filepath):
    result = []
    with open(filepath) as f:
        for line in f.readlines():
            if not line.startswith("#"):
                result.append(line.replace("\n", ""))
    return result
