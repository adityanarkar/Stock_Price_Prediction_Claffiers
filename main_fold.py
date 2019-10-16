import json
import os
from functools import reduce
import definitions
import numpy as np
from ZeroR import zeror
import KNN.knn_fold as knn

import data_prep.data_collection as dc
from RandomForest import rf_fold as rf
from SVM import svm_fold
from data_prep import data_preparation as dp


def selectTop3(top3, x):
    if x["our_test_score"] > top3[-1]["our_test_score"]:
        top3[-1] = x
    return sorted(top3, key=lambda y: y["our_test_score"], reverse=True)


def selectTop(top, x):
    print("x ", x, "top ", top)
    if x["our_test_score"] > top["our_test_score"]:
        return x
    return top


def getInitial():
    initial = []
    for i in range(3):
        initial.append({"our_test_score": 0})
    return initial


def result_in_csv(STOCK, Algo, Estimator=0, Depth=0, Distance_function='0', No_of_features=0, Model_Score=0,
                  Future_day=0, C=-1,
                  Our_test_score=-1):
    return f"{STOCK},{Algo},{Estimator},{Depth},{Distance_function},{No_of_features},{Model_Score},{Future_day},{C},{Our_test_score}\n "


def testRandomForests(STOCK, future_day, data_for_algos, estimator_start, estimator_stop,
                      depth_start, depth_stop,
                      initial_no_of_features, max_features):
    print(f"RF started for {STOCK} {future_day}")
    n_estimators = range(estimator_start, estimator_stop, 10)
    max_depth = range(depth_start, depth_stop, 10)
    top = get_initial_top_rf()
    for no_of_features in [10, 15, 20, 22, data_for_algos.shape[1]]:
        for i in n_estimators:
            for j in max_depth:
                try:
                    selector, score, last_test_score = rf.random_forest_classifier(data_for_algos, i, j, no_of_features,
                                                                                   future_day=future_day)
                    if is_new_model_better(top, score, last_test_score):
                        top = get_top_rf(estimators=i, max_depth=j, model_score=score, future_day=future_day,
                                         no_of_features=no_of_features, last_test_score=last_test_score)
                except:
                    continue
    result = get_top_rf_result_csv_format(STOCK, top)
    print(result)
    print(f"RF ended for {STOCK} {future_day}")
    return result


def get_initial_top_rf():
    return {"estimators": -1, 'max_depth': -1, 'model_score': -1, 'future_day': -1, 'no_of_features': -1, 'our_test_score':-1}


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


def testZeroHour(STOCK, future_day, data_for_algos):
    result = ""
    print(f"ZR started for {STOCK} {future_day}")
    try:
        model, score, last_score = zeror.zr(data_for_algos, future_day)
        result = result_in_csv(STOCK, 'ZR', Future_day=future_day, Model_Score=score, Our_test_score=last_score)
    except:
        result = result_in_csv(STOCK, 'ZR', Future_day=future_day, Model_Score=-1, Our_test_score=-1)
    print(result)
    print(f"ZR ended for {STOCK} {future_day}")
    return result


def create_dir_and_store_result(dir_to_create, result_path, result):
    if not os.path.isdir(dir_to_create):
        os.mkdir(dir_to_create)
        with open(result_path, 'w') as f:
            f.write(json.dumps(result))


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


def testKNN(STOCK, data_for_algos, future_day):
    algos = ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']
    top = get_knn_top("-1", -1, -1, -1)
    print(f"KNN started for {STOCK} {future_day}")
    for no_of_features in [10, 15, 20, 22, "all"]:
        for n_neighbors in [3, 5, 7, 9, 11]:
            for distance_function in algos:
                try:
                    clf, score, last_test_score = knn.knn_classifier(data_for_algos, distance_function, n_neighbors, future_day, no_of_features)
                    if is_new_model_better(top, score, last_test_score):
                        top = get_knn_top(distance_function, score, last_test_score, no_of_features)
                except:
                    continue
    result = get_csv_result_knn(STOCK, top, future_day)
    print(result)
    print(f"KNN ended for {STOCK} {future_day}")
    return result


def get_csv_result_knn(STOCK, top, future_day):
    return result_in_csv(STOCK, 'KNN', Distance_function=top["distance_function"], Model_Score=top['score'],
                  Future_day=future_day,
                  Our_test_score=top['our_test_score'],
                  No_of_features=top['no_of_features'])


def is_new_model_better(top, score, last_test_score):
    return last_test_score > top['our_test_score'] or (
                last_test_score == top['our_test_score'] and score > top['score'])


def get_knn_top(distance_function, score, our_test_score, no_of_features):
    return {"distance_function": distance_function, 'score': score, 'our_test_score': our_test_score, 'no_of_features': no_of_features}


def testSVM(STOCK, data_for_algos, future_day, initial_no_of_features,
            max_features, C):
    print(f"SVM started for {STOCK} {future_day}")
    top = get_top_svm(-1, -1, future_day, -1, -1)
    for no_of_features in [10, 15, 20, 22, data_for_algos.shape[1]]:
        for c_val in C:
            try:
                clf, score, last_test_score = svm_fold.svm_classifier(data_for_algos, no_of_features, c_val, future_day)
                if is_new_model_better(top, score, last_test_score):
                    top = get_top_svm(c_val, score, future_day, no_of_features, last_test_score)
            except:
                continue
    result = get_svm_top_result_csv(STOCK, top)
    print(f"{result}")
    print(f"SVM ended for {STOCK} {future_day}")
    return result


def get_top_svm(C, score, future_day, no_of_features, our_test_score):
    return {'C': C, 'score': score, 'future_day': future_day, 'no_of_features': no_of_features, 'our_test_score': our_test_score}


def get_svm_top_result_csv(STOCK, top):
    return result_in_csv(STOCK, 'SVM', No_of_features=top['no_of_features'], Distance_function="Linear", C=top['C'],
                         Model_Score=top['score'], Future_day=top['future_day'], Our_test_score=top['our_test_score'])


def get_test_score(predictions, test_classes):
    our_test_score = sum([1 if predictions[i] == test_classes[i] else 0 for i in range(len(predictions))])
    return 0 if our_test_score is None else our_test_score


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


        if 'KNN' in algos:
            result += testKNN(STOCK, data_for_algos, future_day)

        if 'RF' in algos:
            result += testRandomForests(STOCK, future_day, data_for_algos, estimator_start,
                                        estimator_stop,
                                        depth_start, depth_stop, initial_no_of_features, max_features)

        if 'SVM' in algos:
            result += testSVM(STOCK, data_for_algos, future_day, initial_no_of_features, max_features, C)

        if 'ZR' in algos:
            result += testZeroHour(STOCK, future_day, data_for_algos)

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
