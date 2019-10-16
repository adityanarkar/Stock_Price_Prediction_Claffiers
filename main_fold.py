import os

import data_prep.data_collection as dc
import definitions
from RF import rf_fold as rf

from data_prep import data_preparation as dp


def testRandomForests(STOCK, future_day, data_for_algos, estimator_start, estimator_stop,
                      depth_start, depth_stop,
                      initial_no_of_features, max_features):
    n_estimators = range(estimator_start, estimator_stop, 10)
    max_depth = range(depth_start, depth_stop, 10)
    top = get_initial_top_rf()
    # for no_of_features in [10, 15, 20, 25, 28]:
    print(f"{STOCK} {future_day}")
    for i in n_estimators:
        for j in max_depth:
            try:
                selector, score, last_test_score = rf.random_forest_classifier(data_for_algos, i, j, -1,
                                                                               future_day=future_day)
                if is_new_model_better(top, score, last_test_score):
                    top = get_top_rf(estimators=i, max_depth=j, model_score=score, future_day=future_day,
                                     no_of_features=-1, last_test_score=last_test_score)
            except:
                continue
    result = get_top_rf_result_csv_format(STOCK, top)
    print(f"final Result RF: {result}")
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


def result_in_csv(STOCK, Algo, Estimator=0, Depth=0, Distance_function='0', No_of_features=0, Model_Score=0,
                  Future_day=0, C=-1,
                  Our_test_score=-1):
    return f"{STOCK},{Algo},{Estimator},{Depth},{Distance_function},{No_of_features},{Model_Score},{Future_day},{C},{Our_test_score}\n "


def get_csv_result_knn(STOCK, top, future_day):
    return result_in_csv(STOCK, 'KNN', Distance_function=top["distance_function"], Model_Score=top['score'],
                         Future_day=future_day,
                         Our_test_score=top['our_test_score'])


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
        except:
            continue

        if 'KNN' in algos:
            result += testKNN(STOCK, data_for_algos, future_day)

        if 'RF' in algos:
            print(f"Predicting {STOCK} for future days: {future_day} using RF")
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
