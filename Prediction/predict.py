import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from data_prep import data_preparation as dp
import os
import definitions
import numpy as np

def type_the_frame(df: pd.DataFrame):
    df = df.copy()
    df['Stock'] = df['Stock'].apply(lambda x: x.strip())
    df['Our_test_score'] = df['Our_test_score'].apply(lambda x: float(x))
    return df

def filter(df: pd.DataFrame):
    df = df.copy()
    df = df[df['Algorithm'] == 'RF']
    df = df[df['Our_test_score'] > 0.69]
    df = df[df['Stock'].isin(['KMT'])]
    return df

def predict_rf(n_estimators, max_depth, X, y, data_to_predict):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X, y)
    print(clf.predict(data_to_predict))

def predict_svm(kernel, degree, c, X, y, data_to_predict):
    clf = svm.SVC(kernel=kernel, gamma='scale', degree=degree, C=c)
    clf.fit(X, y)
    print(clf.predict(data_to_predict))

def get_prepared_data(STOCK_FILE, window_size, feature_window_size, discretize):
    df, actual_data_to_predict = dp.data_preparation(os.path.join(f"{definitions.ROOT_DIR}", f"data/{STOCK_FILE}"),
                                                     window_size=window_size,
                                                     feature_window_size=feature_window_size,
                                                     discretize=discretize).data_frame_with_features()
    # df.drop(columns=['open', 'high', 'low', 'close'], inplace=True)
    data_for_algos = df.to_numpy()
    return data_for_algos, actual_data_to_predict.to_numpy()

def get_data_and_predict(x, algo):
    print(x)
    filename = f"{x['Stock']}.csv"
    future_day = int(x['Future_day'])
    data_for_algos, actual_data_to_predict = get_prepared_data(filename, future_day, 50, True)
    X = np.asarray(list(map(lambda row: row[:-1], data_for_algos)))
    y = np.asarray(list(map(lambda row: row[-1], data_for_algos)))
    n_estimators = int(x['Estimators'])
    max_depth = int(x['Depth'])
    print(filename, future_day)
    predict_rf(n_estimators, max_depth, X, y, actual_data_to_predict)

# df = pd.read_csv('../Results/Fold/All/Only_Discretize/Profit_loss/result.csv')
# df = type_the_frame(df)
# df = filter(df)
# df.apply(lambda x: get_data_and_predict(x), axis = 1)

get_data_and_predict({'Stock': 'CWEN', 'Future_day': 20, 'Estimators': 40, 'Depth': 10})