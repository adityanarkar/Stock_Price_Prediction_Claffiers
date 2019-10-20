def get_max_k_splits(X, k, size_of_each_split):
    max_splits = get_max_no_of_possible_splits(X, k, size_of_each_split)
    X_length = len(X)
    return get_train_test_index(max_splits, X_length, size_of_each_split)


def get_train_test_index(max_splits, X_length, size_of_each_split):
    train_index = []
    test_index = []
    for i in range(1, max_splits + 1):
        test_start_index = X_length - (i * size_of_each_split)
        test_end_index = test_start_index + size_of_each_split
        test_index.append([test_start_index, test_end_index])
        train_start_index = 0
        train_end_index = test_start_index
        train_index.append([train_start_index, train_end_index])
        yield ([train_index for train_index in range(train_start_index, train_end_index)],
               [test_index for test_index in range(test_start_index, test_end_index)])


def get_max_no_of_possible_splits(X, k, size_of_each_split):
    if splits_possible(X, k, size_of_each_split):
        return k
    else:
        test_size = int(len(X) * 0.2)
        return test_size // size_of_each_split


def splits_possible(X, k, size_of_each_split):
    return int(len(X) * 0.2) >= (k * size_of_each_split)


def get_train_test_set(X, y, train_index, test_index):
    X_train = X[train_index[0]:train_index[1]]
    y_train = y[train_index[0]:train_index[1]]
    X_test = X[test_index[0]:test_index[1]]
    y_test = y[test_index[0]:test_index[1]]

    return X_train, y_train, X_test, y_test