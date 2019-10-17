import json
import main_fold
import multiprocessing
from multiprocessing import Process, Lock

file = open('configs/config_all_disc.json')
configs = json.load(file)


def pre_run_tasks(RESULT_FILE, COMPLETED_FILE):
    main_fold.make_missing_dirs(RESULT_FILE)
    main_fold.make_missing_dirs(COMPLETED_FILE)
    main_fold.add_headers(RESULT_FILE)


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


if __name__ == '__main__':
    max_cpus = multiprocessing.cpu_count()
    lock = Lock()
    completed = []

    for dictionary in configs:
        RESULT_FILE, COMPLETED_FILE, algos, future_day_start, future_day_stop, estimator_start, \
        estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features, \
        feature_window_size, discretize, C = get_config_from_dict(dictionary)
        pre_run_tasks(RESULT_FILE, COMPLETED_FILE)

        processes = []
        counter = 0
        files = list(map(lambda x: x.replace("\n", ""), open('55stocks.txt', 'r').readlines()))
        files.reverse()
        print(files)
        for filename in files:
            if filename not in completed:
                p = Process(target=main_fold.runExperiment,
                            args=(
                                lock, filename, RESULT_FILE, algos, future_day_start, future_day_stop, estimator_start,
                                estimator_stop, depth_start, depth_stop, initial_no_of_features, max_features,
                                feature_window_size,
                                discretize, C))
                p.start()
                p.join()
                processes.append({"process": p, "stock": filename})
                if len(processes) % max_cpus == 0:
                    processes[0]["process"].join()
                    open(COMPLETED_FILE, 'a').write(f"{processes[0]['stock']}\n")
                    processes.remove(processes[0])
