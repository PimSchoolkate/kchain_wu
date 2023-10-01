import numpy as np
import time


def aggregate_cv_results_kchain(cv_metrics):

    agg_results = {}

    for metric in cv_metrics.keys():
        if metric not in agg_results:
            agg_results[metric] = {}

        if metric == "time":
            agg_results[metric]["mean"] = np.mean(cv_metrics[metric])
            agg_results[metric]["std"] = np.std(cv_metrics[metric])
            continue

        if metric == "layers":
            agg_results[metric]["mean"] = np.mean(cv_metrics[metric])
            agg_results[metric]["std"] = np.std(cv_metrics[metric])
            continue

        # print(cv_metrics[metric])
        
        results = np.array([array[-1] for array in cv_metrics[metric]])
        

        agg_results[metric]["mean"] = np.mean(results) # only consider the last entry as this is supposed to be the best one
        agg_results[metric]["std"] = np.std(results) # only consider the last entry as this is supposed to be the best one.

    return agg_results  


def cv_experiment_kchain(fold_dict, model, model_kwargs={}, fit_kwargs={}, verbose=False):

    cv_metrics = {"time": []}

    for fold in fold_dict.keys():

        if verbose:
            print(" ============================ ")
            print(f" Fold {fold}")

        X_train = fold_dict[fold]["X_train"]
        y_train = fold_dict[fold]["y_train"]
        X_test = fold_dict[fold]["X_test"]
        y_test = fold_dict[fold]["y_test"]

        s_t = time.time()

        fold_model = model(**model_kwargs)
        fold_model.fit(X_train, y_train, X_test, y_test, **fit_kwargs)

        e_t = time.time()

        fold_metrics = fold_model.metrics()

        for metric in fold_metrics.keys():
            if metric not in cv_metrics:
                cv_metrics[metric] = []
            cv_metrics[metric].append(fold_metrics[metric])

        cv_metrics["time"].append(e_t - s_t)
        cv_metrics["layers"] = len(fold_model.Z)

        if verbose:
            print(" \n")

    agg_metrics = aggregate_cv_results_kchain(cv_metrics)

    return cv_metrics, agg_metrics


def aggregate_cv_results(cv_metrics):
    
    agg_results = {}

    for metric in cv_metrics.keys():
        if metric not in agg_results:
            agg_results[metric] = {}

        # print(cv_metrics[metric])        

        agg_results[metric]["mean"] = np.mean(cv_metrics[metric]) # only consider the last entry as this is supposed to be the best one
        agg_results[metric]["std"] = np.std(cv_metrics[metric]) # only consider the last entry as this is supposed to be the best one.

    return agg_results


def cv_experiment(fold_dict, model, model_kwargs={}, fit_kwargs={}):
    """ Bad coding here basically a copy of the above function but with a different name and a different fit and evaluate function"""
    cv_metrics = {"train_acc": []
                , "test_acc": []}

    for fold in fold_dict.keys():

        X_train = fold_dict[fold]["X_train"]
        y_train = fold_dict[fold]["y_train"]
        X_test = fold_dict[fold]["X_test"]
        y_test = fold_dict[fold]["y_test"]

        if len(y_train.shape) > 1:
            y_train = y_train.reshape(-1)

        if len(y_test.shape) > 1:
            y_test = y_test.reshape(-1)

        fold_model = model(**model_kwargs)
        fold_model.fit(X_train, y_train)

        train_acc = fold_model.score(X_train, y_train)
        test_acc = fold_model.score(X_test, y_test)

        cv_metrics["train_acc"].append(train_acc)
        cv_metrics["test_acc"].append(test_acc)

    agg_metrics = aggregate_cv_results(cv_metrics)

    return cv_metrics, agg_metrics