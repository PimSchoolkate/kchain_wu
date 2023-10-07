import numpy as np
import time

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder

def aggregate_cv_results_kchain(cv_metrics):

    agg_results = {}

    for metric in cv_metrics.keys():
        results = np.array(cv_metrics[metric])

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

        agg_results[metric]["mean"] = np.mean(results) # only consider the last entry as this is supposed to be the best one
        agg_results[metric]["std"] = np.std(results) # only consider the last entry as this is supposed to be the best one.

    return agg_results  


def cv_experiment_kchain(fold_dict, model, model_kwargs={}, fit_kwargs={}, verbose=False, data_transform=None):

    cv_metrics = {"time": [],
                  "layers": [],
                "acc_train": [],
                "acc_test": [],
                "f1_train": [],
                "f1_test": [],
                "mse_train": [],
                "mse_test": [],
                "hsic_train": [],
                "hsic_test": [],
                "knn_acc_train": [],
                "knn_acc_test": [],
                "gnb_acc_train": [],
                "gnb_acc_test": []}

    for fold in fold_dict.keys():

        if verbose:
            print(" ============================ ")
            print(f" Fold {fold}")

        X_train = fold_dict[fold]["X_train"]
        y_train = fold_dict[fold]["y_train"].ravel()
        X_test = fold_dict[fold]["X_test"]
        y_test = fold_dict[fold]["y_test"].ravel()

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

        if data_transform == "standardize":
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        if data_transform == "normalize":
            scaler = Normalizer()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        s_t = time.time()

        fold_model = model(**model_kwargs)
        fold_model.fit(X_train, y_train, X_test, y_test, **fit_kwargs)

        e_t = time.time()

        y_hat_test = np.argmax(fold_model.forward(X_test), axis=1)
        y_hat_train = np.argmax(fold_model.forward(X_train), axis=1)

        cv_metrics['acc_test'].append(accuracy_score(y_test, y_hat_test))
        cv_metrics['acc_train'].append(accuracy_score(y_train, y_hat_train))

        cv_metrics['f1_test'].append(f1_score(y_test, y_hat_test, average='weighted'))
        cv_metrics['f1_train'].append(f1_score(y_train, y_hat_train, average='weighted'))

        cv_metrics['mse_test'].append(mean_squared_error(y_test, y_hat_test))
        cv_metrics['mse_train'].append(mean_squared_error(y_train, y_hat_train))

        cv_metrics["time"].append(e_t - s_t)
        cv_metrics["layers"].append(len(fold_model.layers))

        cv_metrics['hsic_test'].append(fold_model.layers[-1].metrics['hsic_test'])
        cv_metrics['hsic_train'].append(fold_model.layers[-1].metrics['hsic_train'])

        cv_metrics['knn_acc_test'].append(fold_model.layers[-1].metrics['knn_test'])
        cv_metrics['knn_acc_train'].append(fold_model.layers[-1].metrics['knn_train'])

        cv_metrics['gnb_acc_test'].append(fold_model.layers[-1].metrics['gnb_test'])
        cv_metrics['gnb_acc_train'].append(fold_model.layers[-1].metrics['gnb_train'])

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


def cv_experiment(fold_dict, model, model_kwargs={}, fit_kwargs={}, data_transform=None):
    """ Bad coding here basically a copy of the above function but with a different name and a different fit and evaluate function"""
    cv_metrics = {"train_acc": []
                , "test_acc": []}

    for fold in fold_dict.keys():

        X_train = fold_dict[fold]["X_train"]
        y_train = fold_dict[fold]["y_train"]
        X_test = fold_dict[fold]["X_test"]
        y_test = fold_dict[fold]["y_test"]

        if data_transform == "standardize":
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        if data_transform == "normalize":
            scaler = Normalizer()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

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