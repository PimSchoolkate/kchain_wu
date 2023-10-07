import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle as util_shuffle


def load_complete_data(name: str, path: str = './data/complete_data'):
    
    X = pd.read_csv(f'{path}/{name}.csv', header=None).to_numpy()
    Y = pd.read_csv(f'{path}/{name}_label.csv', header=None).to_numpy().ravel()
    
    return X, Y


def load_cross_validation_data(dataset: str, path: str = './data/10_folded'):
    if dataset not in ['adversarial', 'cancer', 'car', 'cifar10', 'divorce', 'face', 'random', 'spiral', 'wine']:
        raise ValueError('Dataset must be one of adversarial, cancer, car, cifar10, divorce, face, random, spiral, wine')

    files = os.listdir(f'{path}/{dataset}')

    fold_dict = {}

    for file in files:
        file_splitted = os.path.splitext(file)[0].split('_')
        fold_num = int(file_splitted[1])
        if fold_num not in fold_dict:
            fold_dict[fold_num] = {}

        if len(file_splitted) == 2:
            fold_dict[fold_num]["X_train"] = pd.read_csv(f'{path}/{dataset}/{file}', header=None).to_numpy()
            continue
        
        if len(file_splitted) == 3:
            if file_splitted[2] == "test":
                fold_dict[fold_num]["X_test"] = pd.read_csv(f'{path}/{dataset}/{file}', header=None).to_numpy()
                continue
            
            elif file_splitted[2] == "label":
                fold_dict[fold_num]["y_train"] = pd.read_csv(f'{path}/{dataset}/{file}', header=None).to_numpy()
                continue

        if len(file_splitted) == 4:
            fold_dict[fold_num]["y_test"] = pd.read_csv(f'{path}/{dataset}/{file}', header=None).to_numpy()
            continue

        raise ValueError(f'Something went wrong. Got {file} and was not able to handle it.')

    return fold_dict


def make_adversarial(n_samples=1000, offset_percentage=0.1):
    half_samples = int(np.ceil(n_samples / 2))
    X_base = np.random.rand(half_samples, 2)
    X_0 = X_base - np.random.randn(half_samples, 2) * offset_percentage
    X_1 = X_base + np.random.randn(half_samples, 2) * offset_percentage
    X = np.concatenate((X_0, X_1), axis=0)
    Y = np.concatenate((np.zeros((half_samples, 1)), np.ones((half_samples, 1))), axis=0)

    return X, Y

def make_spirals(n_classes, n_samples, sep=0.0, noise=0.2, radius=1, len=4, shuffle=True):

    """ Taken and modified from: https://cs231n.github.io/neural-networks-case-study/ """

    samples_per_class = n_samples // n_classes

    X = np.zeros((samples_per_class*n_classes, 2))
    y = np.zeros(samples_per_class*n_classes)

    for j in range(n_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(sep, radius ,samples_per_class)

        t = np.linspace(j*len, (j+1)*len, samples_per_class) + np.random.randn(samples_per_class)*noise
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    if shuffle:
        # Randomly permute samples
        X, y = util_shuffle(X, y)

    return X, y


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def standardize(x):
    return (x - np.mean(x)) / np.std(x)
