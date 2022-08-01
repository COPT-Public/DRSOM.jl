# script from:
# https://github.com/probml/pyprobml/tree/master/scripts
# training code based on
# https://github.com/scikit-learn/scikit-learn/blob/master/benchmarks/bench_mnist.py

import superimport

import numpy as np
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import zero_one_loss

# from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_mnist_data_openml():
    # Returns X_train: (60000, 784), X_test: (10000, 784), scaled [0...1]
    # y_train: (60000,) 0..9 ints, y_test: (10000,)
    print("Downloading mnist...")
    data = fetch_openml('mnist_784', version=1, cache=True)
    print("Done")
    #data = fetch_mldata('MNIST original')
    X = data['data'].astype('float32')
    y = data["target"].astype('int64')
    # Normalize features
    X = X / 255
    # Create train-test split (as [Joachims, 2006])
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_mnist_data_openml()

for max_iter in [10, 40]:

    ESTIMATORS = {
        'LogReg-SAG':
        LogisticRegression(solver='sag',
                           tol=1e-1,
                           multi_class='multinomial',
                           penalty='none',
                           max_iter=max_iter,
                           verbose=1,
                           C=1e4),
        'LogReg-LBFGS':
        LogisticRegression(solver='lbfgs',
                           tol=1e-6,
                           multi_class='multinomial',
                           max_iter=max_iter,
                           penalty='none',
                           verbose=1,
                           C=1e4),
    }

    print("Training Classifiers")
    print("====================")
    error, train_time, test_time = {}, {}, {}
    names = ESTIMATORS.keys()
    for name in names:
        print("Training %s ... " % name, end="")
        estimator = ESTIMATORS[name]
        estimator_params = estimator.get_params()

        time_start = time()
        estimator.fit(X_train, y_train)
        train_time[name] = time() - time_start

        time_start = time()
        y_pred = estimator.predict(X_test)
        test_time[name] = time() - time_start

        error[name] = zero_one_loss(y_train, estimator.predict(X_train)), zero_one_loss(y_test, y_pred)

        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print("{0: <24} {1: >10} {2: >11} {3: >12} {4: >12}"
          "".format("Classifier  ", "train-time", "test-time", "train-error", "test-error"))
    print("-" * 60)
    for name in sorted(names, key=error.get):
        print("{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f} {4: >12.4f}"
              "".format(name, train_time[name], test_time[name], error[name][0], error[name][1]))
    print()
"""
-----------------------------------------------------------
RESULT KEEPER:
@update by C. Zhang, 07/08/22

Classification performance:
10 epochs:
===========================
Classifier               train-time   test-time  train-error   test-error
------------------------------------------------------------
LogReg-SAG                    6.20s       0.02s       0.0699       0.0779
LogReg-LBFGS                  1.51s       0.06s       0.1245       0.1175

40 epochs:
===========================
Classifier               train-time   test-time  train-error   test-error
------------------------------------------------------------
LogReg-SAG                    6.24s       0.01s       0.0695       0.0749
LogReg-LBFGS                  3.67s       0.03s       0.0750       0.0783
"""