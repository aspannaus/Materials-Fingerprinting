#!/usr/bin/env python3
# file: tda_classify.py
# author: Adam Spannaus
# date: 12/08/202
# utility main functions for materials fingerprinting code

import numpy as np
import matplotlib.pyplot as plt
import classify_utils as clf_utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import zero_one_loss, confusion_matrix
from joblib import dump
import datetime
import dist

np.random.seed(42)


def classify(train_data, class_labels, test_data=None, split=0.25, plot=False,
             save_clf=False):
    '''Classify pds from atomic neighborhods.'''

    if test_data is None:
        X_train, X_test, y_train, y_test = train_test_split(
            train_data.X, train_data.y, test_size=split)

    n_estimators = 1000
    learning_rate = 1.05

    dt_stump = DecisionTreeClassifier(max_depth=6, criterion='entropy')

    ada = AdaBoostClassifier(base_estimator=dt_stump,
                             learning_rate=learning_rate,
                             n_estimators=n_estimators,
                             algorithm='SAMME.R')

    if plot:
        ada_err = np.zeros(n_estimators)
        ada_err_train = np.zeros(n_estimators)
        dt_stump.fit(X_train, y_train)
        dt_stump_err = 1 - dt_stump.score(X_test, y_test)  # misclassification

    clf = ada.fit(X_train, y_train)

    if save_clf:
        t = datetime.date.today()
        num = X_train.shape[0]
        out_file = 'classifier_' + str(num) + '_' + t.isoformat() + '.joblib'
        dump(clf, out_file)

    y_predict = ada.predict(X_test)
    ada_score = ada.score(X_test, y_test)
    print(' 75/25 Train test split:')
    print(' Accuracy score = {:5.2f}%'.format(ada_score * 100))
    print(' Confusion Matrix: \n', confusion_matrix(y_test, y_predict))

    scores = cross_validate(ada, train_data.X, train_data.y, cv=10,
                            verbose=0, n_jobs=4,
                            return_train_score=False)

    cv_scores = scores['test_score'].mean()

    print('\n Accuracy 10-fold Cross Validation: ')

    if plot:
        for i, y_pred in enumerate(ada.staged_predict(X_test)):
            ada_err[i] = zero_one_loss(y_test, y_pred)
        for i, y_pred in enumerate(ada.staged_predict(X_train)):
            ada_err_train[i] = zero_one_loss(y_train, y_pred)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k--',
                label='Decision Stump Error')
        ax.plot(np.arange(n_estimators) + 1, ada_err,
                label='AdaBoost Test Error',
                color='orange')
        ax.plot(np.arange(n_estimators) + 1, ada_err_train,
                label='AdaBoost Train Error',
                color='green')

        ax.set_ylim((-0.1, 1.1))
        ax.set_xlabel('Ensemble Size')
        ax.set_ylabel('Error Rate')
        leg = ax.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.7)
        plt.tight_layout()
        plt.show()

    print(' Mean CV accuracy = {:5.2f}%'.format(np.asarray(cv_scores).mean() * 100))

    return None


@clf_utils.MeasureTime
def classify_fn(n=0.50, m=0.5, prop=0.5):
    print('\n Process Started')
    clf_utils.timestamp()

    split = 0.25
    num_pds = 500  # total number of bcc and fcc structures
    classes = []
    noise = [0, 0.25, 0.50, 0.75, 1]
    missing = [0, 0.33, 0.50, 0.67]

    if 0 <= prop <= 1:
        num_b = int(prop * num_pds)
        num_f = int((1 - prop) * num_pds)
    else:
        print(' The proportion given is out of bounds.')
        print(' Program terminating.')
        return None

    if n in noise:
        if n == 0:
            n = '00'
        else:
            n = str(int(n * 100))
        b_train = 'BCC_' + n
        f_train = 'FCC_' + n
    else:
        print(' Please select a noise level from sigma = [0.0, 0.25, 0.5, 0.75, 1.0]')
        print(' Program terminating')
        return None

    if m in missing:
        if m == 0:
            m = '00'
        else:
            m = str(int(m * 100))
        b_train = b_train + '_' + m + '.xyz'
        f_train = f_train + '_' + m + '.xyz'
    else:
        print(' Please select a percent missing from [0, 33, 50, 67]')
        print(' Program terminating')
        return None

    if num_b > 0:
        classes.append('bcc')
    if num_f > 0:
        classes.append('fcc')

    num_train = num_b + num_f
    data = clf_utils.makeDiagrams(num_b, num_f)
    data_pts = clf_utils.getData()

    print(' BCC = {}\tFCC = {}\t'.format(num_b, num_f))
    print(' Total PDs = {:4d}'.format(num_train))
    print(' Std Dev noise: {}\tPercent Missing {} %'.format(float(n)/100., m))

    print('\n Reading Training Data')
    if num_b > 0:
        data_pts.read_data(b_train, data)
        data.make_bcc_dgms()
    if num_f > 0:
        data_pts.read_data(f_train, data)
        data.make_fcc_dgms()
    data.dgm_lists()

    train_dists = dist.distances(data, metric='dpc', classes=classes)
    print('\n Multiprocessing Distances\n\n Training Set')

    train_dists.dists_mp(data)
    train_dists.feature_matrix()

    print(' Complete\n\n Classifying')
    classify(train_dists, classes, split=split, plot=False)
    print('\n Process Concluded')
    clf_utils.timestamp()

    return None


def main():
    noise = 0.25      # std dev of added noise, sigma in manuscript
    missing = 0.33    # gamma in manuscript
    proportion = 0.5  # proportion btwn bcc and fcc
    classify_fn(n=noise, m=missing, prop=proportion)


if __name__ == '__main__':
    main()
