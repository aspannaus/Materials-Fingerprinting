#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import classify_utils as clf_utils
import dist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss
from joblib import dump
import datetime


def classify(train_data, test_data=None, split=0.25, plot=False, save_clf=False):

    if test_data is None:
        X_train, X_test, y_train, y_test = train_test_split(
            train_data.X, train_data.y, test_size=split)
    else:
        X_train = train_data.X
        y_train = train_data.y
        X_test = test_data.X
        y_test = test_data.y

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    n_estimators = 2000
    learning_rate = 1.05

    dt_stump = DecisionTreeClassifier(max_depth=8, criterion='entropy')

    ada = AdaBoostClassifier(base_estimator=dt_stump,
                             learning_rate=learning_rate,
                             n_estimators=n_estimators,
                             algorithm="SAMME.R")
    clf = ada.fit(X_train, y_train)

    if plot:
        ada_err = np.zeros(n_estimators)
        ada_err_train = np.zeros(n_estimators)
        dt_stump.fit(X_train, y_train)
        dt_stump_err = 1 - dt_stump.score(X_test, y_test)  # misclassification

    if save_clf:
        t = datetime.date.today()
        num = X_train.shape[0]
        out_file = "classifier_" + str(num) + "_" + t.isoformat() + ".joblib"
        dump(clf, out_file)

    y_pred = ada.predict(X_test)
    ada_score = ada.score(X_test, y_test)  # correct classification
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test))
    print(' Boosting score = {:5.2f}%'.format(ada_score * 100))

    scores = cross_validate(ada, X, y, cv=10,
                            verbose=1, n_jobs=8,
                            return_train_score=False)

    cv_scores = scores['test_score'].mean()
    print(" Mean CV score {}".format(cv_scores))
    print(' Boosting Cross-Validation: ')
    for s in scores:
        res = ' ' + str(s) + ' = ' + str(scores[s])
        print(res)
    print(' Feature Importance: ', ada.feature_importances_)

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

    print(' Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    return None


@clf_utils.MeasureTime
def main(prop=None):
    print("\n Process Started")
    clf_utils.timestamp()
    multiphase = True
    # scores = []

    # we will split the dataset in the classify function for now
    split = 0.25
    if prop is None:
        num_b = 30
        num_f = 30
    else:
        num_b = int(prop * 5000)
        num_f = int((1 - prop) * 5000)
    # total = num_b + num_f
    train_name = 'cells_50K.xyz'
    print(" Train/test split = {:3.1f}%".format(split * 100))
    print(' Test size = {:4d}'.format(int((num_b + num_f) * split)))
    data = clf_utils.makeDiagrams(num_b, num_f)
    data_pts = clf_utils.getData()

    if multiphase:
        test_name = 'multiphase_apt.xyz'
        test_num = 10
        print(' Test size = {:4d}'.format(test_num))
        test_data = clf_utils.makeDiagrams(num_b, num_f, test_len=test_num)
        test_pts = clf_utils.getData()

    # print('\n Test size split: {:2.1f} %'.format(100 *
    #                                              float(test_b + test_f) /
    #                                              float(num_b + num_f)))
    print(' Total PDs = {:4d}, BCC: {} FCC: {}'.format(num_b + num_f, num_b, num_f))

    print("\n Reading Training Data")
    data_pts.read_data(train_name, data)
    data.make_bcc_dgms()
    data.make_fcc_dgms()
    data.dgm_lists()
    train_dists = dist.distances(data, metric='dpc')

    if multiphase:
        print("\n Reading Testing Data")
        test_pts.read_data(test_name, test_data, multi=True)
        test_data.make_dgms()
        test_dists = dist.distances(test_data, multi=True, metric='dpc')

    print("\n Multiprocessing Distances\n\n Training Set")
    train_dists.dists_mp(data)
    train_dists.feature_matrix(data)

    if multiphase:
        print("\n Computing Distances: Testing Set\n")
        test_dists.dists_mp_multiphase(test_data, data)
        test_dists.feature_matrix(test_data, multiphase=True)

    print(" Complete\n\n Classifying")

    classify(train_dists)

    # avg = np.asarray(scores)
    # print(scores, type(scores))
    # print(' Mean CV accuracy = {:5.2f}%'.format(avg.mean() * 100))

    print("\n Process Concluded")
    clf_utils.timestamp()

    return None


if __name__ == "__main__":
    main()
    # vals = [0.1, 0.25, 0.4, 0.6, 0.75, 0.9]
    # for v in vals:
    #    main(v)
