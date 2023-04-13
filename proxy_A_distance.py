# Compute A-distance using numpy and sklearn
# Reference: Analysis of representations in domain adaptation, NIPS-07.

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

C_list = [10**k for k in range(-10, 6, 1)]
scoring = 'roc_auc'; cv = 3; random_state=42

def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    # C_list = np.logspace(-5, 4, 2)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = LogisticRegression(random_state=random_state, penalty='l2', solver='liblinear', class_weight='balanced',
                                     C=C)
        # hyperparameter tunning for logistic regression model
        # clf =  GridSearchCV(LRmodel, param_grid=param_grid, cv=cv)    

        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)    # error rate

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)