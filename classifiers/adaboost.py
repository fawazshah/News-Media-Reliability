import numpy as np
from sklearn.ensemble import AdaBoostClassifier

from train_precomp import train_model


class AdaBoostWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "adaboost"

    @staticmethod
    def get_classifier():
        return AdaBoostClassifier()

    @staticmethod
    def get_gridsearch_params():
        return dict(learning_rate=np.logspace(-4, 1, 6),
                    n_estimators=np.arange(50, 250, 50),
                    )

    @staticmethod
    def get_best_classifier(clf):
        return AdaBoostClassifier(
            learning_rate=clf.best_estimator_.learning_rate,
            n_estimators=clf.best_estimator_.n_estimators,
        )

train_model(AdaBoostWrapper)
