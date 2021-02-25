import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train_precomp import train_model


class RandomForestWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "random forest"

    @staticmethod
    def get_classifier():
        return RandomForestClassifier()

    @staticmethod
    def get_gridsearch_params():
        return dict(n_estimators=np.arange(50, 250, 25),
                    criterion=["gini", "entropy"],
                    min_samples_leaf=np.arange(1, 10, 1)
                    )

    @staticmethod
    def get_best_classifier(clf):
        return RandomForestClassifier(
            n_estimators=clf.best_estimator_.n_estimators,
            criterion=clf.best_estimator_.criterion,
            min_samples_leaf=clf.best_estimator_.min_samples_leaf,
        )

train_model(RandomForestWrapper)
