import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from train_precomp import train_model


class GradientBoostingWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "gradient boosted forest"

    @staticmethod
    def get_classifier():
        return GradientBoostingClassifier()

    @staticmethod
    def get_gridsearch_params():
        return dict(learning_rate=np.logspace(-4, 1, 5),
                    n_estimators=np.arange(50, 250, 50),
                    min_samples_leaf=np.arange(1, 10, 2)
                    )

    @staticmethod
    def get_best_classifier(clf):
        return GradientBoostingClassifier(
            learning_rate=clf.best_estimator_.learning_rate,
            n_estimators=clf.best_estimator_.n_estimators,
            min_samples_leaf=clf.best_estimator_.min_samples_leaf,
        )

train_model(GradientBoostingWrapper)
