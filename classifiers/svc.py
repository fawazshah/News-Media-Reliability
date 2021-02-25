import numpy as np
from sklearn.svm import SVC

from train_precomp import train_model


class SVCWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "svm"

    @staticmethod
    def get_classifier():
        return SVC()

    @staticmethod
    def get_gridsearch_params():
        return[dict(kernel=["rbf"], gamma=np.logspace(-6, 1, 8), C=np.logspace(-2, 2, 5))]

    @staticmethod
    def get_best_classifier(clf):
        return SVC(
            kernel=clf.best_estimator_.kernel,
            gamma=clf.best_estimator_.gamma,
            C=clf.best_estimator_.C,
            probability=True
        )

train_model(SVCWrapper)
