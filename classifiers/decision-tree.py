import numpy as np
from sklearn.tree import DecisionTreeClassifier

from train_precomp import train_model


class DecisionTreeWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "decision tree"

    @staticmethod
    def get_classifier():
        return DecisionTreeClassifier()

    @staticmethod
    def get_gridsearch_params():
        return dict(criterion=["gini", "entropy"],
                    splitter=["best", "random"],
                    min_samples_leaf=np.arange(1, 10, 1)
                    )

    @staticmethod
    def get_best_classifier(clf):
        return DecisionTreeClassifier(
            criterion=clf.best_estimator_.criterion,
            splitter=clf.best_estimator_.splitter,
            min_samples_leaf=clf.best_estimator_.min_samples_leaf
        )

train_model(DecisionTreeWrapper)
