from sklearn.neural_network import MLPClassifier

from train import train_model


class MLPWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_name():
        return "mlp"

    @staticmethod
    def get_classifier():
        return MLPClassifier()

    @staticmethod
    def get_gridsearch_params():
        return dict(activation=["tanh"],
                    hidden_layer_sizes=[(12,), (24,)],
                    learning_rate=["constant", "adaptive"],
                    early_stopping=[False],
                    alpha=[0.1, 1],
                    max_iter=[500])

    @staticmethod
    def get_best_classifier(clf):
        return MLPClassifier(
            activation=clf.best_estimator_.activation,
            hidden_layer_sizes=clf.best_estimator_.hidden_layer_sizes,
            learning_rate=clf.best_estimator_.learning_rate,
            early_stopping=clf.best_estimator_.early_stopping,
            alpha=clf.best_estimator_.alpha,
            max_iter=clf.best_estimator_.max_iter
        )

train_model(MLPWrapper)
