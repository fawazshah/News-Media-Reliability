import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, mean_absolute_error

label2int = {
    "fact": {"low": 0, "mixed": 1, "high": 2},
    "bias": {"left": 0, "center": 1, "right": 2},
}

int2label = {
    "fact": {0: "low", 1: "mixed", 2: "high"},
    "bias": {0: "left", 1: "center", 2: "right"},
}

NUM_CLASSES = 3

TWITTER_ALL = "has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified"
WIKI_ALL = "has_wikipedia,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc"
ARTICLE_ALL = "articles_body_glove,articles_title_glove"
ALEXA = "alexa"
ALL = ",".join([TWITTER_ALL, WIKI_ALL, ARTICLE_ALL, ALEXA])
FEATURE_MAPPING = {"TWITTER_ALL": TWITTER_ALL,
                   "WIKI_ALL": WIKI_ALL,
                   "ARTICLE_ALL": ARTICLE_ALL,
                   "ALEXA": ALEXA,
                   "ALL": ALL}


def calculate_metrics(actual, predicted):
    """
    Calculate performance metrics given the actual and predicted labels.
    Returns the macro-F1 score, the accuracy, the mean absolute error (MAE)
    and macro-averaged mean absolute error (MAEM).
    MAEM is mean absolute error but with errors weighted by the size of the
    true class.
    """
    # calculate macro-f1
    f1 = f1_score(actual, predicted, average='macro') * 100

    # calculate accuracy
    accuracy = accuracy_score(actual, predicted) * 100

    # calculate mean absolute error (mae)
    mae = mean_absolute_error(actual, predicted)

    # calculate macro-averaged mean absolute error (maem)
    class_sizes = np.zeros(NUM_CLASSES)
    class_errors = np.zeros(NUM_CLASSES)
    for i, label in enumerate(actual):
        class_sizes[label] += 1
        class_errors[label] += abs(actual[i] - predicted[i])
    class_weights = 1 / class_sizes
    maem = 1 / NUM_CLASSES * sum([class_weight * class_error
                                  for class_weight, class_error in zip(class_weights, class_errors)])

    return f1, accuracy, mae, maem


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required command-line arguments
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default="",
        required=True,
        help="the features that will be used in the current experiment (comma-separated)",
    )
    parser.add_argument(
        "-tk",
        "--task",
        type=str,
        default="bias",
        required=True,
        help="the task for which the model is trained: (fact or bias)",
    )

    # Boolean command-line arguments
    parser.add_argument(
        "-cc",
        "--clear_cache",
        action="store_true",
        help="flag to whether the corresponding features file need to be deleted before re-computing",
    )

    # Other command-line arguments
    parser.add_argument(
        "-hd",
        "--home_dir",
        type=str,
        default="/Users/fawaz/Desktop/data/News-Media-Reliability",
        help="the directory that contains the project files"
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        default="emnlp18",
        help="the name of the dataset for which we are building the media objects",
    )
    parser.add_argument(
        "-nl",
        "--num_labels",
        type=int,
        default=3,
        help="the number of classes of the given task",
    )

    return parser.parse_args()
