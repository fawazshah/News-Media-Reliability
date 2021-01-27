import argparse
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

label2int = {
    "fact": {"low": 0, "mixed": 1, "high": 2},
    "bias": {"extreme-left": 0, "left-center": 1, "left": 2, "center": 3, "right-center": 4, "right": 5, "extreme-right": 6},
}

int2label = {
    "fact": {0: "low", 1: "mixed", 2: "high"},
    "bias": {0: "extreme-left", 1: "left-center", 2: "left", 3: "center", 4: "right-center", 5: "right", 6: "extreme-right"},
}

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
    Returns the macro-F1 score, the accuracy, the flip error rate and the
    mean absolute error (MAE).
    The flip error rate is the percentage where an instance was predicted
    as the opposite label (i.e., left-vs-right or high-vs-low).
    """
    # calculate macro-f1
    f1 = f1_score(actual, predicted, average='macro') * 100

    # calculate accuracy
    accuracy = accuracy_score(actual, predicted) * 100

    # calculate the flip error rate
    flip_err = sum([1 for i in range(len(actual)) if abs(actual[i] - predicted[i]) > 1]) / len(actual) * 100

    # calculate mean absolute error (mae)
    mae = sum([abs(actual[i] - predicted[i]) for i in range(len(actual))]) / len(actual)
    mae = mae[0] if not isinstance(mae, float) else mae

    return f1, accuracy, flip_err, mae


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
        default=7,
        help="the number of classes of the given task",
    )

    return parser.parse_args()
