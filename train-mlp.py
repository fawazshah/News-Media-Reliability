# -*- coding: utf-8 -*-
import os
import json
import shutil
import logging
import argparse
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.neural_network import MLPClassifier
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import numpy as np

np.random.seed(16)

import warnings

warnings.filterwarnings("ignore")

# setup the logging environment
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                    datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# We find a mix of the following hyperparams to give the best performance
params_mlp = dict(activation=["tanh"],
                  hidden_layer_sizes=[(12,), (24,)],
                  learning_rate=["constant", "adaptive"],
                  alpha=[0.1, 1],
                  shuffle=[False, True],
                  max_iter=[500])

label2int = {
    "fact": {"low": 0, "mixed": 1, "high": 2},
    "bias": {"extreme-left": 0, "left-center": 1, "left": 2, "center": 3, "right-center": 4, "right": 5,
             "extreme-right": 6},
}

int2label = {
    "fact": {0: "low", 1: "mixed", 2: "high"},
    "bias": {0: "extreme-left", 1: "left-center", 2: "left", 3: "center", 4: "right-center", 5: "right",
             6: "extreme-right"},
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


if __name__ == "__main__":

    # parse the command-line arguments
    args = parse_arguments()

    if not args.features:
        raise ValueError("No Features are specified")

    # create the list of features sorted alphabetically
    original_features = args.features
    args.features = args.features.split(",")
    for i, feature in enumerate(args.features):
        if feature in FEATURE_MAPPING.keys():
            args.features.remove(feature)
            args.features += FEATURE_MAPPING[feature].split(",")
    args.features = sorted(args.features)


    # specify the output directory where the results will be stored
    out_dir = os.path.join(args.home_dir, "data", args.dataset, f"results", f"{args.task}_{original_features}",
                           f"mlp")

    # remove the output directory (if it already exists and args.clear_cache was set to TRUE)
    shutil.rmtree(out_dir) if args.clear_cache and os.path.exists(out_dir) else None

    # create the output directory
    os.makedirs(out_dir, exist_ok=True)

    # display the experiment summary in a tabular format
    summary = PrettyTable()
    summary.add_row(["task", args.task])
    summary.add_row(["classification mode", "single classifier"])
    summary.add_row(["features", original_features])
    print(summary)

    # read the dataset
    df = pd.read_csv(os.path.join(args.home_dir, "data", args.dataset, "corpus.tsv"), sep="\t")

    # create a dictionary: the keys are the media and the values are their corresponding labels (transformed to int)
    labels = {df["source_url_normalized"][i]: label2int[args.task][df[args.task][i]] for i in range(df.shape[0])}

    # load the evaluation splits
    splits = json.load(open(os.path.join(args.home_dir, "data", args.dataset, f"splits.json"), "r"))
    num_folds = len(splits)

    # create the features dictionary: each key corresponds to a feature type, and its value is the pre-computed features dictionary
    features = {
        feature: json.load(open(os.path.join(args.home_dir, "data", args.dataset, "features", f"{feature}.json"), "r"))
        for feature in args.features}

    # create placeholders where predictions will be cumulated over the different folds
    all_test_urls = []
    actual = np.zeros(df.shape[0], dtype=np.int)
    predicted = np.zeros(df.shape[0], dtype=np.int)
    probs = np.zeros((df.shape[0], args.num_labels), dtype=np.float)
    best_params = []

    i = 0

    logger.info("Start training...")
    training_start = time.perf_counter()

    for f in range(num_folds):
        logger.info(f"Fold: {f}")

        # get the training and testing media for the current fold
        urls = {
            "train": splits[str(f)]["train"],
            "test": splits[str(f)]["test"],
        }

        all_test_urls.extend(splits[str(f)]["test"])

        # initialize the features and labels matrices
        X, y = {}, {}

        # concatenate the different features/labels for the training sources
        X["train"] = np.asmatrix(
            [list(itertools.chain(*[features[feat][url] for feat in args.features])) for url in urls["train"]]).astype(
            "float")
        y["train"] = np.array([labels[url] for url in urls["train"]], dtype=np.int)

        # concatenate the different features/labels for the testing sources
        X["test"] = np.asmatrix(
            [list(itertools.chain(*[features[feat][url] for feat in args.features])) for url in urls["test"]]).astype(
            "float")
        y["test"] = np.array([labels[url] for url in urls["test"]], dtype=np.int)

        # normalize the features values
        scaler = MinMaxScaler()
        scaler.fit(X["train"])
        X["train"] = scaler.transform(X["train"])
        X["test"] = scaler.transform(X["test"])

        # fine-tune the model
        clf_cv = GridSearchCV(MLPClassifier(), scoring="f1_macro", cv=num_folds, n_jobs=4, param_grid=params_mlp)
        clf_cv.fit(X["train"], y["train"])
        best_params.append(clf_cv.best_estimator_)

        # train the final classifier using the best parameters during crossvalidation
        clf = MLPClassifier(
            activation=clf_cv.best_estimator_.activation,
            hidden_layer_sizes=clf_cv.best_estimator_.hidden_layer_sizes,
            learning_rate=clf_cv.best_estimator_.learning_rate,
            alpha=clf_cv.best_estimator_.alpha,
            shuffle=clf_cv.best_estimator_.shuffle,
            max_iter=clf_cv.best_estimator_.max_iter
        )
        clf.fit(X["train"], y["train"])
        plt.plot(clf.loss_curve_)
        plt.show()

        # generate predictions
        pred = clf.predict(X["test"])

        # generate probabilites
        prob = clf.predict_proba(X["test"])

        # cumulate the actual and predicted labels, and the probabilities over the different folds.  then, move the index
        actual[i: i + y["test"].shape[0]] = y["test"]
        predicted[i: i + y["test"].shape[0]] = pred
        probs[i: i + y["test"].shape[0], :] = prob
        i += y["test"].shape[0]

    seconds = time.perf_counter() - training_start
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60

    # calculate the performance metrics on the whole set of predictions (5 folds all together)
    results = calculate_metrics(actual, predicted)

    # display the performance metrics
    logger.info(f"Macro-F1: {results[0]}")
    logger.info(f"Accuracy: {results[1]}")
    logger.info(f"Flip Error-rate: {results[2]}")
    logger.info(f"MAE: {results[3]}")
    logger.info(f"Training took {hours} hrs, {minutes} mins, {seconds} seconds.")
    logger.info(f"Best parameters for each fold:")
    for param_set in best_params:
        logger.info(str(param_set) + "\n")

    # map the actual and predicted labels to their categorical format
    predicted = np.array([int2label[args.task][int(l)] for l in predicted])
    actual = np.array([int2label[args.task][int(l)] for l in actual])

    # create a dictionary: the keys are the media, and the values are their actual and predicted labels
    predictions = {all_test_urls[i]: (actual[i], predicted[i]) for i in range(len(all_test_urls))}

    # create a dataframe that contains the list of m actual labels, the predictions with probabilities.  then store it in the output directory
    df_data = {"source_url": all_test_urls, "actual": actual, "predicted": predicted}
    df_probs = {int2label[args.task][i]: probs[:, i] for i in range(args.num_labels)}
    df_data.update(df_probs)

    df_out = pd.DataFrame(df_data)
    columns = ["source_url", "actual", "predicted"] + [int2label[args.task][i] for i in range(args.num_labels)]
    df_out.to_csv(os.path.join(out_dir, "predictions.tsv"), index=False, columns=columns)

    # write the experiment results in a tabular format
    res = PrettyTable()
    res.field_names = ["Macro-F1", "Accuracy", "Flip error-rate", "MAE"]
    res.add_row(results)

    # write the experiment summary and outcome into a text file and save it to the output directory
    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write(summary.get_string(title="Experiment Summary") + "\n")
        f.write(res.get_string(title="Results") + "\n")
        f.write(f"Training took {hours} hrs, {minutes} mins, {seconds} seconds.\n")
        f.write("Best parameters at each fold:" + "\n")
        for param_set in best_params:
            f.write(str(param_set) + "\n")
