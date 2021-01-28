# -*- coding: utf-8 -*-
import os
import json
import shutil
import logging
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from prettytable import PrettyTable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

from shared import label2int, int2label, FEATURE_MAPPING, parse_arguments, calculate_metrics

np.random.seed(16)

import warnings

warnings.filterwarnings("ignore")

# setup the logging environment
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                    datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

params_svm = [dict(kernel=["rbf"], gamma=np.logspace(-6, 1, 8), C=np.logspace(-2, 2, 5))]


if __name__ == "__main__":

    # parse the command-line arguments
    args = parse_arguments()

    if not args.features:
        raise ValueError("No Features are specified")

    # create the list of features sorted alphabetically
    bare_features = []
    for i, feature in enumerate(args.features.split(",")):
        if feature in FEATURE_MAPPING.keys():
            bare_features += FEATURE_MAPPING[feature].split(",")
    bare_features = sorted(bare_features)

    # specify the output directory where the results will be stored
    out_dir = os.path.join(args.home_dir, "data", args.dataset, f"results", f"{args.task}_{args.features}", f"svm")

    # remove the output directory (if it already exists and args.clear_cache was set to TRUE)
    shutil.rmtree(out_dir) if args.clear_cache and os.path.exists(out_dir) else None

    # create the output directory
    os.makedirs(out_dir, exist_ok=True)

    # display the experiment summary in a tabular format
    summary = PrettyTable()
    summary.add_row(["task", args.task])
    summary.add_row(["classification mode", "single classifier"])
    summary.add_row(["features", args.features])
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
        for feature in bare_features
    }

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
            [list(itertools.chain(*[features[feat][url] for feat in bare_features])) for url in urls["train"]]).astype(
            "float")
        y["train"] = np.array([labels[url] for url in urls["train"]], dtype=np.int)

        # concatenate the different features/labels for the testing sources
        X["test"] = np.asmatrix(
            [list(itertools.chain(*[features[feat][url] for feat in bare_features])) for url in urls["test"]]).astype(
            "float")
        y["test"] = np.array([labels[url] for url in urls["test"]], dtype=np.int)

        # normalize the features values
        scaler = MinMaxScaler()
        scaler.fit(X["train"])
        X["train"] = scaler.transform(X["train"])
        X["test"] = scaler.transform(X["test"])

        # fine-tune the model
        clf_cv = GridSearchCV(SVC(), scoring="f1_macro", cv=num_folds, n_jobs=4, param_grid=params_svm)
        clf_cv.fit(X["train"], y["train"])
        best_params.append(clf_cv.best_estimator_)

        # train the final classifier using the best parameters during crossvalidation
        clf = SVC(
            kernel=clf_cv.best_estimator_.kernel,
            gamma=clf_cv.best_estimator_.gamma,
            C=clf_cv.best_estimator_.C,
            probability=True
        )

        train_sizes, train_scores, test_scores = learning_curve(clf, np.vstack((X["train"], X["test"])), np.concatenate([y["train"], y["test"]]), cv=5)
        plt.plot(train_sizes, train_scores, 'o-', color='r', label="Training error")
        plt.plot(train_sizes, test_scores, 'o-', color='g', label="Test error")
        plt.legend(loc="best")
        plt.show()


        clf.fit(X["train"], y["train"])

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
    logger.info(best_params)

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
        f.write("Best parameters at each fold:\n")
        for i in range(num_folds):
            f.write(str(best_params[i]) + "\n")
