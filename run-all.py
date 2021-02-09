import os


if __name__ == "__main__":

    args_list = ["-tk 'bias' -f 'ALL'", "-tk 'bias' -f 'ARTICLE_ALL'"]
    for args in args_list:
        print("USING SVM")
        os.system(f"python3 classifiers/svc.py {args}")
        print("USING MLP")
        os.system(f"python3 classifiers/mlp.py {args}")
        print("USING DECISION TREE")
        os.system(f"python3 classifiers/decision-tree.py {args}")
        print("USING RANDOM FOREST")
        os.system(f"python3 classifiers/random-forest.py {args}")
        # print("USING GRADIENT BOOSTED TREES")
        # os.system(f"python3 classifiers/gradient-boosted-forest.py {args}")
        print("USING ADABOOST")
        os.system(f"python3 classifiers/adaboost.py {args}")
