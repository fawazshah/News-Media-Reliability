import os
import sys


if __name__ == "__main__":

    args = " ".join(sys.argv[1:])
    print("USING SVM")
    os.system(f"python3 classifiers/svc.py {args}")
    print("USING MLP")
    os.system(f"python3 classifiers/mlp.py {args}")
    print("USING DECISION TREE")
    os.system(f"python3 classifiers/decision-tree.py {args}")
    print("USING RANDOM FOREST")
    os.system(f"python3 classifiers/random-forest.py {args}")
