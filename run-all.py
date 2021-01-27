import os
import sys


if __name__ == "__main__":

    args = " ".join(sys.argv[1:])
    print("USING SVM")
    os.system(f"python3 train-svc.py {args}")
    print("USING MLP")
    os.system(f"python3 train-mlp.py {args}")
