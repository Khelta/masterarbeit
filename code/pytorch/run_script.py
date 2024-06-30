import os
from constants import VALID_ALGORITHMS, VALID_DATASETS, ANOMALIE_PERCENTS

abs_path = os.path.dirname(__file__)

pythonpath = os.path.join(abs_path, "../../.venv/Scripts/python.exe")
scriptpath = os.path.join(abs_path, "run.py")

for algorithm in VALID_ALGORITHMS:
    for dataset in VALID_DATASETS:
        for ap in ANOMALIE_PERCENTS:
            os.system(pythonpath + " " + scriptpath + " -a {} -d {} -ap {}".format(algorithm, dataset, ap))
