# Script to convert the combined results into a latex table

import os
import pandas as pd
from copy import deepcopy

from sklearn import metrics
from constants import VALID_DATASETS, ANOMALIE_PERCENTS, LABEL_DICT, CUT_OFF_PERCENTS, AU_METRICS, VALID_ALGORITHMS


absolute_path = os.path.dirname(__file__)

for metric in AU_METRICS:
    for dataset in VALID_DATASETS:
        path = "results/All/{}/{}-train-Combined.xlsx".format(dataset, metric)
        path = os.path.join(absolute_path, path)
        df = pd.read_excel(path)
        df.set_index("Unnamed: 0", inplace=True)

        path = "results/All/{}/{}-train-stds-Combined.xlsx".format(dataset, metric)
        path = os.path.join(absolute_path, path)
        df_std = pd.read_excel(path)
        df_std.set_index("Unnamed: 0", inplace=True)

        df = df.T
        df_std = df_std.T

        df.index = (df.index*100).astype('int').astype('str')+"%"
        df_std.index = (df_std.index*100).astype('int').astype('str')+"%"

        df = (df * 100).astype("str").apply(lambda x: x.str[:5])
        df_std = (df_std * 100).astype("str").apply(lambda x: x.str[:4])
        df = "$" + df + "\pm " + df_std + "$"
        x = df.to_csv(None).replace("\r", ",")
        print(dataset, metric)
        print(x)
        print("")
