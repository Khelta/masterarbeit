import pandas as pd
from sklearn import metrics


def calculate_auroc_from_csv(filepath, label):
    label = str(label)
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 0 if str(x) == label else 1)

    fpr, tpr, _ = metrics.roc_curve(df["Label"], df["Loss"])
    return metrics.auc(fpr, tpr)


def calculate_auprin_from_csv(filepath, label):
    label = str(label)
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 0 if str(x) == label else 1)
    x = df["Label"].replace({0: 1, 1: 0})
    p, r, _ = metrics.precision_recall_curve(x, 1 / df["Loss"])
    return metrics.auc(r, p)


def calculate_auprout_from_csv(filepath, label):
    label = str(label)
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 0 if str(x) == label else 1)
    p, r, _ = metrics.precision_recall_curve(df["Label"], df["Loss"])
    return metrics.auc(r, p)
