import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

absolute_path = os.path.dirname(__file__)

def display_roc(filepath, label):
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 0 if x == label else 1)
    fpr, tpr, _ = metrics.roc_curve(df["Label"], df["Loss"])
    roc_auc = metrics.auc(fpr, tpr)
    
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    plt.show()

#path = os.path.join(absolute_path, "results/10/4-0,25-0,5-30train-loss.csv")
path = os.path.join(absolute_path, "results/train-loss.csv")
display_roc(path, 1)
