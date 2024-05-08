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
    
def display_hist(filepath, label):
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 0 if x == label else 1)
    normal = df[df.Label == 0]["Loss"]
    anomalie = df[df.Label == 1]["Loss"]
    
    bins = 20
    plt.hist(normal, alpha=0.5, bins=bins, label='Data 1', edgecolor='black')
    plt.hist(anomalie, alpha=0.5, bins=bins, label='Data 1', edgecolor='black')
    plt.show()

#path = os.path.join(absolute_path, "results/250-Epochs/0.25/102/2-0,25-0,5-250train-loss.csv")
path = os.path.join(absolute_path, "results/100/3-0,25-0,5-30test-loss.csv")
#path = os.path.join(absolute_path, "results/train-loss.csv")
display_hist(path, 3)
display_roc(path, 3)
