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
    
def display_hist(filepath, label, save=False):
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 0 if x == label else 1)
    normal = df[df.Label == 0]["Loss"]
    anomalie = df[df.Label == 1]["Loss"]

    fpr, tpr, _ = metrics.roc_curve(df["Label"], df["Loss"])
    roc_auc = metrics.auc(fpr, tpr)
    
    bins = 20
    plt.hist(normal, alpha=0.5, bins=bins, label='Data 1', edgecolor='black')
    plt.hist(anomalie, alpha=0.5, bins=bins, label='Data 1', edgecolor='black')
    if save:
        filepath = filepath[:-4] + ".png"
        split = filepath.split("-")
        plt.title("{}/{} {} - ap:{} cop:{} - {} {}".format(split[0].split("/")[-1], split[5], split[1], split[2], split[3], split[6], str(round(roc_auc, 6))))
        plt.savefig(filepath)
        plt.clf()
    else:
        plt.show()

def historun(dirpath, label):
    dir = os.path.join(absolute_path, dirpath)
    for file in os.listdir(dir):
        if file[-3:] == "csv" and file.split("-")[1] == str(label):
            display_hist(os.path.join(dir, file), label, True)
            

algorithm = "DRAE"
dataset = "mnist"
cycle = 0
label = 3
ap = 0.05
cop = 0.5
epochs = 250
train = True


path = os.path.join(absolute_path, "results/{}/{}/{}/{}-{}-{}-{}-{}-{}-{}-loss.csv".format(algorithm,
                                                                                                        dataset,
                                                                                                        cycle,
                                                                                                        dataset,
                                                                                                        label,
                                                                                                        ap,
                                                                                                        cop,
                                                                                                        epochs,
                                                                                                        algorithm,
                                                                                                        "train" if train else "test"))
display_hist(path, label)
display_roc(path, label)

# historun("results/myCAE/svhn/0/", 0)
