import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn import metrics

from constants import VALID_ALGORITHMS, VALID_DATASETS
from run import restricted_float

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
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser_display = subparsers.add_parser('display')
    parser_display.add_argument("-epochs","-e", help="the number of epochs used while training", type=int, default=30)
    parser_display.add_argument("-cop", help="the percentage of cut-off", type=restricted_float, default=0.5)
    parser_display.add_argument("-ap", help="The proportion of anomalies in the total data set", type=restricted_float, default=0.25)
    parser_display.add_argument("-dataset", "-d", help="The dataset to be used", choices=VALID_DATASETS, default="mnist")
    parser_display.add_argument("-algorithm", "-a", help="The algorithm to use for training", choices=VALID_ALGORITHMS, default="myCAE")
    parser_display.add_argument("-cycle", "-c", help="The cycle number", type=int, default=0)
    parser_display.add_argument("-label", "-l", help="The label used as normal data", type=int, default=0)
    parser_display.add_argument("--test", help='The test data will be displayed instead of the training data when used.', action="store_true")
    parser_display.add_argument("--old",  help="Old (false) data will be used", action="store_true")

    parser_historun = subparsers.add_parser('historun')
    parser_historun.add_argument("-algorithm", "-a", help="The algorithm to use for training", choices=VALID_ALGORITHMS, default="myCAE")
    parser_historun.add_argument("-cycle", "-c", help="The cycle number", type=int, default=0)
    parser_historun.add_argument("-dataset", "-d", help="The dataset to be used", choices=VALID_DATASETS, default="mnist")
    parser_historun.add_argument("-label", "-l", help="The label used as normal data", type=int, default=0)
   
    args = parser.parse_args()
    print(args)
    if args.subcommand == "display":
        algorithm = args.algorithm
        dataset = args.dataset
        cycle = args.cycle
        label = args.label
        ap = args.ap
        cop = args.cop
        epochs = args.epochs
        test = args.test
        old = args.old

        path_start = "old/ap_cop_verwechslung/results/" if old else "results/"


        path = os.path.join(absolute_path, path_start + "{}/{}/{}/{}-{}-{}-{}-{}-{}-{}-loss.csv".format(algorithm,
                                                                                                                dataset,
                                                                                                                cycle,
                                                                                                                dataset,
                                                                                                                label,
                                                                                                                ap,
                                                                                                                cop,
                                                                                                                epochs,
                                                                                                                algorithm,
                                                                                                                "test" if test else "train"))
        display_hist(path, label)
        display_roc(path, label)
    elif args.subcommand == "historun":
        algorithm = args.algorithm
        dataset = args.dataset
        cycle = args.cycle
        label = args.label
        historun("results/{}/{}/{}/".format(algorithm, dataset, cycle), label)