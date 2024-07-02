import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn import metrics

from constants import VALID_ALGORITHMS, VALID_DATASETS
from run import restricted_float

absolute_path = os.path.dirname(__file__)


def display_roc(filepath, label, ax):
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 0 if x == label else 1)
    metrics.RocCurveDisplay.from_predictions(df["Label"], df["Loss"], ax=ax)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")


def display_prc_in(filepath, label, ax):
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 1 if x == label else 0)
    metrics.PrecisionRecallDisplay.from_predictions(df["Label"], 1 / df["Loss"], plot_chance_level=True, ax=ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")


def display_prc_out(filepath, label, ax):
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 0 if x == label else 1)
    metrics.PrecisionRecallDisplay.from_predictions(df["Label"], df["Loss"], plot_chance_level=True, ax=ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")


def display_hist_historun(filepath, label, save=False):
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


def display_hist(filepath, label, ax, save=False):
    df = pd.read_csv(filepath)
    df["Label"] = df["Label"].map(lambda x: 0 if x == label else 1)
    normal = df[df.Label == 0]["Loss"]
    anomalie = df[df.Label == 1]["Loss"]

    fpr, tpr, _ = metrics.roc_curve(df["Label"], df["Loss"])
    roc_auc = metrics.auc(fpr, tpr)

    bins = 20
    ax.hist(normal, alpha=0.5, bins=bins, label='Data 1', edgecolor='black')
    ax.hist(anomalie, alpha=0.5, bins=bins, label='Data 1', edgecolor='black')
    if save:
        filepath = filepath[:-4] + ".png"
        split = filepath.split("-")
        ax.title("{}/{} {} - ap:{} cop:{} - {} {}".format(split[0].split("/")[-1], split[5], split[1], split[2], split[3], split[6], str(round(roc_auc, 6))))
        ax.savefig(filepath)
        ax.clf()


def historun(dirpath, label):
    dir = os.path.join(absolute_path, dirpath)
    for file in os.listdir(dir):
        if file[-3:] == "csv" and file.split("-")[1] == str(label):
            display_hist_historun(os.path.join(dir, file), label, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.required = True

    parser_display = subparsers.add_parser('display')
    parser_display.add_argument("-epochs", "-e", help="the number of epochs used while training", type=int, default=30)
    parser_display.add_argument("-cop", help="the percentage of cut-off", type=restricted_float, default=0.5)
    parser_display.add_argument("-ap", help="The proportion of anomalies in the total data set", type=restricted_float, default=0.25)
    parser_display.add_argument("-dataset", "-d", help="The dataset to be used", choices=VALID_DATASETS, default="mnist")
    parser_display.add_argument("-algorithm", "-a", help="The algorithm to use for training", choices=VALID_ALGORITHMS, default="myCAE")
    parser_display.add_argument("-cycle", "-c", help="The cycle number", type=int, default=0)
    parser_display.add_argument("-label", "-l", help="The label used as normal data", type=int, default=0)
    parser_display.add_argument("--test", help='The test data will be displayed instead of the training data when used.', action="store_true")
    parser_display.add_argument("--old", help="Old (false) data will be used", action="store_true")

    parser_display3 = subparsers.add_parser('display3')
    parser_display3.add_argument("-epochs", "-e", help="the number of epochs used while training", type=int, default=30)
    parser_display3.add_argument("-cop", help="the percentage of cut-off", type=restricted_float, default=0.5)
    parser_display3.add_argument("-ap", help="The proportion of anomalies in the total data set", type=restricted_float, default=0.25)
    parser_display3.add_argument("-dataset", "-d", help="The dataset to be used", choices=VALID_DATASETS, default="mnist")
    parser_display3.add_argument("-algorithm", "-a", help="The algorithm to use for training", choices=VALID_ALGORITHMS, default="myCAE")
    parser_display3.add_argument("-algorithm2", "-a2", help="The algorithm to use for training", choices=VALID_ALGORITHMS, default="CAE")
    parser_display3.add_argument("-algorithm3", "-a3", help="The algorithm to use for training", choices=VALID_ALGORITHMS, default="CAEDrop")
    parser_display3.add_argument("-cycle", "-c", help="The cycle number", type=int, default=0)
    parser_display3.add_argument("-label", "-l", help="The label used as normal data", type=int, default=0)
    parser_display3.add_argument("--test", help='The test data will be displayed instead of the training data when used.', action="store_true")
    parser_display3.add_argument("--old", help="Old (false) data will be used", action="store_true")

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
        fig, axs = plt.subplots(1, 3)
        # hist = display_hist(path, label, axs[0])
        roc = display_roc(path, label, axs[0])
        prc_in = display_prc_in(path, label, axs[1])
        prc_out = display_prc_out(path, label, axs[2])

        plt.show()

    elif args.subcommand == "display3":
        algorithm = args.algorithm
        algorithm2 = args.algorithm2
        algorithm3 = args.algorithm3
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
        path2 = os.path.join(absolute_path, path_start + "{}/{}/{}/{}-{}-{}-{}-{}-{}-{}-loss.csv".format(algorithm2,
                                                                                                         dataset,
                                                                                                         cycle,
                                                                                                         dataset,
                                                                                                         label,
                                                                                                         ap,
                                                                                                         cop,
                                                                                                         epochs,
                                                                                                         algorithm2,
                                                                                                         "test" if test else "train"))
        path3 = os.path.join(absolute_path, path_start + "{}/{}/{}/{}-{}-{}-{}-{}-{}-{}-loss.csv".format(algorithm3,
                                                                                                         dataset,
                                                                                                         cycle,
                                                                                                         dataset,
                                                                                                         label,
                                                                                                         ap,
                                                                                                         cop,
                                                                                                         epochs,
                                                                                                         algorithm3,
                                                                                                         "test" if test else "train"))
        fig, axs = plt.subplots(3, 3)
        # hist = display_hist(path, label, axs[0])
        roc = display_roc(path, label, axs[0, 0])
        prc_in = display_prc_in(path, label, axs[0, 1])
        prc_out = display_prc_out(path, label, axs[0, 2])
        roc2 = display_roc(path2, label, axs[1, 0])
        prc_in2 = display_prc_in(path2, label, axs[1, 1])
        prc_out2 = display_prc_out(path2, label, axs[1, 2])
        roc2 = display_roc(path3, label, axs[2, 0])
        prc_in2 = display_prc_in(path3, label, axs[2, 1])
        prc_out2 = display_prc_out(path3, label, axs[2, 2])

        plt.show()

    elif args.subcommand == "historun":
        algorithm = args.algorithm
        dataset = args.dataset
        cycle = args.cycle
        label = args.label
        historun("results/{}/{}/{}/".format(algorithm, dataset, cycle), label)
