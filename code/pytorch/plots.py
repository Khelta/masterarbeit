import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from constants import CUT_OFF_PERCENTS, LABEL_DICT, VALID_DATASETS
from helper import calculate_auroc_from_csv, calculate_auprin_from_csv, calculate_auprout_from_csv

absolute_path = os.path.dirname(__file__)

auroc_not_mnist = (0.91, 0.59)
auroc_not_cifar = (0.66, 0.51)
aupr_in_not_mnist = (1.01, 0.74)
aupr_in_not_cifar = (1.01, 0.74)
aupr_out_not_mnist = (0.78, 0.14)
aupr_out_not_cifar = (0.41, 0.04)
ylim_dict = {False: {"AUROC": {"mnist": auroc_not_mnist,
                               "fashion": auroc_not_mnist,
                               "svhn": auroc_not_cifar,
                               "cifar10": auroc_not_cifar,
                               "cifar100": auroc_not_cifar},
                     "AUPR_IN": {"mnist": aupr_in_not_mnist,
                                 "fashion": aupr_in_not_mnist,
                                 "svhn": aupr_in_not_cifar,
                                 "cifar10": aupr_in_not_cifar,
                                 "cifar100": aupr_in_not_cifar},
                     "AUPR_OUT": {"mnist": aupr_out_not_mnist,
                                  "fashion": aupr_out_not_mnist,
                                  "svhn": aupr_out_not_cifar,
                                  "cifar10": aupr_out_not_cifar,
                                  "cifar100": aupr_out_not_cifar}},
             True: {"AUROC": {"mnist": auroc_not_mnist,
                              "fashion": auroc_not_mnist,
                              "svhn": auroc_not_cifar,
                              "cifar10": auroc_not_cifar,
                              "cifar100": auroc_not_cifar},
                    "AUPR_IN": {"mnist": aupr_in_not_mnist,
                                "fashion": aupr_in_not_mnist,
                                "svhn": aupr_in_not_cifar,
                                "cifar10": aupr_in_not_cifar,
                                "cifar100": aupr_in_not_cifar},
                    "AUPR_OUT": {"mnist": aupr_out_not_mnist,
                                 "fashion": aupr_out_not_mnist,
                                 "svhn": aupr_out_not_cifar,
                                 "cifar10": aupr_out_not_cifar,
                                 "cifar100": aupr_out_not_cifar}}}


def plot_single(dataset, ax, metric="AUROC", all_models=True):
    path = os.path.join(absolute_path, "results/All/" + dataset + "/{}-train-Combined.xlsx".format(metric))
    df = pd.read_excel(path)
    df = df.set_index("Unnamed: 0")

    if all_models is not True:
        best = df.loc["0.5":"0.9"].T.mean().idxmax()
        worst = df.loc["0.5":"0.9"].T.mean().idxmin()

        cops = ["0.5", "0.6", "0.7", "0.8", "0.9"]
        for cop in cops:
            if cop != best and cop != worst:
                df = df.drop(cop)

        best_name = "Best: " + best
        worst_name = "Worst: " + worst
        df = df.rename({best: best_name, worst: worst_name})

    df = df.T

    if "Best" in df.columns[1]:
        df = df.reindex(columns=[df.columns[1], df.columns[0]] + list(df.columns[2:]))

    ax.set_title(dataset)
    if all_models is not True:
        ax.plot(df[best_name], "^-", color="#318e2d")
        ax.plot(df[worst_name], "v-", color="#c10001")
    else:
        for x in CUT_OFF_PERCENTS:
            ax.plot(df[str(x)], "o-")
    ax.plot(df["CAE"], "o--", color="#797979")
    ax.plot(df["CAEDrop"], "X--", color="tab:purple")
    ax.plot(df["DRAE"], "P--", color="tab:blue")
    ax.plot(df["DeepSVDD"], "X--", color="tab:orange")
    ax.set_xticks(np.arange(0.05, 0.26, step=0.05))
    ax.set_xticks(np.arange(0.05, 0.26, step=0.01), minor=True)

    ax.legend(df.columns)
    ax.grid(which="major", alpha=0.5)
    ax.grid(which="minor", alpha=0.2)
    ax.tick_params(which='minor', bottom=False)

    ax.set_xlabel("Anomalierate")
    ax.set_ylabel(metric)

    ymax, ymin = ylim_dict[all_models][metric][dataset]

    ax.set_ylim([ymin, ymax])

    ax.set_yticks(np.arange(ymin, ymax, step=0.05))
    ax.set_yticks(np.arange(ymin, ymax, step=0.01), minor=True)


def plot_all(metric="AUROC", all=True, save=False, as_one=False):
    if not as_one:
        plt.close('all')

        ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 1), colspan=2)
        ax2 = plt.subplot2grid((2, 6), (0, 3), colspan=2)
        ax3 = plt.subplot2grid((2, 6), (1, 0), colspan=2)
        ax4 = plt.subplot2grid((2, 6), (1, 2), colspan=2)
        ax5 = plt.subplot2grid((2, 6), (1, 4), colspan=2)

        plt.subplots_adjust(hspace=0.5, wspace=1)

        plot_single("mnist", ax1, metric, all)
        plot_single("fashion", ax2, metric, all)
        plot_single("svhn", ax3, metric, all)
        plot_single("cifar10", ax4, metric, all)
        plot_single("cifar100", ax5, metric, all)

        if save:
            path = os.path.join(absolute_path, "plots/{}-all.png".format(metric))
            plt.savefig(path, dpi=200)
        else:
            plt.show()
    else:
        for dataset in ["mnist", "fashion", "svhn", "cifar10", "cifar100"]:
            plt.close('all')
            ax1 = plt.subplot()
            plot_single(dataset, ax1, metric, all)
            path = os.path.join(absolute_path, "plots/{}-{}.png".format(metric, dataset))
            if save:
                plt.savefig(path, dpi=200)
            else:
                plt.show()


def plot_historun_across_epochs(filepath, metric="AUROC", starting_point=3, save=True):
    df = pd.read_csv(filepath)
    fig, ax1 = plt.subplots()

    color = "tab:blue"
    i = starting_point
    ax1.set_ylabel('Trainings-Loss', color=color)
    ax1.plot([x + 1 for x in range(i, len(df[i:]) + i)], df["Mean Loss"][i:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel("Epoche")

    ax2 = ax1.twinx()

    color = "tab:orange"

    ax2.plot([x + 1 for x in range(0, len(df))], df[metric], color=color)
    ax2.set_ylabel('AUROC', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 1])
    """color = "tab:green"
    ax2.plot([x + 1 for x in range(0, len(df))], df["AUPR-IN"], color=color)
    color = "tab:purple"
    ax2.plot([x + 1 for x in range(0, len(df))], df["AUPR-OUT"], color=color)"""
    plt.axvline(30, linestyle="--", color="black")

    algo = filepath.split("-")[5]
    dataset = filepath.split("-")[0].split("/")[-1]
    label = filepath.split("-")[1]
    plt.title("{} - {} - Label: {}".format(dataset, algo, label))
    path = os.path.join(absolute_path, "plots/epochs-{}-{}-{}.png".format(metric, dataset, algo))
    if save:
        plt.savefig(path, dpi=200)
    else:
        plt.show()


def calculate_values_for_boxplots(metric, dataset, algorithm, ap, cop, epoch):
    values = []
    for cycle in range(0, 5):
        for label in LABEL_DICT[dataset]:
            if algorithm == "myCAE":
                path = os.path.join(absolute_path, "results/myCAE/{}/{}/{}-{}-{}-{}-{}-myCAE-train-loss.csv".format(dataset, cycle, dataset, label, ap, cop, epoch))
            else:
                cop = cycle * 0.1 + 0.5
                path = os.path.join(absolute_path, "results/{}/{}/0/{}-{}-{}-{}-{}-{}-train-loss.csv".format(algorithm, dataset, dataset, label, ap, cop, epoch, algorithm))

            if metric == "AUROC":
                values.append(calculate_auroc_from_csv(path, label))
            elif metric == "AUPR_IN":
                values.append(calculate_auprin_from_csv(path, label))
            elif metric == "AUPR_OUT":
                values.append(calculate_auprout_from_csv(path, label))
    return values


def boxplots(dataset="mnist", metric="AUROC", save=True):

    epoch = 30
    ap = 0.25
    algorithms = [("myCAE", 0.5), ("myCAE", 0.6), ("myCAE", 0.7), ("myCAE", 0.8), ("myCAE", 0.9), ("CAE", 0.5), ("CAEDrop", 0.5), ("DRAE", 0.5), ("DeepSVDD", 0.5)]

    values = []
    ticklabels = []
    for dataset in VALID_DATASETS:
        for algorithm, cop in algorithms:
            values.append(calculate_values_for_boxplots(metric, dataset, algorithm, ap, cop, epoch))
            if algorithm == "myCAE":
                ticklabels.append(dataset + " " * (9 - len(dataset)) + "-" + " " * 6 + str(cop))
            else:
                ticklabels.append(dataset + " " * (9 - len(dataset)) + "-" + " " * (9 - len(algorithm)) + algorithm)
        values.append([])
        ticklabels.append("")
        values.append([])
        ticklabels.append("")

    values = values[:-2]
    ticklabels = ticklabels[:-2]

    plt.rcParams['font.family'] = 'monospace'
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.boxplot(values, vert=False)
    ax.set_yticklabels(ticklabels)
    ax.vlines(0.5, 1, len(values), color="black", linestyles="--")
    fig.tight_layout()

    if save is True:
        plt.savefig(os.path.join(absolute_path, "plots/boxplots-{}.png".format(metric)), dpi=400)
    plt.show()


if __name__ == "__main__":
    plot_all("AUPR_OUT", False, save=True, as_one=True)

    """for algo in ["myCAE", "CAE", "CAEDrop", "DRAE"]:
        for dataset in VALID_DATASETS: 
            plot_historun_across_epochs("C:/Users/jonas/Desktop/programming/masterarbeit/code/pytorch/results/historun/{}/{}/0/{}-0-0.25-0.5-250-{}-e0-meanLossesPerEpoch.csv".format(algo, dataset, dataset, algo), "AUROC")"""
    #boxplots("cifar100")
