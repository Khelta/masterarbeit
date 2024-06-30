import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from constants import VALID_DATASETS, CUT_OFF_PERCENTS

absolute_path = os.path.dirname(__file__)


def plot_all(metric="auroc", all=True):
    for dataset in VALID_DATASETS:
        path = os.path.join(absolute_path, "results/All/" + dataset + "/{}-train-Combined.xlsx".format(metric))
        df = pd.read_excel(path)
        df = df.set_index("Unnamed: 0")

        if all is not True:
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

        plt.title(dataset)
        if all is not True:
            plt.plot(df[best_name], "^-")
            plt.plot(df[worst_name], "v-")
        else:
            for x in CUT_OFF_PERCENTS:
                plt.plot(df[str(x)], "o-")
        plt.plot(df["CAE"], "o--")
        plt.plot(df["DRAE"], "o--")
        plt.plot(df["DeepSVDD"], "o--")
        plt.xticks(np.arange(0.05, 0.26, step=0.05))
        plt.xticks(np.arange(0.05, 0.26, step=0.01), minor=True)

        plt.legend(df.columns)
        plt.grid(which="major", alpha=0.5)
        plt.grid(which="minor", alpha=0.2)
        plt.tick_params(which='minor', bottom=False)

        plt.xlabel("Anomalierate")
        plt.ylabel("AUROC")

        ax = plt.gca()
        """ymin = 0.60
        ymax = 0.91

        if dataset in ["cifar10", "cifar100", "svhn"]:
            ymax = 0.65
            ymin = 0.5
        ax.set_ylim([ymin, ymax])

        plt.yticks(np.arange(ymin, ymax, step=0.05))
        plt.yticks(np.arange(ymin, ymax, step=0.01), minor=True)"""

        plt.show()


if __name__ == "__main__":
    plot_all("aupr_OUT")
