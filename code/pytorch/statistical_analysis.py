import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn import metrics
from constants import VALID_DATASETS, AU_METRICS, CUT_OFF_PERCENTS, VALID_ALGORITHMS, DASASETS_NUM_LABELS

absolute_path = os.path.dirname(__file__)


def count(files, results, single_label=True, name=None):
    for cop in CUT_OFF_PERCENTS:
        cop = str(cop)
        tempAuroc = []
        tempAuprin = []
        tempAuprout = []

        for f in [file for file in files if file.split("-")[3] == cop]:

            if os.path.isfile(f) and f[-4:] == ".csv":
                data = pd.read_csv(f)
                label, _, cop = f.split("-")[1:4]
                data["Label"] = data["Label"].map(lambda x: 0 if str(x) == label else 1)
                auroc = metrics.roc_auc_score(data["Label"], data["Loss"])
                tempAuroc.append(auroc)

                x = data["Label"].replace({0: 1, 1: 0})
                p, r, _ = metrics.precision_recall_curve(x, 1 / data["Loss"])
                aupr_IN = metrics.auc(r, p)
                tempAuprin.append(aupr_IN)

                p, r, _ = metrics.precision_recall_curve(data["Label"], data["Loss"])
                aupr_OUT = metrics.auc(r, p)
                tempAuprout.append(aupr_OUT)
        if single_label is False:
            if name is not None:
                results["AUROC"][name].append(np.mean(tempAuroc))
                results["AUPR_IN"][name].append(np.mean(tempAuprin))
                results["AUPR_OUT"][name].append(np.mean(tempAuprout))
            else:
                results["AUROC"][cop].append(np.mean(tempAuroc))
                results["AUPR_IN"][cop].append(np.mean(tempAuprin))
                results["AUPR_OUT"][cop].append(np.mean(tempAuprout))
        else:
            if name is not None:
                results["AUROC"][name] += tempAuroc
                results["AUPR_IN"][name] += tempAuprin
                results["AUPR_OUT"][name] += tempAuprout
            else:
                results["AUROC"][cop] += tempAuroc
                results["AUPR_IN"][cop] += tempAuprin
                results["AUPR_OUT"][cop] += tempAuprout
    return results


def evaluation(single_label=True, metric="AUROC"):
    for dataset in VALID_DATASETS:
        for ap in [0.25, 0.2, 0.15, 0.1, 0.05]:
            print(dataset, ap)
            results = {"AUROC": {"0.5": [],
                                 "0.6": [],
                                 "0.7": [],
                                 "0.8": [],
                                 "0.9": [],
                                 "CAE": [],
                                 "CAEDrop": [],
                                 "DRAE": [],
                                 "DeepSVDD": []},
                       "AUPR_IN": {"0.5": [],
                                   "0.6": [],
                                   "0.7": [],
                                   "0.8": [],
                                   "0.9": [],
                                   "CAE": [],
                                   "CAEDrop": [],
                                   "DRAE": [],
                                   "DeepSVDD": []},
                       "AUPR_OUT": {"0.5": [],
                                    "0.6": [],
                                    "0.7": [],
                                    "0.8": [],
                                    "0.9": [],
                                    "CAE": [],
                                    "CAEDrop": [],
                                    "DRAE": [],
                                    "DeepSVDD": []}}
            for algorithm in VALID_ALGORITHMS:
                algorithm
                if algorithm == "myCAE":
                    for cycle in range(5):
                        path = os.path.join(absolute_path, "results/" + algorithm + "/" + dataset + "/" + str(cycle) + "/")
                        files = os.listdir(path)
                        files = [file for file in files if os.path.isfile(path + file)]
                        files = [file for file in files if file.split("-")[2] == str(ap)
                                    and file.split("-")[-2] == "train"]
                        files = [os.path.join(path, file) for file in files]

                        results = count(files, results, single_label)    
                else:
                    cycle = 0
                    path = os.path.join(absolute_path, "results/" + algorithm + "/" + dataset + "/" + str(cycle) + "/")
                    files = os.listdir(path)
                    files = [file for file in files if os.path.isfile(path + file)]
                    files = [file for file in files if file.split("-")[2] == str(ap)
                                and file.split("-")[-2] == "train"]
                    files = [os.path.join(path, file) for file in files]
                    results = count(files, results, single_label, name=algorithm)

            algos = ["0.5", "0.6", "0.7", "0.8", "0.9", "CAE", "CAEDrop", "DRAE", "DeepSVDD"]
            p_values = []
            for i in range(0, len(algos)):
                for j in range(i + 1, len(algos)):
                    a = algos[i]
                    b = algos[j]
                    test = stats.wilcoxon(results[metric][a], results[metric][b], alternative="greater", method="approx")
                    r = test.zstatistic / np.sqrt(DASASETS_NUM_LABELS[dataset] * 5)
                    p_values.append((test.pvalue, r, a, b))
            p_values.sort(key=lambda x: x[0])

            holm_results = []
            data_dict = {}
            m = len(p_values)
            alpha = 0.05
            for i in range(0, len(p_values)):
                p_value, r, a, b = p_values[i]
                test = p_value <= alpha / (m - i)
                holm_results.append((p_value, a, b, test))
                if a not in data_dict:
                    data_dict[a] = {}
                data_dict[a][b] = (p_value, r, test)

            heatmap_data = []
            heatmap_text = []
            for algo in algos:
                data = []
                text = []
                for compAlgo in algos:
                    if algo == compAlgo:
                        data.append(0.5)
                        text.append("")
                    else:
                        if algo not in data_dict:
                            data.append(0.5)
                            text.append("")
                        elif compAlgo not in data_dict[algo]:
                            data.append(0.5)
                            text.append("")
                        else:
                            data.append(1 if data_dict[algo][compAlgo][2] else 0)
                            text.append("{:.2e}".format(data_dict[algo][compAlgo][0]) + "\n" + "{:.2f}".format(data_dict[algo][compAlgo][1]))
                heatmap_data.append(data)
                heatmap_text.append(text)

            heatmap_data = heatmap_data[:-1]
            heatmap_text = heatmap_text[:-1]

            for i in range(0, len(heatmap_data)):
                heatmap_data[i] = heatmap_data[i][1:]
                heatmap_text[i] = heatmap_text[i][1:]

            a = np.array([0.93, 0.38, 0.34, 1])
            b = np.array([0.9, 0.9, 0.9, 1])
            c = np.array([0.38, 0.82, 0.58, 1])
            cmap = ListedColormap([a, b, c])

            if sum([i.count(1) for i in heatmap_data]) == 0:
                cmap = ListedColormap([a, b])
            elif sum([i.count(0) for i in heatmap_data]) == 0:
                cmap = ListedColormap([b, c])

            plt.clf()
            ax = sns.heatmap(heatmap_data, annot=heatmap_text, cbar=False, cmap=cmap, fmt="", linewidths=0.1)
            plt.title(dataset + " - ap: " + str(ap))
            ax.set_xticklabels(algos[1:])
            ax.set_yticklabels(algos[:-1])
            ax.figure.tight_layout()

            path = os.path.join(absolute_path, "plots/stats-{}-{}-{}.png".format(metric, dataset, ap))
            plt.savefig(path, dpi=200)


if __name__ == "__main__":
    for metric in AU_METRICS:
        evaluation(single_label=True, metric=metric.upper())
