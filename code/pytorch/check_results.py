import os

from constants import VALID_ALGORITHMS, VALID_DATASETS, ANOMALIE_PERCENTS, LABEL_DICT, CUT_OFF_PERCENTS


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


absolute_path = os.path.dirname(__file__)
compare_path = os.path.join(absolute_path, "./results")


def check_results_all(epoch=30):
    for algorithm in VALID_ALGORITHMS:
        for dataset in VALID_DATASETS:
            print(algorithm, dataset)
            for ap in ANOMALIE_PERCENTS:
                error = ""
                b = False
                check = True
                if algorithm in ["myCAE", "myCAEsoft"]:
                    for cycle in range(0, 5):
                        for cop in CUT_OFF_PERCENTS:
                            if b:
                                break
                            for label in LABEL_DICT[dataset]:
                                path = os.path.join(compare_path, algorithm + "/" + dataset + "/")
                                path = os.path.join(path, "{}/{}-{}-{}-{}-{}-{}-train-loss.csv".format(cycle,
                                                                                                       dataset,
                                                                                                       label,
                                                                                                       ap,
                                                                                                       cop,
                                                                                                       epoch,
                                                                                                       algorithm))
                                if not os.path.isfile(path):
                                    error = "Cycle {}, Label {}, cop {}".format(cycle, label, cop)
                                    b = True
                                    check = False
                                    break
                else:
                    cycle = 0
                    for cop in CUT_OFF_PERCENTS:
                        if b:
                            break
                        for label in LABEL_DICT[dataset]:
                            path = os.path.join(compare_path, algorithm + "/" + dataset + "/")
                            path = os.path.join(path, "{}/{}-{}-{}-{}-{}-{}-train-loss.csv".format(cycle,
                                                                                                   dataset,
                                                                                                   label,
                                                                                                   ap,
                                                                                                   cop,
                                                                                                   epoch,
                                                                                                   algorithm))
                            if not os.path.isfile(path):
                                error = "Label {}, cop {}".format(label, cop)
                                b = True
                                check = False
                                break
                if len(error) == 0:
                    print("\t", bcolors.OKGREEN, ap, check, bcolors.ENDC)
                elif error == "Cycle 0, Label 0, cop 0.5" or error == "Label 0, cop 0.5":
                    print("\t", bcolors.FAIL, ap, check, error, bcolors.ENDC)
                else:
                    print("\t", bcolors.WARNING, ap, check, error, bcolors.ENDC)


if __name__ == "__main__":
    check_results_all()
