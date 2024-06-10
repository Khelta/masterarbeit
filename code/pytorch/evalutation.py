import os
import pandas as pd

from sklearn import metrics
from constants import VALID_DATASETS, ANOMALIE_PERCENTS, LABEL_DICT, CUT_OFF_PERCENTS, AU_METRICS, VALID_ALGORITHMS

absolute_path = os.path.dirname(__file__)


def create_df(results, ismycae):
    df = pd.DataFrame()
    for label, value in results.items():
        train_loss_percent = label.split("-")[-1]
        actual_label = label.split("-")[0]
        df.at[actual_label, train_loss_percent] = value
    if not ismycae:
        mean = df.mean(axis=1)
        std = df.std(axis=1).mean()
        df["Loss"] = mean
        df.loc["avg mean"] = df["Loss"].mean()
        df.loc["avg std"] = std
        df = df[["Loss"]]
    return df


def merge(paths):
    dfs = [pd.read_csv(path) for path in paths]
    for i in range(1, len(dfs)):
        dfs[i] = dfs[i].drop(columns=["Unnamed: 0"])
    result = pd.concat(dfs, axis=1)
    result.set_index("Unnamed: 0", inplace=True)
    stds = result.T.groupby(by=result.columns).std().T
    result = result.T.groupby(by=result.columns).mean().T
    result.loc["avg mean"] = result.mean()
    result.loc["avg std"] = stds.mean()
    return result


def collect_single(file_prefix, ap, cop, ismycae, has_test=False):
    ap = str(ap)
    results_train = {metric: {ap: {}} for metric in AU_METRICS}
    if has_test:
        results_test = {metric: {ap: {}} for metric in AU_METRICS}

    file_path = os.path.join(absolute_path, "./results/" + file_prefix)
    files = os.listdir(file_path)
    files = [file for file in files if file.split("-")[2] == ap]
    for filename in files:
        f = os.path.join(file_path, filename)

        if os.path.isfile(f) and f[-4:] == ".csv":
            data_path = os.path.join(file_path, filename)
            data = pd.read_csv(data_path)
            label, _, cop = filename.split("-")[1:4]
            data["Label"] = data["Label"].map(lambda x: 0 if str(x) == label else 1)
            fpr, tpr, _ = metrics.roc_curve(data["Label"], data["Loss"])
            auroc = metrics.auc(fpr, tpr)

            p, r, _ = metrics.precision_recall_curve(data["Label"], data["Loss"])
            aupr_IN = metrics.auc(r, p)

            data["Label"] = data["Label"].replace({0: 1, 1: 0})
            p, r, _ = metrics.precision_recall_curve(data["Label"], data["Loss"])
            aupr_OUT = metrics.auc(r, p)

            key = str(label) + "-" + cop
            if filename[-14:] == "train-loss.csv":
                results_train["auroc"][ap][key] = auroc
                results_train["aupr_IN"][ap][key] = aupr_IN
                results_train["aupr_OUT"][ap][key] = aupr_OUT

            elif filename[-13:] == "test-loss.csv":
                results_test["auroc"][ap][key] = auroc
                results_test["aupr_IN"][ap][key] = aupr_IN
                results_test["aupr_OUT"][ap][key] = aupr_OUT

    for metric in AU_METRICS:
        path = "./results/" + file_prefix[:-2] + metric + ("-" + file_prefix[-2] if ismycae else "") + "-" + ap + "-"
        train_path = os.path.join(absolute_path, path + "train.csv")
        create_df(results_train[metric][ap], ismycae).to_csv(train_path)
        if has_test:
            test_path = os.path.join(absolute_path, path + "test.csv")
            create_df(results_test[metric][ap], ismycae).to_csv(test_path)


def collect_all_results():
    for algorithm in ["myCAE", "CAE"]:
        for dataset in VALID_DATASETS:
            has_test = algorithm in ["myCAE", "CAE", "DRAE"]
            path = os.path.join(absolute_path, "results/" + algorithm + "/" + dataset + "/")
            print(algorithm, dataset)
            for i in range(5):
                for ap in ANOMALIE_PERCENTS:
                    cycle = 0 if algorithm != "myCAE" else i
                    files = os.listdir(path + "{}/".format(cycle))
                    files = [filename for filename in files if filename.split("-")[2] == str(ap)]
                    check = len(LABEL_DICT[dataset]) * len(CUT_OFF_PERCENTS) * 2 if has_test else 1

                    check_path = os.path.join(path, "{}-{}-train.csv".format(AU_METRICS[0], ap))
                    if os.path.isfile(check_path):
                        print("\tCycle {}|{} already created.".format(i, ap))
                        continue

                    if check != len(files):
                        print("Check not passed for {} {} ap: {} - {}/{}".format(algorithm, dataset, ap, len(files), check))
                        continue

                    cop = cycle * 0.1 + 0.5
                    collect_single(algorithm + "/" + dataset + "/" + str(cycle) + "/", ap, cop, algorithm == "myCAE", has_test)

            if algorithm == "myCAE":
                for ap in ANOMALIE_PERCENTS:
                    for metric in AU_METRICS:
                        for train in ["train.csv", "test.csv"]:
                            path = os.path.join(absolute_path, "results/" + algorithm + "/" + dataset + "/")
                            files = os.listdir(path)
                            files = [file for file in files if os.path.isfile(path + file)]
                            files = [file for file in files if file.split("-")[2] == str(ap)
                                     and file.split("-")[0] == metric
                                     and file.split("-")[-1] == train]
                            files = [os.path.join(path, file) for file in files]
                            if len(files) == 0:
                                continue
                            merge(files).to_csv(os.path.join(absolute_path, "results/" + algorithm + "/" + dataset + "/" + "-".join([metric, str(ap), train])))
                            for file in files:
                                os.remove(file)

    # Collect all algos in one file
    for dataset in VALID_DATASETS:
        for metric in AU_METRICS:
            for ap in ANOMALIE_PERCENTS:
                for t in ["train", "test"]:
                    result = pd.DataFrame()
                    for algorithm in VALID_ALGORITHMS:
                        path = "./results/{}/{}/{}-{}-{}.csv".format(algorithm, dataset, metric, ap, t)
                        path = os.path.join(absolute_path, path)
                        if os.path.isfile(path):
                            df = pd.read_csv(path)
                            df = df.set_index("Unnamed: 0")
                            if algorithm != "myCAE":
                                df = df.rename(columns={"Loss": algorithm})  
                            result = pd.concat([result, df], axis=1)
                    directory = "./results/All/{}/".format(dataset)
                    directory = os.path.join(absolute_path, directory)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    path = os.path.join(directory, "{}-{}-{}.xlsx".format(metric, ap, t))
                    result.to_excel(path)

    for dataset in VALID_DATASETS:
        for t in ["train", "test"]:
            for metric in AU_METRICS:
                result = pd.DataFrame()
                for ap in ANOMALIE_PERCENTS:
                    path = "results/All/{}/{}-{}-{}.xlsx".format(dataset, metric, ap, t)
                    path = os.path.join(absolute_path, path)
                    df = pd.read_excel(path)
                    if len(df) == 0:
                        continue
                    df = df.T
                    df.columns = df.loc["Unnamed: 0"]
                    df = df.drop("Unnamed: 0")
                    df = df.rename(columns={"avg mean": ap})
                    df = df[[ap]]
                    if len(result) == 0:
                        result = pd.concat([result, df])
                    else:
                        result = pd.merge(result, df, left_index=True, right_index=True, how="outer")
                columns = list(result.columns)
                columns.sort()
                result = result[columns]
                path = "results/All/{}/{}-{}-Combined.xlsx".format(dataset, metric, t)
                path = os.path.join(absolute_path, path)
                result.to_excel(path)


if __name__ == "__main__":
    collect_all_results()
