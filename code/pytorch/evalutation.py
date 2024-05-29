import os
import pandas as pd
import numpy as np

from sklearn import metrics

from constants import VALID_ALGORITHMS, VALID_DATASETS

absolute_path = os.path.dirname(__file__)

def collect_normal(file_prefix, file_path, result_suffix, column_name):
    results_train = {}
    results_test = {}
    
    file_path = os.path.join(absolute_path, "./results/" + file_prefix + file_path)
    for filename in os.listdir(file_path):
        f = os.path.join(file_path, filename)

        if os.path.isfile(f):
            data_path = os.path.join(file_path, filename)
            data = pd.read_csv(data_path)
            label = filename.split("-")[0]
            data["Label"] = data["Label"].map(lambda x: 0 if str(x) == label else 1)
            fpr, tpr, _ = metrics.roc_curve(data["Label"], data["Loss"])
            roc_auc = metrics.auc(fpr, tpr)
            
            if filename[-14:] == "train-loss.csv":
                if label not in results_train:
                    results_train[label] = [roc_auc]
                else:
                    results_train[label].append(roc_auc)
            elif filename[-13:] == "test-loss.csv":
                if label not in results_test:
                    results_test[label] = [roc_auc]
                else:
                    results_test[label].append(roc_auc)
    

    def create_df(results):
        indices = []
        column_values = []
        stds = []
        for label, values in results.items():
            indices.append(label)
            column_values.append(np.mean(results[label]))
            stds.append(np.std(results[label]))

        indices += ["mean", "mean std"]
        column_values += [np.mean(column_values), np.mean(stds)]
        result = pd.DataFrame(column_values, index=indices, columns=[column_name]) 
        return result

    train_path = os.path.join(absolute_path, "./results/"+file_prefix+"train-"+str(result_suffix)+".csv")
    test_path = os.path.join(absolute_path, "./results/"+file_prefix+"test-"+str(result_suffix)+".csv")
    df_train = create_df(results_train).to_csv(train_path)
    df_test = create_df(results_test).to_csv(test_path)
    
def collect_single_my(file_prefix, file_path, result_suffix):
    results_train = {}
    results_test = {}
    
    file_path = os.path.join(absolute_path, "./results/" + file_prefix + file_path)
    for filename in os.listdir(file_path):
        f = os.path.join(file_path, filename)

        if os.path.isfile(f) and f[-4:] == ".csv":
            data_path = os.path.join(file_path, filename)
            data = pd.read_csv(data_path)
            label = filename.split("-")[0]
            train_loss_percent = filename.split("-")[2]
            data["Label"] = data["Label"].map(lambda x: 0 if str(x) == label else 1)
            fpr, tpr, _ = metrics.roc_curve(data["Label"], data["Loss"])
            roc_auc = metrics.auc(fpr, tpr)
            
            key = label + "-" + train_loss_percent
            if filename[-14:] == "train-loss.csv":
                    results_train[key] = roc_auc

            elif filename[-13:] == "test-loss.csv":
                    results_test[key] = roc_auc
    

    def create_df(results):
        df = pd.DataFrame() 
        for label, value in results.items():
            train_loss_percent = label.split("-")[-1]
            actual_label = label.split("-")[0]
            df.at[actual_label, train_loss_percent] = value
        return df

    train_path = os.path.join(absolute_path, "./results/"+file_prefix+"train-"+str(result_suffix)+".csv")
    test_path = os.path.join(absolute_path, "./results/"+file_prefix+"test-"+str(result_suffix)+".csv")
    df_train = create_df(results_train).to_csv(train_path)
    df_test = create_df(results_test).to_csv(test_path)
    
def collect_single_run_my(file_prefix, file_paths, result_suffix):
    
    def run(prefix):
        count = len(file_paths)
        
        dfs = []
        
        for i in range(0, len(file_paths)):
            path = os.path.join(absolute_path, "./results/" + prefix + file_paths[i])
            df = pd.read_csv(path)
            dfs.append(df)
            
        stacked = pd.concat([df.stack() for df in dfs], axis=1)
        stds = stacked.std(axis=1).unstack()
        
        df = dfs[0]
        for i in range(1, len(dfs)):
            df += dfs[i]
            
        df /= count

        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
            
        for column in df:
            df.at["mean", column] = df[column].mean()
            df.loc["mean std", column] = stds[column].mean()
        return df
    
    train_path = os.path.join(absolute_path, "./results/"+file_prefix+"train-" + result_suffix + ".csv")
    test_path = os.path.join(absolute_path, "./results/"+file_prefix+"test-" + result_suffix + ".csv")
    df_train = run(file_prefix + "train-").to_csv(train_path)
    df_test = run(file_prefix + "test-").to_csv(test_path)
    
def collect_all_results(file_prefix, names):
    train = ["train-"+name+"-complete.csv" for name in names]
    test = ["test-"+name+"-complete.csv" for name in names]
    
    train = [pd.read_csv(os.path.join(absolute_path, "./results/"+file_prefix+name)) for name in train]
    for i in range(1, len(train)):
        train[i] = train[i].drop(columns=["Unnamed: 0"])
    result = pd.concat(train, axis=1)
    result.set_index("Unnamed: 0", inplace=True)
    
    result.to_excel(os.path.join(absolute_path, "./results/" + file_prefix + "train-ALL.xlsx"))
    
    test = [pd.read_csv(os.path.join(absolute_path, "./results/"+file_prefix+name)) for name in test]
    for i in range(1, len(test)):
        test[i] = test[i].drop(columns=["Unnamed: 0"])
    result = pd.concat(test, axis=1)
    result.set_index("Unnamed: 0", inplace=True)
    
    result.to_excel(os.path.join(absolute_path, "./results/" + file_prefix + "test-ALL.xlsx"))
    

if __name__ == "__main__":
    collect_all_results()
