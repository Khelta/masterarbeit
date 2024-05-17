import argparse
import os

from datasets import prepare_data
from constants import *
from procedure import complete_run_cae

absolute_path = os.path.dirname(__file__)

def run(algorithm, dataset, ap, epochs, cop=None, cycles=5):
    print(algorithm, dataset, ap, epochs, cop, cycles)
    
    if cop is None: 
        labels = globals()["LABELS_" + dataset.upper()]
        for cycle in range(cycles):
            for i in range(5, 10):
                cop = i/10
                for label in labels:
                    dir_path = "results/{}/{}/{}/".format(algorithm, dataset, cycle)
                    file_name = "{}-{}-{}-{}-{}-{}".format(dataset, label, ap, cop, epochs, algorithm)
                    file_prefix = dir_path + file_name

                    joined_dir_path = os.path.join(absolute_path, dir_path)
                    if not os.path.exists(joined_dir_path):
                        os.makedirs(joined_dir_path)
                    
                    file_prefix = os.path.join(absolute_path, file_prefix)
                    print(file_prefix)
                    if not os.path.exists(file_prefix+"-train-loss.csv"):
                        complete_run_cae(dataset, algorithm, file_prefix, label, cop, ap, epochs)
    

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if not x > 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range (0.0, 1.0]"%(x,))
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs","-e", help="the number of epochs to use while training", type=int, default=30)
    #parser.add_argument("-cop", help="the percentage of cut-off", type=restricted_float, default=0.5)
    parser.add_argument("-ap", help="The proportion of anomalies in the total data set", type=restricted_float, default=0.25)
    parser.add_argument("-dataset", "-d", help="The dataset to be used", choices=VALID_DATASETS, default="mnist")
    parser.add_argument("-algorithm", "-a", help="The algorithm to use for training", choices=VALID_ALGORITHMS, default="myCAE")
    parser.add_argument("-cycles", "-c", help="The number of repetitions", type=int, default=5)
    args = parser.parse_args()
    print(args)
    
    run(**vars(args))
    
    
    