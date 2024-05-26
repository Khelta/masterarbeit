VALID_DATASETS = ["mnist", "fashion", "svhn", "cifar10", "cifar100"]
VALID_ALGORITHMS = ["myCAE", "myCAEsoft", "CAE"]
ANOMALIE_PERCENTS = [0.25, 0.15, 0.05]

DATASETS_IN_CHANNELS = {"mnist": 1, "fashion": 1, "cifar10": 3, "cifar100": 3, "svhn": 3}

LABELS_MNIST = [i for i in range(0, 10)]
LABELS_FASHION = [i for i in range(0, 10)]
LABELS_SVHN = [i for i in range(0, 10)]
LABELS_CIFAR10 = [i for i in range(0, 10)]
LABELS_CIFAR100 = [i for i in range(0, 100)]

LABEL_DICT = {"mnist": LABELS_MNIST,
              "fashion": LABELS_FASHION,
              "svhn": LABELS_SVHN,
              "cifar10": LABELS_CIFAR10,
              "cifar100": LABELS_CIFAR100}

CUT_OFF_PERCENTS = [0.5, 0.6, 0.7, 0.8, 0.9]