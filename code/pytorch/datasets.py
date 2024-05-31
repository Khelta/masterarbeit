import numpy as np
import os
import torch
from torchvision import datasets, transforms

absolute_path = os.path.dirname(__file__)

def prepare_data(dataset, selected_label, ap, batch_size=256):
    transform = transforms.ToTensor()
    
    possible_datasets = ["mnist", "fashion", "svhn", "cifar10", "cifar100"]
    if dataset not in possible_datasets:
        raise ValueError("dataset must be in " + str(possible_datasets))
    elif dataset == "mnist":
        train_data = datasets.MNIST(root=os.path.join(absolute_path, './data'), train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root=os.path.join(absolute_path, './data'), train=False, download=True, transform=transform)
    elif dataset == "fashion":
        train_data = datasets.FashionMNIST(root=os.path.join(absolute_path, './data'), train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root=os.path.join(absolute_path, './data'), train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        train_data = datasets.CIFAR10(root=os.path.join(absolute_path, './data'), train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root=os.path.join(absolute_path, './data'), train=False, download=True, transform=transform)
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root=os.path.join(absolute_path, './data'), train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100(root=os.path.join(absolute_path, './data'), train=False, download=True, transform=transform)
    else:
        train_data = datasets.SVHN(root=os.path.join(absolute_path, './data'), download=True, transform=transform)
        test_data = None
    return create_loader(train_data, test_data, selected_label, ap, batch_size)
    

def create_loader(train_data, test_data, selected_label, ap, batch_size=64):
    # Separate the data set into normal and abnormal data
    label_normal_indices = [i for i, (_, label) in enumerate(train_data) if label == selected_label]
    label_anomalie_indices = [i for i, (_, label) in enumerate(train_data) if label != selected_label]

    num_label_normal =  len(label_normal_indices)
    num_label_anomalie = int((num_label_normal/(1-ap))-num_label_normal)

    #print(num_label_normal, num_label_anomalie, num_label_normal + num_label_anomalie)

    selected_label_anomalie_indices = np.random.choice(label_anomalie_indices, num_label_anomalie, replace=False)

    # Combine the selected indices
    selected_indices_train = np.concatenate([label_normal_indices, selected_label_anomalie_indices])

    # Create a Subset of the original dataset with the selected indices
    filtered_dataset = torch.utils.data.Subset(train_data, selected_indices_train)

    pstring = "Num Normal " + str(num_label_normal) + " Len Train: " + str(len(filtered_dataset))
    if test_data is not None: 
        pstring += " Len Test:" + str(len(test_data))

    print(pstring)

    # Create a DataLoader to iterate through the filtered dataset
    train_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    if test_data is not None:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    else:
        test_loader = None
    
    return train_loader, test_loader

if __name__ == "__main__":
    prepare_data("cifar100", 0, 0.25)