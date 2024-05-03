from multiprocessing import Manager
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.cae_pytorch import CAE_pytorch
import pandas as pd

import torch.utils.data as data

from keras2pytorch_dataset import trainset_pytorch, testset_pytorch

absolute_path = os.path.dirname(__file__)

transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

def prox_l21(S, lmbda):
    """L21 proximal operator."""
    Snorm = np.sqrt((S ** 2).sum(axis=tuple(range(1, S.ndim)), keepdims=False))
    multiplier = 1 - 1 / np.minimum(Snorm/lmbda, 1)
    out = S * multiplier.reshape((S.shape[0],)+(1,)*(S.ndim-1))
    return out

def prepare_mnist(selected_label, p):

    # Download and load the MNIST training data
    path_to_data = os.path.join(absolute_path, "./data")
    train_data = datasets.MNIST(root=path_to_data, train=True, download=True)
    test_data = datasets.MNIST(root=path_to_data, train=False, download=True)

    # Filter the dataset to include only images with label normal and 5
    label_normal_indices = [i for i, (_, label) in enumerate(train_data) if label == selected_label]
    label_anomalie_indices = [i for i, (_, label) in enumerate(train_data) if label != selected_label]

    num_label_normal =  len(label_normal_indices)
    num_label_anomalie = int((num_label_normal/(1-p))-num_label_normal)

    print(num_label_normal, num_label_anomalie, num_label_normal + num_label_anomalie)

    selected_label_anomalie_indices = np.random.choice(label_anomalie_indices, num_label_anomalie, replace=False)

    # Combine the selected indices
    selected_indices_train = np.concatenate([label_normal_indices, selected_label_anomalie_indices])

    # Create a Subset of the original dataset with the selected indices
    filtered_dataset = torch.utils.data.Subset(train_data, selected_indices_train)

    print("Len Train:", len(filtered_dataset))
    print("Len Test:", len(test_data))

    # Create a DataLoader to iterate through the filtered dataset
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return filtered_dataset, test_data

def train_robust_cae(x_train, model, criterion, optimizer, lmbda, inner_epochs, outer_epochs, reinit=True, in_channels=1):
    batch_size = 128
    S = np.zeros_like(x_train)  # reside on numpy as x_train

    def get_reconstruction(loader):
        model.eval()
        rc = []
        for batch, _ in loader:
            with torch.no_grad():
                rc.append(model(batch.cuda()).cpu().numpy())
        out = np.concatenate(rc, axis=0)
        # NOTE: transform_train swaps the channel axis, swap back to yield the same shape
        out = out.transpose((0, 2, 3, 1))
        return out

    for oe in range(outer_epochs):
        # update AE
        if reinit:
            # Since our CAE_pytorch does not implement reset_parameters, regenerate a new model if reinit.
            del model
            model = CAE_pytorch(in_channels=1).cuda()
        model.train()
        trainset = trainset_pytorch(x_train-S, train_labels=np.ones((x_train.shape[0], )), transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        for ie in range(inner_epochs):
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.cuda()
                outputs = model(inputs)
                loss = criterion(inputs, outputs)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_idx + 1) % 10 == 0:
                    print('Epoch: [{} | {} ({} | {})], batch: {}, loss: {}'.format(
                        ie+1, inner_epochs, oe+1, outer_epochs, batch_idx+1, loss.item())
                    )
        # update S via l21 proximal operator
        testloader = data.DataLoader(trainset, batch_size=1024, shuffle=False)
        recon = get_reconstruction(testloader)
        S = prox_l21(x_train - recon, lmbda)

    # get final reconstruction
    finalset = trainset_pytorch(x_train - S, train_labels=np.ones((x_train.shape[0],)), transform=transform_train)
    finalloader = data.DataLoader(finalset, batch_size=1024, shuffle=False)
    reconstruction = get_reconstruction(finalloader)
    losses = ((x_train-S-reconstruction) ** 2).sum(axis=(1, 2, 3), keepdims=False)
    return losses

def _RDAE_experiment(single_class_ind, gpu_q, p, n_channels, cycle):
    gpu_to_use = gpu_q.get()
    #cudnn.benchmark = True
    
    train_loader, test_loader = prepare_mnist(single_class_ind, p)

    n_channels = n_channels
    model = CAE_pytorch(in_channels=n_channels)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
    criterion = nn.MSELoss()
    epochs = 20
    inner_epochs = 1
    lmbda = 0.00065

    
    x_train = np.array([np.array(img) for (img, _) in train_loader])
    
    losses = train_robust_cae(x_train, model, criterion, optimizer, lmbda, inner_epochs, epochs//inner_epochs, False, n_channels)
    losses = losses - losses.min()
    losses = losses / (1e-8 + losses.max())
    scores = 1 - losses
    
    print(losses)


if __name__ == '__main__':
    N_GPUS = 1
    man = Manager()
    q = man.Queue(N_GPUS)
    for g in range(N_GPUS):
        q.put(str(g))
    
    for label in range(0, 10):
        for cycle in range(0, 5):
            _RDAE_experiment(label, q, 0.25,1,cycle=cycle)

