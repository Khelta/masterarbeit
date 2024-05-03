from multiprocessing import Manager
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.cae_pytorch import CAE_pytorch
import pandas as pd

absolute_path = os.path.dirname(__file__)

def prepare_mnist(selected_label, p):
    transform = transforms.ToTensor()

    # Download and load the MNIST training data
    path_to_data = os.path.join(absolute_path, "./data")
    train_data = datasets.MNIST(root=path_to_data, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=path_to_data, train=False, download=True, transform=transform)

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
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_cae(train_loader, model, criterion, optimizer, epochs):
    """Valid for both CAE+MSELoss and CAE+DRAELoss"""
    model.train()
    losses = AverageMeter()
    for epoch in range(epochs):
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = torch.autograd.Variable(inputs.cuda())

            outputs = model(inputs)

            loss = criterion(inputs, outputs)

            losses.update(loss.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 10 == 0:
                print('Epoch: [{} | {}], batch: {}, loss: {}'.format(epoch + 1, epochs, batch_idx + 1, losses.avg))

class DRAELossAutograd(nn.Module):

    def __init__(self, lamb, size_average=True):
        super(DRAELossAutograd, self).__init__()
        self.lamb = lamb
        self.size_average = size_average  # for compatibility, not used

    def forward(self, inputs, targets):
        err = inputs.sub(targets).pow(2).view(inputs.size(0), -1).sum(dim=1, keepdim=False)
        err_sorted, _ = torch.sort(err)
        total_scatter = err.sub(err.mean()).pow(2).sum()
        regul = 1e6
        obj = None
        for i in range(inputs.size(0)-1):
            err_in = err_sorted[:i+1]
            err_out = err_sorted[i+1:]
            within_scatter = err_in.sub(err_in.mean()).pow(2).sum() + \
                             err_out.sub(err_out.mean()).pow(2).sum()
            h = within_scatter.div(total_scatter)
            if h < regul:
                regul = h
                obj = err_in.mean()

        return obj + self.lamb * regul

def test_cae_pytorch(testloader, model):
    """Yield reconstruction loss as well as representations"""
    model.eval()
    losses = []
    #reps = []
    r_labels = []
    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs = torch.autograd.Variable(inputs.cuda())
        rep = model.encode(inputs)
        outputs = model.decode(rep)
        loss = outputs.sub(inputs).pow(2).view(outputs.size(0), -1)
        loss = loss.sum(dim=1, keepdim=False)
        losses.append(loss.data.cpu())
        #reps.append(rep.data.cpu())
        r_labels.extend(labels.detach().numpy())
    losses = torch.cat(losses, dim=0)
    #reps = torch.cat(reps, dim=0)
    return losses.numpy(), r_labels

def _DRAE_experiment(selected_label, p, gpu_q, in_channels=3, cycle=0):
    gpu_to_use = gpu_q.get()

    model = CAE_pytorch(in_channels=in_channels)
    batch_size = 128

    model = model.cuda()
    train_loader, test_loader = prepare_mnist(selected_label, p)
    # cudnn.benchmark = True
    criterion = DRAELossAutograd(lamb=0.1)
    optimizer = optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)
    epochs = 30

    # #########################Training########################
    train_cae(train_loader, model, criterion, optimizer, epochs)

    # #######################Testin############################
    losses, labels = test_cae_pytorch(train_loader, model)
    df = pd.DataFrame({'Loss': losses, 
                       'Label': labels}).sort_values(by="Loss")
    df.to_csv(os.path.join(absolute_path, "results/DRAE/{}-{}-cycle{}-{}-train-loss.csv".format(selected_label, p, cycle, epochs)))
    
    losses, labels = test_cae_pytorch(test_loader, model)
    df = pd.DataFrame({'Loss': losses, 
                       'Label': labels}).sort_values(by="Loss")
    df.to_csv(os.path.join(absolute_path, "results/DRAE/{}-{}-cycle{}-{}-test-loss.csv".format(selected_label, p, cycle, epochs)))

    gpu_q.put(gpu_to_use)

if __name__ == '__main__':
    N_GPUS = 1
    man = Manager()
    q = man.Queue(N_GPUS)
    for g in range(N_GPUS):
        q.put(str(g))
    
    for label in range(0, 10):
        for cycle in range(0, 5):
            _DRAE_experiment(label,0.25,q,1,cycle=cycle)