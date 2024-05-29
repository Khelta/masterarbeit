
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
import os
import multiprocessing

from models.cae_pytorch import CAE_28
from constants import DATASETS_IN_CHANNELS


absolute_path = os.path.dirname(__file__)

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


def train_cae_my(train_loader, model, criterion, optimizer, epochs, ap, device, histopath='', dataset="mnist"):
    def f(batch):
        result = []
        for img in batch:
            img = img.unsqueeze(0)
            recon = model(img)
            loss = criterion(recon, img)
            result.append((img, loss))
        return result
    
    for epoch in range(epochs):
        imgs = []
        num_processes = 4
        imgs = []
        for (batch, _) in train_loader:
            batch = batch.to(device)
            result = f(batch)
            """with multiprocessing.pool.ThreadPool(num_processes) as p:
                result = p.map(f, batch)"""
            imgs += result
        
        #print(len(results))
    
        imgs.sort(key=lambda x: x[1])
        l = int(ap * len(imgs))
        imgs = imgs[:l]
        optimizer.zero_grad()

        for data in imgs:
            img = data[0]
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if len(histopath) > 0:
            torch.save(model.state_dict(), histopath+"-e"+str(epoch+1)+".pt")
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    
    return None


def train_cae_my_soft(train_loader, model, criterion, optimizer, epochs, ap, device):
    def f(batch):
        result = []
        for img in batch:
            img = img.unsqueeze(0).unsqueeze(0)
            recon = model(img)
            loss = criterion(recon, img)
            result.append((img, loss))
        return result
    
    for epoch in range(epochs):
        imgs = []

        num_processes = 4
        processes = []
        imgs = []
        for (batch, _) in train_loader:
            batch = batch.to(device)
            with multiprocessing.pool.ThreadPool(num_processes) as p:
                result = p.map(f, batch)
            imgs += result
            
        #print(len(results))
    
        imgs.sort(key=lambda x: x[0][1])
        optimizer.zero_grad()

        losses = []

        for i in range(0, len(imgs)):
            img = imgs[i][0][0]
            recon = model(img)
            loss = criterion(recon, img)
            #loss *= (np.tanh(-i/len(imgs)*2*np.pi+np.pi)+1)/2
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch+1}, Loss:{sum(losses)/len(losses):.4f}')
    
    return None


def train_cae_single(train_loader, model, criterion, optimizer, epochs, device, histopath=''):
    model = model.to(device)
    for epoch in range(epochs):
        for data in train_loader:
            imgs, _ = data
            for img in imgs:
                img = img.to(device)
                img = img.unsqueeze(0)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, img)
                loss.backward()
                optimizer.step()
        if len(histopath) > 0:
            torch.save(model.state_dict(), histopath+"-e"+str(epoch+1)+".pt")
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    return None


def train_drae(trainloader, model, criterion, optimizer, epochs, ap, device, histopath='', dataset="mnist"):
    model.train()
    losses = AverageMeter()
    for epoch in range(epochs):
        for batch_id, (inputs, _) in enumerate(trainloader):
            #for input in inputs:
                #input = input.unsqueeze(0)
                inputs = torch.autograd.Variable(inputs.cuda())

                outputs = model(inputs)

                loss = criterion(inputs, outputs)

                losses.update(loss.item(), inputs.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
           


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