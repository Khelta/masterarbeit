import os
import pandas as pd
import torch
import torch.nn as nn

from algorithms import *
from datasets import prepare_data
from models.cae_pytorch import CAE_28, CAE_32
from constants import VALID_ALGORITHMS, VALID_DATASETS

absolute_path = os.path.dirname(__file__)

def complete_run_cae(dataset, algorithm, file_prefix, selected_label=9, cop=0.05, ap=0.5, num_epochs=30, historun=False):
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError("Algoritm must be in " + str(VALID_ALGORITHMS))
    
    if dataset not in VALID_DATASETS:
        raise ValueError("Algoritm must be in " + str(VALID_DATASETS))
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if dataset in ["mnist", "fashion"]:
        model = CAE_28(in_channels=1)
    else:
        model = CAE_32(in_channels=3)
    model = model.to(device)

    if historun is True:
        histopath = os.path.join(absolute_path, str(file_prefix))
        torch.save(model.state_dict(), histopath + "-e0.pt")
    else:
        histopath = ""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),eps=1e-7, weight_decay=0.0005)
    
    train_loader, test_loader = prepare_data(dataset, selected_label, ap, batch_size=64 if algorithm=="DRAE" else 1024)
    
    if algorithm == "CAE":
        train_cae_single(train_loader, model, criterion, optimizer, epochs=num_epochs, device=device, histopath=histopath)
    elif algorithm == "myCAE":
        train_cae_my(train_loader, model, criterion, optimizer, epochs=num_epochs, ap=ap, device=device, histopath=histopath, dataset=dataset)
    elif algorithm == "myCAEsoft":
        train_cae_my_soft(train_loader, model, criterion, optimizer, epochs=num_epochs, ap=ap, device=device)
    elif algorithm == "DRAE":
        criterion = DRAELossAutograd(lamb=0.1)
        train_drae(train_loader, model, criterion, optimizer, epochs=num_epochs, ap=ap, device=device, histopath=histopath, dataset=dataset)

    
    def calculate_losses(loader, device):
        normal = []
        anomalie = []
        anomalie_label = []
        for (img, label) in loader:
            img = img.to(device)
            recon = model(img)
            for i in range(0, len(img)):
                loss = criterion(recon[i], img[i])
                if label[i].item() == selected_label:
                    normal.append(loss)
                else:
                    anomalie.append(loss)
                    anomalie_label.append(label[i].item())
                
        normal = [round(tensor.tolist(),5) for tensor in normal]
        anomalie = [round(tensor.tolist(),5) for tensor in anomalie]
        return normal, anomalie, anomalie_label
    
    def calculate_losses_drae(testloader):
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

    if algorithm != "DRAE":
        normal_train, anomalie_train, anomalie_label_train = calculate_losses(train_loader, device)
        train_loss_all = normal_train + anomalie_label_train
        train_labels = [selected_label]*len(normal_train) + anomalie_label_train
        if test_loader is not None:
            normal_test, anomalie_test, anomalie_label_test = calculate_losses(test_loader, device)
            test_loss_all = normal_test + anomalie_label_test
            test_labels = [selected_label]*len(normal_test) + anomalie_label_test
    else:
        train_loss_all, train_labels = calculate_losses_drae(train_loader)
        if test_loader is not None:
            test_loss_all, test_labels = calculate_losses_drae(test_loader)

    """
    min_loss = min((min(normal_train),min(anomalie_train)))
    max_loss = max((max(normal_train),max(anomalie_train)))
    print(min_loss, max_loss)
    """
    
    if historun is False: 
        df = pd.DataFrame({'Loss': train_loss_all, 
                           'Label': train_labels}).sort_values(by="Loss")
        path = os.path.join(absolute_path, str(file_prefix)+"-train-loss.csv")
        df.to_csv(path)

        if test_loader is not None:
            df = pd.DataFrame({'Loss': test_loss_all, 
                               'Label': test_labels}).sort_values(by="Loss")
            path = os.path.join(absolute_path, str(file_prefix)+"-test-loss.csv")
            df.to_csv(path)

    else: 
        for i in range(0, num_epochs+1):
            model.load_state_dict(torch.load(histopath+"-e{}.pt".format(i)))
            model.eval()
            if algorithm != "DRAE":
                normal_train, anomalie_train, anomalie_label_train = calculate_losses(train_loader, device)
                train_loss_all = normal_train + anomalie_train
                train_labels = [selected_label]*len(normal_train) + anomalie_label_train
            else:
                train_loss_all, train_labels = calculate_losses_drae(train_loader)
            df = pd.DataFrame({'Loss': train_loss_all, 
                               'Label': train_labels}).sort_values(by="Loss")
            path = os.path.join(absolute_path, str(file_prefix)+"-e{}-train-loss.csv".format(i))
            df.to_csv(path)

    return
