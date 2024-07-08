import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from algorithms import train_cae_my, train_cae_single, train_drae, DRAELossAutograd
from datasets import prepare_data
from models.cae_pytorch import CAE_28, CAE_32, CAE_drop_28, CAE_drop_32
from constants import VALID_ALGORITHMS, VALID_DATASETS
from helper import calculate_auprin_from_csv, calculate_auprout_from_csv, calculate_auroc_from_csv

absolute_path = os.path.dirname(__file__)


def complete_run_cae(dataset, algorithm, file_prefix, selected_label=9, cop=0.05, ap=0.5, num_epochs=30, historun=False):
    if algorithm == "DeepSVDD":
        raise ValueError()

    if algorithm not in VALID_ALGORITHMS:
        raise ValueError("Algoritm must be in " + str(VALID_ALGORITHMS))

    if dataset not in VALID_DATASETS:
        raise ValueError("Algoritm must be in " + str(VALID_DATASETS))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dataset in ["mnist", "fashion"]:
        if algorithm == "CAEDrop":
            model = CAE_drop_28(in_channels=1)
        else:
            model = CAE_28(in_channels=1)
    else:
        if algorithm == "CAEDrop":
            model = CAE_drop_32(in_channels=3)
        else:
            model = CAE_32(in_channels=3)
    model = model.to(device)

    if historun is True:
        histopath = os.path.join(absolute_path, str(file_prefix))
        torch.save(model.state_dict(), histopath + "-e0.pt")
    else:
        histopath = ""

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), eps=1e-7, weight_decay=0.0005)

    batch_size = 256
    train_loader, test_loader = prepare_data(dataset, selected_label, ap, batch_size=batch_size)

    if algorithm == "CAE":
        losses = train_cae_single(train_loader, model, criterion, optimizer, epochs=num_epochs, device=device, histopath=histopath)
    elif algorithm == "myCAE":
        losses = train_cae_my(train_loader, model, criterion, optimizer, epochs=num_epochs, cop=cop, device=device, histopath=histopath, dataset=dataset)
    elif algorithm == "CAEDrop":
        losses = train_cae_single(train_loader, model, criterion, optimizer, epochs=num_epochs, device=device, histopath=histopath)
    elif algorithm == "DRAE":
        criterion = DRAELossAutograd(lamb=0.1)
        losses = train_drae(train_loader, model, criterion, optimizer, epochs=num_epochs, device=device, histopath=histopath, dataset=dataset)

    def calculate_losses(loader, device):
        normal = []
        anomalie = []
        anomalie_label = []
        for (batch, label) in loader:
            batch = batch.to(device)
            recon = model(batch)
            c = criterion.__class__(reduction="none")
            loss = c(batch, recon).view(batch.shape[0], -1).mean(1)

            mask_normal = label == selected_label
            mask_anomalie = label != selected_label
            indices_normal = torch.nonzero(mask_normal).squeeze()
            indices_anomalie = torch.nonzero(mask_anomalie).squeeze()

            if len(indices_normal.size()) == 0:
                normal += [loss[indices_normal].tolist()]
            else:
                normal += loss[indices_normal].tolist()
            if len(indices_anomalie.size()) == 0:
                anomalie += [loss[indices_anomalie].tolist()]
                anomalie_label += [label[indices_anomalie].tolist()]
            else:
                anomalie += loss[indices_anomalie].tolist()
                anomalie_label += label[indices_anomalie].tolist()

        decimals = 5
        normal = list(np.around(np.array(normal), decimals))
        anomalie = list(np.around(np.array(anomalie), decimals))
        return normal, anomalie, anomalie_label

    def calculate_losses_drae(testloader):
        """Yield reconstruction loss as well as representations"""
        model.eval()
        losses = []
        # reps = []
        r_labels = []
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs = torch.autograd.Variable(inputs.cuda())
            rep = model.encode(inputs)
            outputs = model.decode(rep)
            loss = outputs.sub(inputs).pow(2).view(outputs.size(0), -1)
            loss = loss.sum(dim=1, keepdim=False)
            losses.append(loss.data.cpu())
            # reps.append(rep.data.cpu())
            r_labels.extend(labels.detach().numpy())
        losses = torch.cat(losses, dim=0)
        # reps = torch.cat(reps, dim=0)
        return losses.numpy(), r_labels

    if algorithm == "DRAE":
        train_loss_all, train_labels = calculate_losses_drae(train_loader)
        if test_loader is not None:
            test_loss_all, test_labels = calculate_losses_drae(test_loader)
    else:
        normal_train, anomalie_train, anomalie_label_train = calculate_losses(train_loader, device)
        train_loss_all = normal_train + anomalie_train
        train_labels = [selected_label] * len(normal_train) + anomalie_label_train
        if test_loader is not None:
            normal_test, anomalie_test, anomalie_label_test = calculate_losses(test_loader, device)
            test_loss_all = normal_test + anomalie_test
            test_labels = [selected_label] * len(normal_test) + anomalie_label_test

    if historun is False:
        df = pd.DataFrame({'Loss': train_loss_all,
                           'Label': train_labels}).sort_values(by="Loss")
        path = os.path.join(absolute_path, str(file_prefix) + "-train-loss.csv")
        df.to_csv(path)

        if test_loader is not None:
            df = pd.DataFrame({'Loss': test_loss_all,
                               'Label': test_labels}).sort_values(by="Loss")
            path = os.path.join(absolute_path, str(file_prefix) + "-test-loss.csv")
            df.to_csv(path)

    else:
        for i in range(0, num_epochs + 1):
            model.load_state_dict(torch.load(histopath + "-e{}.pt".format(i)))
            model.eval()
            if algorithm == "DRAE":
                train_loss_all, train_labels = calculate_losses_drae(train_loader)
            elif algorithm == "DeepSVDD":
                raise NotImplementedError()
            else:
                normal_train, anomalie_train, anomalie_label_train = calculate_losses(train_loader, device)
                train_loss_all = normal_train + anomalie_train
                train_labels = [selected_label] * len(normal_train) + anomalie_label_train
            df = pd.DataFrame({'Loss': train_loss_all,
                               'Label': train_labels}).sort_values(by="Loss")
            path = os.path.join(absolute_path, str(file_prefix) + "-e{}-train-loss.csv".format(i))
            df.to_csv(path)

        loss_per_epoch = []
        label = file_prefix.split("-")[1]
        for i in range(0, len(losses)):
            current_loss = losses[i]
            n = sum([size for loss, size in current_loss])
            sum_batch_mean_losses = sum([loss * size for loss, size in current_loss])
            mean_loss = sum_batch_mean_losses / n

            path = histopath + "-e{}-train-loss.csv".format(i + 1)
            auroc = calculate_auroc_from_csv(path, label)
            auprin = calculate_auprin_from_csv(path, label)
            auprout = calculate_auprout_from_csv(path, label)
            loss_per_epoch.append((mean_loss, auroc, auprin, auprout))

        df = pd.DataFrame(loss_per_epoch, columns=["Mean Loss", "AUROC", "AUPR-IN", "AUPR-OUT"])
        df = df.set_index(pd.Index(range(1, len(losses) + 1)))
        path = os.path.join(absolute_path, str(file_prefix) + "-e0-meanLossesPerEpoch.csv")
        df.to_csv(path)
