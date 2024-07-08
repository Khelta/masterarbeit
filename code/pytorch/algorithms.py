
import torch
import torch.nn as nn

import os


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


def train_cae_my(train_loader, model, criterion, optimizer, epochs, cop, device, histopath='', dataset="mnist"):
    c = criterion.__class__(reduction='none')
    is_historun = len(histopath) > 0
    if is_historun:
        all_saved_losses = []
    for epoch in range(epochs):
        if is_historun:
            saved_losses = []
        for (batch, _) in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            losses = c(batch, recon).view(batch.shape[0], -1).mean(1)
            indices = torch.argsort(losses)

            index = int(cop * len(batch))
            batch = batch[indices][:index]
            recon = recon[indices][:index]

            loss = criterion(batch, recon)

            if is_historun:
                saved_losses.append((float(loss), len(batch)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if is_historun:
            all_saved_losses.append(saved_losses)
            torch.save(model.state_dict(), histopath + "-e" + str(epoch + 1) + ".pt")
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')

    if is_historun:
        return all_saved_losses
    else:
        return None


def train_cae_single(train_loader, model, criterion, optimizer, epochs, device, histopath=''):
    for epoch in range(epochs):
        for (batch, _) in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(batch, recon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if len(histopath) > 0:
            torch.save(model.state_dict(), histopath + "-e" + str(epoch + 1) + ".pt")
        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    return None


def train_drae(trainloader, model, criterion, optimizer, epochs, device, histopath='', dataset="mnist"):
    model.train()
    losses = AverageMeter()
    for epoch in range(epochs):
        for batch_id, (inputs, _) in enumerate(trainloader):
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
        obj = err_sorted[:inputs.size(0) - 1 + 1].mean()
        for i in range(inputs.size(0) - 1):
            err_in = err_sorted[:i + 1]
            err_out = err_sorted[i + 1:]
            within_scatter = err_in.sub(err_in.mean()).pow(2).sum() + err_out.sub(err_out.mean()).pow(2).sum()
            h = within_scatter.div(total_scatter)
            if h < regul:
                regul = h
                obj = err_in.mean()

        return obj + self.lamb * regul
