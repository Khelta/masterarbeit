
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
import os

from models.cae_pytorch import CAE_pytorch

absolute_path = os.path.dirname(__file__)

def prepare_mnist(selected_label, p):
    transform = transforms.ToTensor()

    # Download and load the MNIST training data
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

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

def train_cae_my(train_loader, model, criterion, optimizer, epochs, train_loss_percent, device):
    for epoch in range(num_epochs):
        imgs = []
        for (img, _) in train_loader:
            for img in batch:
                img = img.to(device)
                img = img.unsqueeze(0)
                recon = model(img)
                loss = criterion(recon, img)
                imgs.append((img, loss))

        imgs.sort(key=lambda x: x[1])
        l = int(train_loss_percent * len(imgs))
        imgs = imgs[:l]

        optimizer.zero_grad()

        for (img, _) in imgs:
            recon = model(img)
            loss = criterion(recon, img)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    
    return None

def train_cae(train_loader, model, criterion, optimizer, epochs, device):
    model = model.to(device)
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
    return None

def complete_run_cae(file_prefix, selected_label=9, p=0.05, train_loss_percent=0.5, num_epochs=30):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    train_loader, test_loader = prepare_mnist(selected_label, p)
    
    model = CAE_pytorch(in_channels=1)
    model = model.to(device)

    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(),eps=1e-7, weight_decay=0.0005)
    
    outputs = None
    #outputs = train_cae(train_loader, model, criterion, optimizer, epochs=250, device=device)
    outputs = train_cae_my(train_loader, model, criterion, optimizer, epochs=num_epochs, train_loss_percent=train_loss_percent, device=device)

    """
    if outputs is not None:
        k = num_epochs-1
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i+1)
            plt.imshow(item[0])
                
        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9+i+1) # row_length + i + 1
            plt.imshow(item[0])
        plt.savefig(file_prefix+"-lables.png")
        plt.close()
    """
    
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

    normal_train, anomalie_train, anomalie_label_train = calculate_losses(train_loader, device)
    normal_test, anomalie_test, anomalie_label_test = calculate_losses(test_loader, device)

    min_loss = min((min(normal_train),min(anomalie_train)))
    max_loss = max((max(normal_train),max(anomalie_train)))
    print(min_loss, max_loss)

    
    #t = [0] * len(normal_train) + [1] * len(anomalie_train)
    p = normal_train + anomalie_train
    df = pd.DataFrame({'Loss': p, 
                       'Label': [selected_label]*len(normal_train) + anomalie_label_train}).sort_values(by="Loss")
    path = os.path.join(absolute_path, file_prefix+"train-loss.csv")
    df.to_csv(path)

    #t = [0] * len(normal_test) + [1] * len(anomalie_test)
    p = normal_test + anomalie_test
    df = pd.DataFrame({'Loss': p, 
                       'Label': [selected_label]*len(normal_test) + anomalie_label_test}).sort_values(by="Loss")
    path = os.path.join(absolute_path, file_prefix+"test-loss.csv")
    df.to_csv(path)

    return


if __name__ == "__main__": 
    pnum = 100
    #for pnum in range(200, 205):
    prefix_num = pnum
    num_epochs = 250
    p = 0.25

    train_file_name = "results/train-{}.csv".format(prefix_num)
    test_file_name = "results/test-{}.csv".format(prefix_num)

    try: 
        df_train = pd.read_csv(train_file_name)
        df_test = pd.read_csv(test_file_name)
    except:
        df_train = pd.DataFrame(index=[i for i in range(0, 10)],columns=[str(i/10) for i in range(5, 10)])
        df_test  = pd.DataFrame(index=[i for i in range(0, 10)],columns=[str(i/10) for i in range(5, 10)])
        
    for j in range(5, 10):
        train_loss_percent = j/10
        for i in range(0, 10):
            if pd.isna(df_train.at[i, str(train_loss_percent)]) or pd.isna(df_test.at[i, str(train_loss_percent)]):
                print("### " + str(i) + " ### ({})".format(train_loss_percent))
                file_prefix = "results/{}/".format(prefix_num) + "-".join([str(x).replace(".", ",") for x in [i, p, train_loss_percent, num_epochs]])
                path = os.path.join(absolute_path, file_prefix+"train-loss.csv")
                print(path)
                if os.path.isfile(path):
                    continue 
                complete_run_cae(file_prefix=file_prefix, selected_label=i, p=p,train_loss_percent=train_loss_percent, num_epochs=num_epochs)
                