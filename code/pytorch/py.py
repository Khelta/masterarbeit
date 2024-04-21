
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd 

class CAE_pytorch(nn.Module):
        def __init__(self, in_channels = 3, rep_dim = 256):
            super(CAE_pytorch, self).__init__()
            nf = 16
            self.nf = nf

            # Encoder
            self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
            self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
            self.enc_act1 = nn.ReLU(inplace=True)

            self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
            self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
            self.enc_act2 = nn.ReLU(inplace=True)

            self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
            self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
            self.enc_act3 = nn.ReLU(inplace=True)

            self.enc_fc = nn.Linear(nf * 4 * 4 * 4, rep_dim)
            self.rep_act = nn.Tanh()

            # Decoder
            self.dec_fc = nn.Linear(rep_dim, nf * 4 * 4 * 4)
            self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 4 * 4 *4)
            self.dec_act0 = nn.ReLU(inplace=True)

            self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
            self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 2)
            self.dec_act1 = nn.ReLU(inplace=True)

            self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.dec_bn2 = nn.BatchNorm2d(num_features=nf)
            self.dec_act2 = nn.ReLU(inplace=True)

            self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.output_act = nn.Tanh()

        def encode(self, x):
            #print(len(x[0]), len(x[0][0]))
            x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
            #print(len(x[0]), len(x[0][0]))
            x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
            #print(len(x[0]), len(x[0][0]))
            x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
            #print(len(x[0]), len(x[0][0]))
            rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
            #print(len(rep[0]))
            return rep

        def decode(self, rep):
            #print("Decode")
            #print(len(rep[0]))
            x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
            #print(len(x[0]))
            x = x.view(-1, self.nf * 4, 4, 4)
            #print(len(x[0]), len(x[0][0]))
            x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
            #print(len(x[0]), len(x[0][0]))
            x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
            #print(len(x[0]), len(x[0][0]))
            x = self.output_act(self.dec_conv3(x))
            #print(len(x[0]), len(x[0][0]))
            return x

        def forward(self, x):
            return self.decode(self.encode(x))

def complete_run(file_prefix, label=9, p=0.05, train_loss_percent=0.5, num_epochs=30):
    transform = transforms.ToTensor()

    # Download and load the MNIST training data
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    selected_label = label

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
    train_loader_without_batch = torch.utils.data.DataLoader(filtered_dataset, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    model = CAE_pytorch(in_channels=1)

    criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(),eps=1e-7, weight_decay=0.0005)

    # Point to training loop video
    num_epochs = 30
    outputs = []
    losses = []
    for epoch in range(num_epochs):
        imgs = []
        for (img, _) in train_loader:
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

        losses.append(loss)
        # print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, img, recon))
        
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
    
    def calculate_losses(loader):
        normal = []
        anomalie = []
        anomalie_label = []
        for (img, label) in loader:
            recon = model(img)
            for i in range(0, len(img)):
                loss = criterion(recon[i], img[i])
                if label[i].item() == selected_label:
                    normal.append(loss)
                else:
                    anomalie.append(loss)
                    anomalie_label.append(label[i].item())
                
        normal.sort()
        anomalie.sort()

        normal = [round(tensor.tolist(),4) for tensor in normal]
        anomalie = [round(tensor.tolist(),4) for tensor in anomalie]
        return normal, anomalie, anomalie_label

    normal_train, anomalie_train, anomalie_label_train = calculate_losses(train_loader)
    normal_test, anomalie_test, anomalie_label_test = calculate_losses(test_loader)

    min_loss = min((min(normal_train),min(anomalie_train)))
    max_loss = max((max(normal_train),max(anomalie_train)))
    print(min_loss, max_loss)

    
    bins = 20
    plt.hist(normal_train, alpha=0.5, bins=bins, label='Data 1', edgecolor='black')
    plt.hist(anomalie_train, alpha=0.5, bins=bins, label='Data 2', edgecolor='black')

    plt.savefig(file_prefix+"-hist-train.png")
    plt.close()

    plt.hist(normal_test, alpha=0.5, bins=bins, label='Data 1', edgecolor='black')
    plt.hist(anomalie_test, alpha=0.5, bins=bins, label='Data 2', edgecolor='black')
    
    plt.savefig(file_prefix+"-hist-test.png")
    plt.close()
    
    def display_roc(t,p):
        fpr, tpr, _ = metrics.roc_curve(t, p)
        roc_auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        for loss in losses:
            l = loss.item()
            
            # normalize
            l = (l-min_loss) / (max_loss - min_loss)
            
            if l <= 1:
                plt.plot([l,l],(0,1), linestyle='-', color="tab:red")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        print(roc_auc)
        return roc_auc
    
    t = [0] * len(normal_train) + [1] * len(anomalie_train)
    p = normal_train + anomalie_train
    roc_train = display_roc(t,p)
    df = pd.DataFrame({'Loss': p, 
                       'Label': [label]*len(normal_train) + anomalie_label_train}).sort_values(by="Loss")
    df.to_csv(file_prefix+"train-loss.csv")

    t = [0] * len(normal_test) + [1] * len(anomalie_test)
    p = normal_test + anomalie_test
    roc_test = display_roc(t,p)
    df = pd.DataFrame({'Loss': p, 
                       'Label': [label]*len(normal_test) + anomalie_label_test}).sort_values(by="Loss")
    df.to_csv(file_prefix+"test-loss.csv")
    plt.savefig(file_prefix+"-roc.png")
    plt.close()
    
    
    
    return roc_train, roc_test


if __name__ == "__main__":
    prefix_num = 4
    num_epochs = 30
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
                roc_train, roc_test = complete_run(file_prefix=file_prefix, label=i, p=p,train_loss_percent=train_loss_percent, num_epochs=num_epochs)
                
                df_train.at[i, str(train_loss_percent)] = roc_train
                df_train.to_csv(train_file_name)
                df_test.at[i, str(train_loss_percent)] = roc_test
                df_test.to_csv(test_file_name)

    # Save threshold data