import torch
import torch.nn as nn

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

    """def encode(self, x):
        #print(len(x[0]), len(x[0][0]))
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        #print(len(x[0]), len(x[0][0]))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        #print(len(x[0]), len(x[0][0]))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        #print(len(x[0]), len(x[0][0]))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        #print(len(rep[0]))
        return rep"""
        
    def encode(self, x):
        #print(len(x[0]), len(x[0][0]))
        x = self.enc_act1((self.enc_conv1(x)))
        #print(len(x[0]), len(x[0][0]))
        x = self.enc_act2((self.enc_conv2(x)))
        #print(len(x[0]), len(x[0][0]))
        x = self.enc_act3((self.enc_conv3(x)))
        #print(len(x[0]), len(x[0][0]))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        #print(len(rep[0]))
        return rep

    """def decode(self, rep):
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
        return x"""
        
    def decode(self, rep):
        #print("Decode")
        #print(len(rep[0]))
        x = self.dec_act0((self.dec_fc(rep)))
        #print(len(x[0]))
        x = x.view(-1, self.nf * 4, 4, 4)
        #print(len(x[0]), len(x[0][0]))
        x = self.dec_act1((self.dec_conv1(x)))
        #print(len(x[0]), len(x[0][0]))
        x = self.dec_act2((self.dec_conv2(x)))
        #print(len(x[0]), len(x[0][0]))
        x = self.output_act(self.dec_conv3(x))
        #print(len(x[0]), len(x[0][0]))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))