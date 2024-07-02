import torch
import torch.nn as nn

class CAE_28(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super(CAE_28, self).__init__()
        nf = 64
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
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

        
    def decode(self, rep):
        x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        x = x.view(-1, self.nf * 4, 4, 4)
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class CAE_32(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super(CAE_32, self).__init__()
        nf = 64
        self.nf = nf

        #TODO Remove batchnorm,
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

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act1 = nn.ReLU(inplace=True)

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf)
        self.dec_act2 = nn.ReLU(inplace=True)

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_act1(self.enc_bn1(self.enc_conv1(x)))
        x = self.enc_act2(self.enc_bn2(self.enc_conv2(x)))
        x = self.enc_act3(self.enc_bn3(self.enc_conv3(x)))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

    def decode(self, rep):
        if rep.shape[0] == 1:
            x = self.dec_act0((self.dec_fc(rep)))
        else:
            x = self.dec_act0(self.dec_bn0(self.dec_fc(rep)))
        x = x.view(-1, self.nf * 4, 4, 4)
        x = self.dec_act1(self.dec_bn1(self.dec_conv1(x)))
        x = self.dec_act2(self.dec_bn2(self.dec_conv2(x)))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class CAE_drop_28(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super(CAE_drop_28, self).__init__()
        nf = 64
        self.nf = nf

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
        self.enc_act1 = nn.ReLU(inplace=True)
        self.enc_drop1 = nn.Dropout()

        self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.enc_act2 = nn.ReLU(inplace=True)
        self.enc_drop2 = nn.Dropout()

        self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
        self.enc_act3 = nn.ReLU(inplace=True)
        self.enc_drop3 = nn.Dropout()

        self.enc_fc = nn.Linear(nf * 4 * 4 * 4, rep_dim)
        self.rep_act = nn.Tanh()

        # Decoder
        self.dec_fc = nn.Linear(rep_dim, nf * 4 * 4 * 4)
        self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 4 * 4 *4)
        self.dec_act0 = nn.ReLU(inplace=True)
        self.dec_drop1 = nn.Dropout()

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act1 = nn.ReLU(inplace=True)
        self.dec_drop2 = nn.Dropout()

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf)
        self.dec_act2 = nn.ReLU(inplace=True)
        self.dec_drop3 = nn.Dropout()

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_drop1(self.enc_act1(self.enc_bn1(self.enc_conv1(x))))
        x = self.enc_drop2(self.enc_act2(self.enc_bn2(self.enc_conv2(x))))
        x = self.enc_drop3(self.enc_act3(self.enc_bn3(self.enc_conv3(x))))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

    def decode(self, rep):
        x = self.dec_drop1(self.dec_act0(self.dec_bn0(self.dec_fc(rep))))
        x = x.view(-1, self.nf * 4, 4, 4)
        x = self.dec_drop2(self.dec_act1(self.dec_bn1(self.dec_conv1(x))))
        x = self.dec_drop3(self.dec_act2(self.dec_bn2(self.dec_conv2(x))))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


class CAE_drop_32(nn.Module):
    def __init__(self, in_channels = 3, rep_dim = 256):
        super(CAE_drop_32, self).__init__()
        nf = 64
        self.nf = nf

        #TODO Remove batchnorm,
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(num_features=nf)
        self.enc_act1 = nn.ReLU(inplace=True)
        self.enc_drop1 = nn.Dropout()

        self.enc_conv2 = nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(num_features=nf * 2)
        self.enc_act2 = nn.ReLU(inplace=True)
        self.enc_drop2 = nn.Dropout()

        self.enc_conv3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(num_features=nf * 4)
        self.enc_act3 = nn.ReLU(inplace=True)
        self.enc_drop3 = nn.Dropout()

        self.enc_fc = nn.Linear(nf * 4 * 4 * 4, rep_dim)
        self.rep_act = nn.Tanh()

        # Decoder
        self.dec_fc = nn.Linear(rep_dim, nf * 4 * 4 * 4)
        self.dec_bn0 = nn.BatchNorm1d(num_features=nf * 4 * 4 *4)
        self.dec_act0 = nn.ReLU(inplace=True)
        self.dec_drop1 = nn.Dropout()

        self.dec_conv1 = nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm2d(num_features=nf * 2)
        self.dec_act1 = nn.ReLU(inplace=True)
        self.dec_drop2 = nn.Dropout()

        self.dec_conv2 = nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_features=nf)
        self.dec_act2 = nn.ReLU(inplace=True)
        self.dec_drop3 = nn.Dropout()

        self.dec_conv3 = nn.ConvTranspose2d(in_channels=nf, out_channels=in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output_act = nn.Tanh()

    def encode(self, x):
        x = self.enc_drop1(self.enc_act1(self.enc_bn1(self.enc_conv1(x))))
        x = self.enc_drop2(self.enc_act2(self.enc_bn2(self.enc_conv2(x))))
        x = self.enc_drop3(self.enc_act3(self.enc_bn3(self.enc_conv3(x))))
        rep = self.rep_act(self.enc_fc(x.view(x.size(0), -1)))
        return rep

    def decode(self, rep):
        if rep.shape[0] == 1:
            x = self.dec_drop1(self.dec_act0((self.dec_fc(rep))))
        else:
            x = self.dec_drop1(self.dec_act0(self.dec_bn0(self.dec_fc(rep))))
        x = x.view(-1, self.nf * 4, 4, 4)
        x = self.dec_drop2(self.dec_act1(self.dec_bn1(self.dec_conv1(x))))
        x = self.dec_drop3(self.dec_act2(self.dec_bn2(self.dec_conv2(x))))
        x = self.output_act(self.dec_conv3(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
