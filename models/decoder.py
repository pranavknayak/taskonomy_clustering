from torch import outer
import torch.nn as nn

class ClassifierNet(nn.Module):
    def __init__(self, layer_sizes) -> None:
        super(ClassifierNet, self).__init__()
        self.layers = nn.ModuleList([nn.Flatten()])
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i != len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Softmax(dim=1))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder3c(nn.Module):
    def __init__(self):
        super(Decoder3c, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=100352, out_features=8000)
        self.linear2 = nn.Linear(in_features=8000, out_features=8192)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_layers = nn.ModuleList([
            self.conv1,
            self.conv2,
            self.deconv1,
            self.conv3,
            self.deconv2,
            self.conv4,
            self.deconv3,
            self.conv5,
            self.deconv4,
            self.conv6,
            self.deconv5,
            self.conv7,
            self.deconv6,
            self.conv8
        ])


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = x.view(-1, 512, 4, 4)
        for idx, layer in enumerate(self.conv_layers):
            x = layer(x)
            if idx != len(self.conv_layers) - 1:
                x = self.relu(x)
            if idx != 1 and idx % 2 == 1:
                x = self.dropout(x)
        x = self.tanh(x)
        return x


class Decoder2c(nn.Module):
    def __init__(self) -> None:
        super(Decoder2c, self).__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=100352, out_features=8000)
        self.linear2 = nn.Linear(in_features=8000, out_features=8192)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_layers = nn.ModuleList([
            self.conv1,
            self.conv2,
            self.deconv1,
            self.conv3,
            self.deconv2,
            self.conv4,
            self.deconv3,
            self.conv5,
            self.deconv4,
            self.conv6,
            self.deconv5,
            self.conv7,
            self.deconv6,
            self.conv8
        ])

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = x.view(-1, 512, 4, 4)
        for idx, layer in enumerate(self.conv_layers):
            x = layer(x)
            if idx != len(self.conv_layers) - 1:
                x = self.relu(x)
            if idx != 1 and idx % 2 == 1:
                x = self.dropout(x)
        x = self.tanh(x)
        return x


class Decoder1c(nn.Module):
    def __init__(self) -> None:
        super(Decoder1c, self).__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=100352, out_features=8000)
        self.linear2 = nn.Linear(in_features=8000, out_features=8192)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_layers = nn.ModuleList([
            self.conv1,
            self.conv2,
            self.deconv1,
            self.conv3,
            self.deconv2,
            self.conv4,
            self.deconv3,
            self.conv5,
            self.deconv4,
            self.conv6,
            self.deconv5,
            self.conv7,
            self.deconv6,
            self.conv8
        ])

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = x.view(-1, 512, 4, 4)
        for idx, layer in enumerate(self.conv_layers):
            x = layer(x)
            if idx != len(self.conv_layers) - 1:
                x = self.relu(x)
            if idx != 1 and idx % 2 == 1:
                x = self.dropout(x)
        x = self.tanh(x)
        return x


class Decoder64c(nn.Module):
    def __init__(self) -> None:
        super(Decoder64c, self).__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=100352, out_features=8000)
        self.linear2 = nn.Linear(in_features=8000, out_features=8192)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_layers = nn.ModuleList([
            self.conv1,
            self.conv2,
            self.deconv1,
            self.conv3,
            self.deconv2,
            self.conv4,
            self.deconv3,
            self.conv5,
            self.deconv4,
            self.conv6,
            self.deconv5,
            self.conv7,
            self.deconv6,
            self.conv8
        ])

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = x.view(-1, 512, 4, 4)
        for idx, layer in enumerate(self.conv_layers):
            x = layer(x)
            if idx != len(self.conv_layers) - 1:
                x = self.relu(x)
            if idx != 1 and idx % 2 == 1:
                x = self.dropout(x)
        x = self.tanh(x)
        return x


class Decoder17c(nn.Module):
    def __init__(self) -> None:
        super(Decoder17c, self).__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=100352, out_features=8000)
        self.linear2 = nn.Linear(in_features=8000, out_features=8192)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_layers = nn.ModuleList([
            self.conv1,
            self.conv2,
            self.deconv1,
            self.conv3,
            self.deconv2,
            self.conv4,
            self.deconv3,
            self.conv5,
            self.deconv4,
            self.conv6,
            self.deconv5,
            self.conv7,
            self.deconv6,
            self.conv8
        ])

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = x.view(-1, 512, 4, 4)
        for idx, layer in enumerate(self.conv_layers):
            x = layer(x)
            if idx != len(self.conv_layers) - 1:
                x = self.relu(x)
            if idx != 1 and idx % 2 == 1:
                x = self.dropout(x)
        x = self.tanh(x)
        return x

