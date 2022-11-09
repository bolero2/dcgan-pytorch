import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 1, kernel_size=4, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("disc x1 shape :", x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print("disc x2 shape :", x.shape)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print("disc x3 shape :", x.shape)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        # print("disc x4 shape :", x.shape)

        x = self.conv5(x)
        # print("disc x5 shape :", x.shape)

        x = self.sigmoid(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.sampling = nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, padding=0)
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.sampling(x)
        # print("gen x1 shape :", x.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("gen x2 shape :", x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print("gen x3 shape :", x.shape)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print("gen x4 shape :", x.shape)

        x = self.conv4(x)
        x = self.bn4(x)
        # print("gen x5 shape :", x.shape)

        x = self.tanh(x)

        return x
