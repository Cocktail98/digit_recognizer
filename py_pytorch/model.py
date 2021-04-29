from __future__ import print_function, division

import torch.nn as nn


class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()
        # 28*28
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(5, 5), stride=(1, 1))
        self.act1 = nn.ReLU()
        # 24*24
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1))
        self.act2 = nn.ReLU()
        # 20*20
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.5)
        # 10*10
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1))
        self.act3 = nn.ReLU()
        # 6*6
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.5)
        # 3*3
        self.fla1 = nn.Flatten()
        # 1*576
        self.linear1 = nn.Linear(64 * 3 * 3, 512)
        self.act4 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        # 1*512
        self.linear2 = nn.Linear(512, 256)
        self.act5 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.5)
        # 1*256
        self.linear3 = nn.Linear(256, 64)
        self.act6 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.5)
        # 1*64
        self.linear4 = nn.Linear(64, 10)
        self.act7 = nn.Softmax(dim=1)

        # an affine operation: y = Wx + b
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.dropout1(self.pool1(x))
        x = self.act3(self.conv3(x))
        x = self.dropout2(self.pool2(x))
        x = self.fla1(x)
        x = self.dropout3(self.act4(self.linear1(x)))
        x = self.dropout4(self.act5(self.linear2(x)))
        x = self.dropout5(self.act6(self.linear3(x)))
        x = self.act7(self.linear4(x))
        return x
