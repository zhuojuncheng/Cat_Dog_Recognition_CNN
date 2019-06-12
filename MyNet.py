import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu_1 = nn.ReLU()
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2d_2 = nn.Conv2d(16, 32, 3)
        self.relu_2 = nn.ReLU()
        self.maxpool2d_2 = nn.MaxPool2d(2, 2)

        self.conv2d_3 = nn.Conv2d(32, 16, 3)
        self.relu_3 = nn.ReLU()

        self.linear_1 = nn.Linear(21*21*16, 120)
        self.relu_4 = nn.ReLU()

        self.linear_2 = nn.Linear(120, 60)
        self.relu_5 = nn.ReLU()

        self.linear_3 = nn.Linear(60, 2)

    def forward(self, x_input):
        x_conv2d_1 = self.conv2d_1(x_input)
        x_relu_1 = self.relu_1(x_conv2d_1)
        x_maxpool2d_1 = self.maxpool2d_1(x_relu_1)

        x_conv2d_2 = self.conv2d_2(x_maxpool2d_1)
        x_relu_2 = self.relu_2(x_conv2d_2)
        x_maxpool2d_2 = self.maxpool2d_2(x_relu_2)

        x_conv2d_3 = self.conv2d_3(x_maxpool2d_2)
        x_relu_3 = self.relu_3(x_conv2d_3)

        x_reshape = torch.reshape(x_relu_3, shape=(-1, 21*21*16))
        x_linear_1 = self.linear_1(x_reshape)
        x_relu_4 = self.relu_4(x_linear_1)

        x_linear_2 = self.linear_2(x_relu_4)
        x_relu_5 = self.relu_5(x_linear_2)

        x_linear_3 = self.linear_3(x_relu_5)
        return x_linear_3

